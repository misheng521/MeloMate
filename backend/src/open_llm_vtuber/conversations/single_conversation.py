from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from typing import Callable
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    augment_text_with_screen_context,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    screen_vision: Optional[Dict[str, Any]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
    on_reply_started: Optional[Callable[[], None]] = None,
    on_workspace_work_started: Optional[Callable[[], None]] = None,
    on_workspace_work_completed: Optional[Callable[[], None]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here
    reply_started = False

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input. Multiple queued inputs are merged into one model turn.
        input_text = await process_queued_user_inputs(
            user_input, context.asr_engine, websocket_send
        )
        augmented_input_text = await augment_text_with_screen_context(
            input_text, images, screen_vision
        )
        if metadata and metadata.get("workspace_revision_candidate"):
            metadata["workspace_revision"] = looks_like_workspace_revision_text(
                input_text
            )

        if metadata and metadata.get("workspace_revision"):
            augmented_input_text = (
                "The user is giving a modification or guidance for the workspace item "
                "you just created or are creating. Treat this as a revision request for "
                "that workspace work, not as unrelated chat. Update the relevant workspace "
                "artifact before replying.\n"
                f"{augmented_input_text}"
            )

        # Create batch input
        batch_input = create_batch_input(
            input_text=augmented_input_text,
            images=None,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        if (
            not skip_history
            and context.agent_engine
            and hasattr(context.agent_engine, "set_system")
        ):
            refreshed_prompt = await context.construct_system_prompt(
                context.character_config.persona_prompt
            )
            context.agent_engine.set_system(refreshed_prompt)
            context.system_prompt = refreshed_prompt

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    if is_workspace_tool_status(output_item):
                        if output_item.get("status") == "running":
                            on_workspace_work_started and on_workspace_work_started()
                        elif output_item.get("status") in {"completed", "error"}:
                            on_workspace_work_completed and on_workspace_work_completed()

                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    if not reply_started:
                        reply_started = True
                        on_reply_started and on_reply_started()
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.get_current_tts_engine(),
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
            )
            logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)


async def process_queued_user_inputs(
    user_input: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
    asr_engine,
    websocket_send: WebSocketSend,
) -> str:
    if not isinstance(user_input, list):
        return await process_user_input(user_input, asr_engine, websocket_send)

    parts: List[str] = []
    for item in user_input:
        text = (await process_user_input(item, asr_engine, websocket_send)).strip()
        if text:
            parts.append(text)

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]

    joined = "\n".join(f"{index + 1}. {part}" for index, part in enumerate(parts))
    return (
        "The user sent several messages before you replied. "
        "Treat them as one combined request and answer the latest full intent.\n"
        f"{joined}"
    )


def is_workspace_tool_status(output_item: Dict[str, Any]) -> bool:
    tool_name = str(output_item.get("tool_name") or "")
    return tool_name in {
        "create_workspace_folder",
        "write_workspace_file",
        "read_workspace_file",
        "list_workspace",
        "schedule_reminder",
        "open_workspace_item",
    }


def looks_like_workspace_revision_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False

    revision_keywords = (
        "改",
        "修改",
        "换成",
        "换为",
        "变成",
        "做成",
        "加",
        "加上",
        "增加",
        "删",
        "删除",
        "去掉",
        "不要",
        "颜色",
        "风格",
        "可爱",
        "酷",
        "简单点",
        "复杂点",
        "再",
        "也要",
        "还要",
        "按钮",
        "计分",
        "关卡",
        "背景",
        "音效",
        "动画",
        "样式",
        "布局",
        "字体",
        "rewrite",
        "revise",
        "change",
        "make it",
        "add",
        "remove",
        "style",
        "color",
    )
    ordinary_chat_keywords = (
        "在吗",
        "算了",
        "等等",
        "等一下",
        "先别",
        "不用了",
        "我有点",
        "我想你",
        "吃饭",
        "睡觉",
    )

    if any(keyword in normalized for keyword in revision_keywords):
        return True
    if any(keyword in normalized for keyword in ordinary_chat_keywords):
        return False
    return False
