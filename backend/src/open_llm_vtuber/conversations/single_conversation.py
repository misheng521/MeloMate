from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from typing import Callable
from loguru import logger
import numpy as np
import workspace_core

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    augment_text_with_screen_context,
    with_turn_id,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..proactive_conversation import build_return_context_prompt
from ..service_context import ServiceContext
from ..workspace_intent import workspace_live_page_relevant

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


def _attach_live_workspace_context(
    context: ServiceContext,
    input_text: str,
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    next_metadata = dict(metadata or {})
    next_metadata.pop("workspace_awareness", None)
    character = context.character_config
    persona = character.character_name or character.conf_name
    session = context.workspace_agent
    policy = session.begin_user_turn(input_text, persona)
    next_metadata["workspace_tool_policy"] = policy
    if workspace_live_page_relevant(
        input_text, policy.get("user_authorized_workspace_tools")
    ):
        try:
            current = json.loads(workspace_core.read_workspace_state(persona))
            state_file = current.get("state") if isinstance(current, dict) else None
            report = state_file.get("state") if isinstance(state_file, dict) else None
            page = report.get("page") if isinstance(report, dict) else None
            if (
                isinstance(current, dict)
                and current.get("available") is True
                and isinstance(page, dict)
                and page.get("id")
            ):
                session.observe_page({
                    "page": page,
                    "state_version": max(0, int(report.get("state_version") or 0)),
                    "appState": report.get("appState"),
                    "created_ms": int(state_file.get("updated_ms") or 0),
                })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    awareness = session.awareness_for_turn(policy)
    if awareness and workspace_live_page_relevant(
        input_text, policy.get("user_authorized_workspace_tools")
    ):
        next_metadata["workspace_awareness"] = awareness
        allowed = frozenset(policy.get("available_workspace_tools") or ())
        policy.update(
            {
                "enforce": True,
                "filter_workspace_tools": False,
                "allowed_tool_names": allowed,
                "remaining_tool_calls": {name: 64 for name in allowed},
                "workspace_state_tainted": True,
            }
        )
    return next_metadata


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
    input_ids: Optional[List[Optional[str]]] = None,
    queued_items: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    transcription_cache: Optional[Dict[str, str]] = None,
    announced_transcription_ids: Optional[set[str]] = None,
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
    websocket_send_with_turn = with_turn_id(websocket_send, turn_id)

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send_with_turn)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input. Multiple queued inputs are merged into one model turn.
        input_text = await process_queued_user_inputs(
            user_input,
            context.asr_engine,
            websocket_send_with_turn,
            input_ids=input_ids,
            queued_items=queued_items,
            transcription_cache=transcription_cache,
            announced_transcription_ids=announced_transcription_ids,
        )
        metadata = dict(metadata or {})
        augmented_input_text = await augment_text_with_screen_context(
            input_text,
            images,
            screen_vision,
            force=bool(
                metadata.get("proactive_speak") and images and screen_vision
            ),
        )
        return_context = metadata.pop("proactive_return", None)
        trusted_return_utterances = (
            return_context.get("recent_utterances")
            if isinstance(return_context, dict)
            else None
        )
        return_context_prompt = build_return_context_prompt(
            return_context,
            trusted_recent_utterances=trusted_return_utterances,
        )
        if return_context_prompt:
            augmented_input_text = (
                f"{return_context_prompt}\n\n"
                "[用户本次真正说的话]\n"
                f"{augmented_input_text}"
            )
        metadata.pop("workspace_event", None)
        metadata.pop("workspace_event_data", None)
        metadata["workspace_persona"] = str(
            context.character_config.character_name
            or context.character_config.conf_name
            or ""
        )

        if metadata.get("workspace_revision_candidate"):
            metadata["workspace_revision"] = looks_like_workspace_revision_text(
                input_text
            )

        if metadata.get("workspace_revision"):
            augmented_input_text = (
                "The user is giving a modification or guidance for the workspace item "
                "you just created or are creating. Treat this as a revision request for "
                "that workspace work, not as unrelated chat. Update the relevant workspace "
                "artifact before replying.\n"
                f"{augmented_input_text}"
            )

        metadata = _attach_live_workspace_context(
            context, input_text, metadata
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
                context.character_config.persona_prompt,
                current_user_text=input_text,
            )
            context.agent_engine.set_system(refreshed_prompt)
            context.system_prompt = refreshed_prompt

        logger.info(f"User input received (chars={len(input_text)})")
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
                    logger.debug(
                        "Sending tool status update "
                        f"(tool={output_item.get('tool_name')}, "
                        f"status={output_item.get('status')})"
                    )

                    await websocket_send_with_turn(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    if not reply_started:
                        reply_started = True
                        on_reply_started and on_reply_started()
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        avatar_model=context.avatar_model,
                        tts_engine=context.get_current_tts_engine(),
                        websocket_send=websocket_send_with_turn,  # Pass websocket_send for audio/tts messages
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

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send_with_turn(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send_with_turn,
            client_uid=client_uid,
        )

        if (
            full_response
            and metadata.get("proactive_speak")
            and metadata.get("proactive_mode") == "automatic"
        ):
            # Retain only a short per-client window for repetition avoidance and
            # the user's one-shot return reaction.  It is never chat history or
            # long-term character memory.
            context.proactive_utterances.append(full_response.strip()[:180])
            context.proactive_utterances[:] = context.proactive_utterances[-5:]

        if context.history_uid and full_response and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
            )
            logger.info(f"AI response completed (chars={len(full_response)})")
            schedule_memory_review = getattr(
                context.agent_engine, "schedule_core_memory_review", None
            )
            if callable(schedule_memory_review):
                schedule_memory_review()

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send_with_turn(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        await cleanup_conversation(tts_manager, session_emoji)


async def process_workspace_agent_turn(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    runtime: Dict[str, Any],
    turn_id: str,
) -> Dict[str, Any]:
    """Run a page event through the same agent stream used by user conversations."""
    run_workspace_turn = getattr(context.agent_engine, "run_workspace_turn", None)
    if not callable(run_workspace_turn):
        return {"acted": False, "response": ""}

    manager = TTSTaskManager()
    send = with_turn_id(websocket_send, turn_id)
    full_response = ""
    acted = False
    try:
        await send_conversation_start_signals(send)
        async for output_item in run_workspace_turn(runtime):
            if (
                isinstance(output_item, dict)
                and output_item.get("type") == "tool_call_status"
            ):
                output_item["name"] = context.character_config.character_name
                if (
                    output_item.get("tool_name") == "act_workspace_page"
                    and output_item.get("status") == "completed"
                ):
                    try:
                        action_result = json.loads(
                            str(output_item.get("content") or "")
                        )
                    except (json.JSONDecodeError, TypeError):
                        action_result = {}
                    acted = (
                        isinstance(action_result, dict)
                        and action_result.get("confirmed") is True
                    )
                await send(json.dumps(output_item))
            elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                response_part = await process_agent_output(
                    output=output_item,
                    character_config=context.character_config,
                    avatar_model=context.avatar_model,
                    tts_engine=context.get_current_tts_engine(),
                    websocket_send=send,
                    tts_manager=manager,
                    translate_engine=context.translate_engine,
                )
                full_response += str(response_part or "")
            else:
                logger.warning(
                    "Unexpected workspace agent stream item: {}", type(output_item)
                )
        await finalize_conversation_turn(manager, send, client_uid)

        if full_response:
            add_external = getattr(
                context.agent_engine, "add_external_assistant_message", None
            )
            if callable(add_external):
                add_external(full_response)
            if context.history_uid:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=full_response,
                    name=context.character_config.character_name,
                )
        return {"acted": acted, "response": full_response}
    finally:
        await cleanup_conversation(manager, f"workspace-{turn_id}")


async def process_queued_user_inputs(
    user_input: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
    asr_engine,
    websocket_send: WebSocketSend,
    input_ids: Optional[List[Optional[str]]] = None,
    queued_items: Optional[List[Dict[str, Any]]] = None,
    transcription_cache: Optional[Dict[str, str]] = None,
    announced_transcription_ids: Optional[set[str]] = None,
) -> str:
    if not isinstance(user_input, list):
        return await process_queued_user_input_item(
            item=user_input,
            asr_engine=asr_engine,
            websocket_send=websocket_send,
            input_id=input_ids[0] if input_ids else None,
            queued_item=queued_items[0] if queued_items else None,
            transcription_cache=transcription_cache,
            announced_transcription_ids=announced_transcription_ids,
        )

    parts: List[str] = []
    for index, item in enumerate(user_input):
        text = (
            await process_queued_user_input_item(
                item=item,
                asr_engine=asr_engine,
                websocket_send=websocket_send,
                input_id=input_ids[index] if input_ids and index < len(input_ids) else None,
                queued_item=queued_items[index] if queued_items and index < len(queued_items) else None,
                transcription_cache=transcription_cache,
                announced_transcription_ids=announced_transcription_ids,
            )
        ).strip()
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


async def process_queued_user_input_item(
    item: Union[str, np.ndarray],
    asr_engine,
    websocket_send: WebSocketSend,
    input_id: Optional[str] = None,
    queued_item: Optional[Dict[str, Any]] = None,
    transcription_cache: Optional[Dict[str, str]] = None,
    announced_transcription_ids: Optional[set[str]] = None,
) -> str:
    if queued_item and not input_id:
        input_id = queued_item.get("input_id")
    if not input_id:
        input_id = audio_input_fingerprint(item)
        if queued_item is not None:
            queued_item["input_id"] = input_id

    cached_text = queued_item.get("transcription_text") if queued_item else None
    if input_id and transcription_cache and input_id in transcription_cache:
        input_text = transcription_cache[input_id]
        if queued_item is not None:
            queued_item["transcription_text"] = input_text
    elif isinstance(cached_text, str):
        input_text = cached_text
        if input_id and transcription_cache is not None:
            transcription_cache[input_id] = input_text
            trim_transcription_cache(transcription_cache)
    else:
        input_text = await process_user_input(
            item,
            asr_engine,
            websocket_send,
            announce_transcription=False,
        )
        if queued_item is not None:
            queued_item["transcription_text"] = input_text
        if input_id and transcription_cache is not None:
            transcription_cache[input_id] = input_text
            trim_transcription_cache(transcription_cache)

    already_announced = bool(
        (input_id and announced_transcription_ids and input_id in announced_transcription_ids)
        or (queued_item or {}).get("transcription_announced")
    )
    if isinstance(item, np.ndarray) and not already_announced:
        if queued_item is not None:
            queued_item["transcription_announced"] = True
        if input_id and announced_transcription_ids is not None:
            announced_transcription_ids.add(input_id)
            trim_announced_transcription_ids(announced_transcription_ids)
        await websocket_send(
            json.dumps(
                {
                    "type": "user-input-transcription",
                    "text": input_text,
                    "input_id": input_id,
                },
                ensure_ascii=False,
            )
        )

    return input_text


def audio_input_fingerprint(item: Union[str, np.ndarray]) -> Optional[str]:
    if isinstance(item, str):
        normalized = " ".join(item.strip().split())
        return f"text:{normalized}" if normalized else None
    if not isinstance(item, np.ndarray) or item.size == 0:
        return None

    samples = item.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    duration = int(samples.size)
    head = samples[: min(128, samples.size)]
    tail = samples[max(0, samples.size - 128) :]
    checksum = int((float(np.sum(head)) * 1_000_000) + (float(np.sum(tail)) * 1_000_000))
    energy = int(float(np.sum(samples * samples)) * 1000)
    return f"audio:{duration}:{round(peak, 5)}:{energy}:{checksum}"


def trim_transcription_cache(transcription_cache: Dict[str, str], limit: int = 200) -> None:
    while len(transcription_cache) > limit:
        oldest_key = next(iter(transcription_cache), None)
        if oldest_key is None:
            return
        transcription_cache.pop(oldest_key, None)


def trim_announced_transcription_ids(announced_ids: set[str], limit: int = 200) -> None:
    while len(announced_ids) > limit:
        oldest_key = next(iter(announced_ids), None)
        if oldest_key is None:
            return
        announced_ids.discard(oldest_key)


def is_workspace_tool_status(output_item: Dict[str, Any]) -> bool:
    tool_name = str(output_item.get("tool_name") or "")
    return tool_name in {
        "create_workspace_folder",
        "write_workspace_file",
        "append_workspace_file",
        "write_workspace_project",
        "read_workspace_file",
        "list_workspace",
        "replace_workspace_text",
        "move_workspace_item",
        "delete_workspace_item",
        "search_workspace",
        "read_workspace_state",
        "open_workspace_item",
        "inspect_workspace_item",
        "read_workspace_file_range",
        "patch_workspace_file",
        "list_workspace_trash",
        "restore_workspace_item",
        "act_workspace_page",
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
