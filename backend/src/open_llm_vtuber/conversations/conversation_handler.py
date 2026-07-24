import asyncio
import json
from typing import Dict, Optional, Callable, Any, List

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    pending_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    in_flight_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    reply_started_flags: Dict[str, bool],
    workspace_work_flags: Dict[str, bool],
    workspace_revision_flags: Dict[str, bool],
    broadcast_to_group: Callable,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        user_input = str(data.get("text") or "").strip()
        if not user_input:
            try:
                # Get proactive speak prompt from config
                prompt_name = "proactive_speak_prompt"
                prompt_file = context.system_config.tool_prompts.get(prompt_name)
                if prompt_file:
                    user_input = prompt_loader.load_util(prompt_file)
                else:
                    logger.warning("Proactive speak prompt not configured, using default")
                    user_input = "Please say something."
            except Exception as e:
                logger.error(f"Error loading proactive speak prompt: {e}")
                user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")
    screen_vision = data.get("screen_vision")
    session_emoji = np.random.choice(EMOJI_LIST)
    queued_input = {
        "user_input": user_input,
        "images": images,
        "screen_vision": screen_vision,
        "metadata": metadata,
        "session_emoji": session_emoji,
    }

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    screen_vision=screen_vision,
                    session_emoji=session_emoji,
                    metadata=metadata,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        pending_queue = pending_conversation_inputs.setdefault(client_uid, [])
        active_task = current_conversation_tasks.get(client_uid)

        has_content = _queued_input_has_content(queued_input)
        if not has_content and not pending_queue and not in_flight_conversation_inputs.get(client_uid):
            logger.debug("Ignoring empty input with no pending conversation content.")
            return

        if active_task and not active_task.done():
            if workspace_work_flags.get(client_uid):
                if has_content:
                    if _looks_like_workspace_revision(queued_input):
                        queued_input["metadata"] = {
                            **(queued_input.get("metadata") or {}),
                            "workspace_revision": True,
                        }
                    pending_queue.append(queued_input)
                logger.info(
                    f"Queued user input for {client_uid}; workspace work is active."
                )
                return

            if reply_started_flags.get(client_uid):
                if has_content:
                    if workspace_revision_flags.get(client_uid) and _looks_like_workspace_revision(queued_input):
                        queued_input["metadata"] = {
                            **(queued_input.get("metadata") or {}),
                            "workspace_revision": True,
                        }
                    await handle_individual_interrupt(
                        client_uid=client_uid,
                        current_conversation_tasks=current_conversation_tasks,
                        context=context,
                        heard_response="",
                    )
                    await websocket.send_text(
                        json.dumps({"type": "interrupt-signal", "text": ""})
                    )
                    current_conversation_tasks.pop(client_uid, None)
                    pending_queue.append(queued_input)
                    has_content = False
                else:
                    logger.debug("Ignoring empty input while a reply is active.")
                    return
            else:
                if has_content:
                    merged_inputs = [
                        *in_flight_conversation_inputs.get(client_uid, []),
                        *pending_queue,
                        queued_input,
                    ]
                    pending_queue[:] = merged_inputs
                    in_flight_conversation_inputs.pop(client_uid, None)
                    active_task.cancel()
                    logger.info(
                        f"Restarting unreplied turn for {client_uid} with {len(merged_inputs)} merged input(s)."
                    )
                elif pending_queue:
                    logger.info(
                        f"Empty trigger received for {client_uid}; pending unreplied input will be processed."
                    )
                return

        if has_content:
            pending_queue.append(queued_input)

        active_task = current_conversation_tasks.get(client_uid)
        if active_task and not active_task.done():
            logger.info(
                f"Queued user input for {client_uid}; {len(pending_queue)} pending item(s)."
            )
            return

        current_conversation_tasks[client_uid] = asyncio.create_task(
            _drain_single_conversation_queue(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                current_conversation_tasks=current_conversation_tasks,
                pending_conversation_inputs=pending_conversation_inputs,
                in_flight_conversation_inputs=in_flight_conversation_inputs,
                reply_started_flags=reply_started_flags,
                workspace_work_flags=workspace_work_flags,
                workspace_revision_flags=workspace_revision_flags,
            )
        )


def _queued_input_has_content(item: Dict[str, Any]) -> bool:
    user_input = item.get("user_input")
    if isinstance(user_input, str):
        return bool(user_input.strip())
    if isinstance(user_input, np.ndarray):
        return user_input.size > 0
    return user_input is not None


def _looks_like_workspace_revision(item: Dict[str, Any]) -> bool:
    user_input = item.get("user_input")
    if isinstance(user_input, np.ndarray):
        metadata = item.setdefault("metadata", {})
        metadata["workspace_revision_candidate"] = True
        return False
    if not isinstance(user_input, str):
        return False

    text = user_input.strip().lower()
    if not text:
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
        "你",
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

    if any(keyword in text for keyword in revision_keywords):
        return True
    if any(keyword in text for keyword in ordinary_chat_keywords):
        return False
    return False


async def _drain_single_conversation_queue(
    context: ServiceContext,
    websocket_send: Callable,
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    pending_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    in_flight_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    reply_started_flags: Dict[str, bool],
    workspace_work_flags: Dict[str, bool],
    workspace_revision_flags: Dict[str, bool],
) -> None:
    """Process queued individual inputs serially, merging all unreplied inputs per turn."""
    try:
        while True:
            pending_queue = pending_conversation_inputs.setdefault(client_uid, [])
            if not pending_queue:
                return

            batch = pending_queue[:]
            pending_queue.clear()
            in_flight_conversation_inputs[client_uid] = batch
            reply_started_flags[client_uid] = False
            workspace_work_flags[client_uid] = False
            if any((item.get("metadata") or {}).get("workspace_revision") for item in batch):
                workspace_revision_flags[client_uid] = True

            user_inputs = [
                item["user_input"]
                for item in batch
                if _queued_input_has_content(item)
            ]
            if not user_inputs:
                continue

            latest = batch[-1]
            session_emoji = latest.get("session_emoji") or np.random.choice(EMOJI_LIST)
            metadata = _merge_metadata([item.get("metadata") for item in batch])

            logger.info(
                f"Processing {len(user_inputs)} queued input(s) for {client_uid} as one turn."
            )
            await process_single_conversation(
                context=context,
                websocket_send=websocket_send,
                client_uid=client_uid,
                user_input=user_inputs if len(user_inputs) > 1 else user_inputs[0],
                images=latest.get("images"),
                screen_vision=latest.get("screen_vision"),
                session_emoji=session_emoji,
                metadata=metadata,
                on_reply_started=lambda: _mark_reply_started(
                    client_uid, reply_started_flags
                ),
                on_workspace_work_started=lambda: _mark_workspace_work(
                    client_uid, workspace_work_flags, workspace_revision_flags, True
                ),
                on_workspace_work_completed=lambda: _mark_workspace_work(
                    client_uid, workspace_work_flags, workspace_revision_flags, False
                ),
            )
            in_flight_conversation_inputs.pop(client_uid, None)
    finally:
        current_conversation_tasks.pop(client_uid, None)
        in_flight_conversation_inputs.pop(client_uid, None)
        reply_started_flags.pop(client_uid, None)
        workspace_work_flags.pop(client_uid, None)
        if not pending_conversation_inputs.get(client_uid):
            workspace_revision_flags.pop(client_uid, None)
        if pending_conversation_inputs.get(client_uid):
            logger.info(
                f"Restarting conversation queue for {client_uid}; input arrived during drain shutdown."
            )
            current_conversation_tasks[client_uid] = asyncio.create_task(
                _drain_single_conversation_queue(
                    context=context,
                    websocket_send=websocket_send,
                    client_uid=client_uid,
                    current_conversation_tasks=current_conversation_tasks,
                    pending_conversation_inputs=pending_conversation_inputs,
                    in_flight_conversation_inputs=in_flight_conversation_inputs,
                    reply_started_flags=reply_started_flags,
                    workspace_work_flags=workspace_work_flags,
                    workspace_revision_flags=workspace_revision_flags,
                )
            )


def _merge_metadata(items: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged or None


def _mark_reply_started(client_uid: str, reply_started_flags: Dict[str, bool]) -> None:
    reply_started_flags[client_uid] = True


def _mark_workspace_work(
    client_uid: str,
    workspace_work_flags: Dict[str, bool],
    workspace_revision_flags: Dict[str, bool],
    active: bool,
) -> None:
    workspace_work_flags[client_uid] = active
    if active:
        workspace_revision_flags[client_uid] = True


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid and heard_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
            )
        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
