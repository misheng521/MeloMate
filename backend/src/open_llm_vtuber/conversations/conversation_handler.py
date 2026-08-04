import asyncio
import json
import uuid
from typing import Dict, Optional, Callable, Any, List

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_history_manager import store_message
from ..proactive_conversation import (
    build_proactive_prompt,
    normalize_proactive_request,
    sanitize_user_metadata,
)
from ..service_context import ServiceContext
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    pending_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    in_flight_conversation_inputs: Dict[str, List[Dict[str, Any]]],
    transcription_cache: Dict[str, Dict[str, str]],
    announced_transcription_ids: Dict[str, set[str]],
    reply_started_flags: Dict[str, bool],
    workspace_work_flags: Dict[str, bool],
    workspace_revision_flags: Dict[str, bool],
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        # Never trust a browser-supplied hidden prompt.  The client may only send
        # bounded state; the actual instruction is assembled by the server.
        proactive_request = normalize_proactive_request(data.get("proactive"))
        user_input = build_proactive_prompt(
            proactive_request,
            trusted_recent_utterances=context.proactive_utterances,
        )

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "proactive_mode": proactive_request["mode"],
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
        metadata = sanitize_user_metadata(
            data.get("metadata"), preserve_other=False
        )
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])
        # Audio metadata originates entirely in the browser.  Only the one
        # explicitly validated return context is accepted on this path.
        metadata = sanitize_user_metadata(
            data.get("metadata"), preserve_other=False
        )

    if msg_type != "ai-speak-signal":
        proactive_return = metadata.get("proactive_return") if metadata else None
        if isinstance(proactive_return, dict):
            proactive_return["recent_utterances"] = list(
                context.proactive_utterances[-5:]
            )
        # A real user message ends the current silence episode.  Keep this
        # state server-side and consume it even if the reply later fails.
        context.proactive_utterances.clear()

    images = data.get("images")
    screen_vision = data.get("screen_vision")
    session_emoji = np.random.choice(EMOJI_LIST)
    turn_id = data.get("turn_id") or f"server-{uuid.uuid4().hex}"
    queued_input = {
        "user_input": user_input,
        "input_id": data.get("input_id"),
        "turn_id": turn_id,
        "images": images,
        "screen_vision": screen_vision,
        "metadata": metadata,
        "session_emoji": session_emoji,
    }

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
            if not has_content:
                logger.debug("Ignoring empty input while a reply is active.")
                return

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
            pending_queue.append(queued_input)
        else:
            if has_content:
                merged_inputs = [
                    *in_flight_conversation_inputs.get(client_uid, []),
                    *pending_queue,
                    queued_input,
                ]
                pending_queue[:] = merged_inputs
                in_flight_conversation_inputs.pop(client_uid, None)
                await _cancel_conversation_task(
                    client_uid,
                    current_conversation_tasks,
                )
                logger.info(
                    f"Restarting unreplied turn for {client_uid} with {len(merged_inputs)} merged input(s)."
                )
            else:
                if pending_queue:
                    logger.info(
                        f"Empty trigger received for {client_uid}; pending unreplied input will be processed."
                    )
                return
    elif has_content:
        pending_queue.append(queued_input)

    if not pending_queue:
        return

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
            transcription_cache=transcription_cache,
            announced_transcription_ids=announced_transcription_ids,
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
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata
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
    transcription_cache: Dict[str, Dict[str, str]],
    announced_transcription_ids: Dict[str, set[str]],
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

            batch = list(pending_queue)
            pending_queue.clear()
            in_flight_conversation_inputs[client_uid] = batch
            reply_started_flags[client_uid] = False
            workspace_work_flags[client_uid] = False
            if any((item.get("metadata") or {}).get("workspace_revision") for item in batch):
                workspace_revision_flags[client_uid] = True

            content_items = [item for item in batch if _queued_input_has_content(item)]
            user_inputs = [item["user_input"] for item in content_items]
            input_ids = [item.get("input_id") for item in content_items]
            if not user_inputs:
                continue

            latest = batch[-1]
            session_emoji = latest.get("session_emoji") or np.random.choice(EMOJI_LIST)
            turn_id = latest.get("turn_id")
            metadata = _merge_metadata([item.get("metadata") for item in batch])

            logger.info(
                f"Processing {len(user_inputs)} queued input(s) for {client_uid} as one turn."
            )
            await process_single_conversation(
                context=context,
                websocket_send=websocket_send,
                client_uid=client_uid,
                user_input=user_inputs if len(user_inputs) > 1 else user_inputs[0],
                input_ids=input_ids if len(user_inputs) > 1 else input_ids[:1],
                queued_items=content_items,
                turn_id=turn_id,
                transcription_cache=transcription_cache.setdefault(client_uid, {}),
                announced_transcription_ids=announced_transcription_ids.setdefault(client_uid, set()),
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
        current_task = asyncio.current_task()
        if current_conversation_tasks.get(client_uid) is current_task:
            current_conversation_tasks.pop(client_uid, None)
            in_flight_conversation_inputs.pop(client_uid, None)
            reply_started_flags.pop(client_uid, None)
            workspace_work_flags.pop(client_uid, None)
            if not pending_conversation_inputs.get(client_uid):
                workspace_revision_flags.pop(client_uid, None)


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
    task = current_conversation_tasks.get(client_uid)
    if task:
        await _cancel_conversation_task(client_uid, current_conversation_tasks)
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


async def _cancel_conversation_task(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
) -> Optional[asyncio.Task]:
    """Cancel the task currently owned by a client and wait for its finalizer."""
    task = current_conversation_tasks.get(client_uid)
    if not task:
        return None

    if task is asyncio.current_task():
        raise RuntimeError("A conversation worker cannot cancel itself")

    if not task.done():
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning(
            f"Conversation task for {client_uid} failed while being stopped: {exc}"
        )

    if current_conversation_tasks.get(client_uid) is task:
        current_conversation_tasks.pop(client_uid, None)
    return task
