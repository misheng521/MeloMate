from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import base64
import binascii
from collections import deque
import io
import json
import shutil
import time
from enum import Enum
from pathlib import Path
import numpy as np
import soundfile as sf
from loguru import logger

from .service_context import ServiceContext
from .message_handler import message_handler
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .utils.optional_dependencies import (
    missing_voice_clone_dependencies,
    voice_clone_dependencies_available,
)
from .conversations.conversation_handler import (
    _cancel_conversation_task,
    handle_conversation_trigger,
    handle_individual_interrupt,
)
from .workspace_controller import WorkspaceController
from .workspace_security import normalize_workspace_event
from .secure_credentials import (
    CHAT_API_KEY,
    SCREEN_VISION_API_KEY,
    SecureCredentialError,
    SecureCredentialStore,
    validate_profile_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOICE_CLONE_REFERENCE_ROOT = (
    PROJECT_ROOT / "reference_sounds" / "voice_clone_refs"
)
MAX_VOICE_CLONE_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_VOICE_CLONE_REFERENCE_BASE64_LENGTH = (
    (MAX_VOICE_CLONE_REFERENCE_BYTES * 4 // 3) + 8
)
MAX_VOICE_CLONE_DATA_URL_PREFIX_LENGTH = 128
MIN_VOICE_CLONE_DURATION_SECONDS = 2.5
MAX_VOICE_CLONE_DURATION_SECONDS = 10.5
MIN_VOICE_CLONE_SAMPLE_RATE = 8_000
MAX_VOICE_CLONE_SAMPLE_RATE = 96_000
MAX_VOICE_CLONE_CHANNELS = 2
VOICE_CLONE_UPLOADS_PER_MINUTE = 6
VOICE_CLONE_ALLOWED_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg"})
VOICE_CLONE_ALLOWED_CONTAINERS = frozenset({"WAV", "WAVEX", "FLAC", "OGG", "MP3"})


def validate_and_normalize_voice_clone_reference(
    raw_audio: bytes, target: Path
) -> dict[str, float | int | str]:
    """Decode a bounded reference once and store only canonical mono PCM WAV."""
    if not raw_audio or len(raw_audio) > MAX_VOICE_CLONE_REFERENCE_BYTES:
        raise ValueError("参考音频为空或超过 10 MB。")

    try:
        with sf.SoundFile(io.BytesIO(raw_audio)) as source:
            container = str(source.format or "").upper()
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
            declared_frames = int(source.frames)
            if container not in VOICE_CLONE_ALLOWED_CONTAINERS:
                raise ValueError("音频容器不受支持，请使用 WAV、MP3、FLAC 或 OGG。")
            if not MIN_VOICE_CLONE_SAMPLE_RATE <= sample_rate <= MAX_VOICE_CLONE_SAMPLE_RATE:
                raise ValueError("参考音频采样率必须在 8 kHz 到 96 kHz 之间。")
            if channels < 1 or channels > MAX_VOICE_CLONE_CHANNELS:
                raise ValueError("参考音频只能是单声道或双声道。")
            if declared_frames <= 0:
                raise ValueError("参考音频没有可解码的采样。")
            declared_duration = declared_frames / sample_rate
            if not MIN_VOICE_CLONE_DURATION_SECONDS <= declared_duration <= MAX_VOICE_CLONE_DURATION_SECONDS:
                raise ValueError("参考音频时长必须为 3-10 秒。")

            frame_limit = int(MAX_VOICE_CLONE_DURATION_SECONDS * sample_rate) + 1
            decoded = source.read(
                frames=frame_limit,
                dtype="float32",
                always_2d=True,
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("参考音频无法安全解码或文件内容与格式不符。") from exc

    actual_frames = int(decoded.shape[0])
    actual_duration = actual_frames / sample_rate
    if actual_frames != declared_frames or not (
        MIN_VOICE_CLONE_DURATION_SECONDS
        <= actual_duration
        <= MAX_VOICE_CLONE_DURATION_SECONDS
    ):
        raise ValueError("参考音频时长或文件结构不一致。")
    if not np.isfinite(decoded).all():
        raise ValueError("参考音频包含无效采样。")

    mono = np.mean(decoded, axis=1, dtype=np.float32)
    peak = float(np.max(np.abs(mono), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if peak < 1e-4 or rms < 1e-5:
        raise ValueError("参考音频几乎没有可用声音，请重新选择清晰的人声。")

    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), mono, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "container": container,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": actual_duration,
    }


class MessageType(Enum):
    """Enum for WebSocket message types"""

    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    screen_vision: Optional[dict]
    history_uid: Optional[str]
    file: Optional[str]
    path: Optional[str]
    display_text: Optional[dict]
    api_base_url: Optional[str]
    api_key: Optional[str]
    screen_vision_api_key: Optional[str]
    model: Optional[str]
    credential_profile_id: Optional[str]
    credential: Optional[str]
    enabled: Optional[bool]
    audio_base64: Optional[str]
    file_name: Optional[str]
    ref_text: Optional[str]
    language: Optional[str]
    request_id: Optional[str]
    metadata: Optional[dict]
    event: Optional[dict]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.pending_conversation_inputs: Dict[str, list[dict]] = {}
        self.in_flight_conversation_inputs: Dict[str, list[dict]] = {}
        self.transcription_cache: Dict[str, dict[str, str]] = {}
        self.announced_transcription_ids: Dict[str, set[str]] = {}
        self.reply_started_flags: Dict[str, bool] = {}
        self.workspace_work_flags: Dict[str, bool] = {}
        self.workspace_revision_flags: Dict[str, bool] = {}
        self.conversation_locks: Dict[str, asyncio.Lock] = {}
        self.client_cleanup_tasks: Dict[str, asyncio.Task] = {}
        self.voice_clone_reference_dirs: Dict[str, Path] = {}
        self.voice_clone_upload_times: Dict[str, deque[float]] = {}
        self.workspace_controllers: Dict[str, WorkspaceController] = {}
        self.credential_store = SecureCredentialStore()
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self._remove_stale_voice_clone_references()

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    @staticmethod
    def _remove_stale_voice_clone_references() -> None:
        """Clear crash leftovers before any client session is accepted."""
        if VOICE_CLONE_REFERENCE_ROOT.is_symlink():
            logger.error("Refusing to use a symlinked voice clone reference root.")
            return
        try:
            VOICE_CLONE_REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
            resolved_root = VOICE_CLONE_REFERENCE_ROOT.resolve(strict=True)
        except OSError as exc:
            logger.warning(f"Voice clone reference root is unavailable: {exc}")
            return
        for child in resolved_root.iterdir():
            try:
                resolved = child.resolve()
                if resolved.parent != resolved_root:
                    continue
                if resolved.is_dir():
                    shutil.rmtree(resolved, ignore_errors=False)
                else:
                    resolved.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    f"Failed to remove stale voice clone reference {child}: {exc}"
                )

    def _allow_voice_clone_upload(self, client_uid: str) -> bool:
        now = time.monotonic()
        recent = self.voice_clone_upload_times.setdefault(client_uid, deque())
        while recent and now - recent[0] >= 60.0:
            recent.popleft()
        if len(recent) >= VOICE_CLONE_UPLOADS_PER_MINUTE:
            return False
        recent.append(now)
        return True

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "workspace-state-event": self._handle_workspace_state_event,
            "workspace-item-viewed": self._handle_workspace_item_viewed,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "request-init-config": self._handle_init_config_request,
            "credential-status-request": self._handle_credential_status_request,
            "clear-saved-credential": self._handle_clear_saved_credential,
            "client-api-config": self._handle_client_api_config,
            "client-voice-clone-config": self._handle_client_voice_clone_config,
            "heartbeat": self._handle_heartbeat,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store data owned by a connected client."""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])


    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "character_name": session_service_context.character_config.character_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                    "capabilities": {
                        "voice_clone": voice_clone_dependencies_available(),
                        "voice_clone_missing": missing_voice_clone_dependencies(),
                    },
                }
            )
        )

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize a session context with an independent mutable agent and memory."""
        session_service_context = ServiceContext()
        try:
            await session_service_context.load_cache(
                config=self.default_context_cache.config.model_copy(deep=True),
                system_config=self.default_context_cache.system_config.model_copy(
                    deep=True
                ),
                character_config=self.default_context_cache.character_config.model_copy(
                    deep=True
                ),
                live2d_model=self.default_context_cache.live2d_model,
                asr_engine=self.default_context_cache.asr_engine,
                tts_engine=self.default_context_cache.tts_engine,
                vad_engine=self.default_context_cache.vad_engine,
                agent_engine=None,
                translate_engine=self.default_context_cache.translate_engine,
                mcp_server_registery=self.default_context_cache.mcp_server_registery,
                tool_adapter=self.default_context_cache.tool_adapter,
                send_text=send_text,
                client_uid=client_uid,
            )
            await session_service_context.init_agent(
                session_service_context.character_config.agent_config,
                session_service_context.character_config.persona_prompt,
            )
            session_service_context._load_short_memory_into_agent()
            return session_service_context
        except BaseException:
            try:
                await session_service_context.close()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up partially initialized context for {client_uid}: "
                    f"{cleanup_error}"
                )
            raise

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def handle_disconnect(self, client_uid: str) -> None:
        """Release every resource owned by a disconnected client."""
        await self._complete_client_resource_release(client_uid)
        logger.info(f"Client {client_uid} disconnected")

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Release resources left by a partially initialized connection."""
        await self._complete_client_resource_release(client_uid)

    async def _complete_client_resource_release(self, client_uid: str) -> None:
        """Share one shielded cleanup task across concurrent disconnect handlers."""
        cleanup_task = self.client_cleanup_tasks.get(client_uid)
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(
                self._release_client_resources(client_uid)
            )
            self.client_cleanup_tasks[client_uid] = cleanup_task

        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise
        finally:
            if (
                cleanup_task.done()
                and self.client_cleanup_tasks.get(client_uid) is cleanup_task
            ):
                self.client_cleanup_tasks.pop(client_uid, None)

    async def _release_client_resources(self, client_uid: str) -> None:
        """Idempotently stop work, detach state, and close a client's context."""
        lock = self.conversation_locks.setdefault(client_uid, asyncio.Lock())
        context = None
        voice_clone_reference_dir = None
        async with lock:
            try:
                workspace_controller = self.workspace_controllers.pop(
                    client_uid, None
                )
                if workspace_controller:
                    await workspace_controller.close()
                await _cancel_conversation_task(
                    client_uid,
                    self.current_conversation_tasks,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to stop conversation task for {client_uid}: {exc}"
                )
            finally:
                self.current_conversation_tasks.pop(client_uid, None)
                self.client_connections.pop(client_uid, None)
                context = self.client_contexts.pop(client_uid, None)
                self.received_data_buffers.pop(client_uid, None)
                self.pending_conversation_inputs.pop(client_uid, None)
                self.in_flight_conversation_inputs.pop(client_uid, None)
                self.transcription_cache.pop(client_uid, None)
                self.announced_transcription_ids.pop(client_uid, None)
                self.reply_started_flags.pop(client_uid, None)
                self.workspace_work_flags.pop(client_uid, None)
                self.workspace_revision_flags.pop(client_uid, None)
                self.voice_clone_upload_times.pop(client_uid, None)
                voice_clone_reference_dir = self.voice_clone_reference_dirs.pop(
                    client_uid, None
                )

        try:
            if context:
                await context.close()
        except Exception as exc:
            logger.warning(f"Failed to close context for {client_uid}: {exc}")
        finally:
            self._remove_voice_clone_reference_dir(voice_clone_reference_dir)
            message_handler.cleanup_client(client_uid)
            if self.conversation_locks.get(client_uid) is lock:
                self.conversation_locks.pop(client_uid, None)

    @staticmethod
    def _remove_voice_clone_reference_dir(reference_dir: Path | None) -> None:
        if reference_dir is None:
            return
        try:
            if VOICE_CLONE_REFERENCE_ROOT.is_symlink():
                logger.warning("Refusing to clean a symlinked voice clone reference root.")
                return
            resolved_root = VOICE_CLONE_REFERENCE_ROOT.resolve(strict=True)
            resolved = reference_dir.resolve()
            if resolved.parent != resolved_root:
                logger.warning(
                    f"Refusing to remove unexpected voice clone directory: {resolved}"
                )
                return
            shutil.rmtree(resolved, ignore_errors=True)
        except Exception as exc:
            logger.warning(f"Failed to remove voice clone reference directory: {exc}")

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        lock = self.conversation_locks.setdefault(client_uid, asyncio.Lock())
        async with lock:
            heard_response = data.get("text", "")
            context = self.client_contexts[client_uid]
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        # Commit the selection only after the requested history has been validated.
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    @staticmethod
    async def _reject_workspace_event(
        websocket: WebSocket, reason: str
    ) -> None:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "workspace-event-rejected",
                    "reason": reason,
                }
            )
        )

    async def _handle_workspace_state_event(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Submit sanitized page state to the independent real-time controller."""
        event_data = normalize_workspace_event(data.get("event"))
        context = self.client_contexts.get(client_uid)
        character_config = getattr(context, "character_config", None)
        expected_persona = str(
            getattr(character_config, "character_name", "")
            or getattr(character_config, "conf_name", "")
            or ""
        )
        if event_data is None:
            await self._reject_workspace_event(websocket, "invalid")
            return
        if not expected_persona or event_data.get("persona") != expected_persona:
            await self._reject_workspace_event(websocket, "persona_mismatch")
            return
        controller = self._workspace_controller_for(
            websocket, client_uid, context
        )
        controller.submit(event_data)

    def _workspace_controller_for(
        self, websocket: WebSocket, client_uid: str, context: ServiceContext
    ) -> WorkspaceController:
        controller = self.workspace_controllers.get(client_uid)
        if controller is None:
            controller = WorkspaceController(context, websocket.send_text)
            self.workspace_controllers[client_uid] = controller
        return controller

    async def _handle_workspace_item_viewed(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        context = self.client_contexts.get(client_uid)
        if context is None:
            return
        character = context.character_config
        persona = str(character.character_name or character.conf_name or "")
        path = str(data.get("path") or "").strip()[:1000]
        if not persona or not path:
            return
        controller = self._workspace_controller_for(
            websocket, client_uid, context
        )
        await controller.observe_item(persona, path)

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        metadata = data.get("metadata")
        if isinstance(metadata, dict) and metadata.get("workspace_event") is True:
            await self._reject_workspace_event(websocket, "legacy_event_protocol")
            return

        resolved_data = dict(data)
        screen_vision = data.get("screen_vision")
        context = self.client_contexts[client_uid]
        if isinstance(screen_vision, dict):
            resolved_screen_vision = dict(screen_vision)
            if context.screen_vision_api_key:
                resolved_screen_vision["api_key"] = context.screen_vision_api_key
            resolved_data["screen_vision"] = resolved_screen_vision

        lock = self.conversation_locks.setdefault(client_uid, asyncio.Lock())
        async with lock:
            await handle_conversation_trigger(
                msg_type=resolved_data.get("type", ""),
                data=resolved_data,
                client_uid=client_uid,
                context=context,
                websocket=websocket,
                received_data_buffers=self.received_data_buffers,
                current_conversation_tasks=self.current_conversation_tasks,
                pending_conversation_inputs=self.pending_conversation_inputs,
                in_flight_conversation_inputs=self.in_flight_conversation_inputs,
                transcription_cache=self.transcription_cache,
                announced_transcription_ids=self.announced_transcription_ids,
                reply_started_flags=self.reply_started_flags,
                workspace_work_flags=self.workspace_work_flags,
                workspace_revision_flags=self.workspace_revision_flags,
            )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            controller = self.workspace_controllers.pop(client_uid, None)
            if controller:
                await controller.close()
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "character_name": context.character_config.character_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                    "capabilities": {
                        "voice_clone": voice_clone_dependencies_available(),
                        "voice_clone_missing": missing_voice_clone_dependencies(),
                    },
                }
            )
        )

    async def _handle_client_api_config(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Persist new secrets and apply saved per-client API settings."""
        request_id = str(data.get("request_id") or "")[:128]
        context = self.client_contexts.get(client_uid)
        if not context:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Client context is not initialized",
                    }
                )
            )
            return

        base_url = str(data.get("api_base_url") or data.get("base_url") or "").strip()
        model = str(data.get("model") or "").strip()
        new_chat_key = str(data.get("api_key") or "").strip()
        new_screen_key = str(data.get("screen_vision_api_key") or "").strip()

        try:
            profile_id = validate_profile_id(data.get("credential_profile_id") or "")
            if not self.credential_store.available:
                raise SecureCredentialError(
                    "当前系统不支持 Windows 用户绑定的安全密钥存储"
                )
            new_secrets = {}
            if new_chat_key:
                new_secrets[CHAT_API_KEY] = new_chat_key
            if new_screen_key:
                new_secrets[SCREEN_VISION_API_KEY] = new_screen_key
            if new_secrets:
                self.credential_store.update(profile_id, secrets=new_secrets)

            chat_key = new_chat_key or self.credential_store.get(
                profile_id, CHAT_API_KEY
            )
            screen_key = new_screen_key or self.credential_store.get(
                profile_id, SCREEN_VISION_API_KEY
            )
            context.screen_vision_api_key = screen_key or ""
            if not chat_key:
                raise ValueError("请填写聊天 API Key，或先恢复已安全保存的 Key")
            await context.apply_client_api_config(
                base_url=base_url,
                api_key=chat_key,
                model=model,
            )
            await self._send_credential_status(
                websocket,
                profile_id,
                request_id=request_id,
                success=True,
                chat_config_applied=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to apply client API config: {type(e).__name__}"
            )
            profile_id = str(data.get("credential_profile_id") or "")
            await self._send_credential_status(
                websocket,
                profile_id,
                request_id=request_id,
                success=False,
                chat_config_applied=False,
                message=f"API 配置失败：{e}",
            )

    async def _send_credential_status(
        self,
        websocket: WebSocket,
        profile_id: str,
        **extra,
    ) -> None:
        status = {CHAT_API_KEY: False, SCREEN_VISION_API_KEY: False}
        available = self.credential_store.available
        if available:
            try:
                status = self.credential_store.status(profile_id)
            except (ValueError, SecureCredentialError):
                pass
        payload = {
            "type": "credential-status",
            "available": available,
            "chat_api_key_saved": status[CHAT_API_KEY],
            "screen_vision_api_key_saved": status[SCREEN_VISION_API_KEY],
        }
        payload.update(extra)
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def _handle_credential_status_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        del client_uid
        await self._send_credential_status(
            websocket,
            str(data.get("credential_profile_id") or ""),
            request_id=str(data.get("request_id") or "")[:128],
            success=True,
        )

    async def _handle_clear_saved_credential(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        request_id = str(data.get("request_id") or "")[:128]
        profile_id = str(data.get("credential_profile_id") or "")
        credential = str(data.get("credential") or "")
        name = {
            "chat": CHAT_API_KEY,
            "screen_vision": SCREEN_VISION_API_KEY,
        }.get(credential)
        try:
            if name is None:
                raise ValueError("未知的密钥类型")
            profile_id = validate_profile_id(profile_id)
            if not self.credential_store.available:
                raise SecureCredentialError("安全密钥存储不可用")
            self.credential_store.update(profile_id, clear={name})
            context = self.client_contexts.get(client_uid)
            if context is not None:
                if name == SCREEN_VISION_API_KEY:
                    context.screen_vision_api_key = ""
                else:
                    await context.clear_client_api_key()
            await self._send_credential_status(
                websocket,
                profile_id,
                request_id=request_id,
                success=True,
                cleared=credential,
            )
        except Exception as exc:
            logger.error(f"Failed to clear saved credential: {type(exc).__name__}")
            await self._send_credential_status(
                websocket,
                profile_id,
                request_id=request_id,
                success=False,
                message=f"清除已保存的 API Key 失败：{exc}",
            )

    async def _handle_client_voice_clone_config(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Apply per-client OmniVoice voice clone settings."""
        request_id = str(data.get("request_id") or "")[:128]
        context = self.client_contexts.get(client_uid)
        if not context:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "voice-clone-config-applied",
                        "enabled": False,
                        "success": False,
                        "request_id": request_id,
                        "available": voice_clone_dependencies_available(),
                        "message": "客户端上下文尚未初始化，已保持普通语音模式。",
                    }
                )
            )
            return

        enabled = bool(data.get("enabled"))
        ref_audio_path = ""
        reference_dir = None
        ref_text = ""
        language = ""

        try:
            if enabled:
                if not self._allow_voice_clone_upload(client_uid):
                    raise ValueError("语音克隆参考音频提交过于频繁，请一分钟后再试。")
                missing_dependencies = missing_voice_clone_dependencies()
                if missing_dependencies:
                    raise RuntimeError(
                        "语音克隆组件未安装，请重新运行 setup-windows.bat 并选择安装语音克隆。"
                        f"缺少：{', '.join(missing_dependencies)}"
                    )

                audio_base64 = data.get("audio_base64")
                if not isinstance(audio_base64, str) or not audio_base64:
                    raise ValueError("已开启语音克隆，但没有提供参考音频。")

                if len(audio_base64) > (
                    MAX_VOICE_CLONE_REFERENCE_BASE64_LENGTH
                    + MAX_VOICE_CLONE_DATA_URL_PREFIX_LENGTH
                ):
                    raise ValueError("参考音频过大，最大允许 10 MB。")
                if "," in audio_base64:
                    audio_base64 = audio_base64.split(",", 1)[1]
                if len(audio_base64) > MAX_VOICE_CLONE_REFERENCE_BASE64_LENGTH:
                    raise ValueError("参考音频过大，最大允许 10 MB。")

                try:
                    raw_audio = base64.b64decode(audio_base64, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("参考音频编码无效。") from exc
                if not raw_audio:
                    raise ValueError("参考音频为空。")
                if len(raw_audio) > MAX_VOICE_CLONE_REFERENCE_BYTES:
                    raise ValueError("参考音频过大，最大允许 10 MB。")

                file_name = data.get("file_name") or "reference.wav"
                if not isinstance(file_name, str) or len(file_name) > 255:
                    raise ValueError("参考音频文件名无效。")
                suffix = Path(file_name).suffix.lower()
                if suffix not in VOICE_CLONE_ALLOWED_SUFFIXES:
                    raise ValueError("参考音频格式不支持，请使用 WAV、MP3、FLAC 或 OGG。")

                ref_text_value = data.get("ref_text")
                if ref_text_value is not None and not isinstance(ref_text_value, str):
                    raise ValueError("参考文本必须是字符串。")
                ref_text = (ref_text_value or "").strip()
                if len(ref_text) > 1000:
                    raise ValueError("参考文本过长，最大允许 1000 个字符。")
                language_value = data.get("language")
                if language_value is not None and not isinstance(language_value, str):
                    raise ValueError("语种参数必须是字符串。")
                language = (language_value or "").strip()
                if len(language) > 32:
                    raise ValueError("语种参数过长。")

                if VOICE_CLONE_REFERENCE_ROOT.is_symlink():
                    raise ValueError("语音克隆临时目录不安全，已拒绝写入。")
                VOICE_CLONE_REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
                resolved_reference_root = VOICE_CLONE_REFERENCE_ROOT.resolve(strict=True)
                reference_dir = (resolved_reference_root / client_uid).resolve()
                if reference_dir.parent != resolved_reference_root:
                    raise ValueError("客户端参考音频目录无效。")
                self._remove_voice_clone_reference_dir(
                    self.voice_clone_reference_dirs.pop(client_uid, None)
                )
                reference_dir.mkdir(parents=True, exist_ok=False)
                ref_file = reference_dir / "reference.wav"
                audio_info = await asyncio.to_thread(
                    validate_and_normalize_voice_clone_reference,
                    raw_audio,
                    ref_file,
                )
                ref_audio_path = str(ref_file)
                self.voice_clone_reference_dirs[client_uid] = reference_dir
                logger.info(
                    f"Validated voice clone reference for {client_uid}: "
                    f"{audio_info['container']}, {audio_info['duration_seconds']:.2f}s, "
                    f"{audio_info['sample_rate']}Hz, {audio_info['channels']}ch"
                )

            await context.apply_client_voice_clone_config(
                enabled=enabled,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text or None,
                language=language or None,
            )
            if not enabled:
                self._remove_voice_clone_reference_dir(
                    self.voice_clone_reference_dirs.pop(client_uid, None)
                )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "voice-clone-config-applied",
                        "enabled": enabled,
                        "success": True,
                        "request_id": request_id,
                        "available": voice_clone_dependencies_available(),
                    }
                )
            )
        except Exception as e:
            self._remove_voice_clone_reference_dir(
                self.voice_clone_reference_dirs.pop(client_uid, None)
                or reference_dir
            )
            try:
                await context.apply_client_voice_clone_config(enabled=False)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to disable voice clone after configuration error: {cleanup_error}"
                )
            logger.error(f"Failed to apply voice clone config: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "voice-clone-config-applied",
                        "enabled": False,
                        "success": False,
                        "request_id": request_id,
                        "available": voice_clone_dependencies_available(),
                        "message": str(e),
                    }
                )
            )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")
