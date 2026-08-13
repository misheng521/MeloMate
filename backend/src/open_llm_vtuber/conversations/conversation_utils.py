import asyncio
import re
import uuid
from typing import Optional, Union, Any, List, Dict
import numpy as np
import json
import httpx
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput, DisplayText, Actions
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..avatar_model import AvatarModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


PLAYBACK_COMPLETION_TIMEOUT_SECONDS = 180.0
MAX_TRANSLATION_CALLS_PER_RESPONSE = 32
MAX_TRANSLATION_CHARS_PER_RESPONSE = 32_000

# Face-style emoji are visually out of character for the avatar and are also
# spoken inconsistently by different TTS providers.  Keep non-face symbols
# available (for example, chess pieces) while removing the yellow-face ranges.
_FACE_EMOJI_RE = re.compile(
    "[\u2639\u263a"
    "\U0001f600-\U0001f64f"
    "\U0001f910-\U0001f92f"
    "\U0001f970-\U0001f978"
    "\U0001f97a\U0001f9d0"
    "\U0001fae0-\U0001fae8]"
    "[\ufe0e\ufe0f]?"
)


def remove_face_emojis(text: str) -> str:
    """Remove face/yellow-bean emoji without deleting useful symbols."""
    return _FACE_EMOJI_RE.sub("", str(text or ""))


def clean_response_fragment(text: str) -> str:
    """Remove UI/TTS-unfriendly artifacts from streamed response fragments."""
    text = remove_face_emojis(text)
    return re.sub(r"\s+", " ", text.replace("$", "")).strip()


def with_turn_id(websocket_send: WebSocketSend, turn_id: Optional[str]) -> WebSocketSend:
    """Attach one server-owned turn id to every structured response payload."""
    if not turn_id:
        return websocket_send

    async def send(message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            await websocket_send(message)
            return
        if isinstance(payload, dict):
            payload.setdefault("turn_id", turn_id)
            await websocket_send(json.dumps(payload, ensure_ascii=False))
            return
        await websocket_send(message)

    return send


GAME_CONTROL_NARRATION_PATTERNS = (
    r"我先看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)?(?:再说|吧)?[，,。.!！~～]*",
    r"让我看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)?(?:再说|吧)?[，,。.!！~～]*",
    r"我看(?:一下|看)?(?:当前|现在)?(?:局面|棋盘|盘面|情况|后面|画面|状态)[，,。.!！~～]*",
    r"先看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)(?:再说)?[，,。.!！~～]*",
    r"看(?:一下|看)?(?:当前|现在)?(?:局面|棋盘|盘面|情况|后面|画面|状态)(?:再说)?[，,。.!！~～]*",
)


def remove_game_control_narration(text: str) -> str:
    """Remove immersion-breaking tool-control narration from visible/TTS text."""
    cleaned = text or ""
    for pattern in GAME_CONTROL_NARRATION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ，,。.!！~～")


def remove_stage_directions(text: str) -> str:
    """Remove stage directions without deleting Markdown-emphasized dialogue."""
    text = re.sub(r"（[^（）]*）", "", text)
    text = re.sub(r"\([^()]*\)", "", text)
    # Models commonly emphasize names with Markdown (for example ``**小可**``).
    # Two or more asterisks are formatting, so keep their contents.  A single
    # pair remains the app's stage-direction convention and is removed.
    text = re.sub(r"\*{2,}([^*\n]+?)\*{2,}", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_dot_only_fragment(text: str) -> bool:
    return bool(re.fullmatch(r"[\s.\u3002\u2026]+", text or ""))


SCREEN_VISION_PROMPT = (
    "\u8bf7\u7528\u7b80\u6d01\u4e2d\u6587\u63cf\u8ff0\u8fd9\u5f20"
    "\u5c4f\u5e55\u622a\u56fe\u91cc\u548c\u7528\u6237\u5f53\u524d"
    "\u95ee\u9898\u76f8\u5173\u7684\u53ef\u89c1\u4fe1\u606f\u3002"
    "\u91cd\u70b9\u8bf4\u53ef\u89c1\u5e94\u7528\u3001\u753b\u9762"
    "\u5185\u5bb9\u3001\u6587\u5b57\u3001\u72b6\u6001\u3001\u660e"
    "\u663e\u95ee\u9898\u3002\u4e0d\u8981\u7f16\u9020\u770b\u4e0d"
    "\u5230\u7684\u5185\u5bb9\u3002"
)

SCREEN_CONTEXT_LABEL = "\u5f53\u524d\u5c4f\u5e55\u8bc6\u522b\u7ed3\u679c"
SCREEN_CONTEXT_INSTRUCTION = (
    "\u8bf7\u7ed3\u5408\u7528\u6237\u7684\u8bdd\u548c\u5c4f\u5e55"
    "\u8bc6\u522b\u7ed3\u679c\u56de\u7b54\u3002"
)
async def describe_screen_image(
    images: Optional[List[Dict[str, Any]]],
    screen_vision: Optional[Dict[str, Any]],
    user_question: str = "",
) -> Optional[str]:
    if not images or not screen_vision:
        return None

    api_key = str(screen_vision.get("api_key") or "").strip()
    model = str(screen_vision.get("model") or "").strip()
    base_url = str(screen_vision.get("api_base_url") or "").strip().rstrip("/")
    if not base_url or not api_key or not model:
        logger.warning("Screen vision skipped: missing api_base_url, api_key or model")
        return None

    image = images[0]
    image_url = image.get("data")
    if not image_url:
        logger.warning("Screen vision skipped: missing image data")
        return None

    question = re.sub(r"\s+", " ", str(user_question or "")).strip()[:1200]
    prompt = SCREEN_VISION_PROMPT
    if question:
        prompt = f"{prompt}\n用户当前问题：{question}"

    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    if model.lower().startswith("kimi-"):
        payload["thinking"] = {"type": "disabled"}

    try:
        timeout = httpx.Timeout(90.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Screen vision request failed: "
                    f"{response.status_code} {response.text[:1000]}"
                )
                return None
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        logger.warning(f"Screen vision request timed out: {type(exc).__name__}: {exc!r}")
        return None
    except Exception as exc:
        logger.warning(f"Screen vision request failed: {type(exc).__name__}: {exc!r}")
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("Screen vision response did not contain message content")
        return None

    if isinstance(content, list):
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in (None, "text")
        ]
        content = "\n".join(text_parts)

    content = str(content or "").strip()
    return content or None


async def augment_text_with_screen_context(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    screen_vision: Optional[Dict[str, Any]],
    *,
    force: bool = False,
) -> str:
    # Ordinary turns expose a model-controlled built-in screen tool instead.
    # This path remains only for the existing forced proactive-speech behavior.
    if not force:
        return input_text

    if not images or not screen_vision:
        return input_text

    screen_description = await describe_screen_image(
        images, screen_vision, user_question=input_text
    )
    if not screen_description:
        return input_text

    return (
        f"{input_text}\n\n"
        f"[{SCREEN_CONTEXT_LABEL}]\n{screen_description}\n\n"
        f"{SCREEN_CONTEXT_INSTRUCTION}"
    )

# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
        metadata=metadata,
    )


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    avatar_model: AvatarModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                avatar_model,
                tts_engine,
                websocket_send,
                tts_manager,
                getattr(character_config, "voice_style", None),
                translate_engine,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(output, websocket_send)
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response


async def handle_sentence_output(
    output: SentenceOutput,
    avatar_model: AvatarModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    voice_style: Optional[dict] = None,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    translation_calls = 0
    translation_characters = 0
    translation_budget_reported = False
    async for display_text, tts_text, actions in output:
        logger.debug(f"Processing agent sentence (tts_chars={len(tts_text)})")

        display_text.text = avatar_model.remove_action_keywords(display_text.text)
        tts_text = avatar_model.remove_action_keywords(tts_text)
        display_text.text = remove_stage_directions(display_text.text)
        tts_text = remove_stage_directions(tts_text)
        display_text.text = clean_response_fragment(display_text.text)
        tts_text = clean_response_fragment(tts_text)
        display_text.text = remove_game_control_narration(display_text.text)
        tts_text = remove_game_control_narration(tts_text)
        if not display_text.text and not tts_text:
            continue
        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？"\'「」『』（）：；]+', "", tts_text)):
                if (
                    translation_calls >= MAX_TRANSLATION_CALLS_PER_RESPONSE
                    or translation_characters + len(tts_text)
                    > MAX_TRANSLATION_CHARS_PER_RESPONSE
                ):
                    if not translation_budget_reported:
                        logger.warning(
                            "Translation skipped after reaching the per-response budget"
                        )
                        translation_budget_reported = True
                else:
                    translation_calls += 1
                    translation_characters += len(tts_text)
                    try:
                        tts_text = await translate_engine.async_translate(tts_text)
                    except Exception as exc:
                        logger.warning(
                            "Translation unavailable; using original TTS text "
                            f"({type(exc).__name__})"
                        )
        else:
            logger.debug("No translation engine available. Skipping translation.")

        # A translation provider is also untrusted output and may introduce emoji.
        tts_text = clean_response_fragment(tts_text)

        full_response += display_text.text
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            avatar_model=avatar_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
            voice_style=voice_style,
        )
    return full_response

async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        transcript = clean_response_fragment(remove_stage_directions(transcript))
        display_text.text = clean_response_fragment(
            remove_stage_directions(display_text.text)
        )
        full_response += transcript
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        await websocket_send(json.dumps(audio_payload))
    return full_response


async def send_conversation_start_signals(websocket_send: WebSocketSend) -> None:
    """Send initial conversation signals"""
    await websocket_send(
        json.dumps(
            {
                "type": "control",
                "text": "conversation-chain-start",
            }
        )
    )
    await websocket_send(json.dumps({"type": "full-text", "text": "Thinking..."}))


async def speak_text_response(
    context: Any,
    websocket_send: WebSocketSend,
    client_uid: str,
    text: str,
    turn_id: str,
) -> str:
    """Deliver a server-owned text through the normal subtitle, TTS and completion protocol."""
    clean_text = remove_game_control_narration(
        clean_response_fragment(remove_stage_directions(str(text or "")))
    )
    if not clean_text:
        return ""
    send = with_turn_id(websocket_send, turn_id)
    manager = TTSTaskManager()
    try:
        await send_conversation_start_signals(send)
        output = SentenceOutput(
            display_text=DisplayText(
                text=clean_text,
                name=context.character_config.character_name,
            ),
            tts_text=clean_text,
            actions=Actions(),
        )
        response = await process_agent_output(
            output=output,
            character_config=context.character_config,
            avatar_model=context.avatar_model,
            tts_engine=context.get_current_tts_engine(),
            websocket_send=send,
            tts_manager=manager,
            translate_engine=context.translate_engine,
        )
        await finalize_conversation_turn(manager, send, client_uid)
        return response
    finally:
        await cleanup_conversation(manager, f"workspace-{turn_id}")


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
    announce_transcription: bool = True,
    input_id: Optional[str] = None,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        if announce_transcription:
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
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
) -> None:
    """Finalize a conversation turn"""
    await tts_manager.finish()
    playback_request_id = uuid.uuid4().hex
    message_handler.register_response_waiter(
        client_uid,
        "frontend-playback-complete",
        playback_request_id,
    )
    try:
        await websocket_send(
            json.dumps(
                {
                    "type": "backend-synth-complete",
                    "request_id": playback_request_id,
                }
            )
        )
    except BaseException:
        message_handler.cancel_response_waiter(
            client_uid,
            "frontend-playback-complete",
            playback_request_id,
        )
        raise

    response = await message_handler.wait_for_response(
        client_uid,
        "frontend-playback-complete",
        playback_request_id,
        timeout=PLAYBACK_COMPLETION_TIMEOUT_SECONDS,
    )
    if not response:
        logger.warning(
            f"No playback completion response for request {playback_request_id} "
            f"from {client_uid}; ending the turn after timeout"
        )

    await websocket_send(json.dumps({"type": "force-new-message"}))

    await send_conversation_end_signal(websocket_send)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    session_emoji: str = "session",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    logger.info(f"Conversation chain {session_emoji} completed.")


async def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    await tts_manager.clear()
    logger.debug(f"Clearing up conversation {session_emoji}.")


EMOJI_LIST = [
    "session-01",
    "session-02",
    "session-03",
    "session-04",
    "session-05",
    "session-06",
    "session-07",
    "session-08",
    "session-09",
    "session-10",
]
