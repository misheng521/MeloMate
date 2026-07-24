import asyncio
import re
from typing import Optional, Union, Any, List, Dict
import numpy as np
import json
import httpx
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


def clean_response_fragment(text: str) -> str:
    """Remove UI/TTS-unfriendly artifacts from streamed response fragments."""
    return re.sub(r"\s+", " ", text.replace("$", "")).strip()


def remove_stage_directions(text: str) -> str:
    """Remove parenthesized/asterisk action descriptions from visible chat text."""
    text = re.sub(r"（[^（）]*）", "", text)
    text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\*+[^*]*\*+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_dot_only_fragment(text: str) -> bool:
    return bool(re.fullmatch(r"[\s.\u3002\u2026]+", text or ""))


SCREEN_VISION_INTENT_PATTERN = re.compile(
    r"(\u770b(\u4e00\u4e0b|\u4e0b|\u770b)?"
    r"(\u8fd9\u4e2a|\u8fd9\u8fb9|\u8fd9\u91cc|\u5c4f\u5e55|\u753b\u9762|\u6e38\u620f|\u7a97\u53e3|\u7f51\u9875|\u4ee3\u7801|\u754c\u9762)|"
    r"\u5e2e\u6211\u770b|\u5e2e\u5fd9\u770b|\u770b\u770b|\u7785\u7785|"
    r"\u8bc6\u522b(\u4e00\u4e0b|\u4e0b)?(\u5c4f\u5e55|\u753b\u9762|\u56fe\u7247|\u8fd9\u4e2a)|"
    r"\u5c4f\u5e55\u4e0a|\u753b\u9762\u91cc|\u6e38\u620f\u91cc|\u8fd9\u4e2a\u753b\u9762|\u5f53\u524d\u753b\u9762|"
    r"current screen|look at|see this)",
    re.IGNORECASE,
)


def wants_screen_vision(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False

    if SCREEN_VISION_INTENT_PATTERN.search(normalized):
        return True

    intent_keywords = (
        "\u8bc6\u522b\u5c4f\u5e55",
        "\u770b\u5c4f\u5e55",
        "\u770b\u770b\u5c4f\u5e55",
        "\u770b\u4e00\u4e0b\u5c4f\u5e55",
        "\u770b\u753b\u9762",
        "\u770b\u770b\u753b\u9762",
        "\u770b\u6e38\u620f",
        "\u770b\u770b\u6e38\u620f",
        "\u770b\u8fd9\u4e2a",
        "\u770b\u770b\u8fd9\u4e2a",
        "\u770b\u8fd9\u91cc",
        "\u5f53\u524d\u753b\u9762",
        "\u5c4f\u5e55\u4e0a",
        "current screen",
        "look at my screen",
        "see my screen",
    )
    return any(keyword in normalized for keyword in intent_keywords)


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
SCREEN_VISION_FAILED_MESSAGE = (
    "\u7528\u6237\u60f3\u8ba9\u4f60\u770b\u5c4f\u5e55\uff0c\u4f46"
    "\u8fd9\u6b21\u5c4f\u5e55\u8bc6\u522b\u6ca1\u6709\u6210\u529f\u3002"
    "\u8bf7\u76f4\u63a5\u544a\u8bc9\u7528\u6237\uff1a\u5df2\u89e6\u53d1"
    "\u5c4f\u5e55\u8bc6\u522b\uff0c\u4f46\u6ca1\u62ff\u5230\u53ef\u7528"
    "\u7684\u8bc6\u56fe\u7ed3\u679c\uff1b\u8ba9\u7528\u6237\u68c0\u67e5"
    "\u8bc6\u56fe API \u5730\u5740\u3001\u8bc6\u56fe\u6a21\u578b\u662f"
    "\u5426\u652f\u6301\u56fe\u7247\u3001API Key \u548c\u5c4f\u5e55"
    "\u5171\u4eab\u6743\u9650\u3002\u4e0d\u8981\u518d\u8bf4\u4f60"
    "\u6ca1\u6709\u88ab\u6388\u6743\u770b\u5c4f\u5e55\u3002"
)

async def describe_screen_image(
    images: Optional[List[Dict[str, Any]]],
    screen_vision: Optional[Dict[str, Any]],
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

    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCREEN_VISION_PROMPT},
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
) -> str:
    if not wants_screen_vision(input_text):
        return input_text

    if not images:
        return f"{input_text}\n\n[{SCREEN_CONTEXT_LABEL}]\n{SCREEN_VISION_FAILED_MESSAGE}"

    if not screen_vision:
        return f"{input_text}\n\n[{SCREEN_CONTEXT_LABEL}]\n{SCREEN_VISION_FAILED_MESSAGE}"

    screen_description = await describe_screen_image(images, screen_vision)
    if not screen_description:
        return f"{input_text}\n\n[{SCREEN_CONTEXT_LABEL}]\n{SCREEN_VISION_FAILED_MESSAGE}"

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
    live2d_model: Live2dModel,
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
                live2d_model,
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
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    voice_style: Optional[dict] = None,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"Processing output: '''{tts_text}'''...")

        display_text.text = live2d_model.remove_emotion_keywords(display_text.text)
        display_text.text = remove_stage_directions(display_text.text)
        tts_text = remove_stage_directions(tts_text)
        display_text.text = clean_response_fragment(display_text.text)
        tts_text = clean_response_fragment(tts_text)
        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？"\'「」『』（）：；]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"Text after translation: '''{tts_text}'''...")
        else:
            logger.debug("No translation engine available. Skipping translation.")

        full_response += display_text.text
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
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


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
    announce_transcription: bool = True,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        if announce_transcription:
            await websocket_send(
                json.dumps({"type": "user-input-transcription", "text": input_text})
            )
        return input_text
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            return

    await websocket_send(json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "session",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"Conversation chain {session_emoji} completed.")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
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
