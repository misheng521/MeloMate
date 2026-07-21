import asyncio
import inspect
import json
import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


STYLE_SPEED_MULTIPLIERS = {
    "normal": 1.0,
    "happy": 1.08,
    "shy": 0.94,
    "sad": 0.88,
    "excited": 1.16,
}


def select_voice_style(text: str, voice_style: Optional[Dict[str, str]]) -> Tuple[str, Optional[str]]:
    if not voice_style:
        return "normal", None

    clean = text.strip()
    style_key = "normal"

    if re.search(r"(太好了|好耶|哇|真的呀|厉害|开心|高兴|哈哈|嘿嘿|nice|great|happy)", clean, re.I):
        style_key = "happy"
    if re.search(r"(激动|超想|太棒|分享|冲呀|快看|不得了|excited|awesome)", clean, re.I):
        style_key = "excited"
    if re.search(r"(害羞|不好意思|脸红|唔|诶呀|shy|embarrass)", clean, re.I):
        style_key = "shy"
    if re.search(r"(难过|伤心|委屈|抱抱|没事的|陪你|辛苦|累了|sad|sorry|tired)", clean, re.I):
        style_key = "sad"

    if style_key not in voice_style:
        style_key = "normal" if "normal" in voice_style else next(iter(voice_style))

    style_prompt = voice_style.get(style_key)
    if not style_prompt:
        return style_key, None
    return style_key, style_prompt


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        voice_style: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1
        voice_style_key, voice_style_prompt = select_voice_style(tts_text, voice_style)

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                voice_style_key=voice_style_key,
                voice_style_prompt=voice_style_prompt,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.
        """
        buffered_payloads: Dict[int, Dict] = {}

        while True:
            try:
                # Get payload from queue
                payload, sequence_number = await self._payload_queue.get()
                buffered_payloads[sequence_number] = payload

                # Send payloads in order
                while self._next_sequence_to_send in buffered_payloads:
                    next_payload = buffered_payloads.pop(self._next_sequence_to_send)
                    await websocket_send(json.dumps(next_payload))
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        voice_style_key: str,
        voice_style_prompt: Optional[str],
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(
                tts_engine,
                tts_text,
                voice_style_key=voice_style_key,
                voice_style_prompt=voice_style_prompt,
            )
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Queue the payload with its sequence number
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _generate_audio(
        self,
        tts_engine: TTSInterface,
        text: str,
        voice_style_key: str = "normal",
        voice_style_prompt: Optional[str] = None,
    ) -> str:
        """Generate audio file from text"""
        logger.debug(f"🏃Generating audio for '''{text}'''...")
        kwargs = {
            "text": text,
            "file_name_no_ext": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        }
        signature = inspect.signature(tts_engine.async_generate_audio)
        if "voice_style" in signature.parameters:
            kwargs["voice_style"] = voice_style_prompt
        if "voice_style_key" in signature.parameters:
            kwargs["voice_style_key"] = voice_style_key
        return await tts_engine.async_generate_audio(**kwargs)

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
