import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.open_llm_vtuber import websocket_handler as handler_module  # noqa: E402
from src.open_llm_vtuber.websocket_handler import (  # noqa: E402
    MAX_VOICE_CLONE_DURATION_SECONDS,
    VOICE_CLONE_UPLOADS_PER_MINUTE,
    WebSocketHandler,
    validate_and_normalize_voice_clone_reference,
)


def wav_bytes(
    duration: float = 3.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    amplitude: float = 0.2,
) -> bytes:
    frames = int(duration * sample_rate)
    time_axis = np.arange(frames, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 220 * time_axis)
    audio = np.repeat(signal[:, None], channels, axis=1)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class VoiceCloneReferenceValidationTests(unittest.TestCase):
    def test_valid_reference_is_normalized_to_mono_pcm_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "reference.wav"
            info = validate_and_normalize_voice_clone_reference(
                wav_bytes(channels=2), target
            )

            normalized = sf.info(target)
            self.assertEqual(info["channels"], 2)
            self.assertEqual(normalized.channels, 1)
            self.assertEqual(normalized.subtype, "PCM_16")
            self.assertAlmostEqual(normalized.duration, 3.0, places=2)

    def test_arbitrary_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "无法安全解码"):
                validate_and_normalize_voice_clone_reference(
                    b"this is not audio", Path(temp_dir) / "reference.wav"
                )

    def test_compressed_or_declared_long_audio_is_rejected(self):
        duration = MAX_VOICE_CLONE_DURATION_SECONDS + 0.5
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "3-10 秒"):
                validate_and_normalize_voice_clone_reference(
                    wav_bytes(duration=duration), Path(temp_dir) / "reference.wav"
                )

    def test_more_than_two_channels_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "单声道或双声道"):
                validate_and_normalize_voice_clone_reference(
                    wav_bytes(channels=3), Path(temp_dir) / "reference.wav"
                )

    def test_silent_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "可用声音"):
                validate_and_normalize_voice_clone_reference(
                    wav_bytes(amplitude=0.0), Path(temp_dir) / "reference.wav"
                )


class VoiceCloneClientIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_clients_receive_different_normalized_files(self):
        class Context:
            def __init__(self):
                self.calls = []

            async def apply_client_voice_clone_config(self, **kwargs):
                self.calls.append(kwargs)

        class Socket:
            def __init__(self):
                self.messages = []

            async def send_text(self, text):
                self.messages.append(json.loads(text))

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            handler_module, "VOICE_CLONE_REFERENCE_ROOT", Path(temp_dir).resolve()
        ), patch.object(
            handler_module, "missing_voice_clone_dependencies", return_value=[]
        ), patch.object(
            handler_module, "voice_clone_dependencies_available", return_value=True
        ):
            handler = WebSocketHandler.__new__(WebSocketHandler)
            handler.client_contexts = {
                "client-a": Context(),
                "client-b": Context(),
            }
            handler.voice_clone_reference_dirs = {}
            handler.voice_clone_upload_times = {}
            sockets = {"client-a": Socket(), "client-b": Socket()}
            encoded = base64.b64encode(wav_bytes()).decode("ascii")

            for client_uid in sockets:
                await handler._handle_client_voice_clone_config(
                    sockets[client_uid],
                    client_uid,
                    {
                        "enabled": True,
                        "request_id": client_uid,
                        "audio_base64": encoded,
                        "file_name": "voice.wav",
                    },
                )

            first_path = Path(
                handler.client_contexts["client-a"].calls[0]["ref_audio_path"]
            )
            second_path = Path(
                handler.client_contexts["client-b"].calls[0]["ref_audio_path"]
            )
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.parent.name, "client-a")
            self.assertEqual(second_path.parent.name, "client-b")
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())
            self.assertTrue(sockets["client-a"].messages[-1]["success"])
            self.assertTrue(sockets["client-b"].messages[-1]["success"])

    async def test_upload_rate_is_bounded_per_client(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.voice_clone_upload_times = {}
        for _ in range(VOICE_CLONE_UPLOADS_PER_MINUTE):
            self.assertTrue(handler._allow_voice_clone_upload("client-a"))
        self.assertFalse(handler._allow_voice_clone_upload("client-a"))
        self.assertTrue(handler._allow_voice_clone_upload("client-b"))


if __name__ == "__main__":
    unittest.main()
