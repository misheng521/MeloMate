import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_llm_vtuber.tts.edge_tts import TTSEngine


class EdgeTTSRetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temporary.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temporary.cleanup()

    def test_transient_empty_audio_recovers_on_retry(self):
        calls = 0

        class FakeCommunicate:
            def __init__(self, text, voice):
                self.text = text
                self.voice = voice

            def save_sync(self, file_name):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise RuntimeError("No audio was received")
                Path(file_name).write_bytes(b"ID3" + b"audio" * 40)

        engine = TTSEngine("zh-CN-XiaoxiaoNeural")
        with (
            patch("open_llm_vtuber.tts.edge_tts.edge_tts.Communicate", FakeCommunicate),
            patch("open_llm_vtuber.tts.edge_tts.time.sleep") as sleep,
        ):
            result = engine.generate_audio("你好", "retry-success")

        self.assertEqual(calls, 3)
        self.assertTrue(Path(result).is_file())
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.45, 1.2])

    def test_all_failed_attempts_remove_partial_audio(self):
        class FakeCommunicate:
            def __init__(self, text, voice):
                pass

            def save_sync(self, file_name):
                Path(file_name).write_bytes(b"partial")
                raise RuntimeError("No audio was received")

        engine = TTSEngine("zh-CN-XiaoxiaoNeural")
        with (
            patch("open_llm_vtuber.tts.edge_tts.edge_tts.Communicate", FakeCommunicate),
            patch("open_llm_vtuber.tts.edge_tts.time.sleep"),
        ):
            result = engine.generate_audio("你好", "retry-failure")

        self.assertIsNone(result)
        self.assertFalse(Path("cache/retry-failure.mp3").exists())


if __name__ == "__main__":
    unittest.main()
