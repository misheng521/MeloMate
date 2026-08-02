import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.open_llm_vtuber.agent.agents.basic_memory_agent import (  # noqa: E402
    BasicMemoryAgent,
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_CALLS_PER_TURN,
)
from src.open_llm_vtuber.asr.utils import download_file  # noqa: E402
from src.open_llm_vtuber.translate.translate_interface import (  # noqa: E402
    MAX_TRANSLATION_CALLS_PER_MINUTE,
    MAX_TRANSLATION_INPUT_CHARS,
    TranslateInterface,
)
from install_sensevoice_release import _validated_member_names  # noqa: E402


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.headers = {"content-length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]


class BudgetedTranslator(TranslateInterface):
    def __init__(self):
        super().__init__()

    def translate(self, text: str) -> str:
        self._consume_request_budget(text)
        return self._validate_output(text.upper())


class DownloadAndRuntimeLimitTests(unittest.TestCase):
    def test_verified_download_replaces_target_atomically(self):
        import hashlib

        body = b"verified model bytes"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.bin"
            target.write_bytes(b"old")
            with patch(
                "src.open_llm_vtuber.asr.utils.requests.get",
                return_value=FakeResponse(body),
            ):
                download_file(
                    "https://huggingface.co/test/model/resolve/revision/model.bin",
                    target,
                    expected_sha256=hashlib.sha256(body).hexdigest(),
                    expected_size=len(body),
                )
            self.assertEqual(target.read_bytes(), body)
            self.assertEqual(list(Path(temp).glob("*.part")), [])

    def test_failed_hash_keeps_existing_target(self):
        body = b"untrusted bytes"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.bin"
            target.write_bytes(b"known-good")
            with patch(
                "src.open_llm_vtuber.asr.utils.requests.get",
                return_value=FakeResponse(body),
            ):
                with self.assertRaises(ValueError):
                    download_file(
                        "https://huggingface.co/test/model/resolve/revision/model.bin",
                        target,
                        expected_sha256="0" * 64,
                        expected_size=len(body),
                    )
            self.assertEqual(target.read_bytes(), b"known-good")

    def test_archive_member_path_traversal_is_rejected(self):
        result = SimpleNamespace(stdout="../escape.txt\nmodel/model.bin\n")
        with patch("install_sensevoice_release.subprocess.run", return_value=result):
            with self.assertRaises(RuntimeError):
                _validated_member_names("tar", Path("model.7z"))

    def test_tool_call_budget_limits_round_and_turn(self):
        self.assertEqual(
            BasicMemoryAgent._consume_tool_call_budget(0, MAX_TOOL_CALLS_PER_ROUND),
            MAX_TOOL_CALLS_PER_ROUND,
        )
        with self.assertRaises(RuntimeError):
            BasicMemoryAgent._consume_tool_call_budget(0, MAX_TOOL_CALLS_PER_ROUND + 1)
        with self.assertRaises(RuntimeError):
            BasicMemoryAgent._consume_tool_call_budget(MAX_TOOL_CALLS_PER_TURN, 1)

    def test_translation_has_size_and_rate_budgets(self):
        translator = BudgetedTranslator()
        with self.assertRaises(ValueError):
            translator.translate("x" * (MAX_TRANSLATION_INPUT_CHARS + 1))
        for _ in range(MAX_TRANSLATION_CALLS_PER_MINUTE):
            self.assertEqual(translator.translate("ok"), "OK")
        with self.assertRaises(RuntimeError):
            translator.translate("one too many")


if __name__ == "__main__":
    unittest.main()
