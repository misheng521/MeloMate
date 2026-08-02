import ast
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from open_llm_vtuber import chat_history_manager as history
from open_llm_vtuber.config_manager.agent import AgentConfig
from open_llm_vtuber.config_manager.asr import ASRConfig
from open_llm_vtuber.config_manager.tts import TTSConfig
from open_llm_vtuber.config_manager.vad import VADConfig


class ChatHistoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_root = history.CHAT_HISTORY_DIR
        history.CHAT_HISTORY_DIR = Path(self.temporary.name)

    def tearDown(self):
        history.CHAT_HISTORY_DIR = self.original_root
        self.temporary.cleanup()

    def test_concurrent_messages_are_not_lost_or_paired_with_another_client(self):
        uid = history.create_new_history("persona")

        def store(index: int) -> None:
            role = "human" if index % 2 == 0 else "ai"
            history.store_message("persona", uid, role, f"message-{index}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(store, range(history.MAX_MEMORY_MESSAGES)))

        messages = history.get_history("persona", uid)
        self.assertEqual(len(messages), history.MAX_MEMORY_MESSAGES)
        self.assertEqual(
            {message["content"] for message in messages},
            {f"message-{index}" for index in range(history.MAX_MEMORY_MESSAGES)},
        )
        stored = json.loads(
            (Path(self.temporary.name) / "persona" / history.SHORT_MEMORY_FILE).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored["version"], 2)

    def test_metadata_is_real_and_damage_recovers_from_valid_backup(self):
        uid = history.create_new_history("persona")
        self.assertTrue(
            history.update_metadata(
                "persona", uid, {"agent_type": "hume_ai_agent", "resume_id": "resume-1"}
            )
        )
        self.assertEqual(history.get_metadata("persona", uid)["resume_id"], "resume-1")

        history.store_message("persona", uid, "human", "first")
        history.store_message("persona", uid, "ai", "second")
        memory_path = Path(self.temporary.name) / "persona" / history.SHORT_MEMORY_FILE
        memory_path.write_text("{damaged", encoding="utf-8")
        recovered = history.get_history("persona", uid)
        self.assertTrue(recovered)
        json.loads(memory_path.read_text(encoding="utf-8"))

    def test_two_backend_processes_cannot_overwrite_each_others_messages(self):
        script = (
            "import sys; from pathlib import Path; "
            "from open_llm_vtuber import chat_history_manager as h; "
            "h.CHAT_HISTORY_DIR=Path(sys.argv[1]); uid=h.create_new_history('persona'); "
            "[h.store_message('persona',uid,'human',f'{sys.argv[2]}-{i}') for i in range(20)]"
        )
        processes = [
            subprocess.Popen([sys.executable, "-c", script, self.temporary.name, prefix])
            for prefix in ("first", "second")
        ]
        for process in processes:
            self.assertEqual(process.wait(timeout=20), 0)
        messages = history.get_history("persona", history.SINGLE_HISTORY_UID)
        self.assertEqual(len(messages), 40)
        self.assertEqual(
            {message["content"] for message in messages},
            {f"{prefix}-{index}" for prefix in ("first", "second") for index in range(20)},
        )

    def test_invalid_paths_and_unknown_history_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            history.create_new_history("../outside")
        with self.assertRaises(ValueError):
            history.get_history("persona", "../../another-history")
        self.assertFalse(history.delete_history("persona", "another-history"))

    def test_history_manager_has_no_duplicate_function_definitions_or_stubs(self):
        source_path = Path(history.__file__)
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("rename_history_file", names)
        self.assertNotIn("update_metadate", names)


class SelectedConfigurationTests(unittest.TestCase):
    def test_selected_asr_tts_and_vad_sections_are_required(self):
        with self.assertRaises(ValidationError):
            ASRConfig(asr_model="sherpa_onnx_asr")
        with self.assertRaises(ValidationError):
            TTSConfig(tts_model="edge_tts")
        with self.assertRaises(ValidationError):
            VADConfig(vad_model="silero_vad")

    def test_selected_agent_and_llm_sections_are_required(self):
        with self.assertRaises(ValidationError):
            AgentConfig(
                conversation_agent_choice="basic_memory_agent",
                agent_settings={},
                llm_configs={},
            )
        with self.assertRaises(ValidationError):
            AgentConfig(
                conversation_agent_choice="basic_memory_agent",
                agent_settings={
                    "basic_memory_agent": {"llm_provider": "deepseek_llm"}
                },
                llm_configs={},
            )

    def test_removed_fake_agent_is_not_a_valid_provider(self):
        with self.assertRaises(ValidationError):
            AgentConfig(
                conversation_agent_choice="mem0_agent",
                agent_settings={},
                llm_configs={},
            )

    def test_missing_optional_dependency_is_reported_during_validation(self):
        with patch(
            "open_llm_vtuber.config_manager.provider_dependencies.find_spec",
            return_value=None,
        ):
            with self.assertRaisesRegex(ValidationError, "azure-cognitiveservices-speech"):
                ASRConfig(
                    asr_model="azure_asr",
                    azure_asr={"api_key": "key", "region": "eastus"},
                )

    def test_gpt_sovits_uses_the_public_configuration_key(self):
        config = TTSConfig(
            tts_model="gpt_sovits_tts",
            gpt_sovits_tts={
                "api_url": "http://127.0.0.1:9880/tts",
                "text_lang": "zh",
                "ref_audio_path": "reference.wav",
                "prompt_lang": "zh",
                "prompt_text": "",
                "text_split_method": "cut5",
                "batch_size": "1",
                "media_type": "wav",
                "streaming_mode": "false",
            },
        )
        self.assertIsNotNone(config.gpt_sovits_tts)


if __name__ == "__main__":
    unittest.main()
