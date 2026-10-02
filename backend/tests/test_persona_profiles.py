"""MD-only profiles, legacy selections and real existing memory continuity."""
import os
import hashlib
import concurrent.futures
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import test_text_memory as fixtures
from src.open_llm_vtuber import chat_history_manager as memory
from src.open_llm_vtuber.persona_text import persona_path, text_character


class PersonaProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        state = patch.object(memory, "CHAT_HISTORY_DIR", self.root / "memory")
        state.start()
        self.addCleanup(state.stop)
        source = fixtures.BACKEND / "src/open_llm_vtuber/config_manager/utils.py"
        self.load = fixtures.source_method(source, "load_character_profile",
            {"persona_path": persona_path, "text_character": text_character})
        self.scan = fixtures.source_method(source, "scan_config_alts_directory", {
            "Path": Path, "os": os, "load_character_profile": self.load,
            "logger": types.SimpleNamespace(warning=lambda *a: None, debug=lambda *a: None)})

    def test_bundled_md_profiles_keep_real_preexisting_memory_without_yaml(self):
        bundled = fixtures.BACKEND.parent / "characters/profiles"
        for name, uid in [("小可", "avatar_xiaoke_001"), ("小薇", "avatar_xiaowei_001"), ("小鱼", "avatar_xiaoyu_001")]:
            self.assertFalse((bundled / (name + ".yaml")).exists())
            memory.store_message(uid, memory.SINGLE_HISTORY_UID, "human", "以前聊过的事情。")
            result = self.load(str(bundled), name + ".yaml")
            self.assertEqual(result["persona_file"], name + ".md")
            self.assertEqual(result["conf_uid"], name)
            self.assertIn("以前聊过", memory.get_context_history(result["conf_uid"])[0]["content"])
            self.assertTrue((self.root / "memory" / name / "memory.md").is_file())
            self.assertFalse((self.root / "memory" / uid).exists())

    def test_new_persona_memory_folder_uses_its_name(self):
        (self.profiles / "阿澄.md").write_text("你叫阿澄。", encoding="utf-8")
        result = self.load(str(self.profiles), "阿澄.md")
        memory.create_new_history(result["conf_uid"])
        self.assertTrue((self.root / "memory/阿澄/memory.md").is_file())
        self.assertEqual([p.name for p in (self.root / "memory").iterdir()], ["阿澄"])

    def test_hashed_directory_migrates_notes_archive_and_message_ids_once(self):
        uid = "text_" + hashlib.sha256("阿澄".encode()).hexdigest()[:24]
        message_id = memory.store_message(uid, memory.SINGLE_HISTORY_UID, "human", "旧聊天内容。")
        notes = self.root / "memory" / uid / "memory.md"
        notes.write_text("以前保存的记忆。", encoding="utf-8")
        memory.read_memory(uid)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: memory.read_memory("阿澄"), range(4)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual((self.root / "memory/阿澄/memory.md").read_text(encoding="utf-8"), "以前保存的记忆。")
        self.assertEqual(memory.get_context_history("阿澄")[0]["id"], message_id)
        self.assertFalse(notes.parent.exists())

    def test_duplicate_old_and_new_directories_preserve_both(self):
        uid = "avatar_xiaoke_001"
        memory.store_message(uid, memory.SINGLE_HISTORY_UID, "human", "原有历史。")
        target = self.root / "memory/小可"
        target.mkdir()
        (target / "memory.md").write_text("用户新写的记忆。", encoding="utf-8")
        with self.assertRaises(memory.HistoryStorageError):
            memory.read_memory("小可")
        self.assertEqual((target / "memory.md").read_text(encoding="utf-8"), "用户新写的记忆。")
        self.assertIn("原有历史", memory.get_context_history(uid)[0]["content"])

    def test_distinct_unicode_character_names_do_not_share_a_directory(self):
        for name in ("A", "Ａ"):
            memory.store_message(name, memory.SINGLE_HISTORY_UID, "human", name)
        self.assertEqual(memory.get_context_history("A")[0]["content"], "A")
        self.assertEqual(memory.get_context_history("Ａ")[0]["content"], "Ａ")

    def test_stale_yaml_cannot_override_md_or_runtime_defaults(self):
        prompt = self.profiles / "新人.md"
        prompt.write_text("你叫新人。", encoding="utf-8")
        (self.profiles / "新人.yaml").write_text("invalid: [", encoding="utf-8")
        first = self.load(str(self.profiles), "新人.yaml")
        self.assertEqual(first["persona_prompt"], "你叫新人。")
        self.assertNotIn("tts_config", first)
        self.assertNotIn("agent_config", first)
        prompt.write_text("新的背景。", encoding="utf-8")
        second = self.load(str(self.profiles), "新人.md")
        self.assertEqual(second["conf_uid"], first["conf_uid"])
        self.assertEqual(second["persona_prompt"], "新的背景。")
        self.assertEqual(prompt.read_text(encoding="utf-8"), "新的背景。")

    def test_picker_only_lists_valid_top_level_md_once(self):
        (self.profiles / "新人.md").write_text("你叫新人。", encoding="utf-8")
        (self.profiles / "新人.yaml").write_text("ignored", encoding="utf-8")
        (self.profiles / "旧角色.txt").write_text("ignored", encoding="utf-8")
        (self.profiles / "空白.md").write_text(" ", encoding="utf-8")
        (self.profiles / "nested").mkdir()
        (self.profiles / "nested/其他.md").write_text("ignored", encoding="utf-8")
        result = self.scan(str(self.profiles))
        self.assertEqual([(p["filename"], p["name"]) for p in result], [("新人.md", "新人")])

    def test_old_selection_without_md_fails_instead_of_reusing_other_persona(self):
        with self.assertRaises(FileNotFoundError):
            self.load(str(self.profiles), "missing.yaml")
        with self.assertRaises(ValueError):
            self.load(str(self.profiles), "../outside.md")


if __name__ == "__main__":
    unittest.main()
