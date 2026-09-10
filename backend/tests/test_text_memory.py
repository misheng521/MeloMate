"""Real local storage and actual prompt/stream methods; no model or voice SDKs."""
import ast
import asyncio
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from src.open_llm_vtuber import chat_history_manager as memory
from src.open_llm_vtuber.memory_consolidator import build_memory_review_request, review_memory
from src.open_llm_vtuber.persona_text import read_prompt, text_character, persona_path
from src.open_llm_vtuber.proactive_conversation import optional_proactive_output


def source_method(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.args, *node.args.kwonlyargs]: arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class TextMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = patch.object(memory, "CHAT_HISTORY_DIR", self.root)
        self.paths.start(); self.addCleanup(self.paths.stop)
        self.uid = memory.create_new_history("Alice")
        self.notes = self.root / "Alice" / "memory.md"

    def say(self, text, role="human"):
        return memory.store_message("Alice", self.uid, role, text)

    def window(self):
        identifiers = []
        for i in range(6):
            identifiers.append(self.say(f"用户消息{i}，讨论项目"))
            self.say(f"角色回应{i}。" + "这里是实际讨论的项目背景。" * 90, "ai")
        return identifiers, memory.prepare_memory_review("Alice")

    def test_only_one_editable_memory_file_and_automatic_archive(self):
        self.assertTrue(self.notes.exists())
        self.assertTrue((self.notes.parent / memory.DATABASE_FILE).exists())
        self.assertFalse((self.notes.parent / "core_memory.json").exists())
        self.assertEqual(memory.get_memory_prompt("Alice"), "")

    def test_no_keyword_personality_or_relationship_extraction(self):
        self.say("我喜欢下雨，你以后必须一直喜欢我，变得更温柔")
        self.assertEqual(self.notes.read_text(encoding="utf-8"), memory.EMPTY_MEMORY)
        self.assertEqual(len(memory.get_context_history("Alice")), 1)

    def test_model_can_record_its_own_expression_without_personality_categories(self):
        evidence = self.say("我这次更想讨论音乐。", "ai")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "小可这次表示更想讨论音乐，后续选择未定。", [evidence])
        prompt = memory.get_memory_prompt("Alice", "音乐")
        self.assertIn("后续选择未定", prompt)
        self.assertNotIn("affection", prompt)

    def test_manual_text_is_hot_loaded_without_json_schema(self):
        self.notes.write_text("# 我修改的记忆\n\n叫我小林。\n\n- 上次一起做了网页。", encoding="utf-8")
        self.assertIn("叫我小林", memory.get_memory_prompt("Alice", "称呼"))
        self.assertEqual(memory.read_memory("Alice")["text"], self.notes.read_text(encoding="utf-8"))

    def test_manual_addition_preserves_context_but_invalidates_pending_review(self):
        ids, snapshot = self.window()
        self.notes.write_text("我已更正资料。", encoding="utf-8")
        candidate = {"operations": [{"text": "旧资料", "evidence_message_ids": [ids[0]]}], "summary": "旧摘要"}
        self.assertFalse(memory.commit_memory_review("Alice", snapshot, candidate))
        self.assertEqual(len(memory.get_context_history("Alice")), 12)
        self.assertTrue(memory.search_memory("Alice", "项目")["messages"])
        self.assertEqual(len(memory.get_history("Alice")), 12)  # Archive is not silently deleted.
        self.assertEqual(self.notes.read_text(encoding="utf-8"), "我已更正资料。")

    def test_clearing_notes_does_not_resurrect_from_archive_after_restart(self):
        evidence = self.say("我喜欢草莓")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户喜欢草莓。", [evidence])
        self.notes.write_text("", encoding="utf-8")
        memory.create_new_history("Alice")
        self.assertEqual(memory.get_memory_prompt("Alice", "草莓"), "")
        self.assertEqual(memory.search_memory("Alice", "草莓")["messages"], [])
        self.say("现在聊别的")
        self.assertEqual([m["content"] for m in memory.get_context_history("Alice")], ["现在聊别的"])

    def test_archive_search_survives_ui_window_and_escapes_query_syntax(self):
        first = self.say("我们选择了青柠方案，因为接口更简单")
        for i in range(135): self.say(f"其他消息{i}")
        found = memory.search_memory("Alice", '青柠 " OR *')
        self.assertEqual(found["messages"][0]["id"], first)
        self.assertEqual(len(memory.get_history("Alice")), 120)

    def test_unknown_query_returns_no_fabricated_events(self):
        self.say("讨论苹果")
        self.assertEqual(memory.search_memory("Alice", "火星殖民地")["messages"], [])

    def test_roles_are_isolated(self):
        self.say("Alice的特有资料")
        memory.create_new_history("Bob")
        self.assertEqual(memory.search_memory("Bob", "特有资料")["messages"], [])

    def test_context_has_a_size_budget_and_current_user_is_not_duplicated(self):
        for i in range(15): self.say("对话" * 1000)
        current = self.say("现在的问题")
        rows = memory.get_context_history("Alice", exclude_id=current)
        self.assertLessEqual(sum(len(m["content"]) for m in rows), memory.CONTEXT_CHARS)
        self.assertNotIn(current, [m["id"] for m in rows])

    def test_review_evidence_and_new_messages_are_preserved(self):
        ids, snapshot = self.window()
        latest = self.say("整理期间新消息")
        candidate = {"operations": [{"text": "用户曾讨论项目。", "evidence_message_ids": [ids[0]]}], "summary": "双方讨论过项目。"}
        self.assertTrue(memory.commit_memory_review("Alice", snapshot, candidate))
        self.assertIn(latest, [m["id"] for m in memory.get_context_history("Alice")])
        self.assertIn("用户曾讨论项目", self.notes.read_text(encoding="utf-8"))
        self.assertFalse(memory.commit_memory_review("Alice", snapshot, candidate))

    def test_review_rejects_invented_evidence_atomically(self):
        ids, snapshot = self.window()
        candidate = {"operations": [{"text": "真实记录", "evidence_message_ids": [ids[0]]}, {"text": "伪造", "evidence_message_ids": ["not-real"]}], "summary": "summary"}
        with self.assertRaises(ValueError): memory.commit_memory_review("Alice", snapshot, candidate)
        self.assertEqual(self.notes.read_text(encoding="utf-8"), memory.EMPTY_MEMORY)

    def test_model_edit_requires_current_revision_and_exact_old_text(self):
        evidence = self.say("记住我叫小林")
        state = memory.read_memory("Alice")
        first = memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [evidence])
        with self.assertRaises(ValueError): memory.edit_memory("Alice", state["revision"], "", "过期修改", [evidence])
        with self.assertRaises(ValueError): memory.edit_memory("Alice", first["revision"], "不存在的句子", "修改", [evidence])

    def test_explicit_forget_excludes_older_messages(self):
        first = self.say("我叫小林")
        state = memory.read_memory("Alice")
        state = memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [first])
        forget = self.say("忘记我的名字")
        memory.edit_memory("Alice", state["revision"], "用户叫小林。", "", [forget])
        self.assertEqual(memory.search_memory("Alice", "小林")["messages"], [])
        self.assertNotIn("小林", memory.get_memory_prompt("Alice"))

    def test_background_cannot_delete_or_overwrite_manual_edit(self):
        self.notes.write_text("用户手工备注。", encoding="utf-8")
        ids, snapshot = self.window()
        with self.assertRaises(ValueError):
            memory.commit_memory_review("Alice", snapshot, {"operations": [{"old_text": "用户手工备注。", "text": "", "evidence_message_ids": [ids[0]]}], "summary": ""})

    def test_clear_history_clears_notes_and_cannot_remigrate_legacy(self):
        self.say("旧消息")
        (self.notes.parent / "short_memory.json").write_text(json.dumps({"messages": [{"role": "human", "content": "不应迁回"}]}), encoding="utf-8")
        self.assertTrue(memory.delete_history("Alice", self.uid))
        memory.create_new_history("Alice")
        self.assertEqual(memory.get_history("Alice"), [])
        self.assertEqual(self.notes.read_text(encoding="utf-8"), memory.EMPTY_MEMORY)

    def test_legacy_migration_preserves_source_and_ignores_adaptation(self):
        path = self.root / "Legacy"; path.mkdir()
        raw = {"profile": {"preferred_name": "小林", "likes": [{"value": "音乐", "status": "active"}, {"value": "旧偏好", "status": "forgotten"}]}, "adaptation": {"affection": "affectionate"}, "character_self": {"preferences": [{"value": "雨天"}]}}
        old = path / "core_memory.json"; old.write_text(json.dumps(raw), encoding="utf-8")
        archive = path / "short_memory.json"; archive.write_text(json.dumps({"messages": [{"role": "human", "content": "过去对话"}]}), encoding="utf-8")
        before = old.read_bytes()
        memory.create_new_history("Legacy")
        text = memory.read_memory("Legacy")["text"]
        self.assertIn("小林", text); self.assertIn("角色曾表达", text)
        self.assertNotIn("affectionate", text); self.assertNotIn("旧偏好", text)
        self.assertEqual(before, old.read_bytes())
        memory.create_new_history("Legacy")
        self.assertEqual(len(memory.get_history("Legacy")), 1)

    def test_invalid_legacy_is_preserved_and_reported(self):
        path = self.root / "Broken"; path.mkdir()
        (path / "core_memory.json").write_text("{bad", encoding="utf-8")
        with self.assertRaises(memory.HistoryStorageError): memory.create_new_history("Broken")
        self.assertEqual((path / "core_memory.json").read_text(), "{bad")

    def test_oversized_or_bad_encoding_notes_are_never_replaced(self):
        for value in (b"\xff", b"x" * (memory.MAX_NOTES_CHARS + 1)):
            self.notes.write_bytes(value)
            with self.assertRaises(memory.HistoryStorageError): memory.read_memory("Alice")
            self.assertEqual(self.notes.read_bytes(), value)

    def test_threaded_writes_and_metadata_are_transactional(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda i: self.say(f"message-{i}"), range(60)))
        self.assertEqual(len(memory.get_history("Alice")), 60)
        memory.update_metadata("Alice", self.uid, {"resume_id": "one"})
        memory.update_metadata("Alice", self.uid, {"agent_type": "hume"})
        self.assertEqual(memory.get_metadata("Alice", self.uid), {"resume_id": "one", "agent_type": "hume"})

    def test_two_processes_do_not_lose_messages(self):
        script = "import sys;from pathlib import Path;from src.open_llm_vtuber import chat_history_manager as h;h.CHAT_HISTORY_DIR=Path(sys.argv[1]);[h.store_message('Alice',h.SINGLE_HISTORY_UID,'human',sys.argv[2]+str(i)) for i in range(12)]"
        env = {**os.environ, "PYTHONPATH": str(BACKEND)}
        processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(self.root), name], env=env) for name in ("A", "B")]
        for process in processes: self.assertEqual(process.wait(timeout=20), 0)
        self.assertEqual(len(memory.get_history("Alice")), 24)

    def test_invalid_paths_and_unknown_history_ids(self):
        for uid in ("../outside", "CON", "C:/tmp", "a\\b"):
            with self.assertRaises(ValueError): memory.create_new_history(uid)
        with self.assertRaises(ValueError): memory.get_history("Alice", "another")
        self.assertFalse(memory.delete_history("Alice", "another"))

    def test_modified_message_updates_search_index(self):
        self.say("草莓方案")
        memory.modify_latest_message("Alice", self.uid, "human", "香蕉方案")
        self.assertEqual(memory.search_memory("Alice", "草莓")["messages"], [])
        self.assertIn("香蕉", memory.search_memory("Alice", "香蕉")["messages"][0]["content"])

    def test_real_agent_message_builder_reloads_notes_boundary(self):
        builder = source_method(BACKEND / "src/open_llm_vtuber/agent/agents/basic_memory_agent.py", "_to_messages", {"get_context_history": memory.get_context_history})
        self.say("较早消息", "ai")
        current = self.say("当前问题")
        agent = types.SimpleNamespace(_memory_conf_uid="Alice", _memory=[{"role": "assistant", "content": "过期内存"}],
            _to_text_prompt=lambda *a, **k: "当前问题", _add_message=lambda *a: None)
        data = types.SimpleNamespace(metadata={"memory_message_id": current}, images=None)
        messages = builder(agent, data)
        self.assertEqual(messages[0]["content"], "较早消息")
        self.assertEqual(len(messages), 2)
        self.notes.write_text("新资料", encoding="utf-8")
        self.assertEqual(len(builder(agent, data)), 2)

    def test_plain_persona_file_is_sufficient_and_hot_reloadable(self):
        (self.root / "新人.md").write_text("你叫新人。", encoding="utf-8")
        first = text_character(self.root, "新人.md")
        (self.root / "新人.md").write_text("你叫新人，使用中文交流。", encoding="utf-8")
        self.assertIn("中文", read_prompt(self.root, "新人.md"))
        self.assertEqual(first["conf_uid"], text_character(self.root, "新人.md")["conf_uid"])
        with self.assertRaises(ValueError): persona_path(self.root, "../outside.md")
        loader = source_method(BACKEND / "src/open_llm_vtuber/config_manager/utils.py", "load_character_profile",
            {"persona_path": persona_path, "read_prompt": read_prompt, "text_character": text_character})
        self.assertEqual(loader(str(self.root), "新人.md")["persona_prompt"], "你叫新人，使用中文交流。")

    def test_late_response_cannot_reintroduce_notes_edited_during_generation(self):
        self.notes.write_text("旧资料", encoding="utf-8")
        self.say("旧资料")
        epoch = memory.memory_epoch("Alice")
        self.notes.write_text("新资料", encoding="utf-8")
        result = memory.store_message("Alice", self.uid, "ai", "旧资料的迟到回答", expected_epoch=epoch)
        self.assertIsNone(result)
        self.assertEqual(memory.get_context_history("Alice"), [])

    def test_heading_without_blank_line_and_long_paragraph_can_be_retrieved(self):
        self.notes.write_text("# 记忆\n姓名是小林。\n\n" + "普通背景。" * 1500 + "特殊的青柠方案。", encoding="utf-8")
        self.assertIn("青柠方案", memory.get_memory_prompt("Alice", "青柠方案"))
        self.assertIn("小林", memory.get_memory_prompt("Alice", "小林"))

    def test_existing_text_overrides_legacy_context_on_first_migration(self):
        path = self.root / "Manual"; path.mkdir()
        (path / "memory.md").write_text("只使用这份新资料。", encoding="utf-8")
        (path / "short_memory.json").write_text(json.dumps({"messages": [{"role": "human", "content": "旧资料"}]}), encoding="utf-8")
        memory.create_new_history("Manual")
        self.assertEqual(memory.get_context_history("Manual"), [])
        self.assertIn("新资料", memory.get_memory_prompt("Manual"))

    def test_editing_name_preserves_project_in_same_message_and_raw_archive(self):
        evidence = self.say("我叫小林。项目使用 Python，接口还没写完。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [evidence])
        self.notes.write_text("用户叫小陈。", encoding="utf-8")
        context = memory.get_context_history("Alice")
        self.assertNotIn("小林", json.dumps(context, ensure_ascii=False))
        self.assertIn("接口还没写完", context[0]["content"])
        self.assertIn("小陈", memory.get_memory_prompt("Alice"))
        self.assertIn("小林", memory.get_history("Alice")[0]["content"])
        self.assertEqual(memory.search_memory("Alice", "小林")["messages"], [])
        self.assertIn("Python", memory.search_memory("Alice", "Python")["messages"][0]["content"])

    def test_forget_last_note_does_not_reset_unrelated_chat(self):
        evidence = self.say("我喜欢草莓。")
        self.say("网页还需要加一个按钮。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户喜欢草莓。", [evidence])
        forget = self.say("忘记这项偏好。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "用户喜欢草莓。", "", [forget])
        self.assertIn("按钮", json.dumps(memory.get_context_history("Alice"), ensure_ascii=False))
        self.assertEqual(memory.search_memory("Alice", "草莓")["messages"], [])
        self.say("今天再聊草莓。")
        self.assertEqual(len(memory.search_memory("Alice", "草莓")["messages"]), 1)

    def test_forgetting_updated_note_does_not_hide_old_source_remaining_project(self):
        first = self.say("我叫小林。项目仍在进行。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [first])
        current = self.say("现在叫我小陈。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "用户叫小林。", "用户叫小陈。", [current])
        forget = self.say("忘记名字。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "用户叫小陈。", "", [forget])
        visible = json.dumps(memory.get_context_history("Alice"), ensure_ascii=False)
        self.assertIn("项目仍在进行", visible)
        self.assertNotIn("小林", visible)
        self.assertNotIn("小陈", visible)

    def test_removing_one_preference_preserves_other_recorded_preference(self):
        source = self.say("用户喜欢音乐。用户喜欢绘画。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户喜欢音乐。\n用户喜欢绘画。", [source])
        self.notes.write_text("用户喜欢绘画。", encoding="utf-8")
        visible = json.dumps(memory.get_context_history("Alice"), ensure_ascii=False)
        self.assertIn("喜欢绘画", visible)
        self.assertNotIn("喜欢音乐", visible)

    def test_formatting_and_reordering_notes_do_not_hide_history(self):
        self.notes.write_text("- 用户叫小林。\n- 项目使用 Python。", encoding="utf-8")
        self.say("我叫小林，项目使用 Python。")
        self.notes.write_text("# 我的笔记\n\n* 项目使用 Python。\n\n* 用户叫小林。", encoding="utf-8")
        self.assertIn("小林", memory.get_context_history("Alice")[0]["content"])

    def test_deleted_paraphrase_uses_provenance_and_does_not_reach_next_review(self):
        source = self.say("我最享受下班后戴上耳机听几首歌。")
        state = memory.read_memory("Alice")
        memory.edit_memory("Alice", state["revision"], "", "用户喜欢音乐。", [source])
        self.say("另外，我们正在做日历页面。")
        self.notes.write_text("目前正在做日历页面。", encoding="utf-8")
        ids, snapshot = self.window()
        self.assertIsNotNone(snapshot)
        self.assertNotIn(source, [message["id"] for message in snapshot["messages"]])
        self.assertIn("日历页面", json.dumps(memory.get_context_history("Alice"), ensure_ascii=False))
        self.assertEqual(memory.search_memory("Alice", "耳机")["messages"], [])

    def test_manual_note_without_provenance_filters_matching_text_with_no_fts(self):
        self.notes.write_text("用户叫小林。\n项目使用 Python。", encoding="utf-8")
        self.say("我叫小林。还要继续检查项目。")
        with memory._session("Alice") as (db, _, __): memory._set(db, "fts", False)
        self.notes.write_text("项目使用 Python。", encoding="utf-8")
        self.assertNotIn("小林", json.dumps(memory.get_context_history("Alice"), ensure_ascii=False))
        self.assertIn("继续检查项目", memory.search_memory("Alice", "项目")["messages"][0]["content"])

    def test_search_alternatives_return_evidence_with_neighboring_dialogue(self):
        before = self.say("昨天讨论的那个页面怎么样了？")
        found = self.say("日历页面已经加好了月份切换。", "ai")
        after = self.say("接着完善移动端。")
        result = memory.search_memory("Alice", "日期选择器", alternative_queries=["日历", "月份切换"])
        self.assertEqual(result["messages"][0]["id"], found)
        self.assertEqual([m["id"] for m in result["messages"][0]["context"]], [before, after])
        with self.assertRaises(ValueError): memory.search_memory("Alice", "日历", alternative_queries=["a"] * 4)

    def test_review_waits_for_volume_and_reduces_empty_review_frequency(self):
        for i in range(6): self.say("嗯")
        self.assertIsNone(memory.prepare_memory_review("Alice"))
        for i in range(18): self.say("好")
        snapshot = memory.prepare_memory_review("Alice")
        self.assertIsNotNone(snapshot)
        memory.commit_memory_review("Alice", snapshot, {"operations": [], "summary": "简短回应。"})
        for i in range(24): self.say("嗯")
        self.assertIsNone(memory.prepare_memory_review("Alice"))
        for i in range(24): self.say("好的")
        self.assertIsNotNone(memory.prepare_memory_review("Alice"))

    def test_failed_review_has_persisted_backoff_and_keeps_raw_history(self):
        ids, snapshot = self.window()
        with patch.object(memory.time, "time", return_value=1000):
            memory.record_review_failure("Alice", snapshot)
            self.assertIsNone(memory.prepare_memory_review("Alice"))
            memory.create_new_history("Alice")
            self.assertIsNone(memory.prepare_memory_review("Alice"))
        with patch.object(memory.time, "time", return_value=1061):
            self.assertIsNotNone(memory.prepare_memory_review("Alice"))
        self.assertEqual(len(memory.get_history("Alice")), 12)


class AsyncMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_that_corrects_memory_still_archives_its_reply(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory) / "memory"):
            root = Path(directory)
            (root / "Alice.md").write_text("你叫 Alice。", encoding="utf-8")
            runner, context, sent = self.conversation_fixture(root)
            source = memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "human", "我叫小林。")
            state = memory.read_memory("Alice")
            memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [source])
            context.pc_tools = types.SimpleNamespace(memory_edit_epoch=None)
            original_chat = context.agent_engine.chat
            async def chat(data):
                state = memory.read_memory("Alice")
                result = memory.edit_memory("Alice", state["revision"], "用户叫小林。", "用户叫小陈。", [data.metadata["memory_message_id"]])
                context.pc_tools.memory_edit_epoch = result["_memory_epoch"]
                async for item in original_chat(data): yield item
            context.agent_engine.chat = chat
            await runner(context, sent.append_async, "client", "现在叫我小陈。")
            self.assertEqual(memory.get_history("Alice")[-1]["role"], "ai")
            self.assertIn("小陈", memory.get_memory_prompt("Alice"))

    async def test_conversation_turn_archives_real_text_and_reloads_persona(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory) / "memory"):
            root = Path(directory)
            (root / "Alice.md").write_text("你叫 Alice。", encoding="utf-8")
            runner, context, sent = self.conversation_fixture(root)
            await runner(context, sent.append_async, "client", "你好")
            self.assertEqual([m["role"] for m in memory.get_history("Alice")], ["human", "ai"])
            self.assertEqual(context.agent_engine.messages[-1]["content"][0]["text"], "你好")
            (root / "Alice.md").write_text("你叫 Alice，使用中文。", encoding="utf-8")
            await runner(context, sent.append_async, "client", "继续")
            self.assertIn("使用中文", context.agent_engine.system)
            self.assertEqual(len(context.agent_engine.messages), 3)

    async def test_queued_proactive_turn_drops_old_notes_and_silence_is_not_archived(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory) / "memory"):
            root = Path(directory)
            (root / "Alice.md").write_text("你叫 Alice。", encoding="utf-8")
            runner, context, sent = self.conversation_fixture(root)
            memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "human", "旧资料")
            (root / "memory/Alice/memory.md").write_text("旧资料", encoding="utf-8")
            epoch = memory.memory_epoch("Alice")
            (root / "memory/Alice/memory.md").write_text("新资料", encoding="utf-8")
            context.proactive_utterances = ["旧资料"]
            context.agent_engine.outputs = [{"type": "proactive-silence"}]
            await runner(context, sent.append_async, "client", "含有旧资料的排队提示", metadata={
                "memory_epoch_at_queue": epoch, "proactive_speak": True, "proactive_mode": "automatic", "skip_history": True,
                "proactive_request": {}, "proactive_return": {"recent_utterances": ["旧资料"]}})
            self.assertNotIn("旧资料", json.dumps(context.agent_engine.messages, ensure_ascii=False))
            self.assertIn("新资料", context.agent_engine.system)
            self.assertEqual(context.proactive_utterances, [])
            self.assertEqual(len(memory.get_history("Alice")), 1)
            self.assertTrue(any(json.loads(item).get("type") == "proactive-silence" for item in sent))

    def conversation_fixture(self, root):
        from src.open_llm_vtuber.companion_session import event_reaction_prompt
        from src.open_llm_vtuber.persona_text import CONVERSATION_GUIDANCE
        from src.open_llm_vtuber.proactive_conversation import build_return_context_prompt, build_proactive_prompt
        async def noop(*args, **kwargs): pass
        async def identity(value, *args, **kwargs): return value
        class Sent(list):
            async def append_async(self, text): self.append(text)
        class Sentence:
            text = "你好，很高兴认识你。"
        logger = types.SimpleNamespace(**{key: lambda *a: None for key in ("info", "debug", "warning", "error", "exception")})
        builder = source_method(BACKEND / "src/open_llm_vtuber/agent/agents/basic_memory_agent.py", "_to_messages", {"get_context_history": memory.get_context_history})
        class Agent:
            _memory_conf_uid = "Alice"
            _memory = []
            outputs = [Sentence()]
            _to_text_prompt = staticmethod(lambda data, **kwargs: data.text)
            _add_message = staticmethod(lambda *args: None)
            def set_system(self, value): self.system = value
            async def chat(self, data):
                self.messages = builder(self, data)
                for item in self.outputs: yield item
        async def process_output(**kwargs): return kwargs["output"].text
        namespace = {"asyncio": asyncio, "json": json, "logger": logger, "memory_epoch": memory.memory_epoch, "store_message": memory.store_message,
            "TTSTaskManager": object, "with_turn_id": lambda send, turn: send, "send_conversation_start_signals": noop,
            "process_queued_user_inputs": identity, "augment_text_with_screen_context": identity,
            "build_return_context_prompt": build_return_context_prompt, "build_proactive_prompt": build_proactive_prompt,
            "event_reaction_prompt": event_reaction_prompt,
            "_attach_live_workspace_context": lambda ctx, text, meta: meta,
            "create_batch_input": lambda **kw: types.SimpleNamespace(text=kw["input_text"], images=kw["images"], metadata=kw["metadata"]),
            "SentenceOutput": Sentence, "AudioOutput": type("Audio", (), {}), "process_agent_output": process_output,
            "finalize_conversation_turn": noop, "cleanup_conversation": noop,
            "np": types.SimpleNamespace(random=types.SimpleNamespace(choice=lambda items: items[0])), "EMOJI_LIST": ["test"]}
        runner = source_method(BACKEND / "src/open_llm_vtuber/conversations/single_conversation.py", "process_single_conversation", namespace)
        prompt = source_method(BACKEND / "src/open_llm_vtuber/service_context.py", "construct_system_prompt",
            {"logger": logger, "read_prompt": read_prompt, "get_memory_prompt": memory.get_memory_prompt, "CONVERSATION_GUIDANCE": CONVERSATION_GUIDANCE})
        context = types.SimpleNamespace(character_config=types.SimpleNamespace(conf_uid="Alice", conf_name="Alice", character_name="Alice", human_name="用户", persona_file="Alice.md", persona_prompt="备用提示"),
            system_config=types.SimpleNamespace(config_alts_dir=str(root), tool_prompts={}), runtime_control=types.SimpleNamespace(settings={"project_folder": ""}),
            history_uid=memory.SINGLE_HISTORY_UID, proactive_utterances=[], asr_engine=None, agent_engine=Agent(), avatar_model=None, translate_engine=None, get_current_tts_engine=lambda: None)
        context.construct_system_prompt = types.MethodType(prompt, context)
        return runner, context, Sent()

    async def test_real_agent_background_review_reaches_next_prompt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory)):
            for i in range(6):
                memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "human", f"讨论青柠方案{i}")
                memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "ai", "一起检查了接口。" * 150)
            class LLM:
                async def chat_completion(self, **kwargs):
                    data = json.loads(kwargs["messages"][0]["content"])
                    yield json.dumps({"operations": [{"text": "用户讨论过青柠方案。", "evidence_message_ids": [data["messages"][0]["id"]]}], "summary": "双方检查过接口。"})
            path = BACKEND / "src/open_llm_vtuber/agent/agents/basic_memory_agent.py"
            namespace = {"asyncio": asyncio, "review_memory": review_memory,
                "prepare_memory_review": memory.prepare_memory_review, "commit_memory_review": memory.commit_memory_review, "record_review_failure": memory.record_review_failure,
                "logger": types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)}
            run = source_method(path, "_run_memory_review", namespace)
            schedule = source_method(path, "schedule_memory_review", namespace)
            agent = types.SimpleNamespace(_memory_conf_uid="Alice", _memory_character_name="Alice", _llm=LLM(), _memory_review_task=None)
            agent._run_memory_review = types.MethodType(run, agent)
            self.assertTrue(schedule(agent))
            self.assertFalse(schedule(agent))  # No concurrent duplicate review.
            await agent._memory_review_task
            self.assertIn("青柠方案", memory.get_memory_prompt("Alice"))
            self.assertIn("检查过接口", memory.get_memory_prompt("Alice"))
            self.assertFalse(schedule(agent))

    async def test_review_provider_failure_does_not_change_snapshot(self):
        class LLM:
            async def chat_completion(self, **kwargs):
                raise ConnectionError("offline")
                yield ""
        snapshot = {"notes": "原记忆", "summary": "", "messages": []}
        with self.assertRaises(ConnectionError): await review_memory(LLM(), snapshot, "小可")
        self.assertEqual(snapshot["notes"], "原记忆")

    async def test_review_uses_same_model_and_closes_stream(self):
        closed, calls = [], []
        class LLM:
            model = "configured-model"
            max_tokens = 9000
            async def chat_completion(self, **kwargs):
                calls.append((self.model, self.max_tokens))
                try: yield '{"operations":[],"summary":"记录"}'
                finally: closed.append(True)
        llm = LLM()
        result = await review_memory(llm, {"notes": "", "summary": "", "messages": []}, "小可")
        self.assertEqual(result["summary"], "记录")
        self.assertEqual(calls, [("configured-model", 4096)])
        self.assertEqual(llm.max_tokens, 9000)
        self.assertTrue(closed)

    async def test_proactive_silence_is_filtered_before_speech(self):
        @optional_proactive_output()
        async def response(data):
            for chunk in (" ", "<sil", "ence/", "> "): yield chunk
        data = types.SimpleNamespace(metadata={"proactive_speak": True})
        self.assertEqual([x async for x in response(data)], [{"type": "proactive-silence"}])
        data.metadata = {}
        self.assertEqual("".join([x async for x in response(data)]), " <silence/> ")

    async def test_proactive_normal_response_streams_without_rewriting(self):
        @optional_proactive_output()
        async def response(data):
            for chunk in ("今天", "想聊音乐。"): yield chunk
        result = [x async for x in response(types.SimpleNamespace(metadata={"proactive_speak": True}))]
        self.assertEqual("".join(result), "今天想聊音乐。")


if __name__ == "__main__": unittest.main()
