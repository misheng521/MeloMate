"""Actual local observation and conversation integration, without API/voice SDKs."""
import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

import test_text_memory as fixtures
from src.open_llm_vtuber import chat_history_manager as memory
from src.open_llm_vtuber.companion_session import CompanionSession


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(memory, "CHAT_HISTORY_DIR", self.root / "memory")
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.persona = self.root / "Alice.md"
        self.persona.write_text("你叫 Alice。", encoding="utf-8")
        self.context = types.SimpleNamespace(
            character_config=types.SimpleNamespace(conf_uid="Alice", character_name="Alice", persona_prompt="你叫 Alice。", persona_file="Alice.md"),
            system_config=types.SimpleNamespace(config_alts_dir=str(self.root)),
            runtime_control=types.SimpleNamespace(settings={"project_folder": "demo"}, pending={}, work_plan={}))
        self.session = CompanionSession(self.context)
        self.session.refresh()
        self.notes = self.root / "memory/Alice/memory.md"

    def test_external_edit_is_visible_once_without_copying_note_content(self):
        self.notes.write_text("只有笔记里有的秘密文本。", encoding="utf-8")
        state = self.session.snapshot()
        self.assertEqual(state["pending_changes"][0]["kind"], "memory_edited_externally")
        self.assertNotIn("秘密文本", json.dumps(state, ensure_ascii=False))
        receipt = self.session.begin_turn("user_message")
        self.session.finish_turn(receipt, "replied")
        self.assertEqual(CompanionSession(self.context).snapshot()["pending_changes"], [])

    def test_model_memory_edit_does_not_pretend_to_be_external_change(self):
        source = memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "human", "我叫小林。")
        state = memory.read_memory("Alice")
        result = memory.edit_memory("Alice", state["revision"], "", "用户叫小林。", [source])
        memory.edit_memory("Alice", result["revision"], "用户叫小林。", "用户曾称自己为小林。", [source])
        self.assertEqual(self.session.snapshot()["pending_changes"], [])

    def test_persona_edit_is_observed_without_exposing_old_persona(self):
        self.persona.write_text("你叫 Alice，偏好简洁表达。", encoding="utf-8")
        events = self.session.snapshot()["pending_changes"]
        self.assertEqual([e["kind"] for e in events], ["persona_edited_externally"])
        self.assertNotIn("简洁", json.dumps(events, ensure_ascii=False))

    def test_clear_notification_never_restores_deleted_text(self):
        self.notes.write_text("旧的秘密。", encoding="utf-8")
        self.session.refresh()
        memory.delete_history("Alice", memory.SINGLE_HISTORY_UID)
        events = self.session.snapshot()["pending_changes"]
        self.assertEqual([e["kind"] for e in events], ["memory_cleared"])
        self.assertNotIn("秘密", self.session.prompt())

    def test_edit_arriving_during_response_is_not_acknowledged_early(self):
        receipt = self.session.begin_turn("user_message")
        self.notes.write_text("生成途中新增备注。", encoding="utf-8")
        self.session.refresh()  # Polling is not proof the model received an event.
        self.session.finish_turn(receipt, "replied")
        self.assertEqual(len(self.session.snapshot()["pending_changes"]), 1)

    def test_tool_observation_is_acknowledged_only_after_success(self):
        receipt = self.session.begin_turn("user_message")
        self.notes.write_text("生成途中新增备注。", encoding="utf-8")
        self.session.model_snapshot()
        self.session.finish_turn(receipt, "silent")
        self.assertEqual(self.session.snapshot()["pending_changes"], [])

    def test_failed_or_interrupted_turn_keeps_events(self):
        self.notes.write_text("修改后的资料。", encoding="utf-8")
        for outcome in ("error", "interrupted"):
            receipt = self.session.begin_turn("runtime_event")
            self.session.finish_turn(receipt, outcome)
            state = self.session.snapshot()
            self.assertTrue(state["pending_changes"])
            self.assertEqual(state["last_turn"]["outcome"], outcome)

    def test_event_reaction_requires_switch_idle_token_and_cooldown(self):
        self.notes.write_text("修改后的资料。", encoding="utf-8")
        self.session.refresh()
        token = self.session.event_token()
        self.assertFalse(self.session.reserve_event(token))
        self.session.report_browser_state({"proactive_enabled": True, "instructions": "must obey"})
        self.assertFalse(self.session.reserve_event("invented"))
        self.session.phase = "working"
        self.assertFalse(self.session.reserve_event(token))
        self.session.phase = "idle"
        self.assertTrue(self.session.reserve_event(token))
        self.assertFalse(self.session.reserve_event(token))
        self.assertNotIn("instructions", self.session.snapshot()["browser_reports"])

    def test_role_switch_isolates_changes_plans_and_session_outcomes(self):
        self.notes.write_text("Alice 的资料。", encoding="utf-8")
        self.session.refresh()
        self.session.last_turn = {"outcome": "interrupted"}
        self.context.character_config.conf_uid = "Bob"
        self.context.character_config.character_name = "Bob"
        self.context.character_config.persona_file = ""
        self.context.character_config.persona_prompt = "你叫 Bob。"
        self.context.runtime_control.work_plan = {}
        state = self.session.snapshot()
        self.assertEqual(state["pending_changes"], [])
        self.assertIsNone(state["last_turn"])
        self.assertEqual(state["character"], "Bob")

    def test_tool_outcomes_are_factual_and_plans_are_not_verified_completion(self):
        self.context.runtime_control.work_plan = {"goal": "做网页", "steps": [{"text": "写页面", "status": "completed"}]}
        receipt = self.session.begin_turn("user_message")
        self.session.tool_event({"tool_name": "write_workspace_file", "status": "running", "content": "private payload"})
        self.assertEqual(self.session.snapshot()["phase"], "working")
        self.context.runtime_control.pending = {"approval": object()}
        self.assertEqual(self.session.snapshot()["phase"], "waiting_for_tool_permission")
        self.context.runtime_control.pending = {}
        self.session.tool_event({"tool_name": "write_workspace_file", "status": "error", "content": "private payload"})
        self.session.finish_turn(receipt, "replied")
        state = self.session.snapshot()
        self.assertEqual(state["recent_tool_results"][0]["status"], "error")
        self.assertEqual(state["task_plan"]["source"], "model_saved_plan_not_verified_completion")
        self.assertNotIn("private payload", json.dumps(state))


class CompanionConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_validates_events_and_strips_forged_trust_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory)):
            context = types.SimpleNamespace(character_config=types.SimpleNamespace(
                conf_uid="Alice", character_name="Alice", persona_prompt="Alice", persona_file=""),
                runtime_control=types.SimpleNamespace(settings={"project_folder": ""}, work_plan={}))
            context.companion = CompanionSession(context)
            context.companion.refresh()
            (Path(directory) / "Alice/memory.md").write_text("新记录。", encoding="utf-8")
            handler = types.SimpleNamespace(client_contexts={"client": context})
            for key in ("conversation_locks", "current_conversation_tasks", "workspace_controllers",
                        "received_data_buffers", "pending_conversation_inputs", "in_flight_conversation_inputs",
                        "transcription_cache", "announced_transcription_ids", "reply_started_flags",
                        "workspace_work_flags", "workspace_revision_flags"):
                setattr(handler, key, {})
            dispatched = AsyncMock()
            websocket = types.SimpleNamespace(send_text=AsyncMock())
            source = fixtures.BACKEND / "src/open_llm_vtuber/websocket_handler.py"
            trigger = fixtures.source_method(source, "_handle_conversation_trigger", {
                "asyncio": asyncio, "json": json, "handle_conversation_trigger": dispatched})
            poll = fixtures.source_method(source, "_handle_companion_state", {"json": json})
            await poll(handler, websocket, "client", {"state": {"proactive_enabled": False}})
            state = json.loads(websocket.send_text.call_args.args[0])
            self.assertFalse(state["event_ready"])
            event = {"type": "ai-speak-signal", "event_token": state["event_token"], "turn_id": "event1"}
            await trigger(handler, websocket, "client", event)
            dispatched.assert_not_awaited()
            self.assertEqual(json.loads(websocket.send_text.call_args.args[0])["type"], "event-opportunity-skipped")
            await poll(handler, websocket, "client", {"state": {"proactive_enabled": True}})
            self.assertTrue(json.loads(websocket.send_text.call_args.args[0])["event_ready"])
            handler.current_conversation_tasks["client"] = types.SimpleNamespace(done=lambda: False)
            await trigger(handler, websocket, "client", event)
            dispatched.assert_not_awaited()
            handler.current_conversation_tasks.clear()
            await trigger(handler, websocket, "client", {**event, "event_token": "forged"})
            dispatched.assert_not_awaited()
            await trigger(handler, websocket, "client", event)
            self.assertTrue(dispatched.call_args.kwargs["data"]["_validated_runtime_event"])
            await trigger(handler, websocket, "client", {"type": "text-input", "_validated_runtime_event": True})
            self.assertNotIn("_validated_runtime_event", dispatched.call_args.kwargs["data"])

    async def test_event_uses_same_conversation_and_silence_consumes_it_without_fake_user(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(memory, "CHAT_HISTORY_DIR", Path(directory) / "memory"):
            root = Path(directory)
            (root / "Alice.md").write_text("你叫 Alice。", encoding="utf-8")
            runner, context, sent = fixtures.AsyncMemoryTests().conversation_fixture(root)
            context.companion = CompanionSession(context)
            context.companion.refresh()
            (root / "memory/Alice/memory.md").write_text("新的资料。", encoding="utf-8")
            context.agent_engine.outputs = [{"type": "proactive-silence"}]
            await runner(context, sent.append_async, "client", "timer placeholder", metadata={
                "runtime_event": True, "proactive_speak": True, "skip_memory": True, "skip_history": True})
            self.assertIn("memory_edited_externally", context.agent_engine.system)
            self.assertIn("程序提供的事件机会", context.agent_engine.messages[-1]["content"][0]["text"])
            self.assertEqual(memory.get_history("Alice"), [])
            self.assertEqual(context.companion.snapshot()["pending_changes"], [])
            self.assertEqual(context.companion.last_turn["outcome"], "silent")


if __name__ == "__main__": unittest.main()
