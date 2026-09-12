"""Source-only regressions. No server, model key, Docker or voice packages needed.

MCP transport and logger imports use the existing policy-test stubs; assertions
exercise the real policy, executor, storage, adapter and runtime implementations.
"""
import asyncio
import ast
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_daily_tool_executor_policy  # noqa: F401: optional-dependency stubs
import workspace_core as workspace
import project_runtime
from src.open_llm_vtuber.runtime_control import RuntimeControl, scope_arguments
from src.open_llm_vtuber.workspace_agent import WorkspaceAgentSession
from src.open_llm_vtuber.mcpp.json_detector import StreamJSONDetector
from src.open_llm_vtuber.mcpp.results import collect_result, image_messages
from src.open_llm_vtuber.mcpp.tool_executor import ToolExecutor
from src.open_llm_vtuber.mcpp.tool_adapter import ToolAdapter
from src.open_llm_vtuber.mcpp.tool_manager import ToolManager
from src.open_llm_vtuber.mcpp.types import FormattedTool
from src.open_llm_vtuber.mcpp.types import ToolCallObject
from src.open_llm_vtuber.mcpp.server_registry import ServerRegistry
from src.open_llm_vtuber.mcpp.mcp_client import MCPClient


class ProjectScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.multiple(workspace, ROOT=self.root, WORKSPACE_ROOT=self.root / "workspace")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.runtime = RuntimeControl()
        self.runtime.configure({"project_folder": "projects/demo"})
        self.policy = self.runtime.policy("Alice")
        workspace.create_workspace_folder("Alice", "projects/demo")

    def test_scope_handles_relative_and_returned_paths_without_double_prefix(self):
        for path in ("index.html", "projects/demo/index.html", "Alice/projects/demo/index.html"):
            self.assertEqual(scope_arguments("read_workspace_file", {"path": path}, self.policy),
                             {"persona": "Alice", "path": "projects/demo/index.html"})
        self.assertEqual(scope_arguments("list_workspace", {}, self.policy)["folder"], "projects/demo")
        self.assertEqual(scope_arguments("run_workspace_command", {"argv": ["python3", "app.py"]}, self.policy)["cwd"], "projects/demo")
        self.assertEqual(scope_arguments("get_workspace_runtime", {}, self.policy), {})

    def test_rejects_traversal_absolute_and_other_persona(self):
        for path in ("../other.txt", "C:/Windows/a", "/outside", ".control/state.json", "a/../../x"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                scope_arguments("read_workspace_file", {"path": path}, self.policy)
        with self.assertRaises(ValueError):
            scope_arguments("list_workspace", {"persona": "Bob"}, self.policy)

    def test_recovery_cannot_list_or_restore_sibling_project(self):
        workspace.write_workspace_file("Alice", "projects/demo", "a.txt", "own")
        workspace.write_workspace_file("Alice", "other", "b.txt", "sibling")
        own = json.loads(workspace.delete_workspace_item("Alice", "projects/demo/a.txt"))
        other = json.loads(workspace.delete_workspace_item("Alice", "other/b.txt"))
        entries = json.loads(workspace.list_workspace_trash("Alice", "projects/demo"))["entries"]
        self.assertEqual([e["id"] for e in entries], [own["trash_id"]])
        with self.assertRaises(ValueError):
            workspace.restore_workspace_item("Alice", other["trash_id"], folder="projects/demo")
        with self.assertRaises(ValueError):
            workspace.restore_workspace_item("Alice", own["trash_id"], destination="other/a.txt", folder="projects/demo")
        workspace.restore_workspace_item("Alice", own["trash_id"], folder="projects/demo")
        self.assertEqual(workspace.workspace_path("Alice", "projects/demo/a.txt").read_text(), "own")

    def test_live_page_scope_and_action_normalization(self):
        report = {"state": {"page": {"path": "other/index.html"}}}
        with patch.object(workspace, "read_workspace_state_file", return_value=report):
            with self.assertRaises(ValueError):
                workspace.read_workspace_state("Alice", "page", "projects/demo")
            with self.assertRaises(ValueError):
                workspace.send_workspace_action("Alice", expected_page_id="page", folder="projects/demo")
        executor = ToolExecutor(None, None)
        arguments = {"persona": "Alice", "page_id": "page", "state_version": 1, "action_id": "save"}
        scoped = scope_arguments("act_workspace_page", arguments, self.policy)
        normalized, error = executor.apply_tool_policy("act_workspace_page", scoped, self.policy)
        self.assertIsNone(error)
        self.assertEqual(normalized["folder"], "projects/demo")

    def test_runtime_policy_does_not_depend_on_question_wording(self):
        session = WorkspaceAgentSession(types.SimpleNamespace(runtime_control=self.runtime))
        for request in ("做个网页", "能不能做个网页？", "帮我做个介绍如何养猫的网页", "十分钟后叫我喝水"):
            policy = session.begin_user_turn(request, "Alice")
            self.assertIn("write_workspace_file", policy["available_workspace_tools"])
            self.assertIn("create_reminder", policy["user_authorized_daily_tools"])
        self.assertFalse(session.page_action_authorized("Alice", "untrusted-page", claim=True))
        session.finish_task()
        self.assertTrue(session.active_task.completed)

    def test_static_check_reports_errors_and_never_executes_project(self):
        project = workspace.workspace_path("Alice", "projects/demo")
        marker = project / "executed.txt"
        (project / "valid.py").write_text("from pathlib import Path\nPath('executed.txt').write_text('bad')", encoding="utf-8")
        (project / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        (project / "data.json").write_text("{}", encoding="utf-8")
        result = project_runtime.validate_project("Alice", "projects/demo")
        self.assertFalse(result["ok"])
        self.assertIn("valid.py", result["checked"])
        self.assertEqual(result["errors"][0]["path"], "bad.py")
        self.assertFalse(marker.exists())

    def test_runner_mounts_only_project_and_never_falls_back_to_host(self):
        command = project_runtime.command_argv("Alice", "projects/demo", ["python3", "app.py"], "test:local", "test")
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--pull=never", command)
        self.assertEqual(command.count("--mount"), 1)
        self.assertIn(str(workspace.workspace_path("Alice", "projects/demo")), command[command.index("--mount") + 1])
        with patch.object(project_runtime, "runtime_info", return_value={"available": False, "reason": "missing"}), patch.object(project_runtime.subprocess, "Popen") as run:
            self.assertFalse(project_runtime.run_command("Alice", ["python3", "app.py"])["executed"])
            run.assert_not_called()

    def test_progress_is_scoped_redacted_and_journal_failure_does_not_fail_tool(self):
        self.runtime.record({"tool_name": "write", "status": "completed", "content": "api_key=secret123"})
        restored = RuntimeControl()
        restored.configure({"project_folder": "projects/demo"})
        restored.load_progress("Alice")
        self.assertEqual(len(restored.events), 1)
        self.assertNotIn("secret123", restored.progress_prompt())
        restored.load_progress("Bob")
        self.assertEqual(restored.events, [])
        with patch.object(workspace, "_atomic_write_text", side_effect=OSError("readonly")):
            self.runtime.record({"status": "completed"})


class ProtocolTests(unittest.TestCase):
    def test_json_stream_handles_code_braces_and_does_not_execute_nested_objects(self):
        detector = StreamJSONDetector()
        value = {"tool": "write", "arguments": {"content": 'if (x) { console.log("}"); }', "example": {"tool": "delete", "arguments": {}}}}
        encoded = json.dumps(value)
        results = []
        for char in encoded:
            results.extend(detector.process_chunk(char))
        self.assertEqual(results, [value])

    def test_prompt_fallback_resolves_known_alias_and_accepts_empty_arguments(self):
        manager = ToolManager(initial_tools_dict={"remote_read": FormattedTool({}, "remote", original_name="read")})
        executor = ToolExecutor(None, manager)
        calls = executor.process_tool_from_prompt_json([{"tool": "remote_read", "arguments": {}}])
        self.assertEqual(calls[0]["server"], "remote")
        self.assertEqual(calls[0]["args"], {})
        self.assertEqual(executor.process_tool_from_prompt_json([{"tool": "unknown", "arguments": {}}, {"tool": "remote_read", "mcp_server": "other", "arguments": {}}]), [])

    def test_nested_nullable_schema_survives_both_adapters(self):
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {"anyOf": [{"type": "object", "properties": {"x": {"enum": [1, 2]}}}, {"type": "null"}]}}}, "required": ["items"], "additionalProperties": False}
        adapter = ToolAdapter.__new__(ToolAdapter)
        openai, claude = adapter.format_tools_for_api({"tool": FormattedTool(schema, "server")})
        self.assertEqual(openai[0]["function"]["parameters"], schema)
        self.assertEqual(claude[0]["input_schema"], schema)
        self.assertIsNot(openai[0]["function"]["parameters"], schema)

    def test_all_content_structured_errors_and_images_survive(self):
        error, text, images = collect_result({"content_items": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}, {"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}, {"type": "resource", "resource": {"text": "embedded"}}], "structured_content": {"ok": False, "message": "failed"}})
        self.assertTrue(error)
        for word in ("first", "second", "embedded", "failed"):
            self.assertIn(word, text)
        self.assertEqual(image_messages(images, text, "Claude", "id")[1]["type"], "image")
        self.assertEqual(image_messages(images, text, "OpenAI", "id")[0]["content"][1]["type"], "image_url")
        self.assertTrue(collect_result({"content_items": [{"type": "text", "text": '{"ok":false}'}]})[0])



class AsyncRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def isolated_method(file, class_name, method_name, globals):
        """Run an actual streaming method without importing unrelated voice SDKs."""
        source = Path(__file__).resolve().parents[1] / "src/open_llm_vtuber" / file
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == method_name)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"logger": test_daily_tool_executor_policy.loguru.logger, **globals}
        exec(compile(module, str(source), "exec"), namespace)
        return namespace[method_name]

    async def test_native_stream_keeps_partial_names_and_ignores_empty_deltas(self):
        class APIError(Exception): pass
        class ConnectionError(APIError): pass
        class RateError(APIError): pass
        method = self.isolated_method("agent/stateless_llm/openai_compatible_llm.py", "AsyncLLM", "chat_completion",
            {"NOT_GIVEN": object(), "ToolCallObject": ToolCallObject, "APIError": APIError, "APIConnectionError": ConnectionError, "RateLimitError": RateError})
        def delta(name="", args="", finish=None):
            calls = [types.SimpleNamespace(index=0, id="id", type="function", function=types.SimpleNamespace(name=name, arguments=args))] if name or args else None
            return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish, delta=types.SimpleNamespace(tool_calls=calls, content=None))])
        class Stream:
            def __init__(self, chunks): self.chunks, self.closed = chunks, False
            async def __aiter__(self):
                for chunk in self.chunks: yield chunk
            async def close(self): self.closed = True
        stream = Stream([delta("read_", '{"x":'), delta(), delta("file", '1}'), delta(finish="tool_calls")])
        async def create(**kwargs): return stream
        llm = types.SimpleNamespace(support_tools=True, model="test", temperature=0.7, max_tokens=8192,
            client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
        output = [event async for event in method(llm, [], tools=[])]
        calls = [event for event in output if isinstance(event, list)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].function.name, "read_file")
        self.assertEqual(json.loads(calls[0][0].function.arguments), {"x": 1})
        self.assertTrue(stream.closed)
        stream = Stream([delta("write", '{"code":"partial'), delta(finish="length")])
        output = [event async for event in method(llm, [], tools=[])]
        self.assertFalse(any(isinstance(event, list) for event in output))
        self.assertTrue(stream.closed)

    async def test_agent_switches_to_prompt_tools_then_resumes_answer(self):
        method = self.isolated_method("agent/agents/basic_memory_agent.py", "BasicMemoryAgent", "_openai_tool_interaction_loop",
            {"DEFAULT_MAX_TOOL_ROUNDS": 8, "MAX_TOOL_CALLS_PER_TURN": 16, "AGENTIC_TASK_GUIDANCE": "task guidance",
             "SCREEN_VISION_TOOL_NAME": "screen", "ToolCallObject": ToolCallObject, "TOOL_LIMIT_MESSAGE": "limit"})
        requests, executions = [], []
        class LLM:
            async def chat_completion(self, messages, system, tools=None):
                requests.append(list(messages))
                if len(requests) == 1: yield "__API_NOT_SUPPORT_TOOLS__"
                elif len(requests) == 2: yield '{"tool":"known","arguments":{}}'
                else: yield "done"
        class Executor:
            def process_tool_from_prompt_json(self, items): return items
            async def execute_tools(self, **kwargs):
                executions.append(kwargs)
                yield {"type": "final_tool_results", "results": [{"content": "actual result"}]}
        agent = types.SimpleNamespace(prompt_mode_flag=False, _llm=LLM(), _json_detector=StreamJSONDetector(),
            _mcp_prompt_string="available tools", _tool_executor=Executor(),
            _secure_system_prompt_for_policy=lambda system, policy: system,
            _filter_tools_for_policy=lambda tools, mode, policy: tools,
            _consume_tool_call_budget=lambda total, count, limit: total + count)
        output = [event async for event in method(agent, [{"role": "user", "content": "do it"}], [], "persona", remember_turn=False)]
        self.assertEqual(output, ["done"])
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["caller_mode"], "Prompt")
        self.assertEqual(requests[-1][-2]["role"], "assistant")
        self.assertIn("actual result", requests[-1][-1]["content"])

    async def test_all_tools_allow_without_prompts_and_old_settings_are_migrated(self):
        messages = []
        async def send(raw):
            messages.append(json.loads(raw))
        runtime = RuntimeControl(send)
        policy = runtime.policy("Alice")
        from src.open_llm_vtuber.runtime_control import PERMISSION_FIELDS
        for name in ("write_workspace_file", "delete_workspace_item", "run_workspace_command", "create_reminder", "browser_open", "request_connected_service", "external_tool"):
            self.assertEqual(runtime.level(name), "allow")
            self.assertTrue(await runtime.authorize(name, {"network": True}, policy))
        runtime.configure({**dict.fromkeys(PERMISSION_FIELDS, "forbid"), "tools": {"external_tool": "ask"},
                           "project_folder": "new-project", "temperature": 0.8})
        self.assertTrue(all(runtime.settings[field] == "allow" for field in PERMISSION_FIELDS))
        self.assertEqual(runtime.settings["tools"], {})
        self.assertEqual(runtime.settings["project_folder"], "new-project")
        self.assertFalse(await runtime.authorize("external", {}, policy))
        self.assertTrue(await runtime.authorize("external_tool", {}, runtime.policy("Alice")))
        self.assertFalse(await runtime.authorize("external_tool", {}, {"runtime_revision": runtime.revision}))
        self.assertFalse(runtime.resolve("old-approval", True))
        self.assertEqual(messages, [])
        self.assertEqual(runtime.pending, {})

    async def test_parallel_reads_keep_native_reply_order_and_allow_followup_write(self):
        runtime = RuntimeControl()
        runtime.persona = ""  # No persistent journal needed for this protocol test.
        policy = runtime.policy("Alice")
        runtime.record = lambda event: None
        active, peak = 0, 0
        async def call_tool(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"content_items": [{"type": "text", "text": "found"}, {"type": "image", "mimeType": "image/png", "data": "aGVsbG8="}]}
        manager = ToolManager(initial_tools_dict={"search_workspace": FormattedTool({}, "workspace")})
        executor = ToolExecutor(types.SimpleNamespace(call_tool=call_tool), manager)
        calls = [{"name": "search_workspace", "id": str(i), "args": {"persona": "Alice", "query": str(i)}} for i in range(3)]
        events = [event async for event in executor.execute_tools(calls, "OpenAI", policy)]
        results = events[-1]["results"]
        self.assertGreater(peak, 1)
        self.assertEqual([r["tool_call_id"] for r in results[:3]], ["0", "1", "2"])
        self.assertTrue(all(r["role"] == "user" for r in results[3:]))
        executor._restrict_after_network_result("search_web", policy)
        self.assertFalse(policy["enforce"])
        self.assertIn("write_workspace_file", policy["available_workspace_tools"])

    async def test_runtime_info_has_no_persona_parameter_and_stale_turn_prevents_execution(self):
        async def call_tool(**kwargs):
            self.assertEqual(kwargs["tool_args"], {})
            return {"content_items": [{"type": "text", "text": '{"available":false}'}]}
        manager = ToolManager(initial_tools_dict={"get_workspace_runtime": FormattedTool({}, "workspace")})
        executor = ToolExecutor(types.SimpleNamespace(call_tool=call_tool), manager)
        runtime = RuntimeControl()
        runtime.record = lambda event: None
        call = {"name": "get_workspace_runtime", "id": "id", "args": {}}
        events = [e async for e in executor.execute_tools([call], "OpenAI", runtime.policy("Alice"))]
        self.assertEqual(events[-2]["status"], "completed")
        old_policy = runtime.policy("Alice")
        runtime.configure({"project_folder": "changed"})
        events = [e async for e in executor.execute_tools([call], "OpenAI", old_policy)]
        self.assertEqual(events[0]["status"], "error")

    async def test_mcp_connection_has_single_owner_for_parallel_requests_and_close(self):
        client = MCPClient(ServerRegistry.__new__(ServerRegistry))
        owners = []
        class Context:
            async def __aenter__(self):
                owners.append(asyncio.current_task())
            async def __aexit__(self, *args):
                owners.append(asyncio.current_task())
        session = object()
        async def connect(name, stack):
            await stack.enter_async_context(Context())
            await asyncio.sleep(0)
            client.active_sessions[name] = session
            return session
        client._connect = connect
        results = await asyncio.gather(*(client._ensure_server_running_and_get_session("test") for _ in range(4)))
        self.assertEqual(results, [session] * 4)
        await client.aclose()
        self.assertEqual(len(owners), 2)
        self.assertIs(owners[0], owners[1])

    async def test_command_timeout_reaches_the_mcp_sdk(self):
        client = MCPClient(ServerRegistry.__new__(ServerRegistry))
        captured = []
        async def call(name, args, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(content=[], isError=False)
        client.active_sessions["workspace"] = types.SimpleNamespace(call_tool=call)
        await client.call_tool("workspace", "run_workspace_command", {"timeout_seconds": 120})
        self.assertEqual(captured[0]["read_timeout_seconds"].total_seconds(), 165)



if __name__ == "__main__":
    unittest.main()
