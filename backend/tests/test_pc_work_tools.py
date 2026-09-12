"""PC work regression tests: real local HTTP fixture, storage and tool dispatch.

No external accounts/devices, model keys, Docker or browser installation required.
"""
import asyncio
import json
import socket
import stat
import tempfile
import threading
import types
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, patch

import test_daily_tool_executor_policy  # optional-dependency stubs
import workspace_core as workspace
import workspace_extras as extras
from src.open_llm_vtuber import pc_network as network
from src.open_llm_vtuber.runtime_control import RuntimeControl, scope_arguments
from src.open_llm_vtuber.pc_tools import PCWorkTools, definitions
from src.open_llm_vtuber.pc_browser import PCBrowser
from src.open_llm_vtuber.mcpp.tool_executor import ToolExecutor
from src.open_llm_vtuber.mcpp.tool_manager import ToolManager
from src.open_llm_vtuber.secure_credentials import SecureCredentialStore, CHAT_API_KEY


class FixtureHandler(BaseHTTPRequestHandler):
    requests = []
    def do_GET(self): self.handle_request()
    def do_POST(self): self.handle_request()
    def handle_request(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).requests.append((self.command, self.path, self.headers.get("Authorization"), body))
        payload = json.dumps({"state": "ready", "received": body.decode(), "echo": self.headers.get("Authorization", "")}).encode()
        self.send_response(302 if self.path == "/api/redirect" else 200)
        self.send_header("Content-Type", "application/json")
        if self.path == "/api/redirect": self.send_header("Location", "/api/unrequested")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args): pass


class PCWorkTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = patch.multiple(workspace, ROOT=self.root, WORKSPACE_ROOT=self.root / "workspace")
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.runtime = RuntimeControl()
        self.runtime.configure({"project_folder": "demo", "services": [{"id": "fixture", "base_url": self.base,
            "paths": ["/api/"], "methods": ["GET", "POST"], "auth": "bearer"}]})
        self.policy = self.runtime.policy("Alice")
        self.tools = PCWorkTools(self.runtime)
        self.tools.credentials = SecureCredentialStore(self.root / "vault.json", protector=lambda b: b[::-1], unprotector=lambda b: b[::-1])
        self.tools.set_secret("fixture", "fixture-secret-123")
        FixtureHandler.requests.clear()
        workspace.create_workspace_folder("Alice", "demo")

    async def test_real_local_service_injects_secret_without_returning_it(self):
        result = await self.tools.call("request_connected_service", {"service_id": "fixture", "method": "POST", "path": "/api/jobs", "json_body": {"task": "test"}})
        self.assertFalse(result["is_error"])
        self.assertEqual(FixtureHandler.requests[0][:3], ("POST", "/api/jobs", "Bearer fixture-secret-123"))
        self.assertNotIn("fixture-secret-123", json.dumps(result))
        listed = await self.tools.call("list_connected_services", {})
        self.assertNotIn("fixture-secret-123", json.dumps(listed))
        self.assertNotIn("fixture-secret-123", (self.root / "vault.json").read_text())

    async def test_service_redirect_is_not_followed_and_auth_is_not_forwarded(self):
        result = await self.tools.dispatch("request_connected_service", {"service_id": "fixture", "method": "GET", "path": "/api/redirect"})
        self.assertEqual(result["status"], 302)
        self.assertFalse(result["redirect_followed"])
        self.assertEqual(len(FixtureHandler.requests), 1)

    async def test_service_cannot_change_host_path_or_allowed_methods(self):
        for path in ("//evil.test/api", "https://evil.test/api", "/api/../admin", "/api/%2e%2e/admin", "/api/%252e%252e/admin", "/api2/other"):
            with self.subTest(path=path):
                result = await self.tools.call("request_connected_service", {"service_id": "fixture", "method": "GET", "path": path})
                self.assertTrue(result["is_error"])
        result = await self.tools.call("request_connected_service", {"service_id": "fixture", "method": "DELETE", "path": "/api/jobs"})
        self.assertTrue(result["is_error"])
        self.assertEqual(FixtureHandler.requests, [])

    async def test_changing_service_origin_does_not_reuse_credential(self):
        self.runtime.configure({"services": [{"id": "fixture", "base_url": "http://127.0.0.1:1", "auth": "bearer"}]})
        self.assertFalse(self.tools.credential_ready(self.tools.service("fixture")))

    async def test_stale_turn_never_contacts_service(self):
        policy = self.runtime.policy("Alice")
        self.runtime.configure({"project_folder": "changed"})
        manager = ToolManager(initial_tools_dict=definitions())
        executor = ToolExecutor(None, manager, self.tools)
        call = {"name": "request_connected_service", "id": "test", "args": {"service_id": "fixture", "method": "POST", "path": "/api/jobs", "json_body": {}}}
        events = [event async for event in executor.execute_tools([call], "OpenAI", policy)]
        self.assertEqual(events[0]["status"], "error")
        self.assertEqual(FixtureHandler.requests, [])

    async def test_native_tools_execute_through_normal_model_tool_loop(self):
        manager = ToolManager(initial_tools_dict=definitions())
        executor = ToolExecutor(None, manager, self.tools)
        workspace.write_workspace_file("Alice", "demo", "a.txt", "actual contents")
        call = {"name": "copy_workspace_item", "id": "copy1", "args": {"persona": "Alice", "source": "a.txt", "destination": "b.txt"}}
        events = [event async for event in executor.execute_tools([call], "OpenAI", self.policy)]
        self.assertEqual(events[-2]["status"], "completed")
        self.assertEqual(workspace.workspace_path("Alice", "demo/b.txt").read_text(), "actual contents")
        self.assertFalse(workspace.workspace_path("Alice", "b.txt").exists())

    async def test_memory_tools_use_current_character_through_model_loop(self):
        from src.open_llm_vtuber import chat_history_manager as memory
        with patch.object(memory, "CHAT_HISTORY_DIR", self.root / "memory"):
            self.tools.memory_conf_uid = "Alice"
            evidence = memory.store_message("Alice", memory.SINGLE_HISTORY_UID, "human", "记住青柠方案")
            state = await self.tools.dispatch("read_memory", {})
            manager = ToolManager(initial_tools_dict=definitions())
            executor = ToolExecutor(None, manager, self.tools)
            call = {"name": "edit_memory", "id": "note1", "args": {"revision": state["revision"], "old_text": "", "text": "用户选择青柠方案。", "evidence_message_ids": [evidence]}}
            events = [event async for event in executor.execute_tools([call], "OpenAI", self.policy)]
            self.assertEqual(events[-2]["status"], "completed")
            self.assertIn("青柠方案", memory.get_memory_prompt("Alice"))
            self.assertEqual(self.tools.memory_edit_epoch, memory.memory_epoch("Alice"))
            self.assertNotIn("_memory_epoch", json.dumps(events))
            found = await self.tools.dispatch("search_memory", {"query": "青柠"})
            self.assertEqual(found["messages"][0]["id"], evidence)
            self.tools.memory_conf_uid = "Bob"
            self.assertEqual((await self.tools.dispatch("search_memory", {"query": "青柠"}))["messages"], [])
            denied = await self.tools.call("read_memory", {"conf_uid": "Alice"})
            self.assertTrue(denied["is_error"])

    async def test_memory_contents_are_not_copied_into_project_progress(self):
        self.runtime.load_progress("Alice")
        for name in ("read_memory", "search_memory", "edit_memory", "get_session_state"):
            self.runtime.record({"tool_name": name, "status": "completed", "content": "不该留在项目日志的记忆"})
        self.assertNotIn("不该留在", self.runtime.progress_prompt())
        self.runtime.load_progress("Alice")
        self.assertNotIn("不该留在", self.runtime.progress_prompt())
        for path in (self.root / "backend/cache/task-progress").glob("*.json"):
            self.assertNotIn("不该留在", path.read_text(encoding="utf-8"))

    async def test_event_can_observe_session_but_cannot_write_or_use_stale_scope(self):
        from test_text_memory import source_method, BACKEND
        attach = source_method(BACKEND / "src/open_llm_vtuber/conversations/single_conversation.py",
                               "_attach_live_workspace_context", {})
        context = types.SimpleNamespace(runtime_control=self.runtime,
            character_config=types.SimpleNamespace(character_name="Alice", conf_name="Alice"),
            workspace_agent=types.SimpleNamespace(awareness_for_turn=lambda policy: None))
        self.tools.session_state_provider = lambda: {"character": "Alice", "phase": "responding"}
        executor = ToolExecutor(None, ToolManager(initial_tools_dict=definitions()), self.tools)
        policy = attach(context, "事件不是操作授权", {"skip_history": True})["workspace_tool_policy"]
        call = {"name": "get_session_state", "id": "state1", "args": {}}
        events = [event async for event in executor.execute_tools([call], "OpenAI", policy)]
        self.assertEqual(events[-2]["status"], "completed")
        self.assertIn("Alice", json.dumps(events))
        write = {"name": "update_work_plan", "id": "write1", "args": {"goal": "unauthorized", "steps": [], "next_step": ""}}
        events = [event async for event in executor.execute_tools([write], "OpenAI", policy)]
        self.assertEqual(events[0]["status"], "error")
        self.assertEqual(self.runtime.work_plan, {})
        self.runtime.configure({"project_folder": "changed"})
        events = [event async for event in executor.execute_tools([call], "OpenAI", policy)]
        self.assertEqual(events[0]["status"], "error")

    async def test_work_plan_persists_per_project_without_starting_an_action(self):
        plan = {"goal": "Investigate a problem", "steps": [{"text": "inspect", "status": "in_progress"}], "next_step": "read the real error"}
        result = await self.tools.dispatch("update_work_plan", plan)
        self.assertFalse(result["execution_triggered"])
        reloaded = RuntimeControl()
        reloaded.configure({"project_folder": "demo"})
        reloaded.load_progress("Alice")
        self.assertEqual(reloaded.work_plan, plan)
        reloaded.configure({"project_folder": "other"})
        reloaded.load_progress("Alice")
        self.assertEqual(reloaded.work_plan, {})
        self.assertEqual(FixtureHandler.requests, [])

    async def test_network_execution_allows_legacy_settings_but_requires_strict_boolean(self):
        self.runtime.configure({"execution": "allow", "network_execution": "forbid"})
        policy = self.runtime.policy("Alice")
        self.assertTrue(await self.runtime.authorize("run_workspace_command", {"network": False}, policy))
        self.assertTrue(await self.runtime.authorize("run_workspace_command", {"network": True}, policy))
        with self.assertRaises(ValueError): scope_arguments("run_workspace_command", {"network": "true"}, policy)

    async def test_browser_gateway_blocks_private_network_and_project_traversal(self):
        browser = PCBrowser(self.runtime)
        route = types.SimpleNamespace(request=types.SimpleNamespace(url=self.base + "/api/jobs", method="GET", headers={}, post_data_buffer=None), fulfill=AsyncMock(), abort=AsyncMock())
        await browser.route(route)
        route.abort.assert_awaited_once()
        route.fulfill.assert_not_awaited()
        browser.project_root = ("Alice", "demo")
        route.request.url = "https://melomate-project.invalid/%2e%2e/other.txt"
        route.abort.reset_mock()
        await browser.route(route)
        route.abort.assert_awaited_once()

    async def test_browser_gateway_serves_only_selected_project_files(self):
        workspace.write_workspace_file("Alice", "demo", "index.html", "<h1>actual page</h1>")
        browser = PCBrowser(self.runtime)
        browser.project_root = ("Alice", "demo")
        route = types.SimpleNamespace(request=types.SimpleNamespace(url="https://melomate-project.invalid/index.html", method="GET"), fulfill=AsyncMock(), abort=AsyncMock())
        await browser.route(route)
        self.assertEqual(route.fulfill.await_args.kwargs["body"], b"<h1>actual page</h1>")
        route.abort.assert_not_awaited()

    async def test_browser_writes_require_current_action_origin(self):
        browser = PCBrowser(self.runtime)
        route = types.SimpleNamespace(request=types.SimpleNamespace(url="https://example.com/api", method="POST", headers={}, post_data_buffer=b"{}"), fulfill=AsyncMock(), abort=AsyncMock())
        with patch.object(network, "request", return_value=(200, {}, b"ok")) as transport:
            await browser.route(route)
            transport.assert_not_called()
            browser.write_origin = "https://another.example"
            await browser.route(route)
            transport.assert_not_called()
            browser.write_origin = "https://example.com"
            await browser.route(route)
            transport.assert_called_once()
            route.fulfill.assert_awaited_once()

    async def test_browser_close_clears_project_and_pending_write_origin(self):
        browser = PCBrowser(self.runtime)
        browser.project_root = ("Alice", "demo")
        browser.write_origin = "https://example.com"
        browser.browser = types.SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("already disconnected")))
        browser.driver = types.SimpleNamespace(stop=AsyncMock())
        driver = browser.driver
        with self.assertRaises(RuntimeError): await browser.close()
        self.assertIsNone(browser.project_root)
        self.assertIsNone(browser.context)
        self.assertEqual(browser.write_origin, "")
        driver.stop.assert_awaited_once()

    async def test_project_change_discards_existing_browser_session_before_open(self):
        self.tools.browser_scope = ("Alice", "demo")
        self.runtime.configure({"project_folder": "other"})
        with patch.object(self.tools.browser, "close", new_callable=AsyncMock) as close:
            async def open_page(url):
                close.assert_awaited_once()
                return {"ok": True}
            with patch.object(self.tools.browser, "open", side_effect=open_page):
                await self.tools.dispatch("browser_open", {"url": "https://example.com"})
        self.assertEqual(self.tools.browser_scope, ("Alice", "other"))

    def test_zip_roundtrip_and_non_overwriting_copy(self):
        workspace.write_workspace_file("Alice", "demo/source", "a.txt", "hello")
        extras.archive("Alice", "demo/source", "demo/export.zip")
        extras.extract("Alice", "demo/export.zip", "demo/restored")
        self.assertEqual(workspace.workspace_path("Alice", "demo/restored/a.txt").read_text(), "hello")
        with self.assertRaises(ValueError): extras.copy_item("Alice", "demo/source", "demo/restored")

    def test_zip_rejects_traversal_links_and_windows_case_collisions(self):
        for index, name in enumerate(("../outside.txt", "/outside.txt", ".control/state.json", "C:/outside.txt")):
            path = workspace.workspace_path("Alice", f"demo/bad{index}.zip")
            with zipfile.ZipFile(path, "w") as archive: archive.writestr(name, "bad")
            with self.assertRaises(ValueError): extras.extract("Alice", f"demo/bad{index}.zip", f"demo/out{index}")
            self.assertFalse(workspace.workspace_path("Alice", f"demo/out{index}").exists())
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        path = workspace.workspace_path("Alice", "demo/link.zip")
        with zipfile.ZipFile(path, "w") as archive: archive.writestr(link, "/outside")
        with self.assertRaises(ValueError): extras.extract("Alice", "demo/link.zip", "demo/links")
        path = workspace.workspace_path("Alice", "demo/case.zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("file.txt", "a")
            archive.writestr("FILE.txt", "b")
        with self.assertRaises(ValueError): extras.extract("Alice", "demo/case.zip", "demo/cases")

    def test_zip_bomb_rejected_before_creating_destination(self):
        path = workspace.workspace_path("Alice", "demo/bomb.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive: archive.writestr("large.txt", "x" * 3000)
        with patch.object(extras, "MAX_TOTAL", 1000), self.assertRaises(ValueError):
            extras.extract("Alice", "demo/bomb.zip", "demo/expanded")
        self.assertFalse(workspace.workspace_path("Alice", "demo/expanded").exists())

    def test_public_download_never_reaches_local_service(self):
        with self.assertRaises(ValueError): network.request(self.base + "/api/jobs")
        self.assertEqual(FixtureHandler.requests, [])

    def test_special_addresses_blocked_even_for_configured_services(self):
        for ip in ("169.254.169.254", "0.0.0.0", "224.0.0.1"):
            with patch.object(socket, "getaddrinfo", return_value=[(None, None, None, None, (ip, 0))]), self.assertRaises(ValueError):
                network.addresses("test", True)


if __name__ == "__main__": unittest.main()
