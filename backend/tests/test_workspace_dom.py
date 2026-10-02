"""Integration tests against the real injected bridge; uses existing browsers only."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import unittest
from uuid import uuid4
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import workspace_core as core


class WorkspaceDomValidationTests(unittest.TestCase):
    def test_invalid_or_unscoped_commands_never_reach_queue(self):
        with patch.object(core, "send_workspace_action") as send:
            for page, version, operation, payload in [
                ("", 1, "click", {"selector": "button"}),
                ("page", 0, "click", {"selector": "button"}),
                ("page", 1, "shell", {}),
                ("page", 1, "evaluate", {}),
                ("page", 1, "fill", {"selector": "input"}),
                ("page", 1, "click", {}),
                ("page", 1, "scroll", {"dy": "invalid"}),
            ]:
                with self.subTest(operation=operation, payload=payload), self.assertRaises(ValueError):
                    core.control_workspace_page("Test", page, version, operation, payload)
            send.assert_not_called()


class WorkspaceDomBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from browser_environment import browser_info
        info = browser_info()
        node = shutil.which("node")
        if not info["available"] or not node:
            raise unittest.SkipTest("Existing Playwright, browser and Node are required; no downloads performed.")
        from playwright.sync_api import sync_playwright
        cls.persona = "dom-test-" + uuid4().hex
        cls.root = core.WORKSPACE_ROOT.resolve() / cls.persona
        # Only remove this newly created, unique fixture, inside the workspace.
        assert cls.root.parent == core.WORKSPACE_ROOT.resolve() and not cls.root.exists()
        cls.root.mkdir(parents=True)
        cls.addClassCleanup(lambda: shutil.rmtree(cls.root))
        cls.source = """<!doctype html><html><head><title>Plain HTML</title></head><body>
<button id="counter" onclick="this.textContent=String(++window.count)">Count</button>
<input id="field" aria-label="Name" oninput="document.querySelector('#output').textContent=this.value">
<select id="choice"><option value="a">Apple</option><option value="b">Banana</option></select>
<input type="checkbox" id="check"><div id="output"></div><canvas id="canvas"></canvas>
<div id="shadow"></div><div style="height:2000px">Scrollable</div>
<script>window.count=0;document.querySelector('#shadow').attachShadow({mode:'open'}).innerHTML='<button>Shadow</button>';</script>
</body></html>"""
        (cls.root / "plain.htm").write_text(cls.source, encoding="utf-8")
        cls.env = patch.dict(os.environ, {"MELOMATE_SESSION_TOKEN": "dom-test-session"})
        cls.env.start()
        cls.addClassCleanup(cls.env.stop)
        def port():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                return sock.getsockname()[1]
        main_port, workspace_port = port(), port()
        while workspace_port == main_port:
            workspace_port = port()
        child = subprocess.Popen([node, "server.mjs"], cwd=BACKEND.parent,
            env={**os.environ, "PORT": str(main_port), "MELOMATE_WORKSPACE_PORT": str(workspace_port)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def stop():
            child.terminate()
            child.wait(timeout=10)
        cls.addClassCleanup(stop)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", workspace_port), .1):
                    break
            except OSError:
                time.sleep(.05)
        driver = sync_playwright().start()
        cls.addClassCleanup(driver.stop)
        cls.browser = driver.chromium.launch(executable_path=info["executable"], headless=True)
        cls.addClassCleanup(cls.browser.close)
        cls.page = cls.browser.new_page()
        cls.url = f"http://127.0.0.1:{workspace_port}/workspace-files/{cls.persona}/{core.workspace_access_token(cls.persona)}/plain.htm"

    def setUp(self):
        self.page.goto(self.url)
        self.page.wait_for_function("Boolean(window.MeloMateWorkspaceControl)")
        self.page_id = self.page.evaluate("window.MeloMateWorkspaceControl.pageId")
        self.wait_state(lambda s: core.state_protocol_available(s) and bool(core.state_payload(s).get("appState", {}).get("melomate_dom", {}).get("elements")))

    def wait_state(self, predicate):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            state = core.read_workspace_state_file(self.persona, self.page_id)
            if state and predicate(state):
                return state
            self.page.wait_for_timeout(50)
        self.fail("Page state did not arrive")

    def act(self, operation, payload):
        state = self.wait_state(lambda s: core.state_is_fresh(s))
        result = json.loads(core.control_workspace_page(self.persona, self.page_id,
            core.state_version(state), operation, payload, wait_ms=4000))
        self.assertTrue(result["confirmed"], result)
        return result

    def test_plain_html_controls_same_page_and_preserves_source(self):
        state = core.read_workspace_state_file(self.persona, self.page_id)
        elements = core.state_payload(state)["appState"]["melomate_dom"]["elements"]
        counter = next(item for item in elements if item["name"] == "Count")
        self.act("click", {"selector": counter["selector"]})
        self.assertEqual(self.page.locator("#counter").inner_text(), "1")
        self.act("fill", {"selector": "#field", "value": "小可"})
        self.assertEqual(self.page.locator("#output").inner_text(), "小可")
        self.act("select", {"selector": "#choice", "value": "b"})
        self.assertEqual(self.page.locator("#choice").input_value(), "b")
        self.act("click", {"selector": "#check"})
        self.assertTrue(self.page.locator("#check").is_checked())
        self.act("press", {"selector": "#counter", "value": "Enter"})
        self.assertEqual(self.page.locator("#counter").inner_text(), "2")
        result = self.act("evaluate", {"script": "const c=document.querySelector('canvas'); const ctx=c.getContext('2d');ctx.fillStyle='red';ctx.fillRect(0,0,10,10);return Array.from(ctx.getImageData(0,0,1,1).data);"})
        self.assertEqual(result["action_result"]["result"], [255, 0, 0, 255])
        result = self.act("evaluate", {"script": "const b=document.querySelector('#shadow').shadowRoot.querySelector('button');b.onclick=()=>b.textContent='Clicked shadow';return true;"})
        elements = core.state_payload(result["state"])["appState"]["melomate_dom"]["elements"]
        shadow = next(item for item in elements if item["name"] == "Shadow")
        self.act("click", {"selector": shadow["selector"]})
        self.assertEqual(self.page.locator("#shadow button").inner_text(), "Clicked shadow")
        self.act("scroll", {"dy": 350})
        self.assertGreater(self.page.evaluate("window.scrollY"), 0)
        self.assertEqual(len(self.page.context.pages), 1)
        self.assertEqual((self.root / "plain.htm").read_text(encoding="utf-8"), self.source)

    def test_mcp_schema_exposes_generic_operations_and_result_keeps_identity(self):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import mcp_workspace
        from src.open_llm_vtuber.workspace_security import harden_workspace_tool_result
        # Playwright's sync API already owns an event loop on this thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            schema = pool.submit(lambda: asyncio.run(mcp_workspace.mcp.list_tools())).result()
        tool = next(t for t in schema if t.name == "act_workspace_page")
        self.assertNotIn("action_id", tool.inputSchema["required"])
        self.assertIn("operation", tool.inputSchema["properties"])
        self.assertIn("payload", tool.inputSchema["properties"])
        error, hardened = harden_workspace_tool_result("read_workspace_state", core.read_workspace_state(self.persona, self.page_id))
        result = json.loads(hardened)
        self.assertFalse(error)
        self.assertEqual(result["page_id"], self.page_id)
        dom = result["state"]["state"]["appState"]["melomate_dom"]
        self.assertTrue(dom["elements"][0]["selector"].startswith("ref:"))

    def test_stale_wrong_scope_and_failed_scripts(self):
        state = core.read_workspace_state_file(self.persona, self.page_id)
        version = core.state_version(state)
        self.page.evaluate("document.querySelector('#counter').textContent='Changed'")
        self.wait_state(lambda s: core.state_version(s) > version)
        result = json.loads(core.control_workspace_page(self.persona, self.page_id, version, "click", {"selector": "#counter"}))
        self.assertFalse(result["sent"])
        with self.assertRaisesRegex(ValueError, "outside the selected project"):
            core.control_workspace_page(self.persona, self.page_id, version, "click", {"selector": "#counter"}, folder="other")
        # Browser also checks a queued revision immediately before applying it.
        core.append_workspace_command(self.persona, {"id": "stale-queued", "type": "action", "page_id": self.page_id,
            "expected_state_version": version, "action": "dom.click", "payload": {"selector": "#counter"}, "created_ms": int(time.time()*1000)})
        latest = self.wait_state(lambda s: core.find_action_result(s, "stale-queued") is not None)
        self.assertFalse(core.find_action_result(latest, "stale-queued")["accepted"])
        self.assertEqual(self.page.evaluate("window.count"), 0)
        latest = core.read_workspace_state_file(self.persona, self.page_id)
        result = json.loads(core.control_workspace_page(self.persona, self.page_id, core.state_version(latest), "evaluate", {"script": "throw new Error('test failure')"}, wait_ms=4000))
        self.assertFalse(result["confirmed"])
        self.assertIn("test failure", result["action_result"]["error"])

    def test_custom_actions_still_work_alongside_dom(self):
        self.page.evaluate("""() => {
          window.MeloMateWorkspaceState=()=>({availableActions:[{id:'add',action:'add',payload:{}}]});
          window.MeloMateWorkspaceAction=()=>{window.count+=3;return {ok:true};};
        }""")
        state = self.wait_state(lambda s: bool(core.advertised_workspace_actions(s)))
        result = json.loads(core.send_workspace_action(self.persona, expected_page_id=self.page_id,
            expected_state_version=core.state_version(state), action_id="add", wait_ms=4000))
        self.assertTrue(result["confirmed"], result)
        self.assertEqual(self.page.evaluate("window.count"), 3)
        self.assertIn("melomate_dom", core.state_payload(result["state"])["appState"])


if __name__ == "__main__":
    unittest.main()
