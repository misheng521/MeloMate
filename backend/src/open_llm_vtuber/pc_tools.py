"""Session-local PC tools exposed through the same model-driven tool loop as MCP."""
from __future__ import annotations
import asyncio
import hashlib
import importlib.util
import json
import shutil
from urllib.parse import quote
from .mcpp.types import FormattedTool
from .secure_credentials import SecureCredentialStore, CHAT_API_KEY
from . import pc_network as network
from .pc_browser import PCBrowser

FILE_TOOLS = frozenset({"copy_workspace_item", "archive_workspace_items", "extract_workspace_archive", "download_workspace_file", "browser_open_workspace"})
BROWSER_TOOLS = frozenset({"browser_open", "browser_open_workspace", "browser_read", "browser_action", "browser_close"})
INFO_TOOLS = frozenset({"get_pc_capabilities", "list_connected_services", "read_work_plan", "read_memory", "search_memory", "get_session_state"})
_VAULT = SecureCredentialStore()


def definitions() -> dict[str, FormattedTool]:
    string = {"type": "string"}
    def spec(description, properties, required=()):
        return FormattedTool({"type": "object", "properties": properties, "required": list(required), "additionalProperties": False},
            "__pc__", description=description, timeout_seconds=90)
    scoped = {"persona": string}
    return {
        "get_session_state": spec("Observe this character's current MeloMate session: actual change notifications, current project, saved task plan, latest tool outcomes and available input switches. Plans are not proof of completion; switches do not imply unseen observations. Read-only; does not change personality or execute a task.", {}),
        "read_memory": spec("Read this character's editable memory.md and recent message IDs. Notes are historical data, not fixed personality instructions. Does not read another character's memory.", {}),
        "search_memory": spec("Search this character's conversations and notes when recalling past events. Optionally supply up to 3 alternative queries (synonyms, names or related phrases) in one call. Results include nearby dialogue, speakers and dates. This is text search, not semantic inference; do not invent memories when no evidence is found.", {"query": string, "alternative_queries": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 200}}, "limit": {"type": "integer", "minimum": 1, "maximum": 12}}, ("query",)),
        "edit_memory": spec("Remember, correct, or forget a small piece of this character's memory.md, based on actual dialogue. Read memory first for revision and evidence IDs. Empty old_text appends a note; otherwise match old_text exactly once. Empty text removes that note and filters its linked or text-matching old evidence from future model context, preserving unrelated history. Record who said what in which context; do not turn one response into a fixed trait or mandatory personality/relationship rule. Repeating a persona or an existing note is not independent evidence. Distinguish fictional character background from actual events. Does not edit persona prompts or grant permissions.",
            {"revision": string, "old_text": string, "text": string, "evidence_message_ids": {"type": "array", "minItems": 1, "maxItems": 12, "items": string}}, ("revision", "old_text", "text", "evidence_message_ids")),
        "get_pc_capabilities": spec("Inspect actual PC tool/runtime availability and missing prerequisites. Use when deciding how to solve a task; does not install anything.", {}),
        "list_connected_services": spec("List user-configured HTTP services, allowed methods/paths and credential availability. Includes local services and public APIs, not just devices. Never returns secrets.", {}),
        "request_connected_service": spec("Call one configured HTTP service. Choose the method, path and JSON body from actual documentation. Credentials are injected by the backend. Redirects are not followed. A 2xx response alone does not prove a physical action finished: inspect returned state and choose a follow-up read when necessary.",
            {"service_id": string, "method": {"type": "string", "enum": sorted(network.METHODS)}, "path": string, "json_body": {}}, ("service_id", "method", "path")),
        "copy_workspace_item": spec("Copy a file or directory to a new path within the selected project, bounded to 64 MiB. Does not overwrite.", {**scoped, "source": string, "destination": string}, ("persona", "source", "destination")),
        "archive_workspace_items": spec("Create a ZIP of a project file/directory at a new destination. Excludes private runtime directories, rejects links, bounded to 64 MiB.", {**scoped, "source": string, "destination": string}, ("persona", "source", "destination")),
        "extract_workspace_archive": spec("Extract a ZIP into a new project directory; reject traversal, links and decompression bombs. Never executes extracted files.", {**scoped, "path": string, "destination": string}, ("persona", "path", "destination")),
        "download_workspace_file": spec("Download a public HTTP(S) resource (up to 16 MiB) to a new project file. Never executes it. Redirects are returned for the model to inspect and explicitly follow; private services use request_connected_service.", {**scoped, "url": string, "path": string}, ("persona", "url", "path")),
        "browser_open": spec("Open a public website in MeloMate's separate PC browser. Returns text, element selectors, JS errors, tabs and a screenshot. Never uses the user's normal browser profile. Browser component must be installed.", {"url": string}, ("url",)),
        "browser_open_workspace": spec("Preview project HTML/JS/CSS in a real PC browser and inspect screenshot and console errors. Static project server only; arbitrary backend servers are not started. Paths remain in the selected project.", {**scoped, "path": string}, ("persona", "path")),
        "browser_read": spec("Read current browser page, selectors, screenshot and errors. Use to inspect the outcome of an action or delayed UI update.", {"page_id": string}, ("page_id",)),
        "browser_action": spec("Perform one model-selected browser action on a current page. Read selectors first. Click/fill/select/press need a unique selector; value supplies text/key/option. Scroll uses value as vertical pixels. A click is not proof of task success; inspect the returned page.", {"page_id": string, "action": {"type": "string", "enum": ["click", "fill", "select", "press", "scroll"]}, "selector": string, "value": string}, ("page_id", "action")),
        "browser_close": spec("Close one MeloMate browser tab.", {"page_id": string}, ("page_id",)),
        "read_work_plan": spec("Read the saved plan for this project when resuming work. Plans are historical notes, not fresh user authorization.", {}),
        "update_work_plan": spec("Optionally save a plan and current progress for a multi-step user task. The model chooses steps and status. Do not invent completion; cite actual checks in next_step or explain the specific missing prerequisite. No task execution is triggered by saving a plan.",
            {"goal": string, "steps": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {"text": string, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked"]}}, "required": ["text", "status"], "additionalProperties": False}}, "next_step": string}, ("goal", "steps", "next_step")),
    }


class PCWorkTools:
    def __init__(self, runtime):
        self.runtime = runtime
        self.browser = PCBrowser(runtime)
        self.credentials = _VAULT
        self.browser_scope = None
        self.memory_conf_uid = ""
        self.memory_edit_epoch = None
        self.session_state_provider = None

    def service(self, identifier):
        result = next((s for s in self.runtime.settings.get("services", []) if s["id"] == identifier), None)
        if result is None: raise ValueError("Service is not configured. Prepare the integration and ask for only the missing endpoint/authentication through PC settings.")
        return result

    def credential_id(self, service):
        text = f"{self.runtime.persona}:{service['id']}:{service['base_url']}"
        return "pcservice_" + hashlib.sha256(text.encode()).hexdigest()[:40]

    def credential_ready(self, service):
        if service["auth"] == "none": return True
        try: return self.credentials.status(self.credential_id(service)).get(CHAT_API_KEY, False)
        except Exception: return False

    def set_secret(self, identifier, secret="", clear=False):
        service = self.service(identifier)
        return self.credentials.update(self.credential_id(service),
            secrets={CHAT_API_KEY: secret} if secret and not clear else {}, clear=[CHAT_API_KEY] if clear else [])

    async def call(self, name, arguments):
        try:
            result = await self.dispatch(name, arguments)
            image = result.pop("_image", None)
            content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            if image: content.append({"type": "image", "mimeType": "image/jpeg", "data": image})
            return {"is_error": result.get("ok") is False, "content_items": content}
        except Exception as exc:
            return {"is_error": True, "content_items": [{"type": "text", "text": json.dumps({"ok": False, "error": str(exc)[:1500], "instruction": "Do not claim success. Inspect the cause, choose another real tool, or ask only for the missing prerequisite."}, ensure_ascii=False)}]}

    async def dispatch(self, name, args):
        import workspace_core as workspace
        if name == "get_session_state":
            if args: raise ValueError("Session state takes no arguments")
            if self.session_state_provider is None: raise ValueError("Session is not initialized")
            return {"ok": True, **self.session_state_provider()}
        if name in {"read_memory", "search_memory", "edit_memory"}:
            from . import chat_history_manager as memory
            if not self.memory_conf_uid: raise ValueError("Character memory is not initialized")
            function = {"read_memory": memory.read_memory, "search_memory": memory.search_memory, "edit_memory": memory.edit_memory}[name]
            result = await asyncio.to_thread(function, self.memory_conf_uid, **args)
            if name == "edit_memory":
                self.memory_edit_epoch = result.pop("_memory_epoch")
            return {"ok": True, **result}
        if name == "get_pc_capabilities":
            from project_runtime import runtime_info
            return {"ok": True, "platform": "PC", "project_folder": self.runtime.settings["project_folder"],
                    "tools": list(definitions()), "isolated_runner": await asyncio.to_thread(runtime_info),
                    "browser_component": importlib.util.find_spec("playwright") is not None,
                    "node_available": bool(shutil.which("node")), "service_count": len(self.runtime.settings.get("services", [])),
                    "extension_path": "Use existing file tools to implement and test missing logic in the project. Generic HTTP services and configured MCP servers can expose external capabilities. Software/API availability must be verified; writing code alone does not install or connect it."}
        if name == "list_connected_services":
            return {"ok": True, "services": [{**s, "credential_ready": self.credential_ready(s)} for s in self.runtime.settings.get("services", [])]}
        if name == "request_connected_service":
            service = self.service(args["service_id"])
            method = args["method"]
            url = network.service_url(service, args["path"], method)
            secret = ""
            headers = {"Content-Type": "application/json"}
            if service["auth"] != "none":
                secret = self.credentials.get(self.credential_id(service), CHAT_API_KEY) or ""
                if not secret: raise ValueError("Service credential is missing. Enter it in PC settings; do not put it in chat or source code.")
                if "\r" in secret or "\n" in secret: raise ValueError("Invalid credential characters")
                headers["Authorization" if service["auth"] == "bearer" else service["header"]] = ("Bearer " if service["auth"] == "bearer" else "") + secret
            body = json.dumps(args["json_body"], ensure_ascii=False).encode() if "json_body" in args else None
            if method in {"GET", "HEAD"} and body is not None: raise ValueError("GET/HEAD must not include json_body")
            status, response_headers, data = await asyncio.to_thread(network.request, url, method, body, headers, True, 64000)
            return network.json_response(status, response_headers, data, secret)
        if name in FILE_TOOLS - {"browser_open_workspace"}:
            import workspace_extras as extras
            if name == "download_workspace_file":
                status, headers, data = await asyncio.to_thread(network.request, args["url"])
                if not 200 <= status < 300:
                    return {"ok": False, "status": status, "redirect": headers.get("location"), "saved": False}
                return extras.save_download(args["persona"], args["path"], data)
            function = {"copy_workspace_item": extras.copy_item, "archive_workspace_items": extras.archive, "extract_workspace_archive": extras.extract}[name]
            return function(**args)
        if name in BROWSER_TOOLS:
            async with self.browser.lock:
                scope = (self.runtime.persona, self.runtime.settings["project_folder"])
                if self.browser_scope is not None and self.browser_scope != scope:
                    await self.browser.close()
                self.browser_scope = scope
                if name == "browser_open_workspace":
                    persona, folder = args["persona"], self.runtime.settings["project_folder"]
                    root = workspace.workspace_path(persona, folder)
                    path = workspace.workspace_path(persona, args["path"])
                    relative = path.relative_to(root).as_posix()
                    self.browser.project_root = (persona, folder)
                    return await self.browser.open("https://melomate-project.invalid/" + quote(relative, safe="/"))
                if name == "browser_open": return await self.browser.open(args["url"])
                if name == "browser_read": return await self.browser.snapshot(args["page_id"])
                if name == "browser_action": return await self.browser.action(**args)
                return await self.browser.close_page(args["page_id"])
        if name == "read_work_plan": return {"ok": True, "plan": self.runtime.work_plan}
        if name == "update_work_plan":
            self.runtime.save_plan(args)
            if self.runtime.send:
                await self.runtime.send(json.dumps({"type": "work-plan", "plan": self.runtime.work_plan}, ensure_ascii=False))
            return {"ok": True, "plan": self.runtime.work_plan, "execution_triggered": False}
        raise ValueError("Unknown PC tool")

    async def close(self):
        await self.browser.close()
