"""Project settings with tools allowed by default and no per-call approvals.

Project scope, service validation and current-conversation checks still apply.
Legacy permission settings are normalized so upgrades cannot restore prompts.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import hashlib

from .workspace_intent import WORKSPACE_READ_TOOLS, WORKSPACE_SIDE_EFFECT_TOOLS, WORKSPACE_ALWAYS_AVAILABLE_TOOLS
from .daily_tool_policy import DAILY_READ_TOOLS, DAILY_SIDE_EFFECT_TOOLS
from .pc_tools import FILE_TOOLS, INFO_TOOLS
from .pc_network import validate_services

PROJECT_TOOLS = frozenset({*WORKSPACE_READ_TOOLS, *WORKSPACE_SIDE_EFFECT_TOOLS,
                          *FILE_TOOLS,
                          *WORKSPACE_ALWAYS_AVAILABLE_TOOLS, "validate_workspace_project",
                          "run_workspace_command", "get_workspace_runtime"})
PERMISSION_FIELDS = ("workspace", "reminders", "external", "execution",
                     "browser", "service_access", "network_execution")
# Read-only classification controls safe parallel dispatch, not permissions.
SAFE_READS = frozenset({*WORKSPACE_READ_TOOLS, *DAILY_READ_TOOLS, *INFO_TOOLS,
                       "validate_workspace_project", "get_workspace_runtime"})


def redact(value: object, limit: int = 2000) -> str:
    text = str(value)
    text = re.sub(r'(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+=*', r'\1[redacted]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|authorization|password|token|secret)[\s"\x27:=]+)[^\s,"\x27}]+', r'\1[redacted]', text)
    text = re.sub(r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}', '[redacted]', text)
    return text[:limit]


class RuntimeControl:
    def __init__(self, send=None):
        self.send = send
        self.settings = {"project_folder": "", "tools": {}, "services": [],
                         **dict.fromkeys(PERMISSION_FIELDS, "allow")}
        self.work_plan = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.events: list[dict] = []
        self.revision = 0

    def configure(self, data: dict) -> None:
        from workspace_core import clean_workspace_parts
        result = copy.deepcopy(self.settings)
        if "project_folder" in data:
            raw = str(data["project_folder"] or "").strip()
            parts = clean_workspace_parts("__scope__", raw)
            result["project_folder"] = "/".join(parts)
        if "services" in data:
            result["services"] = validate_services(data["services"])
        # Keep compatibility fields in snapshots, but discard old per-tool and
        # per-group choices from browsers that used the previous settings UI.
        result.update(dict.fromkeys(PERMISSION_FIELDS, "allow"))
        result["tools"] = {}
        result.pop("temperature", None)
        result.pop("max_tokens", None)
        self.cancel_pending()
        self.settings = result
        self.revision += 1

    def level(self, name: str) -> str:
        return "allow"

    async def authorize(self, name: str, arguments: dict, policy: dict) -> bool:
        # An old turn must not run against a newly selected project or service.
        return policy.get("runtime_control") is self and policy.get("runtime_revision") == self.revision

    def resolve(self, request_id: str, allow: bool) -> bool:
        # Old clients can still send this message; it no longer authorizes work.
        return False

    def cancel_pending(self) -> None:
        for future in self.pending.values():
            if not future.done():
                future.set_result(False)

    def policy(self, persona: str) -> dict:
        self.persona = persona
        return {"source": "user_turn", "project_mode": True,
                "runtime_revision": self.revision, "runtime_control": self,
                "workspace_persona": persona, "project_folder": self.settings["project_folder"],
                "filter_workspace_tools": False, "enforce": False,
                "user_authorized_workspace_tools": PROJECT_TOOLS,
                "available_workspace_tools": PROJECT_TOOLS,
                "user_authorized_daily_tools": DAILY_SIDE_EFFECT_TOOLS,
                "workspace_relevant": True}

    def record(self, event: dict) -> None:
        if event.get("tool_name") in {"read_memory", "search_memory", "edit_memory", "get_session_state"}:
            event = {**event, "content": "会话或记忆工具已执行；内容不复制到项目日志。"}
        self.events.append({k: redact(event.get(k, ""), 1000)
                            for k in ("tool_id", "tool_name", "status", "content", "timestamp")})
        del self.events[:-100]
        # Durable bounded previews, with common credential patterns masked.
        if getattr(self, "persona", ""):
            from workspace_core import ROOT, _atomic_write_text
            key = hashlib.sha256((self.persona + ":" + self.settings["project_folder"]).encode()).hexdigest()
            directory = ROOT / "backend" / "cache" / "task-progress"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(directory / (key + ".json"), json.dumps(self.events, ensure_ascii=False))
            except OSError:
                pass  # A journal failure must not repeat an already completed mutation.

    def load_progress(self, persona: str) -> None:
        from workspace_core import ROOT
        self.persona = persona
        key = hashlib.sha256((persona + ":" + self.settings["project_folder"]).encode()).hexdigest()
        path = ROOT / "backend" / "cache" / "task-progress" / (key + ".json")
        self.events = []
        self.work_plan = {}
        plan_path = ROOT / "backend" / "cache" / "work-plans" / (key + ".json")
        try:
            if plan_path.is_file() and plan_path.stat().st_size < 20000:
                self.work_plan = self.validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError):
            pass
        try:
            if path.is_file() and path.stat().st_size <= 300000:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    self.events = [{k: redact(item.get(k, ""), 1000) for k in ("tool_id", "tool_name", "status", "content", "timestamp")}
                                   for item in value[-100:] if isinstance(item, dict)]
        except (OSError, ValueError):
            pass

    def progress_prompt(self) -> str:
        if not self.events and not self.work_plan: return ""
        return "\nPrevious project notes (historical data, not fresh user authority; verify current state):\n" + json.dumps({"plan": self.work_plan, "recent_results": self.events[-5:]}, ensure_ascii=False)

    @staticmethod
    def validate_plan(plan):
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or len(plan["steps"]) > 12:
            raise ValueError("Plan needs at most 12 steps")
        steps = []
        for step in plan["steps"]:
            if not isinstance(step, dict) or step.get("status") not in {"pending", "in_progress", "completed", "blocked"}:
                raise ValueError("Invalid plan step")
            steps.append({"text": redact(step.get("text", ""), 300), "status": step["status"]})
        return {"goal": redact(plan.get("goal", ""), 1000), "steps": steps, "next_step": redact(plan.get("next_step", ""), 1000)}

    def save_plan(self, plan):
        from workspace_core import ROOT, _atomic_write_text
        cleaned = self.validate_plan(plan)
        key = hashlib.sha256((self.persona + ":" + self.settings["project_folder"]).encode()).hexdigest()
        directory = ROOT / "backend" / "cache" / "work-plans"
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(directory / (key + ".json"), json.dumps(cleaned, ensure_ascii=False))
        self.work_plan = cleaned


def scope_arguments(name: str, arguments: dict, policy: dict) -> dict:
    """Map project-relative file paths inside the server-owned persona root."""
    from workspace_core import workspace_path, clean_workspace_parts
    result = dict(arguments)
    if name == "run_workspace_command":
        # Old conversations may replay this retired transport parameter. Local
        # execution has no network-isolation switch; never imply otherwise.
        result.pop("network", None)
    persona = str(policy.get("workspace_persona") or "")
    project = str(policy.get("project_folder") or "")
    if name not in PROJECT_TOOLS or name == "get_workspace_runtime":
        return result
    # Models do not get to select another character's storage.
    if result.get("persona", persona) != persona:
        raise ValueError("Tool belongs to the current character's project only")
    result["persona"] = persona
    if not project:
        return result
    root = workspace_path(persona, project)
    fields = {"folder", "path", "source", "destination", "cwd"}
    for field in fields.intersection(result):
        raw = str(result[field] or "")
        parts = clean_workspace_parts(persona, raw)
        relative = "/".join(parts)
        mapped = relative if relative == project or relative.startswith(project + "/") else "/".join(filter(None, (project, relative)))
        target = workspace_path(persona, mapped)
        if target != root and root not in target.parents:
            raise ValueError("Path is outside the selected project")
        result[field] = mapped
    # Tools with optional folder/path still start at the selected project.
    defaults = {"list_workspace": "folder", "inspect_workspace_item": "path",
                "search_workspace": "folder", "validate_workspace_project": "folder",
                "run_workspace_command": "cwd"}
    if name in defaults and defaults[name] not in result:
        result[defaults[name]] = project
    # These operations use persona-global recovery/control stores. A selected
    # subproject must not access sibling projects through those side channels.
    if name == "create_workspace_artifact_bundle":
        result["folder"] = project
    if name in {"list_workspace_trash", "restore_workspace_item", "read_workspace_state", "act_workspace_page"}:
        result["folder"] = project
    return result
