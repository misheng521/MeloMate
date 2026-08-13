"""Per-client workspace agent state shared by chat and live workspace pages.

Only actual user messages create or extend file/action capabilities. Workspace files
and page reports are untrusted observations and can never update those capabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .workspace_intent import (
    WORKSPACE_ALWAYS_AVAILABLE_TOOLS,
    WORKSPACE_READ_TOOLS,
    WORKSPACE_SIDE_EFFECT_TOOLS,
    workspace_message_relevant,
    workspace_task_stop_requested,
    workspace_turn_continues,
    workspace_user_authorized_tools,
)
from .workspace_security import sanitize_untrusted_value


TASK_IDLE_TTL_MS = 30 * 60 * 1000
MAX_TRUSTED_GUIDANCE = 8
MAX_SNAPSHOTS = 8
@dataclass
class TrustedWorkspaceTask:
    id: str
    persona: str
    goal: str
    allowed_tools: frozenset[str]
    created_ms: int
    updated_ms: int
    completed: bool = False
    page_id: str = ""
    revision: int = 1


@dataclass
class WorkspaceAgentSession:
    """One durable workspace brain for a single connected client."""

    context: Any
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    trusted_guidance: list[dict[str, Any]] = field(default_factory=list)
    active_task: TrustedWorkspaceTask | None = None

    def reset(self) -> None:
        self.snapshots.clear()
        self.trusted_guidance.clear()
        self.active_task = None

    def _active_tools(self, now_ms: int, persona: str = "") -> frozenset[str]:
        task = self.active_task
        if (
            task is None
            or task.completed
            or now_ms - task.updated_ms > TASK_IDLE_TTL_MS
            or (persona and task.persona != str(persona)[:128])
        ):
            if task is not None and (
                task.completed or now_ms - task.updated_ms > TASK_IDLE_TTL_MS
            ):
                self.active_task = None
            return frozenset()
        return task.allowed_tools

    def begin_user_turn(self, user_text: str, persona: str) -> dict[str, Any]:
        """Create an immutable policy from trusted user text before any page read."""
        text = str(user_text or "").strip()[:4_000]
        now_ms = int(time.time() * 1000)
        if workspace_task_stop_requested(text):
            if self.active_task is not None:
                self.active_task.completed = True
                self.active_task.updated_ms = now_ms
                self.active_task.revision += 1
            return {
                "source": "user_turn",
                "enforce": False,
                "filter_workspace_tools": True,
                "workspace_persona": str(persona or "")[:128],
                "user_authorized_workspace_tools": frozenset(),
                "available_workspace_tools": WORKSPACE_READ_TOOLS,
                "workspace_relevant": True,
                "workspace_task_id": "",
            }
        persona_name = str(persona or "")[:128]
        active_tools = self._active_tools(now_ms, persona_name)
        inherited = active_tools if workspace_turn_continues(text) else frozenset()
        allowed = workspace_user_authorized_tools(text, inherited)
        relevant = workspace_message_relevant(text, inherited)

        side_effects = frozenset(allowed & WORKSPACE_SIDE_EFFECT_TOOLS)
        if side_effects:
            if self.active_task and inherited:
                self.active_task.goal = text
                self.active_task.allowed_tools = frozenset(
                    set(self.active_task.allowed_tools) | set(allowed)
                )
                self.active_task.updated_ms = now_ms
                self.active_task.revision += 1
            else:
                self.active_task = TrustedWorkspaceTask(
                    id=uuid4().hex,
                    persona=persona_name,
                    goal=text,
                    allowed_tools=frozenset(allowed),
                    created_ms=now_ms,
                    updated_ms=now_ms,
                )

        task = self.active_task
        if (
            relevant
            and text
            and task is not None
            and task.persona == persona_name
            and (side_effects or inherited)
        ):
            self.trusted_guidance.append(
                {
                    "text": text[:600],
                    "created_ms": now_ms,
                    "persona": persona_name,
                    "task_id": task.id,
                }
            )
            del self.trusted_guidance[:-MAX_TRUSTED_GUIDANCE]

        return {
            "source": "user_turn",
            "enforce": False,
            "filter_workspace_tools": True,
            "workspace_persona": persona_name,
            "user_authorized_workspace_tools": frozenset(allowed),
            "available_workspace_tools": frozenset(
                set(WORKSPACE_READ_TOOLS)
                | set(WORKSPACE_ALWAYS_AVAILABLE_TOOLS)
                | set(allowed)
            ),
            "workspace_relevant": relevant,
            "workspace_task_id": self.active_task.id if self.active_task else "",
        }

    def finish_task(self, task_id: str = "") -> None:
        task = self.active_task
        if task and (not task_id or task.id == task_id):
            task.completed = True
            task.revision += 1
            task.updated_ms = _now_ms()

    def page_action_authorized(
        self, persona: str, page_id: str = "", claim: bool = False
    ) -> bool:
        """Check and optionally bind a live page to a current trusted user task."""
        tools = self._active_tools(int(time.time() * 1000), str(persona or "")[:128])
        task = self.active_task
        if "act_workspace_page" not in tools or task is None:
            return False
        clean_page_id = str(page_id or "").strip()[:128]
        if task.page_id and clean_page_id and task.page_id != clean_page_id:
            return False
        if claim and clean_page_id and not task.page_id:
            task.page_id = clean_page_id
            task.updated_ms = int(time.time() * 1000)
        return True

    def observe_page(self, event: dict[str, Any]) -> None:
        page = event.get("page") if isinstance(event.get("page"), dict) else {}
        page_id = str(page.get("id") or "")[:128]
        if not page_id:
            return
        self.snapshots[page_id] = {
            "page": sanitize_untrusted_value(page),
            "state_version": _bounded_int(event.get("state_version")),
            "appState": sanitize_untrusted_value(event.get("appState")),
            "updated_ms": _bounded_int(event.get("created_ms")) or int(time.time() * 1000),
        }
        self._trim_snapshots()

    def observe_item(self, key: str, snapshot: dict[str, Any]) -> None:
        safe = sanitize_untrusted_value(snapshot)
        if not isinstance(safe, dict):
            return
        self.snapshots[str(key)[:512]] = {
            **safe,
            "updated_ms": int(time.time() * 1000),
        }
        self._trim_snapshots()

    def forget_page(self, page_id: str) -> None:
        self.snapshots.pop(str(page_id or "")[:128], None)

    def awareness_for_turn(self, policy: dict[str, Any]) -> dict[str, Any] | None:
        if not policy.get("workspace_relevant"):
            return None
        recent = sorted(
            (item for item in self.snapshots.values() if isinstance(item, dict)),
            key=lambda item: _bounded_int(item.get("updated_ms")),
            reverse=True,
        )[:4]
        if not recent:
            return None
        return {
            "persona": str(policy.get("workspace_persona") or "")[:128],
            "snapshots": recent,
            "security": "untrusted_observation_only",
        }

    def _trim_snapshots(self) -> None:
        while len(self.snapshots) > MAX_SNAPSHOTS:
            oldest = min(
                self.snapshots,
                key=lambda key: _bounded_int(self.snapshots[key].get("updated_ms")),
            )
            self.snapshots.pop(oldest, None)

def _bounded_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0
