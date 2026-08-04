"""Real-time controller for interactive workspace pages."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

import workspace_core

from .workspace_security import (
    extract_workspace_action_grants,
)


ReadState = Callable[[str, str], Awaitable[str]]
SendAction = Callable[
    [str, str, dict[str, Any], int, str, int, str], Awaitable[str]
]
SendText = Callable[[str], Awaitable[None]]
SpeakReply = Callable[[str, str, int], Awaitable[bool]]

STATE_DEBOUNCE_SECONDS = 0.12
FAILURE_COOLDOWN_SECONDS = 10.0
MAX_DECISION_ACTION_CHARS = 18_000
MAX_DECISION_ACTIONS = 72
MAX_DECISIONS_PER_MINUTE = 20

_EMPTY_BOARD_VALUES = {None, False, 0, "", ".", "-", "empty", "none", "null"}


def _bounded_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _workspace_report(result_text: str) -> dict[str, Any] | None:
    try:
        response = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(response, dict) or response.get("available") is not True:
        return None
    state_file = response.get("state")
    if not isinstance(state_file, dict):
        return None
    report = state_file.get("state")
    return report if isinstance(report, dict) else state_file


def _page_id(report: dict[str, Any]) -> str:
    page = report.get("page")
    return str(page.get("id") or "")[:128] if isinstance(page, dict) else ""


def _state_version(report: dict[str, Any]) -> int:
    return _bounded_int(report.get("state_version"))


def _agent_should_act(app_state: Any, persona: str) -> bool:
    """Require an explicit agent turn instead of clicking every advertised control."""
    if not isinstance(app_state, dict):
        return False
    if "agentShouldAct" in app_state:
        return app_state.get("agentShouldAct") is True
    if "agent_should_act" in app_state:
        return app_state.get("agent_should_act") is True
    current_turn = str(
        app_state.get("currentTurn") or app_state.get("current_turn") or ""
    ).strip().casefold()
    accepted_turns = {
        "melomate",
        "ai",
        "assistant",
        str(persona or "").strip().casefold(),
    }
    return bool(current_turn and current_turn in accepted_turns)


def _grid_position(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None
    pairs = (("row", "col"), ("r", "c"), ("y", "x"))
    for first, second in pairs:
        if first not in payload or second not in payload:
            continue
        try:
            row = int(payload[first])
            col = int(payload[second])
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= row <= 999 and 0 <= col <= 999:
            return row, col
    return None


def _occupied_board_positions(app_state: Any) -> list[tuple[int, int]]:
    if not isinstance(app_state, dict):
        return []
    board = app_state.get("board")
    if not isinstance(board, list):
        return []
    occupied: list[tuple[int, int]] = []
    for row_index, row in enumerate(board[:100]):
        if not isinstance(row, list):
            continue
        for col_index, cell in enumerate(row[:100]):
            comparable = cell.lower() if isinstance(cell, str) else cell
            try:
                is_empty = comparable in _EMPTY_BOARD_VALUES
            except TypeError:
                is_empty = False
            if not is_empty:
                occupied.append((row_index, col_index))
    return occupied


def _prioritize_action_grants(
    grants: list[dict[str, Any]], app_state: Any
) -> list[dict[str, Any]]:
    """Bound dense grid choices while leaving the final choice to the model."""
    if len(grants) <= MAX_DECISION_ACTIONS:
        return grants
    grid: list[tuple[dict[str, Any], tuple[int, int]]] = []
    other: list[dict[str, Any]] = []
    for grant in grants:
        position = _grid_position(grant.get("payload"))
        if position is None:
            other.append(grant)
        else:
            grid.append((grant, position))
    if len(grid) < MAX_DECISION_ACTIONS:
        return grants[:MAX_DECISION_ACTIONS]

    occupied = _occupied_board_positions(app_state)
    max_row = max(position[0] for _, position in grid)
    max_col = max(position[1] for _, position in grid)
    center_row = max_row / 2
    center_col = max_col / 2

    def score(item: tuple[dict[str, Any], tuple[int, int]]) -> tuple[float, float, int, int]:
        _, (row, col) = item
        nearest = (
            min(max(abs(row - used_row), abs(col - used_col)) for used_row, used_col in occupied)
            if occupied
            else 0
        )
        center_distance = abs(row - center_row) + abs(col - center_col)
        return nearest, center_distance, row, col

    keep_other = other[: min(len(other), 8)]
    grid_limit = max(1, MAX_DECISION_ACTIONS - len(keep_other))
    return [item[0] for item in sorted(grid, key=score)[:grid_limit]] + keep_other


def _compact_action_choices(
    grants: list[dict[str, Any]], app_state: Any = None
) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    size = 0
    for grant in _prioritize_action_grants(grants[:256], app_state):
        choice = {
            "id": str(grant.get("id") or "")[:128],
            "action": str(grant.get("action") or "")[:120],
            "payload": grant.get("payload") if isinstance(grant.get("payload"), dict) else {},
        }
        encoded = json.dumps(choice, ensure_ascii=False, separators=(",", ":"))
        if size + len(encoded) > MAX_DECISION_ACTION_CHARS:
            break
        choices.append(choice)
        size += len(encoded)
    return choices


class WorkspaceController:
    """Coalesce page updates and execute one validated action per state revision."""

    def __init__(
        self,
        context: Any,
        send_text: SendText,
        read_state: ReadState | None = None,
        send_action: SendAction | None = None,
        speak_reply: SpeakReply | None = None,
        debounce_seconds: float = STATE_DEBOUNCE_SECONDS,
    ) -> None:
        self._context = context
        self._send_text = send_text
        self._read_state = read_state or self._default_read_state
        self._send_action = send_action or self._default_send_action
        self._speak_reply = speak_reply
        self._debounce_seconds = max(0.0, debounce_seconds)
        self._pending: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._speech_tasks: set[asyncio.Task] = set()
        self._status_tasks: set[asyncio.Task] = set()
        self._last_processed: dict[str, tuple[int, int, str]] = {}
        self._last_acted_version: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._decision_times: deque[float] = deque()
        self._closed = False

    @staticmethod
    async def _default_read_state(persona: str, page_id: str) -> str:
        return await asyncio.to_thread(
            workspace_core.read_workspace_state, persona, page_id
        )

    @staticmethod
    async def _default_send_action(
        persona: str,
        action: str,
        payload: dict[str, Any],
        wait_ms: int,
        page_id: str,
        state_version: int,
        action_id: str,
    ) -> str:
        return await asyncio.to_thread(
            workspace_core.send_workspace_action,
            persona,
            action,
            payload,
            wait_ms,
            page_id,
            state_version,
            action_id,
        )

    def submit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        page = event.get("page")
        page_id = str(page.get("id") or "") if isinstance(page, dict) else ""
        if not page_id:
            return
        if event.get("type") == "workspace-page-closed" or bool(page.get("closed")):
            self._pending.pop(page_id, None)
            task = self._tasks.pop(page_id, None)
            if task and not task.done():
                task.cancel()
            self._clear_awareness(page_id)
            self._last_processed.pop(page_id, None)
            self._last_acted_version.pop(page_id, None)
            self._failures.pop(page_id, None)
            self._cooldown_until.pop(page_id, None)
            status_task = asyncio.create_task(
                self._status("closed", page_id, event, "工作区页面已关闭。")
            )
            self._status_tasks.add(status_task)
            status_task.add_done_callback(self._status_tasks.discard)
            return

        previous = self._pending.get(page_id)
        if previous and self._event_order(previous) >= self._event_order(event):
            return
        self._pending[page_id] = event
        self._set_awareness(event)
        task = self._tasks.get(page_id)
        if task is None or task.done():
            self._tasks[page_id] = asyncio.create_task(self._run_page(page_id))

    @staticmethod
    def _event_order(event: dict[str, Any]) -> tuple[int, int, str]:
        return (
            _bounded_int(event.get("state_version")),
            _bounded_int(event.get("created_ms")),
            str(event.get("id") or ""),
        )

    def _set_awareness(self, event: dict[str, Any]) -> None:
        session = getattr(self._context, "workspace_agent", None)
        if session is not None:
            session.observe_page(event)

    def _clear_awareness(self, page_id: str) -> None:
        session = getattr(self._context, "workspace_agent", None)
        if session is not None:
            session.forget_page(page_id)

    async def observe_item(self, persona: str, path: str) -> None:
        """Expose the currently viewed workspace file/folder to relevant chat turns."""
        if any(part.startswith(".") for part in str(path).replace("\\", "/").split("/") if part):
            return

        def read_item() -> dict[str, Any]:
            target = workspace_core.workspace_path(persona, path)
            if target.is_file():
                try:
                    payload = json.loads(workspace_core.read_workspace_file(persona, path))
                    content = str(payload.get("content") or "")[:12_000]
                except UnicodeDecodeError:
                    content = "[binary file; content preview unavailable]"
                return {
                    "kind": "file",
                    "path": path,
                    "contentChunks": [
                        content[index : index + 600]
                        for index in range(0, len(content), 600)
                    ],
                }
            if target.is_dir():
                payload = json.loads(workspace_core.list_workspace(persona, path))
                return {
                    "kind": "folder",
                    "path": path,
                    "entries": payload.get("entries") or [],
                }
            raise FileNotFoundError("workspace item was not found")

        try:
            snapshot = await asyncio.to_thread(read_item)
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            logger.debug(f"Workspace item awareness failed for {path}: {exc}")
            return
        session = getattr(self._context, "workspace_agent", None)
        if session is not None:
            session.observe_item(f"item:{path}", snapshot)

    async def _run_page(self, page_id: str) -> None:
        try:
            while not self._closed:
                if self._debounce_seconds:
                    await asyncio.sleep(self._debounce_seconds)
                event = self._pending.pop(page_id, None)
                if event is None:
                    return
                event_order = self._event_order(event)
                if event_order <= self._last_processed.get(page_id, (0, 0, "")):
                    continue
                self._last_processed[page_id] = event_order

                last_action = event.get("lastAction")
                if (
                    event.get("actionEvent") is True
                    and isinstance(last_action, dict)
                    and last_action.get("accepted") is True
                ):
                    continue
                if not _agent_should_act(event.get("appState"), event.get("persona", "")):
                    continue
                if not extract_workspace_action_grants(event.get("appState")):
                    continue

                cooldown = self._cooldown_until.get(page_id, 0.0) - time.monotonic()
                if cooldown > 0:
                    await self._status(
                        "paused",
                        page_id,
                        event,
                        "连续控制失败，稍后自动恢复。",
                    )
                    await asyncio.sleep(cooldown)
                    self._failures[page_id] = 0
                await self._process_event(page_id, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"Workspace controller failed for page {page_id}: {exc}")
            await self._status("error", page_id, None, "工作区实时控制发生错误。")
        finally:
            current = asyncio.current_task()
            if self._tasks.get(page_id) is current:
                self._tasks.pop(page_id, None)
            if not self._closed and page_id in self._pending and page_id not in self._tasks:
                self._tasks[page_id] = asyncio.create_task(self._run_page(page_id))

    async def _process_event(self, page_id: str, event: dict[str, Any]) -> None:
        persona = str(event.get("persona") or "")
        session = getattr(self._context, "workspace_agent", None)
        if session is None or not session.page_action_authorized(persona, page_id):
            return
        report = _workspace_report(await self._read_state(persona, page_id))
        if report is None or _page_id(report) != page_id:
            return
        version = _state_version(report)
        if version <= 0 or version != _bounded_int(event.get("state_version")):
            return
        if self._last_acted_version.get(page_id) == version:
            return

        app_state = report.get("appState")
        if not _agent_should_act(app_state, persona):
            return
        grants = _compact_action_choices(
            extract_workspace_action_grants(app_state), app_state
        )
        if not grants:
            return
        if not session.page_action_authorized(persona, page_id, claim=True):
            return
        authorized_task = session.active_task
        if authorized_task is None:
            return
        authorization = (authorized_task.id, authorized_task.revision)
        self._set_awareness(
            {
                **event,
                "state_version": version,
                "appState": app_state,
                "created_ms": _bounded_int(report.get("reported_ms")),
            }
        )
        await self._status("thinking", page_id, event, "正在观察并决定下一步。")

        now = time.monotonic()
        while self._decision_times and now - self._decision_times[0] >= 60:
            self._decision_times.popleft()
        if len(self._decision_times) >= MAX_DECISIONS_PER_MINUTE:
            await self._status(
                "paused",
                page_id,
                event,
                "页面变化过快，实时控制已暂时限速。",
            )
            return
        self._decision_times.append(now)

        try:
            selected_id, comment = await self._choose_action(
                persona, app_state, grants
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Workspace decision failed for {page_id}: {exc}")
            await self._record_failure(page_id, event, "暂时无法决定下一步。")
            return
        grant = next((item for item in grants if item["id"] == selected_id), None)
        if grant is None:
            await self._record_failure(page_id, event, "没有选出有效的页面操作。")
            return

        latest = _workspace_report(await self._read_state(persona, page_id))
        if latest is None or _page_id(latest) != page_id or _state_version(latest) != version:
            return
        current_task = session.active_task
        if (
            current_task is None
            or (current_task.id, current_task.revision) != authorization
            or not session.page_action_authorized(persona, page_id)
        ):
            return
        try:
            result_text = await self._send_action(
                persona,
                grant["action"],
                grant["payload"],
                900,
                page_id,
                version,
                grant["id"],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Workspace action failed for {page_id}: {exc}")
            await self._record_failure(page_id, event, "页面暂时无法执行当前操作。")
            return
        try:
            result = json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            result = {}
        if not isinstance(result, dict) or result.get("confirmed") is not True:
            if isinstance(result, dict) and result.get("stale") is True:
                return
            await self._record_failure(page_id, event, "页面没有确认当前操作。")
            return

        self._last_acted_version[page_id] = version
        self._failures[page_id] = 0
        self._cooldown_until.pop(page_id, None)
        if not comment:
            comment = (
                "我这边完成了，轮到你。",
                "这个操作我做好了，你继续。",
                "我已经处理好了，看看接下来有什么变化。",
                "我完成当前操作了，你来。",
            )[version % 4]
        await self._status(
            "acted",
            page_id,
            event,
            comment,
            action=grant["action"],
        )
        self._start_speech(comment, page_id, version)

    def _start_speech(self, text: str, page_id: str, version: int) -> None:
        if self._closed or self._speak_reply is None or not text:
            return

        async def speak() -> None:
            try:
                await self._speak_reply(text, page_id, version)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Workspace spoken reply failed for {page_id}: {exc}")

        task = asyncio.create_task(speak())
        self._speech_tasks.add(task)
        task.add_done_callback(self._speech_tasks.discard)

    async def _choose_action(
        self,
        persona: str,
        app_state: Any,
        grants: list[dict[str, Any]],
    ) -> tuple[str, str]:
        session = getattr(self._context, "workspace_agent", None)
        if session is None:
            return (str(grants[0]["id"]), "") if len(grants) == 1 else ("", "")
        return await session.choose_page_action(persona, app_state, grants)

    async def _record_failure(
        self, page_id: str, event: dict[str, Any], message: str
    ) -> None:
        failures = self._failures.get(page_id, 0) + 1
        self._failures[page_id] = failures
        if failures >= 3:
            self._cooldown_until[page_id] = time.monotonic() + FAILURE_COOLDOWN_SECONDS
        await self._status("error", page_id, event, message)

    async def _status(
        self,
        status: str,
        page_id: str,
        event: dict[str, Any] | None,
        message: str,
        action: str = "",
    ) -> None:
        payload = {
            "type": "workspace-control-status",
            "status": status,
            "page_id": page_id,
            "state_version": _bounded_int((event or {}).get("state_version")),
            "message": message[:300],
        }
        if action:
            payload["action"] = action[:120]
        try:
            await self._send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.debug(f"Workspace status could not be delivered: {exc}")

    async def close(self) -> None:
        self._closed = True
        self._pending.clear()
        tasks = [
            *self._tasks.values(),
            *self._speech_tasks,
            *self._status_tasks,
        ]
        self._tasks.clear()
        self._speech_tasks.clear()
        self._status_tasks.clear()
        self._last_processed.clear()
        self._last_acted_version.clear()
        self._failures.clear()
        self._cooldown_until.clear()
        self._decision_times.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        snapshots = getattr(self._context, "workspace_awareness", None)
        if isinstance(snapshots, dict):
            snapshots.clear()

    async def interrupt_speech(self) -> None:
        """Stop workspace-owned speech without cancelling a pending page decision."""
        tasks = list(self._speech_tasks)
        self._speech_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_idle(self) -> None:
        """Wait until currently queued controller work is drained (primarily for tests)."""
        while self._tasks or self._speech_tasks or self._status_tasks:
            await asyncio.gather(
                *list(self._tasks.values()),
                *list(self._speech_tasks),
                *list(self._status_tasks),
                return_exceptions=True,
            )
