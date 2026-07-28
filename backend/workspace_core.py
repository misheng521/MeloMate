import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = ROOT / "workspace"
MAX_FILE_BYTES = 1024 * 1024
MAX_PROJECT_FILES = 20
FRESH_STATE_MS = 5000
MAX_CONTROL_LINES = 200


def safe_name(value: str, fallback: str = "default") -> str:
    value = str(value or "").strip()
    value = re.sub(r"\.(ya?ml)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" .")
    return value or fallback


def ensure_inside(base: Path, target: Path) -> Path:
    base = base.resolve()
    target = target.resolve()
    if target == base or base in target.parents:
        return target
    raise ValueError("Path is outside the persona workspace.")


def persona_root(persona: str) -> Path:
    return ensure_inside(WORKSPACE_ROOT, WORKSPACE_ROOT / safe_name(persona))


def clean_workspace_parts(persona: str, relative_path: str = "") -> list[str]:
    persona_name = safe_name(persona)
    clean_parts = [
        safe_name(part, "")
        for part in Path(str(relative_path or "")).parts
        if part not in {"", ".", ".."}
    ]
    while clean_parts and clean_parts[0] == persona_name:
        clean_parts.pop(0)
    return clean_parts


def workspace_path(persona: str, relative_path: str = "") -> Path:
    clean_parts = clean_workspace_parts(persona, relative_path)
    return ensure_inside(persona_root(persona), persona_root(persona).joinpath(*clean_parts))


def response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def safe_slug(value: str, fallback: str = "reminder") -> str:
    slug = safe_name(value, fallback)
    slug = re.sub(r"_+", "_", slug)[:36].strip("_")
    return slug or fallback


def create_workspace_folder(persona: str, folder: str) -> str:
    target = workspace_path(persona, folder)
    target.mkdir(parents=True, exist_ok=True)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )


def write_workspace_file(persona: str, folder: str, filename: str, content: str) -> str:
    safe_filename = safe_name(filename)
    if "." not in safe_filename:
        raise ValueError("filename must include an extension such as .txt, .svg, .html, .css, .js, or .json.")

    text = str(content or "")
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("file content is too large.")

    directory = workspace_path(persona, folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = ensure_inside(persona_root(persona), directory / safe_filename)
    target.write_text(text, encoding="utf-8")
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )


def append_workspace_file(
    persona: str,
    folder: str,
    filename: str,
    content: str,
    reset: bool = False,
) -> str:
    safe_filename = safe_name(filename)
    if "." not in safe_filename:
        raise ValueError("filename must include an extension such as .txt, .svg, .html, .css, .js, or .json.")

    text = str(content or "")
    directory = workspace_path(persona, folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = ensure_inside(persona_root(persona), directory / safe_filename)

    existing_size = 0 if reset or not target.exists() else target.stat().st_size
    if existing_size + len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("file content is too large.")

    mode = "w" if reset else "a"
    with target.open(mode, encoding="utf-8") as file:
        file.write(text)

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "mode": "reset" if reset else "append",
        }
    )


def write_workspace_project(persona: str, folder: str, files: list[dict[str, Any]]) -> str:
    if not files:
        raise ValueError("files is required.")
    if len(files) > MAX_PROJECT_FILES:
        raise ValueError(f"too many files. maximum is {MAX_PROJECT_FILES}.")

    project_dir = workspace_path(persona, folder)
    project_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for item in files:
        if not isinstance(item, dict):
            raise ValueError("each project file must be an object with path and content.")

        relative_file = str(item.get("path") or "").strip()
        content = str(item.get("content") or "")
        if not relative_file:
            raise ValueError("each project file requires path.")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"{relative_file} is too large.")

        safe_parts = clean_workspace_parts(persona, relative_file)
        if not safe_parts or "." not in safe_parts[-1]:
            raise ValueError("each project file path must include a filename with an extension.")

        target = ensure_inside(persona_root(persona), project_dir.joinpath(*safe_parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target.relative_to(WORKSPACE_ROOT).as_posix())

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "branch": project_dir.relative_to(WORKSPACE_ROOT).as_posix(),
            "files_written": len(written),
            "paths": written,
        }
    )


def read_workspace_file(persona: str, path: str) -> str:
    target = workspace_path(persona, path)
    if not target.is_file():
        raise FileNotFoundError("workspace file was not found.")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("workspace file is too large to read.")
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "content": target.read_text(encoding="utf-8"),
        }
    )


def list_workspace(persona: str, folder: str = "") -> str:
    target = workspace_path(persona, folder)
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        if child.name.startswith("."):
            continue
        entries.append(
            {
                "name": child.name,
                "path": child.relative_to(WORKSPACE_ROOT).as_posix(),
                "type": "directory" if child.is_dir() else "file",
            }
        )
    return response({"ok": True, "persona": safe_name(persona), "entries": entries})


def schedule_reminder(persona: str, message: str, delay_minutes: float = 0, due_at: str = "") -> str:
    now = datetime.now().astimezone()
    if delay_minutes and delay_minutes > 0:
        due_time = now + timedelta(minutes=float(delay_minutes))
    elif due_at:
        normalized_due_at = str(due_at).strip()
        if normalized_due_at.endswith("Z"):
            normalized_due_at = f"{normalized_due_at[:-1]}+00:00"
        try:
            due_time = datetime.fromisoformat(normalized_due_at)
            if due_time.tzinfo is None:
                due_time = due_time.astimezone()
        except ValueError:
            due_time = now
    else:
        due_time = now

    reminder_text = str(message or "").strip()
    if not reminder_text:
        raise ValueError("message is required.")

    reminder_dir = workspace_path(persona, "reminders/pending")
    reminder_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{due_time.strftime('%Y%m%d-%H%M%S')}-{safe_slug(reminder_text)}.json"
    target = ensure_inside(persona_root(persona), reminder_dir / filename)
    payload = {
        "type": "reminder",
        "status": "pending",
        "persona": safe_name(persona),
        "message": reminder_text,
        "created_at": now.isoformat(timespec="seconds"),
        "due_at": due_time.isoformat(timespec="seconds"),
        "source_time": "device-local-time",
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "due_at": payload["due_at"],
            "message": reminder_text,
        }
    )


def send_workspace_key(
    persona: str,
    key: str,
    code: str = "",
    duration_ms: int = 80,
    repeat: int = 1,
) -> str:
    clean_key = str(key or "").strip()
    if not clean_key:
        raise ValueError("key is required, for example ArrowLeft, ArrowRight, ArrowUp, ArrowDown, Space, Enter, w, a, s, or d.")

    clean_code = str(code or "").strip()
    safe_duration = max(20, min(int(duration_ms or 80), 2000))
    safe_repeat = max(1, min(int(repeat or 1), 20))
    current_state = read_workspace_state_file(persona)
    page_id = state_page_id(current_state)
    now = datetime.now().astimezone()
    command = {
        "id": uuid4().hex,
        "type": "key",
        "page_id": page_id,
        "key": clean_key,
        "code": clean_code,
        "duration_ms": safe_duration,
        "repeat": safe_repeat,
        "created_ms": int(now.timestamp() * 1000),
        "created_at": now.isoformat(timespec="milliseconds"),
    }

    append_workspace_command(persona, command)

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "sent": True,
            "confirmed": False,
            "message": "KEY_SENT_EFFECT_NOT_CONFIRMED: Keyboard events were sent to the page, but their game/app effect is unknown. Do not claim a move, click, score, selection, or UI change unless a later read_workspace_state confirms it.",
            "command": {
                "id": command["id"],
                "type": command["type"],
                "page_id": command["page_id"],
                "key": command["key"],
                "code": command["code"],
                "duration_ms": command["duration_ms"],
                "repeat": command["repeat"],
            },
        }
    )


def workspace_control_dir(persona: str) -> Path:
    target = ensure_inside(persona_root(persona), persona_root(persona) / ".control")
    target.mkdir(parents=True, exist_ok=True)
    return target


def append_workspace_command(persona: str, command: dict[str, Any]) -> None:
    target = ensure_inside(persona_root(persona), workspace_control_dir(persona) / "commands.jsonl")
    lines = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    lines = [*lines[-MAX_CONTROL_LINES + 1 :], json.dumps(command, ensure_ascii=False)]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_workspace_state_file(persona: str) -> dict[str, Any] | None:
    target = ensure_inside(persona_root(persona), workspace_control_dir(persona) / "state.json")
    if not target.is_file():
        return None
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("workspace state is too large to read.")
    try:
        state = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("workspace state is not valid JSON.") from exc
    return state if isinstance(state, dict) else {"state": state}


def state_updated_ms(state: dict[str, Any] | None) -> int | None:
    if not state:
        return None
    value = state.get("updated_ms")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def state_age_ms(state: dict[str, Any] | None) -> int | None:
    updated_ms = state_updated_ms(state)
    if updated_ms is None:
        return None
    return max(0, int(time.time() * 1000) - updated_ms)


def state_payload(state: dict[str, Any] | None) -> dict[str, Any]:
    payload = state.get("state") if state else None
    return payload if isinstance(payload, dict) else {}


def state_page_id(state: dict[str, Any] | None) -> str:
    page = state_payload(state).get("page")
    return str(page.get("id") or "") if isinstance(page, dict) else ""


def state_protocol_available(state: dict[str, Any] | None) -> bool:
    payload = state_payload(state)
    return bool(payload.get("protocolAvailable") and payload.get("appState") is not None)


def state_is_fresh(state: dict[str, Any] | None) -> bool:
    age_ms = state_age_ms(state)
    return age_ms is not None and age_ms < FRESH_STATE_MS


def find_action_result(state: dict[str, Any] | None, command_id: str) -> dict[str, Any] | None:
    if not state:
        return None
    payload = state.get("state")
    candidates: list[Any] = []
    if isinstance(payload, dict):
        candidates.extend([payload.get("lastAction"), payload.get("last_action")])
        actions = payload.get("actions") or payload.get("actionResults") or payload.get("action_results")
        if isinstance(actions, list):
            candidates.extend(actions)
    candidates.extend([state.get("lastAction"), state.get("last_action")])
    for item in candidates:
        if isinstance(item, dict) and str(item.get("id") or "") == command_id:
            return item
    return None


def action_result_confirmed(
    state: dict[str, Any] | None,
    action_result: dict[str, Any] | None,
    command_id: str,
    page_id: str = "",
) -> bool:
    if not action_result:
        return False
    if str(action_result.get("id") or "") != command_id:
        return False
    if page_id and state_page_id(state) != page_id:
        return False
    if action_result.get("handled") is not True:
        return False
    if action_result.get("accepted") is not True:
        return False
    if not state_protocol_available(state):
        return False
    if not state_is_fresh(state):
        return False
    return True


def wait_for_action_result(
    persona: str,
    command_id: str,
    previous_updated_ms: int | None,
    wait_ms: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.monotonic() + max(0, min(int(wait_ms or 0), 5000)) / 1000
    latest_state = read_workspace_state_file(persona)
    latest_result = find_action_result(latest_state, command_id)
    while not latest_result and time.monotonic() < deadline:
        updated_ms = state_updated_ms(latest_state)
        if updated_ms is not None and previous_updated_ms is not None and updated_ms > previous_updated_ms:
            latest_result = find_action_result(latest_state, command_id)
            if latest_result:
                break
        time.sleep(0.05)
        latest_state = read_workspace_state_file(persona)
        latest_result = find_action_result(latest_state, command_id)
    return latest_result, latest_state


def send_workspace_action(
    persona: str,
    action: str,
    payload: dict[str, Any] | None = None,
    wait_ms: int = 900,
) -> str:
    clean_action = str(action or "").strip()
    if not clean_action:
        raise ValueError("action is required, for example place-piece, select-cell, click, move, restart, or pass.")
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload must be an object.")

    previous_state = read_workspace_state_file(persona)
    previous_updated_ms = state_updated_ms(previous_state)
    page_id = state_page_id(previous_state)
    now = datetime.now().astimezone()
    command = {
        "id": uuid4().hex,
        "type": "action",
        "page_id": page_id,
        "action": clean_action,
        "payload": payload or {},
        "created_ms": int(now.timestamp() * 1000),
        "created_at": now.isoformat(timespec="milliseconds"),
    }
    append_workspace_command(persona, command)
    action_result, latest_state = wait_for_action_result(
        persona,
        command["id"],
        previous_updated_ms,
        wait_ms,
    )
    confirmed = action_result_confirmed(latest_state, action_result, command["id"], page_id)

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "sent": True,
            "confirmed": confirmed,
            "control_ready": state_protocol_available(latest_state) and state_is_fresh(latest_state),
            "action_result": action_result,
            "state": latest_state,
            "message": (
                "Action was accepted by the open workspace page. Use the returned state/action_result for your reply."
                if confirmed
                else "CONTROL_NOT_CONFIRMED: The action was not proven to run in the open workspace page. You must not say or imply that you clicked, moved, placed, chose, changed, scored, won, or completed the action. Say briefly that the workspace did not confirm the action, then ask the user to reopen it through MeloMate or revise the app protocol."
            ),
            "command": {
                "id": command["id"],
                "type": command["type"],
                "page_id": command["page_id"],
                "action": command["action"],
                "payload": command["payload"],
            },
        }
    )


def read_workspace_state(persona: str) -> str:
    state = read_workspace_state_file(persona)
    if state is None:
        return response(
            {
                "ok": True,
                "persona": safe_name(persona),
                "available": False,
                "state": None,
                "message": "No workspace app has reported state yet. You cannot see the board or game state. Do not claim any move, coordinate, score, winner, or board position. Ask the user to open the workspace HTML through MeloMate or update the app to publish MeloMateGameState.",
            }
        )
    age_ms = state_age_ms(state)
    fresh = state_is_fresh(state)
    protocol_available = state_protocol_available(state)
    control_ready = protocol_available and fresh

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "available": control_ready,
            "fresh": fresh,
            "protocol_available": protocol_available,
            "control_ready": control_ready,
            "age_ms": age_ms,
            "message": (
                "Workspace control is ready. Use only this reported state for game/app claims."
                if control_ready
                else "CONTROL_NOT_READY: The workspace page is stale or does not expose MeloMateGameState. Do not claim any move, click, choice, score, winner, or current UI state."
            ),
            "state": state,
        }
    )


def open_workspace_item(persona: str, path: str) -> str:
    target = workspace_path(persona, path)
    if not target.exists():
        raise FileNotFoundError("workspace item was not found.")

    opened_url = ""
    if target.is_file() and target.suffix.lower() == ".html":
        persona_name = safe_name(persona)
        relative_item = target.relative_to(persona_root(persona)).as_posix()
        base_url = os.getenv("MELOMATE_FRONTEND_URL", "http://127.0.0.1:5178").rstrip("/")
        opened_url = f"{base_url}/workspace-files/{quote(persona_name)}/{quote(relative_item, safe='/')}"
        open_target = opened_url
    else:
        open_target = str(target)

    if sys.platform == "win32":
        os.startfile(open_target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", open_target])
    else:
        subprocess.Popen(["xdg-open", open_target])

    branch = target.parent if target.is_file() else target
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "opened": True,
            "url": opened_url,
            "branch": branch.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )
