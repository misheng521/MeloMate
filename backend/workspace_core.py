import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = ROOT / "workspace"
MAX_FILE_BYTES = 1024 * 1024
MAX_PROJECT_FILES = 20


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
    now = datetime.now().astimezone()
    command = {
        "id": uuid4().hex,
        "type": "key",
        "key": clean_key,
        "code": clean_code,
        "duration_ms": safe_duration,
        "repeat": safe_repeat,
        "created_ms": int(now.timestamp() * 1000),
        "created_at": now.isoformat(timespec="milliseconds"),
    }

    control_dir = ensure_inside(persona_root(persona), persona_root(persona) / ".control")
    control_dir.mkdir(parents=True, exist_ok=True)
    target = ensure_inside(persona_root(persona), control_dir / "commands.jsonl")
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(command, ensure_ascii=False) + "\n")

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "sent": True,
            "command": {
                "type": command["type"],
                "key": command["key"],
                "code": command["code"],
                "duration_ms": command["duration_ms"],
                "repeat": command["repeat"],
            },
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
