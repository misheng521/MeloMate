import base64
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = ROOT / "workspace"
MAX_FILE_BYTES = 1024 * 1024
MAX_PROJECT_FILES = 64
MAX_PROJECT_BYTES = 4 * 1024 * 1024
MAX_SEARCH_FILES = 1000
MAX_SEARCH_RESULTS = 100
MAX_ACTION_PAYLOAD_BYTES = 32 * 1024
FRESH_STATE_MS = 5000
MAX_CONTROL_LINES = 200
MAX_TRASH_BYTES = 64 * 1024 * 1024
MAX_TRASH_ENTRY_OVERHEAD_BYTES = 64 * 1024
MAX_TRASH_ITEMS = 100
MAX_TRASH_AGE_SECONDS = 7 * 24 * 60 * 60
RESERVED_WORKSPACE_PARTS = frozenset({".control", ".trash"})
_COMMAND_LOCK = threading.Lock()
_WORKSPACE_MUTATION_LOCK = threading.RLock()
_ACTION_LOCKS_GUARD = threading.Lock()
_ACTION_LOCKS: dict[str, threading.Lock] = {}


def _locked_workspace_mutation(function):
    """Serialize agent-owned mutations so checked edits cannot race each other."""
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        with _WORKSPACE_MUTATION_LOCK:
            return function(*args, **kwargs)

    return guarded


def safe_name(value: str, fallback: str = "default") -> str:
    value = str(value or "").strip()
    value = re.sub(r"\.(ya?ml)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" .")
    return value or fallback


def workspace_access_token(persona: str) -> str:
    """Create the same launch-scoped, persona-bound token as the frontend server."""
    secret = os.getenv("MELOMATE_SESSION_TOKEN", "")
    if not secret:
        raise RuntimeError(
            "Workspace files must be opened through the MeloMate launcher."
        )
    persona_name = safe_name(persona)
    digest = hmac.new(
        secret.encode("utf-8"),
        f"melomate-workspace:{persona_name}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def ensure_inside(base: Path, target: Path) -> Path:
    base = base.resolve()
    target = target.resolve()
    if target == base or base in target.parents:
        return target
    raise ValueError("Path is outside the persona workspace.")


def persona_root(persona: str) -> Path:
    return ensure_inside(WORKSPACE_ROOT, WORKSPACE_ROOT / safe_name(persona))


def clean_workspace_parts(persona: str, relative_path: str = "") -> list[str]:
    raw_path = str(relative_path or "").strip().replace("\\", "/")
    if "\x00" in raw_path or raw_path.startswith("/") or re.match(r"^[A-Za-z]:", raw_path):
        raise ValueError("Workspace paths must be relative paths.")
    persona_name = safe_name(persona)
    clean_parts: list[str] = []
    for raw_part in raw_path.split("/"):
        part = raw_part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("Parent path segments are not allowed in the workspace.")
        cleaned = safe_name(part, "")
        if not cleaned or cleaned != part:
            raise ValueError(f"Invalid workspace path segment: {part!r}.")
        if cleaned.casefold() in RESERVED_WORKSPACE_PARTS:
            raise ValueError("The workspace runtime control directory is not user-editable.")
        clean_parts.append(cleaned)
    while clean_parts and clean_parts[0] == persona_name:
        clean_parts.pop(0)
    return clean_parts


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    reparse_attribute = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_attribute)


def _reject_reparse_components(root: Path, target: Path) -> None:
    current = root
    if current.exists() and _is_reparse_point(current):
        raise ValueError("Reparse points are not allowed in the workspace path.")
    try:
        parts = target.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("Path is outside the persona workspace.") from exc
    for part in parts:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            raise ValueError("Reparse points are not allowed in the workspace path.")


def workspace_path(persona: str, relative_path: str = "") -> Path:
    clean_parts = clean_workspace_parts(persona, relative_path)
    root = persona_root(persona)
    lexical_target = root.joinpath(*clean_parts)
    _reject_reparse_components(root, lexical_target)
    return ensure_inside(root, lexical_target)


def _atomic_write_text(target: Path, text: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_filename(filename: str) -> str:
    raw = str(filename or "").strip()
    cleaned = safe_name(raw, "")
    if not cleaned or cleaned != raw or cleaned.casefold() in RESERVED_WORKSPACE_PARTS:
        raise ValueError("Invalid workspace filename.")
    if "." not in cleaned:
        raise ValueError(
            "filename must include an extension such as .txt, .svg, .html, .css, .js, or .json."
        )
    return cleaned


def response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@_locked_workspace_mutation
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


@_locked_workspace_mutation
def write_workspace_file(persona: str, folder: str, filename: str, content: str) -> str:
    safe_filename = _validated_filename(filename)

    text = str(content or "")
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("file content is too large.")

    directory = workspace_path(persona, folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = ensure_inside(persona_root(persona), directory / safe_filename)
    _atomic_write_text(target, text)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )


@_locked_workspace_mutation
def append_workspace_file(
    persona: str,
    folder: str,
    filename: str,
    content: str,
    reset: bool = False,
) -> str:
    safe_filename = _validated_filename(filename)

    text = str(content or "")
    directory = workspace_path(persona, folder)
    directory.mkdir(parents=True, exist_ok=True)
    target = ensure_inside(persona_root(persona), directory / safe_filename)

    existing_size = 0 if reset or not target.exists() else target.stat().st_size
    if existing_size + len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("file content is too large.")

    existing = "" if reset or not target.exists() else target.read_text(encoding="utf-8")
    _atomic_write_text(target, existing + text)

    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "mode": "reset" if reset else "append",
        }
    )


@_locked_workspace_mutation
def write_workspace_project(persona: str, folder: str, files: list[dict[str, Any]]) -> str:
    if not files:
        raise ValueError("files is required.")
    if len(files) > MAX_PROJECT_FILES:
        raise ValueError(f"too many files. maximum is {MAX_PROJECT_FILES}.")

    project_dir = workspace_path(persona, folder)
    prepared: list[tuple[Path, str]] = []
    total_bytes = 0

    for item in files:
        if not isinstance(item, dict):
            raise ValueError("each project file must be an object with path and content.")

        relative_file = str(item.get("path") or "").strip()
        content = str(item.get("content") or "")
        if not relative_file:
            raise ValueError("each project file requires path.")
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_FILE_BYTES:
            raise ValueError(f"{relative_file} is too large.")
        total_bytes += content_bytes
        if total_bytes > MAX_PROJECT_BYTES:
            raise ValueError("project content is too large.")

        safe_parts = clean_workspace_parts(persona, relative_file)
        if not safe_parts or "." not in safe_parts[-1]:
            raise ValueError("each project file path must include a filename with an extension.")

        target = ensure_inside(persona_root(persona), project_dir.joinpath(*safe_parts))
        _reject_reparse_components(persona_root(persona), target)
        prepared.append((target, content))

    project_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for target, content in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
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


def _sha256_file(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_workspace_item(persona: str, path: str = "") -> str:
    """Return bounded metadata for one item without reading its contents."""
    target = workspace_path(persona, path)
    if not target.exists():
        raise FileNotFoundError("workspace item was not found.")
    root = persona_root(persona)
    stat = target.stat()
    payload: dict[str, Any] = {
        "ok": True,
        "persona": safe_name(persona),
        "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
        "type": "directory" if target.is_dir() else "file",
        "size": stat.st_size if target.is_file() else _item_size(target),
        "modified_ms": int(stat.st_mtime * 1000),
    }
    if target.is_file():
        payload["sha256"] = _sha256_file(target)
    else:
        payload["entries"] = sum(
            1
            for child in target.iterdir()
            if child.name.casefold() not in RESERVED_WORKSPACE_PARTS
            and not _is_reparse_point(child)
        )
    ensure_inside(root, target)
    return response(payload)


def read_workspace_file_range(
    persona: str,
    path: str,
    offset: int = 0,
    max_chars: int = 64_000,
) -> str:
    """Read a stable text range so large edits do not require repeated full reads."""
    target = workspace_path(persona, path)
    if not target.is_file():
        raise FileNotFoundError("workspace file was not found.")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("workspace file is too large to read.")
    start = max(0, int(offset or 0))
    limit = max(1, min(int(max_chars or 64_000), 64_000))
    content = target.read_text(encoding="utf-8")
    chunk = content[start : start + limit]
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "offset": start,
            "next_offset": start + len(chunk),
            "eof": start + len(chunk) >= len(content),
            "total_chars": len(content),
            "sha256": _sha256_file(target),
            "content": chunk,
        }
    )


@_locked_workspace_mutation
def patch_workspace_file(
    persona: str,
    path: str,
    expected_sha256: str,
    replacements: list[dict[str, Any]],
) -> str:
    """Atomically apply checked exact-text replacements to one current file version."""
    target = workspace_path(persona, path)
    if not target.is_file():
        raise FileNotFoundError("workspace file was not found.")
    expected = str(expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_sha256 must be the hash returned by the latest read.")
    current_hash = _sha256_file(target)
    if not hmac.compare_digest(current_hash, expected):
        raise ValueError("workspace file changed; inspect or read it again before patching.")
    if not isinstance(replacements, list) or not 1 <= len(replacements) <= 64:
        raise ValueError("replacements must contain 1-64 exact text edits.")
    updated = target.read_text(encoding="utf-8")
    applied = 0
    for edit in replacements:
        if not isinstance(edit, dict):
            raise ValueError("each replacement must be an object.")
        old_text = str(edit.get("old_text") or "")
        new_text = str(edit.get("new_text") or "")
        replace_all = edit.get("replace_all") is True
        if not old_text:
            raise ValueError("replacement old_text must not be empty.")
        matches = updated.count(old_text)
        if matches == 0:
            raise ValueError("replacement text was not found; read the file again.")
        if matches > 1 and not replace_all:
            raise ValueError("replacement text is not unique; include more context.")
        updated = updated.replace(old_text, new_text, -1 if replace_all else 1)
        applied += matches if replace_all else 1
    if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("patched file would exceed the size limit.")
    _atomic_write_text(target, updated)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "replacements": applied,
            "sha256": _sha256_file(target),
        }
    )


def list_workspace(persona: str, folder: str = "") -> str:
    target = workspace_path(persona, folder)
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        if child.name.casefold() in RESERVED_WORKSPACE_PARTS or _is_reparse_point(child):
            continue
        entries.append(
            {
                "name": child.name,
                "path": child.relative_to(WORKSPACE_ROOT).as_posix(),
                "type": "directory" if child.is_dir() else "file",
            }
        )
    return response({"ok": True, "persona": safe_name(persona), "entries": entries})


@_locked_workspace_mutation
def replace_workspace_text(
    persona: str,
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    target = workspace_path(persona, path)
    if not target.is_file():
        raise FileNotFoundError("workspace file was not found.")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("workspace file is too large to edit.")
    needle = str(old_text or "")
    if not needle:
        raise ValueError("old_text must not be empty.")
    existing = target.read_text(encoding="utf-8")
    matches = existing.count(needle)
    if matches == 0:
        raise ValueError("old_text was not found; read the file again before editing.")
    if matches > 1 and not replace_all:
        raise ValueError(
            "old_text is not unique; provide more surrounding text or set replace_all=true."
        )
    updated = existing.replace(needle, str(new_text or ""), -1 if replace_all else 1)
    if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("edited file would exceed the size limit.")
    _atomic_write_text(target, updated)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "replacements": matches if replace_all else 1,
        }
    )


@_locked_workspace_mutation
def move_workspace_item(
    persona: str,
    source: str,
    destination: str,
) -> str:
    source_target = workspace_path(persona, source)
    destination_target = workspace_path(persona, destination)
    root = persona_root(persona)
    if source_target == root:
        raise ValueError("The persona workspace root cannot be moved.")
    if not source_target.exists():
        raise FileNotFoundError("workspace source item was not found.")
    if destination_target.exists():
        raise FileExistsError("workspace destination already exists.")
    if source_target.is_dir() and source_target in destination_target.parents:
        raise ValueError("A folder cannot be moved inside itself.")
    destination_target.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_components(root, destination_target.parent)
    source_target.rename(destination_target)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "source": source_target.relative_to(WORKSPACE_ROOT).as_posix(),
            "destination": destination_target.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )


def _item_size(target: Path) -> int:
    if _is_reparse_point(target):
        raise ValueError("Reparse points cannot be archived or restored.")
    if target.is_file():
        return target.stat().st_size
    total = 0
    for child in target.rglob("*"):
        if _is_reparse_point(child):
            raise ValueError("Reparse points cannot be archived or restored.")
        if child.is_file():
            total += child.stat().st_size
    return total


def _trash_root(persona: str) -> Path:
    target = ensure_inside(persona_root(persona), persona_root(persona) / ".trash")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _trash_entries(persona: str) -> list[Path]:
    root = _trash_root(persona)
    return sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and not _is_reparse_point(child)
        ),
        key=lambda child: child.stat().st_mtime,
    )


def _prune_trash(persona: str) -> None:
    now = time.time()
    entries = _trash_entries(persona)
    for entry in list(entries):
        if now - entry.stat().st_mtime > MAX_TRASH_AGE_SECONDS:
            shutil.rmtree(entry)
            entries.remove(entry)
    sizes = {entry: _item_size(entry) for entry in entries}
    total = sum(sizes.values())
    while entries and (len(entries) > MAX_TRASH_ITEMS or total > MAX_TRASH_BYTES):
        oldest = entries.pop(0)
        total -= sizes.get(oldest, 0)
        shutil.rmtree(oldest)


@_locked_workspace_mutation
def delete_workspace_item(
    persona: str,
    path: str,
    recursive: bool = False,
) -> str:
    target = workspace_path(persona, path)
    root = persona_root(persona)
    if target == root:
        raise ValueError("The persona workspace root cannot be deleted.")
    if not target.exists():
        raise FileNotFoundError("workspace item was not found.")
    item_type = "directory" if target.is_dir() else "file"
    if target.is_dir() and any(target.iterdir()) and not recursive:
        raise ValueError("The folder is not empty; set recursive=true to archive it.")
    item_size = _item_size(target)
    if item_size > MAX_TRASH_BYTES - MAX_TRASH_ENTRY_OVERHEAD_BYTES:
        raise ValueError("workspace item is too large for recoverable deletion.")
    _prune_trash(persona)
    trash_id = f"{int(time.time() * 1000)}-{uuid4().hex}"
    entry = ensure_inside(root, _trash_root(persona) / trash_id)
    entry.mkdir(parents=False, exist_ok=False)
    payload = ensure_inside(root, entry / "payload")
    original_path = target.relative_to(root).as_posix()
    try:
        target.rename(payload)
        _atomic_write_text(
            entry / "metadata.json",
            json.dumps(
                {
                    "id": trash_id,
                    "original_path": original_path,
                    "type": item_type,
                    "size": item_size,
                    "deleted_ms": int(time.time() * 1000),
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        if payload.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            payload.rename(target)
        shutil.rmtree(entry, ignore_errors=True)
        raise
    _prune_trash(persona)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "type": item_type,
            "deleted": True,
            "recoverable": True,
            "trash_id": trash_id,
        }
    )


def list_workspace_trash(persona: str) -> str:
    _prune_trash(persona)
    entries: list[dict[str, Any]] = []
    for entry in reversed(_trash_entries(persona)):
        metadata = entry / "metadata.json"
        if not metadata.is_file() or metadata.stat().st_size > 16_384:
            continue
        try:
            item = json.loads(metadata.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(item, dict):
            entries.append(item)
    return response({"ok": True, "persona": safe_name(persona), "entries": entries})


@_locked_workspace_mutation
def restore_workspace_item(
    persona: str,
    trash_id: str,
    destination: str = "",
) -> str:
    clean_id = str(trash_id or "").strip()
    if not re.fullmatch(r"[0-9]{10,16}-[0-9a-f]{32}", clean_id):
        raise ValueError("invalid trash_id.")
    root = persona_root(persona)
    entry = ensure_inside(root, _trash_root(persona) / clean_id)
    metadata_path = ensure_inside(root, entry / "metadata.json")
    payload = ensure_inside(root, entry / "payload")
    if not metadata_path.is_file() or not payload.exists():
        raise FileNotFoundError("recoverable workspace item was not found.")
    if metadata_path.stat().st_size > 16_384:
        raise ValueError("recoverable workspace metadata is invalid.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("recoverable workspace metadata is invalid.") from exc
    if not isinstance(metadata, dict) or metadata.get("id") != clean_id:
        raise ValueError("recoverable workspace metadata is invalid.")
    restore_path = str(destination or metadata.get("original_path") or "")
    target = workspace_path(persona, restore_path)
    if target.exists():
        raise FileExistsError("workspace restore destination already exists.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_components(root, target.parent)
    payload.rename(target)
    shutil.rmtree(entry)
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "trash_id": clean_id,
            "path": target.relative_to(WORKSPACE_ROOT).as_posix(),
            "restored": True,
        }
    )


def search_workspace(
    persona: str,
    query: str,
    folder: str = "",
    max_results: int = 50,
) -> str:
    needle = str(query or "").strip()
    if not needle or len(needle) > 200:
        raise ValueError("query must contain 1-200 characters.")
    target = workspace_path(persona, folder)
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError("workspace folder was not found.")
    limit = max(1, min(int(max_results or 50), MAX_SEARCH_RESULTS))
    matches: list[dict[str, Any]] = []
    files_checked = 0
    for candidate in sorted(target.rglob("*")):
        if files_checked >= MAX_SEARCH_FILES or len(matches) >= limit:
            break
        if not candidate.is_file() or _is_reparse_point(candidate):
            continue
        if any(part.casefold() in RESERVED_WORKSPACE_PARTS for part in candidate.parts):
            continue
        files_checked += 1
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(lines, start=1):
            if needle.casefold() not in line.casefold():
                continue
            matches.append(
                {
                    "path": candidate.relative_to(WORKSPACE_ROOT).as_posix(),
                    "line": line_number,
                    "text": line.strip()[:300],
                }
            )
            if len(matches) >= limit:
                break
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "query": needle,
            "matches": matches,
            "files_checked": files_checked,
            "truncated": files_checked >= MAX_SEARCH_FILES or len(matches) >= limit,
        }
    )


def workspace_control_dir(persona: str) -> Path:
    target = ensure_inside(persona_root(persona), persona_root(persona) / ".control")
    target.mkdir(parents=True, exist_ok=True)
    return target


def workspace_page_state_path(persona: str, page_id: str) -> Path:
    """Return an internal per-page state file without exposing page ids as paths."""
    clean_page_id = str(page_id or "").strip()
    if not clean_page_id or len(clean_page_id) > 128:
        raise ValueError("page_id must contain 1-128 characters.")
    pages_dir = ensure_inside(
        persona_root(persona), workspace_control_dir(persona) / "pages"
    )
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_key = hashlib.sha256(clean_page_id.encode("utf-8")).hexdigest()
    return ensure_inside(persona_root(persona), pages_dir / f"{page_key}.json")


def append_workspace_command(persona: str, command: dict[str, Any]) -> None:
    target = ensure_inside(
        persona_root(persona), workspace_control_dir(persona) / "commands.jsonl"
    )
    with _COMMAND_LOCK:
        lines = (
            target.read_text(encoding="utf-8").splitlines()
            if target.is_file()
            else []
        )
        previous_created_ms = 0
        if lines:
            try:
                previous_created_ms = int(json.loads(lines[-1]).get("created_ms") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                previous_created_ms = 0
        bounded_command = dict(command)
        bounded_command["created_ms"] = max(
            int(bounded_command.get("created_ms") or 0), previous_created_ms + 1
        )
        lines = [
            *lines[-MAX_CONTROL_LINES + 1 :],
            json.dumps(bounded_command, ensure_ascii=False),
        ]
        _atomic_write_text(target, "\n".join(lines) + "\n")


def read_workspace_state_file(
    persona: str, page_id: str = ""
) -> dict[str, Any] | None:
    target = (
        workspace_page_state_path(persona, page_id)
        if page_id
        else ensure_inside(
            persona_root(persona), workspace_control_dir(persona) / "state.json"
        )
    )
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


def state_version(state: dict[str, Any] | None) -> int:
    value = state_payload(state).get("state_version")
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def state_protocol_available(state: dict[str, Any] | None) -> bool:
    payload = state_payload(state)
    return bool(payload.get("protocolAvailable") and payload.get("appState") is not None)


def state_is_fresh(state: dict[str, Any] | None) -> bool:
    age_ms = state_age_ms(state)
    return age_ms is not None and age_ms < FRESH_STATE_MS


def advertised_workspace_actions(
    state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return exact semantic actions advertised by the current page state."""
    app_state = state_payload(state).get("appState")
    if not isinstance(app_state, dict):
        return []
    advertised = app_state.get("availableActions")
    if not isinstance(advertised, list):
        advertised = app_state.get("available_actions")
    if not isinstance(advertised, list):
        return []
    return [item for item in advertised[:256] if isinstance(item, dict)]


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
    page_id: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.monotonic() + max(0, min(int(wait_ms or 0), 5000)) / 1000
    latest_state = read_workspace_state_file(persona, page_id)
    latest_result = find_action_result(latest_state, command_id)
    while not latest_result and time.monotonic() < deadline:
        updated_ms = state_updated_ms(latest_state)
        if updated_ms is not None and previous_updated_ms is not None and updated_ms > previous_updated_ms:
            latest_result = find_action_result(latest_state, command_id)
            if latest_result:
                break
        time.sleep(0.05)
        latest_state = read_workspace_state_file(persona, page_id)
        latest_result = find_action_result(latest_state, command_id)
    return latest_result, latest_state


def _workspace_action_lock(persona: str, page_id: str) -> threading.Lock:
    # One lock per persona prevents an unbounded number of page ids from growing
    # this process-owned map and also serializes commands across that workspace.
    key = safe_name(persona)
    with _ACTION_LOCKS_GUARD:
        lock = _ACTION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ACTION_LOCKS[key] = lock
        return lock


def _send_workspace_action_unlocked(
    persona: str,
    action: str = "",
    payload: dict[str, Any] | None = None,
    wait_ms: int = 900,
    expected_page_id: str = "",
    expected_state_version: int | None = None,
    action_id: str = "",
) -> str:
    previous_state = read_workspace_state_file(persona, expected_page_id)
    selected_action_id = str(action_id or "").strip()[:128]
    if selected_action_id:
        selected = next(
            (
                item
                for item in advertised_workspace_actions(previous_state)
                if str(item.get("id") or "") == selected_action_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "action_id is not advertised by the current workspace page state."
            )
        action = str(selected.get("action") or "")
        selected_payload = selected.get("payload")
        payload = selected_payload if isinstance(selected_payload, dict) else {}

    clean_action = str(action or "").strip()
    if not clean_action:
        raise ValueError(
            "action or a current page-advertised action_id is required."
        )
    if len(clean_action) > 120:
        raise ValueError("workspace action must be at most 120 characters.")
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload must be an object.")
    try:
        encoded_payload = json.dumps(
            payload or {}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must contain valid finite JSON values.") from exc
    if len(encoded_payload) > MAX_ACTION_PAYLOAD_BYTES:
        raise ValueError("workspace action payload is too large.")

    current_page_id = state_page_id(previous_state)
    current_version = state_version(previous_state)
    if expected_page_id and current_page_id != str(expected_page_id):
        return response(
            {
                "ok": False,
                "sent": False,
                "confirmed": False,
                "stale": True,
                "message": "STALE_WORKSPACE_STATE: the open page changed before the action was sent.",
            }
        )
    if expected_state_version is not None and current_version != int(
        expected_state_version
    ):
        return response(
            {
                "ok": False,
                "sent": False,
                "confirmed": False,
                "stale": True,
                "message": "STALE_WORKSPACE_STATE: the page state changed before the action was sent.",
            }
        )
    previous_updated_ms = state_updated_ms(previous_state)
    page_id = current_page_id
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
        page_id,
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
                "action_id": selected_action_id,
            },
        }
    )


def send_workspace_action(
    persona: str,
    action: str = "",
    payload: dict[str, Any] | None = None,
    wait_ms: int = 900,
    expected_page_id: str = "",
    expected_state_version: int | None = None,
    action_id: str = "",
) -> str:
    """Serialize commands per page and revalidate inside the critical section."""
    with _workspace_action_lock(persona, expected_page_id):
        return _send_workspace_action_unlocked(
            persona,
            action,
            payload,
            wait_ms,
            expected_page_id,
            expected_state_version,
            action_id,
        )


def read_workspace_state(persona: str, page_id: str = "") -> str:
    state = read_workspace_state_file(persona, page_id)
    if state is None:
        return response(
            {
                "ok": True,
                "persona": safe_name(persona),
                "available": False,
                "state": None,
                "message": "No workspace app has reported state yet. Do not claim any visible value or page operation. Ask the user to open the workspace HTML through MeloMate or update the app to publish MeloMateWorkspaceState.",
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
                else "CONTROL_NOT_READY: The workspace page is stale or does not expose MeloMateWorkspaceState. Do not claim any operation or current UI state."
            ),
            "state": state,
        }
    )


def open_workspace_item(persona: str, path: str) -> str:
    target = workspace_path(persona, path)
    if not target.exists():
        raise FileNotFoundError("workspace item was not found.")

    opened_url = ""
    if target.is_file():
        persona_name = safe_name(persona)
        relative_item = target.relative_to(persona_root(persona)).as_posix()
        base_url = os.getenv(
            "MELOMATE_WORKSPACE_URL", "http://127.0.0.1:5179"
        ).rstrip("/")
        access_token = workspace_access_token(persona_name)
        opened_url = (
            f"{base_url}/workspace-files/{quote(persona_name)}/"
            f"{access_token}/{quote(relative_item, safe='/')}"
        )
        if not webbrowser.open(opened_url):
            raise RuntimeError("the workspace item could not be opened in a browser.")
    else:
        if sys.platform == "win32":
            windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
            explorer = windows_root / "explorer.exe"
            if not explorer.is_file():
                raise RuntimeError("Windows Explorer was not found.")
            subprocess.Popen([str(explorer), str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/open", str(target)])
        else:
            subprocess.Popen(["/usr/bin/xdg-open", str(target)])

    branch = target.parent if target.is_file() else target
    return response(
        {
            "ok": True,
            "persona": safe_name(persona),
            "opened": True,
            "url": "",
            "branch": branch.relative_to(WORKSPACE_ROOT).as_posix(),
        }
    )
