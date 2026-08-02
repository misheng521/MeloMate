"""Crash-safe storage for the single short conversation history of each persona.

The UI intentionally exposes one rolling history per persona.  The public functions in
this module keep that contract while making every read/modify/write operation atomic.
Legacy ``[{user, bot, timestamp}]`` files are migrated on the next write.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, TypedDict
from uuid import uuid4

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHAT_HISTORY_DIR = PROJECT_ROOT / "characters" / "memory"
SHORT_MEMORY_FILE = "short_memory.json"
CORE_MEMORY_FILE = "core_memory.json"
SINGLE_HISTORY_UID = "short_memory"
MAX_MEMORY_ROUNDS = 20
MAX_MEMORY_MESSAGES = MAX_MEMORY_ROUNDS * 2
CORE_MEMORY_REVIEW_ROUNDS = 20
MAX_MESSAGE_CHARS = 100_000
MAX_METADATA_BYTES = 16_384
FILE_LOCK_TIMEOUT_SECONDS = 10.0

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_locks_guard = threading.Lock()
_conf_locks: dict[str, threading.RLock] = {}


class HistoryStorageError(RuntimeError):
    """Raised when neither a memory file nor its backup can be decoded."""


class HistoryMessage(TypedDict):
    role: Literal["human", "ai"]
    timestamp: str
    content: str
    name: Optional[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _safe_conf_uid(conf_uid: str) -> str:
    if not isinstance(conf_uid, str):
        raise ValueError("conf_uid must be a string")
    value = unicodedata.normalize("NFKC", conf_uid.strip())
    if not value or len(value) > 128:
        raise ValueError("conf_uid must contain between 1 and 128 characters")
    if value in {".", ".."} or value.endswith((" ", ".")):
        raise ValueError("conf_uid is not a valid directory name")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
        raise ValueError("conf_uid contains invalid path characters")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("conf_uid uses a reserved directory name")
    return value


def _validate_history_uid(history_uid: str) -> None:
    if history_uid != SINGLE_HISTORY_UID:
        raise ValueError(f"Unknown history_uid: {history_uid!r}")


def _conf_lock(conf_uid: str) -> tuple[str, threading.RLock]:
    safe_uid = _safe_conf_uid(conf_uid)
    with _locks_guard:
        lock = _conf_locks.setdefault(safe_uid, threading.RLock())
    return safe_uid, lock


@contextmanager
def _cross_process_lock(safe_uid: str):
    """Serialize transactions if somebody starts a second backend process."""
    lock_path = _path(safe_uid, ".history.lock")
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + FILE_LOCK_TIMEOUT_SECONDS
        acquired = False
        while not acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise HistoryStorageError("Timed out waiting for the history storage lock") from error
                time.sleep(0.025)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_conf(conf_uid: str):
    safe_uid, thread_lock = _conf_lock(conf_uid)
    with thread_lock:
        with _cross_process_lock(safe_uid):
            yield safe_uid


def _conf_dir(safe_uid: str) -> Path:
    directory = (CHAT_HISTORY_DIR / safe_uid).resolve()
    memory_root = CHAT_HISTORY_DIR.resolve()
    if directory.parent != memory_root:
        raise ValueError("conf_uid escapes the memory directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _path(safe_uid: str, filename: str) -> Path:
    return _conf_dir(safe_uid) / filename


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary history file: {}", temporary.name)


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_json_unlocked(path: Path, default: object) -> object:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as primary_error:
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            recovered = json.loads(backup.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as backup_error:
            raise HistoryStorageError(
                f"Memory file {path.name} is unreadable and no valid backup exists"
            ) from backup_error
        logger.error(
            "Recovered damaged memory file {} from its last valid backup: {}",
            path.name,
            type(primary_error).__name__,
        )
        _replace_bytes(path, _json_bytes(recovered))
        return recovered


def _write_json_unlocked(path: Path, data: object) -> None:
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HistoryStorageError(
                f"Refusing to overwrite unreadable memory file {path.name}"
            ) from error
        _replace_bytes(path.with_suffix(path.suffix + ".bak"), _json_bytes(current))
    _replace_bytes(path, _json_bytes(data))


def _default_history() -> dict:
    return {"version": 2, "messages": [], "metadata": {}}


def _default_core_memory() -> dict:
    return {
        "timestamp": _now(),
        "nickname": "",
        "likes": [],
        "dislikes": [],
        "preferences": [],
        "facts": [],
        "turns_since_core_review": 0,
        "last_core_review_at": "",
    }


def _validate_message(message: object) -> dict:
    if not isinstance(message, dict):
        raise HistoryStorageError("History contains a non-object message")
    role = message.get("role")
    content = message.get("content")
    timestamp = message.get("timestamp", "")
    name = message.get("name")
    if role not in {"human", "ai"} or not isinstance(content, str):
        raise HistoryStorageError("History contains an invalid role or content")
    if not isinstance(timestamp, str) or len(timestamp) > 128:
        raise HistoryStorageError("History contains an invalid timestamp")
    if name is not None and (not isinstance(name, str) or len(name) > 200):
        raise HistoryStorageError("History contains an invalid speaker name")
    if len(content) > MAX_MESSAGE_CHARS:
        raise HistoryStorageError("History contains an oversized message")
    return {
        "id": str(message.get("id") or uuid4().hex),
        "role": role,
        "timestamp": timestamp,
        "content": content,
        "name": name,
    }


def _validate_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise HistoryStorageError("History metadata must be a JSON object with string keys")
    try:
        encoded = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HistoryStorageError("History metadata contains a non-JSON value") from error
    if len(encoded) > MAX_METADATA_BYTES:
        raise HistoryStorageError("History metadata is too large")
    return copy.deepcopy(metadata)


def _load_history_unlocked(safe_uid: str) -> dict:
    raw = _read_json_unlocked(_path(safe_uid, SHORT_MEMORY_FILE), _default_history())
    messages: list[dict] = []
    metadata: dict = {}
    if isinstance(raw, list):
        for legacy in raw:
            if not isinstance(legacy, dict):
                raise HistoryStorageError("Legacy history contains a non-object entry")
            timestamp = legacy.get("timestamp", "")
            if legacy.get("user"):
                messages.append(
                    _validate_message(
                        {"role": "human", "content": legacy["user"], "timestamp": timestamp}
                    )
                )
            if legacy.get("bot"):
                messages.append(
                    _validate_message(
                        {"role": "ai", "content": legacy["bot"], "timestamp": timestamp}
                    )
                )
    elif isinstance(raw, dict) and raw.get("version") == 2 and isinstance(raw.get("messages"), list):
        messages = [_validate_message(message) for message in raw["messages"]]
        metadata = _validate_metadata(raw.get("metadata", {}))
    else:
        raise HistoryStorageError("Unsupported history file format")
    return {
        "version": 2,
        "messages": messages[-MAX_MEMORY_MESSAGES:],
        "metadata": metadata,
    }


def _load_core_unlocked(safe_uid: str) -> dict:
    raw = _read_json_unlocked(_path(safe_uid, CORE_MEMORY_FILE), _default_core_memory())
    if not isinstance(raw, dict):
        raise HistoryStorageError("Core memory must be a JSON object")
    core = _default_core_memory()
    for key in ("nickname", "last_core_review_at", "timestamp"):
        if isinstance(raw.get(key), str):
            core[key] = raw[key]
    for key in ("likes", "dislikes", "preferences", "facts"):
        values = raw.get(key)
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            core[key] = values[-30:]
    turns = raw.get("turns_since_core_review", 0)
    if isinstance(turns, int) and 0 <= turns <= CORE_MEMORY_REVIEW_ROUNDS:
        core["turns_since_core_review"] = turns
    return core


def _ensure_memory_files_unlocked(safe_uid: str) -> None:
    short_path = _path(safe_uid, SHORT_MEMORY_FILE)
    core_path = _path(safe_uid, CORE_MEMORY_FILE)
    if not short_path.exists():
        _write_json_unlocked(short_path, _default_history())
    else:
        _load_history_unlocked(safe_uid)
    if not core_path.exists():
        _write_json_unlocked(core_path, _default_core_memory())
    else:
        _load_core_unlocked(safe_uid)


def _normalize_item(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip(" ，。！？；;,.!?\n\t"))


def _split_items(text: str) -> list[str]:
    return [
        item
        for item in (_normalize_item(part) for part in re.split(r"[、，,。；;！？!?\n]+", text))
        if item
    ]


def _is_stable_memory_value(value: object) -> bool:
    clean = _normalize_item(value)
    if len(clean) <= 1:
        return False
    unstable = (
        "吗",
        "么",
        "?",
        "？",
        "什么",
        "为什么",
        "怎么",
        "能不能",
        "可不可以",
        "想玩",
        "想看",
        "看看",
        "试试",
        "现在",
        "这次",
        "刚才",
    )
    return not any(marker in clean for marker in unstable)


def _extract_core_updates(message: str) -> dict:
    text = _normalize_item(message)
    updates: dict[str, object] = {
        "nickname": "",
        "likes": [],
        "dislikes": [],
        "preferences": [],
        "facts": [],
    }
    if not text:
        return updates

    for pattern in (
        r"(?:以后|之后|往后)(?:叫我|喊我|称呼我)(?:为|叫)?[:： ]*([^，。！？；;,.!?\n]{1,20})",
        r"(?:我的名字是|我叫)[:： ]*([^，。！？；;,.!?\n]{1,20})",
    ):
        match = re.search(pattern, text)
        if match:
            updates["nickname"] = _normalize_item(match.group(1))
            break

    for target, pattern in (
        ("likes", r"我(?:很|最|特别|非常)?喜欢[:： ]*([^。！？；;\n]{1,80})"),
        ("dislikes", r"我(?:很|最|特别|非常)?(?:不喜欢|讨厌|不爱)[:： ]*([^。！？；;\n]{1,80})"),
    ):
        for match in re.finditer(pattern, text):
            values = [value for value in _split_items(match.group(1)) if _is_stable_memory_value(value)]
            updates[target].extend(values)  # type: ignore[union-attr]

    for pattern in (
        r"(?:以后|之后|往后)(?:不要|别|不许)[:： ]*([^。！？；;\n]{1,80})",
        r"(?:以后|之后|往后)(?:要|希望你|你要|请你)[:： ]*([^。！？；;\n]{1,80})",
        r"(?:请记住|记住|记得)[:： ]*([^。！？；;\n]{1,100})",
    ):
        for match in re.finditer(pattern, text):
            value = _normalize_item(match.group(1))
            if _is_stable_memory_value(value):
                updates["preferences"].append(value)  # type: ignore[union-attr]

    for pattern in (
        r"我的(?:生日|生辰)是[:： ]*([^。！？；;\n]{1,40})",
        r"我住在[:： ]*([^。！？；;\n]{1,60})",
        r"我是(?:一个|一名)[:： ]*([^。！？；;\n]{1,60})",
    ):
        for match in re.finditer(pattern, text):
            value = _normalize_item(match.group(0))
            if _is_stable_memory_value(value):
                updates["facts"].append(value)  # type: ignore[union-attr]
    return updates


def _add_unique(items: list[str], values: list[str], max_items: int = 30) -> list[str]:
    clean_items = [_normalize_item(item) for item in items if _normalize_item(item)]
    existing = set(clean_items)
    for value in values:
        clean = _normalize_item(value)
        if clean and clean not in existing:
            clean_items.append(clean)
            existing.add(clean)
    return clean_items[-max_items:]


def _merge_core_updates(core: dict, updates: dict) -> dict:
    if updates.get("nickname"):
        core["nickname"] = updates["nickname"]
    for key in ("likes", "dislikes", "preferences", "facts"):
        core[key] = _add_unique(core.get(key, []), updates.get(key, []))
    core["timestamp"] = _now()
    return core


def _review_core_memory_unlocked(safe_uid: str, messages: list[dict]) -> None:
    core = _load_core_unlocked(safe_uid)
    turns = int(core.get("turns_since_core_review") or 0) + 1
    if turns < CORE_MEMORY_REVIEW_ROUNDS:
        core["turns_since_core_review"] = turns
        _write_json_unlocked(_path(safe_uid, CORE_MEMORY_FILE), core)
        return

    combined: dict[str, object] = {
        "nickname": "",
        "likes": [],
        "dislikes": [],
        "preferences": [],
        "facts": [],
    }
    human_messages = [message for message in messages if message["role"] == "human"]
    for message in human_messages[-CORE_MEMORY_REVIEW_ROUNDS:]:
        updates = _extract_core_updates(message["content"])
        if updates.get("nickname"):
            combined["nickname"] = updates["nickname"]
        for key in ("likes", "dislikes", "preferences", "facts"):
            combined[key].extend(updates.get(key, []))  # type: ignore[union-attr]
    core = _merge_core_updates(core, combined)
    core["turns_since_core_review"] = 0
    core["last_core_review_at"] = _now()
    _write_json_unlocked(_path(safe_uid, CORE_MEMORY_FILE), core)


def create_new_history(conf_uid: str) -> str:
    """Idempotently ensure the one rolling history exists and return its UID."""
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
    return SINGLE_HISTORY_UID


def store_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    content: str,
    name: str | None = None,
) -> None:
    if role == "system":
        return
    _validate_history_uid(history_uid)
    if role not in {"human", "ai"}:
        raise ValueError(f"Unsupported history role: {role!r}")
    if not isinstance(content, str) or not content.strip():
        return
    if len(content) > MAX_MESSAGE_CHARS:
        raise ValueError(f"History message exceeds {MAX_MESSAGE_CHARS} characters")
    if name is not None and (not isinstance(name, str) or len(name) > 200):
        raise ValueError("History speaker name is invalid")

    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        state = _load_history_unlocked(safe_uid)
        state["messages"].append(
            {
                "id": uuid4().hex,
                "role": role,
                "timestamp": _now(),
                "content": content,
                "name": name,
            }
        )
        state["messages"] = state["messages"][-MAX_MEMORY_MESSAGES:]
        if role == "human":
            _review_core_memory_unlocked(safe_uid, state["messages"])
        _write_json_unlocked(_path(safe_uid, SHORT_MEMORY_FILE), state)


def get_history(
    conf_uid: str, history_uid: str = SINGLE_HISTORY_UID
) -> list[HistoryMessage]:
    _validate_history_uid(history_uid)
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        state = _load_history_unlocked(safe_uid)
        return [
            {
                "role": message["role"],
                "timestamp": message["timestamp"],
                "content": message["content"],
                "name": message["name"],
            }
            for message in state["messages"]
        ]


def get_history_list(conf_uid: str) -> list[dict]:
    messages = get_history(conf_uid, SINGLE_HISTORY_UID)
    latest = messages[-1] if messages else None
    return [
        {
            "uid": SINGLE_HISTORY_UID,
            "latest_message": latest,
            "timestamp": latest["timestamp"] if latest else "",
        }
    ]


def delete_history(conf_uid: str, history_uid: str) -> bool:
    try:
        _validate_history_uid(history_uid)
    except ValueError:
        return False
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        _write_json_unlocked(_path(safe_uid, SHORT_MEMORY_FILE), _default_history())
    return True


def modify_latest_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    _validate_history_uid(history_uid)
    if role not in {"human", "ai"} or not isinstance(new_content, str) or not new_content.strip():
        return False
    if len(new_content) > MAX_MESSAGE_CHARS:
        raise ValueError(f"History message exceeds {MAX_MESSAGE_CHARS} characters")
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        state = _load_history_unlocked(safe_uid)
        for message in reversed(state["messages"]):
            if message["role"] == role:
                message["content"] = new_content
                message["timestamp"] = _now()
                _write_json_unlocked(_path(safe_uid, SHORT_MEMORY_FILE), state)
                return True
    return False


def get_metadata(conf_uid: str, history_uid: str) -> dict:
    _validate_history_uid(history_uid)
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        state = _load_history_unlocked(safe_uid)
        return copy.deepcopy(state["metadata"])


def update_metadata(conf_uid: str, history_uid: str, metadata: dict) -> bool:
    _validate_history_uid(history_uid)
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise ValueError("History metadata must be a JSON object with string keys")
    try:
        validated_metadata = _validate_metadata(metadata)
    except HistoryStorageError as error:
        raise ValueError("History metadata must contain JSON-serializable values") from error

    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        state = _load_history_unlocked(safe_uid)
        state["metadata"].update(validated_metadata)
        if len(json.dumps(state["metadata"], ensure_ascii=False).encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError("Merged history metadata is too large")
        _write_json_unlocked(_path(safe_uid, SHORT_MEMORY_FILE), state)
    return True


def get_core_memory(conf_uid: str) -> dict:
    with _locked_conf(conf_uid) as safe_uid:
        _ensure_memory_files_unlocked(safe_uid)
        return copy.deepcopy(_load_core_unlocked(safe_uid))


def get_core_memory_prompt(conf_uid: str) -> str:
    core = get_core_memory(conf_uid)
    lines: list[str] = []
    nickname = _normalize_item(core.get("nickname"))
    if nickname:
        lines.append(f"称呼用户：{nickname}")
    for title, key in (
        ("用户喜欢", "likes"),
        ("用户不喜欢", "dislikes"),
        ("用户希望", "preferences"),
        ("用户事实", "facts"),
    ):
        clean = [
            _normalize_item(value)
            for value in core.get(key, [])
            if _is_stable_memory_value(value)
        ]
        if clean:
            lines.append(f"{title}：" + "，".join(clean))
    return "# 核心记忆\n" + "\n".join(lines) if lines else ""
