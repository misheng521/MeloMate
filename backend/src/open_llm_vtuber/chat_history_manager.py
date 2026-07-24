import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, List, TypedDict, Optional

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHAT_HISTORY_DIR = str(PROJECT_ROOT / "characters" / "memory")
SHORT_MEMORY_FILE = "short_memory.json"
CORE_MEMORY_FILE = "core_memory.json"
SINGLE_HISTORY_UID = "short_memory"
MAX_MEMORY_ROUNDS = 20


class HistoryMessage(TypedDict):
    role: Literal["human", "ai"]
    timestamp: str
    content: str
    name: Optional[str]


def _is_safe_filename(filename: str) -> bool:
    if not filename or len(filename) > 255:
        return False
    pattern = re.compile(r"^[\w\-_\u0020-\u007E\u00A0-\uFFFF]+$")
    return bool(pattern.match(filename))


def _sanitize_path_component(component: str) -> str:
    sanitized = os.path.basename(component.strip())
    if not _is_safe_filename(sanitized):
        raise ValueError(f"Invalid characters in path component: {component}")
    return sanitized


def _ensure_conf_dir(conf_uid: str) -> str:
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")

    safe_conf_uid = _sanitize_path_component(conf_uid)
    base_dir = os.path.normpath(os.path.join(CHAT_HISTORY_DIR, safe_conf_uid))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_short_memory_path(conf_uid: str) -> str:
    return os.path.join(_ensure_conf_dir(conf_uid), SHORT_MEMORY_FILE)


def _get_core_memory_path(conf_uid: str) -> str:
    return os.path.join(_ensure_conf_dir(conf_uid), CORE_MEMORY_FILE)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read memory file {path}: {e}")
        return default


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ensure_memory_files(conf_uid: str) -> None:
    short_path = _get_short_memory_path(conf_uid)
    core_path = _get_core_memory_path(conf_uid)

    if not os.path.exists(short_path):
        _write_json(short_path, [])

    if not os.path.exists(core_path):
        _write_json(
            core_path,
            {
                "timestamp": _now(),
                "nickname": "",
                "likes": [],
                "dislikes": [],
                "preferences": [],
                "facts": [],
            },
        )


def _normalize_item(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" ，。！？；;,.!?\n\t"))


def _split_items(text: str) -> list[str]:
    return [
        _normalize_item(part)
        for part in re.split(r"[、,，;；。.!！?\n]+", text)
        if _normalize_item(part)
    ]


def _add_unique(items: list[str], values: list[str], max_items: int = 30) -> list[str]:
    existing = set(items)
    for value in values:
        value = _normalize_item(value)
        if value and value not in existing:
            items.append(value)
            existing.add(value)
    return items[-max_items:]


def _extract_core_updates(message: str) -> dict:
    text = message.strip()
    updates = {
        "nickname": "",
        "likes": [],
        "dislikes": [],
        "preferences": [],
        "facts": [],
    }

    nickname_patterns = [
        r"(?:以后|之后|往后)?(?:叫我|喊我|称呼我)(?:为|叫)?[：: ]*([^，。！？；;,.!?\n]{1,20})",
        r"(?:我的名字是|我叫)[：: ]*([^，。！？；;,.!?\n]{1,20})",
    ]
    for pattern in nickname_patterns:
        match = re.search(pattern, text)
        if match:
            updates["nickname"] = _normalize_item(match.group(1))
            break

    for match in re.finditer(r"我(?:很|最|特别|挺|也)?喜欢[：: ]*([^。！？；;\n]+)", text):
        updates["likes"].extend(_split_items(match.group(1)))

    for match in re.finditer(r"我(?:很|最|特别|挺)?(?:不喜欢|讨厌|不爱)[：: ]*([^。！？；;\n]+)", text):
        updates["dislikes"].extend(_split_items(match.group(1)))

    preference_patterns = [
        r"(?:以后|之后|往后)?(?:不要|别)([^。！？；;\n]{1,60})",
        r"(?:以后|之后|往后)?(?:要|请|希望你|你要)([^。！？；;\n]{1,60})",
        r"(?:记住|记得)[：: ]*([^。！？；;\n]{1,80})",
    ]
    for pattern in preference_patterns:
        for match in re.finditer(pattern, text):
            value = _normalize_item(match.group(1))
            if value:
                updates["preferences"].append(value)

    fact_patterns = [
        r"我的(?:生日|生曰)是[：: ]*([^。！？；;\n]{1,40})",
        r"我(?:是|今年|现在)[：: ]*([^。！？；;\n]{1,60})",
        r"我住在[：: ]*([^。！？；;\n]{1,60})",
    ]
    for pattern in fact_patterns:
        for match in re.finditer(pattern, text):
            value = _normalize_item(match.group(0))
            if value:
                updates["facts"].append(value)

    return updates


def _update_core_memory_from_user_message(conf_uid: str, message: str) -> None:
    updates = _extract_core_updates(message)
    if not any(updates.values()):
        return

    _ensure_memory_files(conf_uid)
    core_path = _get_core_memory_path(conf_uid)
    core = _read_json(core_path, {})
    if not isinstance(core, dict):
        core = {}

    core.setdefault("timestamp", _now())
    core.setdefault("nickname", "")
    core.setdefault("likes", [])
    core.setdefault("dislikes", [])
    core.setdefault("preferences", [])
    core.setdefault("facts", [])

    if updates["nickname"]:
        core["nickname"] = updates["nickname"]

    for key in ("likes", "dislikes", "preferences", "facts"):
        if not isinstance(core.get(key), list):
            core[key] = []
        core[key] = _add_unique(core[key], updates[key])

    core["timestamp"] = _now()
    _write_json(core_path, core)


def get_core_memory(conf_uid: str) -> dict:
    _ensure_memory_files(conf_uid)
    core = _read_json(_get_core_memory_path(conf_uid), {})
    return core if isinstance(core, dict) else {}


def get_core_memory_prompt(conf_uid: str) -> str:
    core = get_core_memory(conf_uid)
    lines = []

    nickname = core.get("nickname")
    if nickname:
        lines.append(f"称呼用户：{nickname}")

    sections = [
        ("用户喜欢", core.get("likes")),
        ("用户不喜欢", core.get("dislikes")),
        ("用户希望", core.get("preferences")),
        ("用户事实", core.get("facts")),
    ]
    for title, values in sections:
        if isinstance(values, list) and values:
            lines.append(f"{title}：" + "；".join(values))

    if not lines:
        return ""

    return "# 核心记忆\n" + "\n".join(lines)


def create_new_history(conf_uid: str) -> str:
    if not conf_uid:
        logger.warning("No conf_uid provided")
        return ""
    _ensure_memory_files(conf_uid)
    return SINGLE_HISTORY_UID


def store_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    content: str,
    name: str | None = None,
):
    if not conf_uid or not content:
        return
    if role == "system":
        return

    _ensure_memory_files(conf_uid)
    short_path = _get_short_memory_path(conf_uid)
    short_memory = _read_json(short_path, [])
    if not isinstance(short_memory, list):
        short_memory = []

    timestamp = _now()
    if role == "human":
        short_memory.append(
            {
                "timestamp": timestamp,
                "user": content,
                "bot": "",
            }
        )
        _update_core_memory_from_user_message(conf_uid, content)
    elif role == "ai":
        if short_memory and short_memory[-1].get("user") and not short_memory[-1].get("bot"):
            short_memory[-1]["bot"] = content
            short_memory[-1]["timestamp"] = timestamp
        else:
            short_memory.append(
                {
                    "timestamp": timestamp,
                    "user": "",
                    "bot": content,
                }
            )

    short_memory = short_memory[-MAX_MEMORY_ROUNDS:]
    _write_json(short_path, short_memory)


def get_history(conf_uid: str, history_uid: str = SINGLE_HISTORY_UID) -> List[HistoryMessage]:
    _ensure_memory_files(conf_uid)
    short_memory = _read_json(_get_short_memory_path(conf_uid), [])
    if not isinstance(short_memory, list):
        return []

    messages: list[HistoryMessage] = []
    for item in short_memory[-MAX_MEMORY_ROUNDS:]:
        timestamp = item.get("timestamp", "")
        user_text = item.get("user", "")
        bot_text = item.get("bot", "")
        if user_text:
            messages.append(
                {
                    "role": "human",
                    "timestamp": timestamp,
                    "content": user_text,
                    "name": None,
                }
            )
        if bot_text:
            messages.append(
                {
                    "role": "ai",
                    "timestamp": timestamp,
                    "content": bot_text,
                    "name": None,
                }
            )
    return messages


def get_history_list(conf_uid: str) -> List[dict]:
    _ensure_memory_files(conf_uid)
    messages = get_history(conf_uid, SINGLE_HISTORY_UID)
    latest_message = messages[-1] if messages else None
    return [
        {
            "uid": SINGLE_HISTORY_UID,
            "latest_message": latest_message,
            "timestamp": latest_message["timestamp"] if latest_message else "",
        }
    ]


def delete_history(conf_uid: str, history_uid: str) -> bool:
    _ensure_memory_files(conf_uid)
    _write_json(_get_short_memory_path(conf_uid), [])
    return True


def modify_latest_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    _ensure_memory_files(conf_uid)
    short_path = _get_short_memory_path(conf_uid)
    short_memory = _read_json(short_path, [])
    if not isinstance(short_memory, list) or not short_memory:
        return False

    target_key = "user" if role == "human" else "bot"
    for item in reversed(short_memory):
        if item.get(target_key):
            item[target_key] = new_content
            item["timestamp"] = _now()
            _write_json(short_path, short_memory[-MAX_MEMORY_ROUNDS:])
            return True
    return False


def rename_history_file(conf_uid: str, old_history_uid: str, new_history_uid: str) -> bool:
    return True


def get_metadata(conf_uid: str, history_uid: str) -> dict:
    return {}


def update_metadate(conf_uid: str, history_uid: str, metadata: dict) -> bool:
    return True
