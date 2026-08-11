"""Trusted prompts for proactive conversation and one-shot return reactions.

The browser only supplies small pieces of state.  Prompt text is always built on
the server so a modified client cannot smuggle arbitrary instructions into the
model through the proactive-conversation protocol.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_PROACTIVE_MODES = {"automatic", "manual"}
_WHITESPACE = re.compile(r"\s+")


def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, parsed))


def _clean_text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()[:limit]


def _recent_utterances(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:5]:
        cleaned = _clean_text(item, limit=180)
        if cleaned:
            result.append(cleaned)
    return result


def normalize_proactive_request(value: Any) -> dict[str, Any]:
    """Return a strict, size-limited representation of a proactive request."""

    source: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    mode = source.get("mode")
    return {
        "mode": mode if mode in _PROACTIVE_MODES else "automatic",
        "elapsed_seconds": _bounded_int(
            source.get("elapsed_seconds"), minimum=0, maximum=7 * 24 * 60 * 60
        ),
        "unanswered_count": _bounded_int(
            source.get("unanswered_count"), minimum=0, maximum=100
        ),
        "recent_utterances": _recent_utterances(source.get("recent_utterances")),
    }


def normalize_return_context(value: Any) -> dict[str, Any] | None:
    """Validate the ephemeral state attached to the user's first return turn."""

    if not isinstance(value, Mapping):
        return None
    unanswered_count = _bounded_int(
        value.get("unanswered_count"), minimum=0, maximum=100
    )
    if unanswered_count <= 0:
        return None
    return {
        "elapsed_seconds": _bounded_int(
            value.get("elapsed_seconds"), minimum=0, maximum=7 * 24 * 60 * 60
        ),
        "unanswered_count": unanswered_count,
        "last_proactive_seconds_ago": _bounded_int(
            value.get("last_proactive_seconds_ago"),
            minimum=0,
            maximum=7 * 24 * 60 * 60,
        ),
        # This field is filled from the per-client server session later.  Never
        # accept model-output text echoed by the browser as a hidden prompt.
        "recent_utterances": [],
    }


def sanitize_user_metadata(
    value: Any, *, preserve_other: bool = True
) -> dict[str, Any]:
    """Keep ordinary metadata while replacing proactive state with validated data."""

    if not isinstance(value, Mapping):
        return {}
    result = (
        {
            str(key): item
            for key, item in value.items()
            if key != "proactive_return"
        }
        if preserve_other
        else {}
    )
    proactive_return = normalize_return_context(value.get("proactive_return"))
    if proactive_return is not None:
        result["proactive_return"] = proactive_return
    return result


def build_proactive_prompt(
    value: Any, *, trusted_recent_utterances: list[str] | None = None
) -> str:
    """Build the only model instruction accepted for proactive speech."""

    state = normalize_proactive_request(value)
    mode = state["mode"]
    elapsed = state["elapsed_seconds"]
    count = state["unanswered_count"]

    recent = _recent_utterances(trusted_recent_utterances or [])
    recent_block = "\n".join(f"- {item}" for item in recent) if recent else "- 无"
    trigger = "用户手动给了你一次主动开口的机会" if mode == "manual" else "系统给了你一次主动开口的机会"
    return f"""[可信运行时事实：这不是用户说的话]
{trigger}。距离用户上次开口约 {elapsed} 秒；这段时间里你已经主动说过 {count} 次，但用户尚未回应。
这些事实本身不规定任何情绪、语气或话题，也不表示用户一定离开、忽视或拒绝了你。

最近几次主动说过的话：
{recent_block}

结合你自己的人设、真实记忆和最近对话，自行判断此刻会有什么感受、是否延续原话题，以及最自然会说什么。不要解释触发机制，也不要把经过时间直接当成某种情绪的理由。只输出角色真正会说的话。"""


def build_return_context_prompt(
    value: Any, *, trusted_recent_utterances: list[str] | None = None
) -> str:
    """Build a one-turn-only instruction for the user's return message."""

    state = normalize_return_context(value)
    if state is None:
        return ""

    elapsed = state["elapsed_seconds"]
    count = state["unanswered_count"]
    recent = _recent_utterances(trusted_recent_utterances or [])
    recent_block = "\n".join(f"- {item}" for item in recent) if recent else "- 无"
    return f"""[可信运行时事实：这不是用户说的话，仅本轮有效]
距离用户上次开口约 {elapsed} 秒；期间你主动说过 {count} 次但没有收到回应，现在用户重新开口了。
这些事实本身不规定任何情绪，也不表示用户一定离开、忽视或拒绝了你。结合你自己的人设、真实记忆、最近对话和用户这次真正说的内容，自行判断此刻的感受与回应方式。不要解释触发机制，也不要把经过时间直接当成某种情绪的理由。
你在等待期间最近说过：
{recent_block}"""
