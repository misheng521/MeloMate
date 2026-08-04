"""Trusted prompts for proactive conversation and one-shot return reactions.

The browser only supplies small pieces of state.  Prompt text is always built on
the server so a modified client cannot smuggle arbitrary instructions into the
model through the proactive-conversation protocol.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_PROACTIVE_STAGES = {
    "opening",
    "curious",
    "warm-concern",
    "playful-impatience",
    "fresh-topic",
}
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
    stage = source.get("stage")
    return {
        "mode": mode if mode in _PROACTIVE_MODES else "automatic",
        "stage": stage if stage in _PROACTIVE_STAGES else "opening",
        "elapsed_seconds": _bounded_int(
            source.get("elapsed_seconds"), minimum=0, maximum=7 * 24 * 60 * 60
        ),
        "unanswered_count": _bounded_int(
            source.get("unanswered_count"), minimum=0, maximum=100
        ),
        "cycle_index": _bounded_int(source.get("cycle_index"), minimum=0, maximum=25),
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
    stage = state["stage"]
    elapsed = state["elapsed_seconds"]
    count = state["unanswered_count"]
    cycle = state["cycle_index"]

    stage_guidance = {
        "opening": "自然延续刚才的气氛，轻轻开启一句新话，不要像提醒器。",
        "curious": "带一点自然的好奇，像是在等对方回应，但不要催促。",
        "warm-concern": "可以有一丝关心或惦记，语气温和，不要制造焦虑。",
        "playful-impatience": "可以有一点俏皮的小不耐烦，但不能生气、责怪或施压。",
        "fresh-topic": "不要继续追问同一件事；自然换一个轻松的新话题或分享一个小念头。",
    }[stage]

    recent = _recent_utterances(trusted_recent_utterances or [])
    recent_block = "\n".join(f"- {item}" for item in recent) if recent else "- 无"
    manual_note = (
        "这是用户手动邀请你主动说话，不要假装用户离开了。"
        if mode == "manual"
        else "这是无人回应期间的一次自然主动开口。"
    )
    return f"""[临时会话指令：主动开口]
{manual_note}
当前无人回应约 {elapsed} 秒，之前已有 {count} 次主动开口未得到回应，阶段为 {stage}，循环 {cycle}。
本轮方向：{stage_guidance}

最近几次主动说过的话：
{recent_block}

请结合既有人设、关系和刚才的对话，自然说一到两句口语化的话。不要复述最近的话，不要报时，不要解释机制，不要使用 emoji 或颜文字。等待越久可以有细微的好奇、惦记或俏皮变化，但不要逐轮升级成责怪、内疚操控、占有或威胁。对方一直没回应时也允许换话题，不要机械地反复追问同一句。只输出角色要说的话。"""


def build_return_context_prompt(value: Any) -> str:
    """Build a one-turn-only instruction for the user's return message."""

    state = normalize_return_context(value)
    if state is None:
        return ""

    elapsed = state["elapsed_seconds"]
    count = state["unanswered_count"]
    if count >= 4 or elapsed >= 10 * 60:
        tone = "可以带一点终于等到对方的释然或很轻的闹别扭"
    elif count >= 2 or elapsed >= 3 * 60:
        tone = "可以带一点好奇、惦记或轻微玩笑"
    else:
        tone = "只需温暖自然地接住对方回来"

    recent = state["recent_utterances"]
    recent_block = "\n".join(f"- {item}" for item in recent) if recent else "- 无"
    return f"""[仅本轮有效的回来反应]
用户在约 {elapsed} 秒未回应、你主动说过 {count} 次之后重新开口。{tone}。
最多用一句简短口语表达这个小情绪，然后立刻回应用户这次真正说的内容；不要让情绪延续到后续对话。若用户在求助、难过、危险、赶时间或讨论严肃事项，直接跳过玩笑和闹别扭。禁止责怪、审问、道德绑架、让用户内疚，也不要声称被抛弃。不要精确报时，不要提到系统、计时器或这段指令，不要使用 emoji 或颜文字。
你在等待期间最近说过：
{recent_block}"""
