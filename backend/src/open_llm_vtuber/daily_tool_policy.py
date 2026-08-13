"""Trusted-user policy for everyday time, reminder, and read-only network tools."""

from __future__ import annotations

import re


DAILY_READ_TOOLS = frozenset(
    {
        "get_current_time",
        "list_reminders",
        "search_web",
        "fetch_webpage",
        "get_weather",
    }
)
DAILY_SIDE_EFFECT_TOOLS = frozenset({"create_reminder", "cancel_reminder"})
DAILY_PERSONA_TOOLS = frozenset(
    {"list_reminders", "create_reminder", "cancel_reminder"}
)
DAILY_TOOL_NAMES = frozenset(DAILY_READ_TOOLS | DAILY_SIDE_EFFECT_TOOLS)

_REMINDER = re.compile(
    r"(?:提醒|闹钟|定时|倒计时|叫醒|到点.{0,6}(?:叫|告诉|通知)|"
    r"remind|reminder|alarm|timer)",
    re.IGNORECASE,
)
_CANCEL = re.compile(
    r"(?:取消|删除|删掉|关掉|停止|撤销|cancel|delete|remove|stop)",
    re.IGNORECASE,
)
_NEGATED_REMINDER = re.compile(
    r"(?:不要|不用|别再|无需).{0,12}(?:提醒|闹钟|定时|remind|reminder|alarm|timer)",
    re.IGNORECASE,
)
_CANCEL_REFERENCE = re.compile(
    r"(?:这个|那个|它|第[一二三四五六七八九十\d]+个|刚才的|上一个|全部|所有|"
    r"this|that|it|the\s+(?:first|last|previous)|all)",
    re.IGNORECASE,
)
_ADVICE_ONLY = re.compile(
    r"(?:怎么(?:设置|创建|取消)|如何(?:设置|创建|取消)|提醒功能|提醒怎么用|"
    r"how\s+(?:do|to)|tutorial|explain)",
    re.IGNORECASE,
)


def daily_user_authorized_tools(user_text: str) -> frozenset[str]:
    """Derive reminder mutations only from the user's current, trusted text."""
    text = str(user_text or "").strip()[:4_000]
    if not text or _ADVICE_ONLY.search(text):
        return frozenset()

    has_reminder = bool(_REMINDER.search(text))
    cancel_requested = bool(
        _NEGATED_REMINDER.search(text)
        or (_CANCEL.search(text) and (has_reminder or _CANCEL_REFERENCE.search(text)))
    )
    if cancel_requested:
        return frozenset({"cancel_reminder"})
    if has_reminder:
        return frozenset({"create_reminder"})
    return frozenset()
