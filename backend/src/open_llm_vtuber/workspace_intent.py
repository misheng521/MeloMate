"""Small, deterministic hints for responsive workspace task acknowledgements."""

from __future__ import annotations

import re


_BUILD_VERB = re.compile(
    r"(?:做(?:一|1)?个|制作|创建|生成|开发|搭建|写(?:一|1)?个|设计|实现|新建|"
    r"修改|更新|重做|改造|增加|添加|删除|build|create|make|generate|develop|"
    r"design|implement|update|modify|revise|add|remove)",
    re.IGNORECASE,
)
_ARTIFACT = re.compile(
    r"(?:工作区|游戏|对战|棋|应用|网页|页面|网站|工具|文件|文档|项目|程序|代码|"
    r"表格|图表|计划|笔记|海报|html|workspace|game|app|website|web\s*page|"
    r"tool|file|document|project|program|code|spreadsheet|chart|plan|note|poster)",
    re.IGNORECASE,
)
_REVISION = re.compile(
    r"(?:修改|更新|重做|改造|增加|添加|删除|update|modify|revise|add|remove)",
    re.IGNORECASE,
)
_GAME = re.compile(
    r"(?:游戏|对战|棋|五子棋|象棋|围棋|扑克|game|play|chess|gomoku|go\s*game)",
    re.IGNORECASE,
)


def workspace_fast_ack_text(user_text: str) -> str:
    """Return an immediate acknowledgement only for explicit artifact work."""
    text = str(user_text or "").strip()[:2_000]
    if not text or not _BUILD_VERB.search(text) or not _ARTIFACT.search(text):
        return ""
    if _REVISION.search(text):
        return "好，我马上改。"
    if _GAME.search(text):
        return "好，我现在就准备，做好我们马上开始。"
    return "好，我现在就做，完成后马上告诉你。"
