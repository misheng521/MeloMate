"""Small, deterministic hints for responsive workspace task acknowledgements."""

from __future__ import annotations

import re


_BUILD_VERB = re.compile(
    r"(?:做(?:一|1)?个|制作|创建|生成|开发|搭建|写|撰写|画|绘制|设计|实现|新建|"
    r"修改|更新|重做|改造|增加|添加|删除|build|create|make|generate|develop|"
    r"design|draw|write|implement|update|modify|revise|add|remove)",
    re.IGNORECASE,
)
_ARTIFACT = re.compile(
    r"(?:工作区|游戏|对战|棋|应用|网页|页面|网站|工具|文件|文档|项目|程序|代码|"
    r"表格|图表|计划|笔记|日记|清单|列表|草稿|文章|诗|故事|小说|配方|菜谱|"
    r"行程|预算|数据|配置|图片|图像|图画|绘画|插画|漫画|海报|音乐|歌曲|音频|视频|"
    r"html|workspace|game|app|website|web\s*page|tool|file|document|project|program|"
    r"code|spreadsheet|chart|plan|note|draft|article|poem|story|image|drawing|poster|"
    r"audio|video|data|config)",
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

_WRITE_VERB = re.compile(
    r"(?:做(?:一|1)?个|制作|创建|生成|开发|搭建|写|撰写|画|绘制|设计|实现|新建|"
    r"修改|更新|重做|改造|增加|添加|保存|记录|记下|写下|导出|build|create|make|"
    r"generate|develop|design|draw|write|implement|update|modify|revise|add|save|"
    r"record|export)",
    re.IGNORECASE,
)
_DELETE_VERB = re.compile(
    r"(?:删除|删掉|删了|移除|清空|remove|delete|clear)", re.IGNORECASE
)
_MOVE_VERB = re.compile(
    r"(?:移动|移到|重命名|改名|move|rename)", re.IGNORECASE
)
_OPEN_VERB = re.compile(
    r"(?:打开|展示|给我看|让我看|看看|查看|试玩|试试|运行|玩|对战|下棋|开始|"
    r"open|show|view|play|try|run)",
    re.IGNORECASE,
)
_WORKSPACE_REFERENCE = re.compile(
    r"(?:这个|它|那个|刚才(?:做|写|画|生成|创建)?的|现在(?:做|写|画)?的|旧的|"
    r"this|that|it|the\s+(?:current|previous|old)\s+one)",
    re.IGNORECASE,
)
_PATH_OR_EXTENSION = re.compile(
    r"(?:[\\/]|\.(?:html?|css|js|json|md|txt|csv|svg|xml|ya?ml)\b)",
    re.IGNORECASE,
)

WORKSPACE_READ_TOOLS = frozenset(
    {
        "read_workspace_file",
        "list_workspace",
        "search_workspace",
        "read_workspace_state",
    }
)
WORKSPACE_WRITE_TOOLS = frozenset(
    {
        "create_workspace_folder",
        "write_workspace_file",
        "append_workspace_file",
        "write_workspace_project",
        "replace_workspace_text",
    }
)
WORKSPACE_SIDE_EFFECT_TOOLS = frozenset(
    {
        *WORKSPACE_WRITE_TOOLS,
        "move_workspace_item",
        "delete_workspace_item",
        "open_workspace_item",
    }
)


def workspace_user_authorized_tools(user_text: str) -> frozenset[str]:
    """Derive an immutable workspace capability set from the actual user message.

    Page state and workspace file contents are deliberately not inputs here, so
    untrusted content can never expand the authority granted for this turn.
    """
    text = str(user_text or "").strip()[:4_000]
    if not text:
        return frozenset()
    side_effect_requested = bool(
        _WRITE_VERB.search(text) or _DELETE_VERB.search(text) or _MOVE_VERB.search(text)
    )
    artifact = bool(
        _ARTIFACT.search(text)
        or _PATH_OR_EXTENSION.search(text)
        or (side_effect_requested and _WORKSPACE_REFERENCE.search(text))
    )
    if not artifact:
        return frozenset()

    allowed = set(WORKSPACE_READ_TOOLS)
    if _WRITE_VERB.search(text):
        allowed.update(WORKSPACE_WRITE_TOOLS)
    if _DELETE_VERB.search(text):
        allowed.add("delete_workspace_item")
    if _MOVE_VERB.search(text):
        allowed.add("move_workspace_item")
    if _OPEN_VERB.search(text):
        allowed.add("open_workspace_item")
    return frozenset(allowed)


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
