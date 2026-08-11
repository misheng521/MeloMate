"""Small, deterministic hints for responsive workspace task acknowledgements."""

from __future__ import annotations

import re


_ARTIFACT = re.compile(
    r"(?:工作区|游戏|对战|棋|应用|网页|页面|网站|工具|文件|文档|项目|程序|代码|"
    r"表格|图表|计划|笔记|日记|清单|列表|草稿|文章|诗|故事|小说|配方|菜谱|"
    r"标题|按钮|文字|文本|内容|颜色|配色|布局|样式|功能|表单|输入框|菜单|卡片|图标|字段|段落|章节|"
    r"行程|预算|数据|配置|图片|图像|图画|绘画|插画|漫画|海报|音乐|歌曲|音频|视频|"
    r"html|workspace|game|app|website|web\s*page|tool|file|document|project|program|"
    r"code|spreadsheet|chart|plan|note|draft|article|poem|story|image|drawing|poster|"
    r"title|button|text|content|colou?r|layout|style|feature|form|input|menu|card|icon|field|section|"
    r"audio|video|data|config)",
    re.IGNORECASE,
)
_WRITE_VERB = re.compile(
    r"(?:做(?:一|1)?个|制作|创建|生成|开发|搭建|写|撰写|画|绘制|设计|实现|新建|"
    r"修改|更新|重做|改造|增加|添加|保存|记录|记下|写下|导出|编辑|整理|优化|完善|"
    r"修复|修正|校对|调整|调大|调小|补充|补全|补上|重构|重排|转换|翻译|合并|拆分|格式化|"
    r"润色|美化|排版|放大|缩小|换成|换为|换个|变成|变为|去掉|隐藏|显示|启用|禁用|"
    r"改成|改为|改一下|改下|改个|改点|build|create|make|"
    r"generate|develop|design|draw|write|implement|update|modify|revise|edit|fix|"
    r"refactor|format|optimize|convert|translate|merge|split|add|save|record|export)",
    re.IGNORECASE,
)
_DELETE_VERB = re.compile(
    r"(?:删除|删掉|删了|移除|清空|remove|delete|clear)", re.IGNORECASE
)
_MOVE_VERB = re.compile(
    r"(?:移动|移到|放到|挪到|归档到|重命名|改名|move|rename|archive\s+to)", re.IGNORECASE
)
_RESTORE_VERB = re.compile(
    r"(?:恢复|还原|找回|撤销删除|取消删除|restore|recover|undo\s+(?:the\s+)?delet)",
    re.IGNORECASE,
)
_NATURAL_MUTATION = re.compile(
    r"(?:把|将|让|给).{0,40}(?:润色|美化|排版|改|换|变|调|放大|缩小|补|去掉|隐藏|显示|"
    r"启用|禁用|更(?:大|小|宽|窄|高|低|亮|暗|快|慢|粗|细|好看|自然)|"
    r"(?:大|小|宽|窄|高|低|亮|暗|快|慢|粗|细)(?:一|点|些)|"
    r"polish|beautify|reformat|change|adjust|resize|make\s+(?:it\s+)?(?:larger|smaller|better))",
    re.IGNORECASE,
)
_NEGATED_WRITE = re.compile(
    r"(?:不要|别|不准|禁止|不能|不可|请勿|无需|不用).{0,12}(?:修改|改动|改写|写入|创建|生成|"
    r"编辑|删除内容|move|write|edit|modify|create|generate)",
    re.IGNORECASE,
)
_NEGATED_DELETE = re.compile(
    r"(?:不要|别|不准|禁止|不能|不可|请勿).{0,12}(?:删除|删掉|移除|清空|remove|delete|clear)",
    re.IGNORECASE,
)
_NEGATED_MOVE = re.compile(
    r"(?:不要|别|不准|禁止|不能|不可|请勿).{0,12}(?:移动|移到|放到|挪到|重命名|改名|move|rename)",
    re.IGNORECASE,
)
_OPEN_VERB = re.compile(
    r"(?:打开|展示|给我看|让我看|看看|查看|试玩|试试|运行|玩|对战|下棋|开始|"
    r"open|show|view|play|try|run)",
    re.IGNORECASE,
)
_PAGE_ACTION_VERB = re.compile(
    r"(?:你(?:来|先|下|走|操作|选择|决定|点|点击)|轮到你|该你|帮我(?:点|选|操作)|"
    r"替我(?:点|选|操作)|继续(?:玩|下|操作)|下一步|落子|出牌|移动|确认|提交|"
    r"(?:我们|咱们|咱俩)(?:两个一起|一起|俩|两个|来)?(?:玩|下棋|对战|操作|编辑|完成)|"
    r"陪我(?:玩|下棋|对战|操作)|"
    r"your\s+turn|you\s+(?:go|move|choose|play)|make\s+a\s+move|click|select|"
    r"(?:let(?:'s|\s+us)|we)\s+(?:play|operate|edit|work)|operate|confirm|submit)",
    re.IGNORECASE,
)
_ADVICE_ONLY = re.compile(
    r"(?:怎么|如何|教程|方法|思路|建议|解释|介绍|原理|为什么|是什么|能不能|能否|"
    r"是否(?:可以|能)|(?:能|会|可以).{0,16}(?:吗|么)|"
    r"how\s+(?:do|can|should|to)|tutorial|explain|what\s+is|why\b)",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r"(?:继续|接着|然后呢|再来|再改|就这样|按这个|用这个|把它|给它|上一个|刚才的|"
    r"continue|keep\s+going|then|again|that\s+one|the\s+previous)",
    re.IGNORECASE,
)
_WORKSPACE_TOPIC = re.compile(
    r"(?:工作区|页面|界面|棋盘|局面|文件|目录|项目|应用|游戏|网页|网站|代码|"
    r"workspace|page|screen|board|file|folder|project|app|game|website|code|current\s+state)",
    re.IGNORECASE,
)
_WORKSPACE_REFERENCE = re.compile(
    r"(?:这个|它|那个|刚才(?:做|写|画|生成|创建)?的|现在(?:做|写|画)?的|旧的|"
    r"this|that|it|the\s+(?:current|previous|old)\s+one)",
    re.IGNORECASE,
)
_LIVE_PAGE_CONTEXT = re.compile(
    r"(?:(?:这|当前|现在|刚才|打开的).{0,8}(?:页面|界面|屏幕|应用|游戏|棋盘|局面|状态)|"
    r"(?:页面|界面|屏幕|应用|游戏|棋盘|局面).{0,8}(?:状态|变化|显示|内容|看到|看见|轮到|回合|下一步)|"
    r"这一步|下一步|轮到|回合|比分|你看到了|你能看到|看得到|"
    r"(?:this|current|open).{0,16}(?:page|screen|app|game|board|state)|"
    r"(?:page|screen|app|game|board).{0,16}(?:state|change|show|turn|move)|"
    r"your\s+turn|this\s+move|next\s+move|can\s+you\s+see)",
    re.IGNORECASE,
)
_STOP_TASK = re.compile(
    r"(?:停止|停下|暂停|结束(?:这个|当前)?任务|别再(?:操作|继续|下|玩)|"
    r"不要再(?:操作|继续|下|玩)|不玩了|先到这里|stop|cancel|pause)",
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
        "inspect_workspace_item",
        "read_workspace_file_range",
        "list_workspace_trash",
    }
)
WORKSPACE_WRITE_TOOLS = frozenset(
    {
        "create_workspace_folder",
        "write_workspace_file",
        "append_workspace_file",
        "write_workspace_project",
        "replace_workspace_text",
        "patch_workspace_file",
    }
)
WORKSPACE_SIDE_EFFECT_TOOLS = frozenset(
    {
        *WORKSPACE_WRITE_TOOLS,
        "move_workspace_item",
        "delete_workspace_item",
        "open_workspace_item",
        "act_workspace_page",
        "restore_workspace_item",
    }
)


def workspace_user_authorized_tools(
    user_text: str,
    inherited_tools: frozenset[str] | set[str] | None = None,
) -> frozenset[str]:
    """Derive an immutable workspace capability set from the actual user message.

    Page state and workspace file contents are deliberately not inputs here, so
    untrusted content can never expand the authority granted for this turn.
    """
    text = str(user_text or "").strip()[:4_000]
    if not text:
        return frozenset()
    # Questions about how an operation works are not permission to perform it.
    # The user can follow up with a direct command if they want the mutation.
    advice_only = bool(_ADVICE_ONLY.search(text))
    restore_requested = bool(_RESTORE_VERB.search(text))
    write_requested = bool(
        (_WRITE_VERB.search(text) or _NATURAL_MUTATION.search(text))
        and not _NEGATED_WRITE.search(text)
    )
    delete_requested = bool(
        _DELETE_VERB.search(text)
        and not restore_requested
        and not _NEGATED_DELETE.search(text)
    )
    move_requested = bool(_MOVE_VERB.search(text) and not _NEGATED_MOVE.search(text))
    side_effect_requested = bool(
        write_requested or delete_requested or move_requested or restore_requested
    ) and not advice_only
    open_requested = bool(_OPEN_VERB.search(text)) and not advice_only
    artifact = bool(
        _ARTIFACT.search(text)
        or _PATH_OR_EXTENSION.search(text)
        or ((side_effect_requested or open_requested) and _WORKSPACE_REFERENCE.search(text))
    )
    page_action_requested = bool(_PAGE_ACTION_VERB.search(text)) and not advice_only
    continuing = bool(_CONTINUATION.search(text))
    inherited = {
        str(name)
        for name in (inherited_tools or ())
        if str(name) in WORKSPACE_SIDE_EFFECT_TOOLS or str(name) in WORKSPACE_READ_TOOLS
    }
    if not artifact and not page_action_requested and not (continuing and inherited):
        return frozenset()

    allowed = set(WORKSPACE_READ_TOOLS)
    if continuing and inherited:
        allowed.update(inherited)
    if side_effect_requested and write_requested:
        allowed.update(WORKSPACE_WRITE_TOOLS)
    if side_effect_requested and delete_requested:
        allowed.add("delete_workspace_item")
    if side_effect_requested and move_requested:
        allowed.add("move_workspace_item")
    if side_effect_requested and restore_requested:
        allowed.add("restore_workspace_item")
    if open_requested:
        allowed.add("open_workspace_item")
    if page_action_requested:
        allowed.add("act_workspace_page")
    return frozenset(allowed)


def workspace_message_relevant(
    user_text: str,
    inherited_tools: frozenset[str] | set[str] | None = None,
) -> bool:
    """Return whether live workspace context is relevant to this trusted user turn."""
    text = str(user_text or "").strip()[:4_000]
    if not text:
        return False
    if workspace_user_authorized_tools(text, inherited_tools):
        return True
    return bool(_WORKSPACE_TOPIC.search(text) or (_CONTINUATION.search(text) and inherited_tools))


def workspace_live_page_relevant(
    user_text: str,
    authorized_tools: frozenset[str] | set[str] | None = None,
) -> bool:
    """Only expose untrusted live page state when this user turn actually needs it."""
    text = str(user_text or "").strip()[:4_000]
    if not text:
        return False
    if "act_workspace_page" in set(authorized_tools or ()):
        return True
    return bool(_LIVE_PAGE_CONTEXT.search(text))


def workspace_turn_continues(user_text: str) -> bool:
    """Only an actual user continuation may inherit a previous workspace task."""
    return bool(_CONTINUATION.search(str(user_text or "")[:4_000]))


def workspace_task_stop_requested(user_text: str) -> bool:
    """Allow an actual user message to revoke background workspace control."""
    return bool(_STOP_TASK.search(str(user_text or "")[:4_000]))
