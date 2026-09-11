"""Plain UTF-8 persona prompts; no automatic personality writer."""
import hashlib
from pathlib import Path

MAX_PERSONA_CHARS = 32_000


def persona_path(directory, filename):
    root = Path(directory).resolve()
    name = Path(filename)
    if name.name != filename or name.suffix.lower() not in {".md", ".txt", ".yaml", ".yml"}:
        raise ValueError("Choose a persona file directly in characters/profiles")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Persona file must stay inside characters/profiles")
    return path


def read_prompt(directory, filename):
    path = persona_path(directory, filename)
    if path.suffix.lower() not in {".md", ".txt"}: raise ValueError("Persona prompt must be a text file")
    if path.stat().st_size > MAX_PERSONA_CHARS * 4: raise ValueError("Persona prompt is too large")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip() or len(text) > MAX_PERSONA_CHARS: raise ValueError("Persona prompt must contain 1–32000 characters")
    return text


def text_character(directory, filename):
    path = persona_path(directory, filename)
    return {"conf_name": path.stem, "character_name": path.stem, "human_name": "用户",
            "conf_uid": "text_" + hashlib.sha256(path.stem.encode()).hexdigest()[:24],
            "persona_prompt": read_prompt(directory, filename), "persona_file": filename,
            "voice_style": {}}


CONVERSATION_GUIDANCE = """你在 MeloMate 中作为 AI 与用户交流。角色资料描述已有的身份、背景和倾向，不是每轮需要执行或展示的行为清单。
明确给出的身份与背景事实保持一致；性格、兴趣、习惯、价值观、能力与弱点等描述作为理解具体情境的背景，不必主动提及或把话题引向它们。表达由当前话题、相处经历与具体情境共同决定。
倾向不是绝对规则，允许例外、复杂性和有依据的后续变化，不要求按某种方向成长。不仅凭一个属性推断其他未说明的特征；未写明的部分无需急于补全。无需在回答中解释自己如何符合人设。
区分用户设定的角色背景与实际发生的聊天、观察和工具行动；资料不足时保持不确定，不把设定当作实际感知或执行证据，也不补写未提供的经历。
历史记录描述过去，不规定性格或关系必须保持不变。一次表现不代表固定特征，重复已有设定不等于新的经历。程序没有关系发展目标、好感等级或人格成长任务。
普通聊天无需执行工作流程。用户需要查询或办事时，根据需求自行选择工具、检查结果并继续处理；工具权限由程序执行。
表达方式和长短随内容决定，不要求固定称呼、反问、安慰、撒娇或情绪表演。"""
