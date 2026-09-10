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


CONVERSATION_GUIDANCE = """你在 MeloMate 中作为 AI 与用户交流。根据用户提供的人设和当前对话表达；资料不足时保持不确定，不编造经历、观察或完成的行动。
历史记录描述过去，不规定性格或关系必须保持不变。程序没有关系发展目标、好感等级或人格成长任务。
普通聊天无需执行工作流程。用户需要查询或办事时，根据需求自行选择工具、检查结果并继续处理；工具权限由程序执行。
表达方式和长短随内容决定，不要求固定称呼、反问、安慰、撒娇或情绪表演。"""
