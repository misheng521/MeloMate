# config_manager/utils.py
import yaml
from pathlib import Path
from typing import Union, Dict, Any, TypeVar
from pydantic import BaseModel, ValidationError
import os
import re
import chardet
from loguru import logger

from .main import Config

T = TypeVar("T", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHARACTER_CONFIG_NAME = "小可.yaml"


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """
    Recursively merge dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_yaml(config_path: str) -> Dict[str, Any]:
    """
    Read the specified YAML configuration file with environment variable substitution
    and guess encoding. Return the configuration data as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration data as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        IOError: If the configuration file cannot be read.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    content = load_text_file_with_guess_encoding(config_path)
    if not content:
        raise IOError(f"Failed to read configuration file: {config_path}")

    # Replace environment variables
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))

    content = pattern.sub(replacer, content)

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML file: {e}")
        raise e


def validate_config(config_data: dict) -> Config:
    """
    Validate configuration data against the Config model.

    Args:
        config_data: Configuration data to validate.

    Returns:
        Validated Config object.

    Raises:
        ValidationError: If the configuration fails validation.
    """
    try:
        return Config(**config_data)
    except ValidationError as e:
        logger.critical(f"Error validating configuration: {e}")
        logger.error("Configuration data:")
        logger.error(config_data)
        raise e


def load_config_with_character(
    config_path: str = "conf.yaml",
    character_file_name: str = DEFAULT_CHARACTER_CONFIG_NAME,
) -> dict:
    """
    Load the base config and merge in a character YAML from config_alts_dir.
    Persona prompts live only in character YAML files.
    """
    base_config = read_yaml(config_path)
    config_alts_dir = base_config.get("system_config", {}).get("config_alts_dir")
    if not config_alts_dir:
        raise ValueError("system_config.config_alts_dir is required")

    if character_file_name in {"xyu.yaml", "xyua.yaml"}:
        character_file_name = DEFAULT_CHARACTER_CONFIG_NAME

    character_path = os.path.normpath(os.path.join(config_alts_dir, character_file_name))
    character_data = read_yaml(character_path)
    alt_character_config = character_data.get("character_config")
    if not alt_character_config:
        raise ValueError(f"Missing character_config in {character_path}")

    base_character_config = base_config.get("character_config", {})
    base_config["character_config"] = deep_merge(
        base_character_config, alt_character_config
    )
    return base_config


def load_text_file_with_guess_encoding(file_path: str) -> str | None:
    """
    Load a text file with guessed encoding.

    Parameters:
    - file_path (str): The path to the text file.

    Returns:
    - str: The content of the text file or None if an error occurred.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii", "cp936"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    # If common encodings fail, try chardet to guess the encoding
    try:
        with open(file_path, "rb") as file:
            raw_data = file.read()
        detected = chardet.detect(raw_data)
        if detected["encoding"]:
            return raw_data.decode(detected["encoding"])
    except Exception as e:
        logger.error(f"Error detecting encoding for config file {file_path}: {e}")
    return None


def save_config(config: BaseModel, config_path: Union[str, Path]):
    """
    Saves a Pydantic model to a YAML configuration file.

    Args:
        config: The Pydantic model to save.
        config_path: Path to the YAML configuration file.
    """
    config_file = Path(config_path)
    config_data = config.model_dump(
        by_alias=True, exclude_unset=True, exclude_none=True
    )

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error writing YAML file: {e}")


def scan_config_alts_directory(config_alts_dir: str) -> list[dict]:
    """
    Scan the config_alts directory and return a list of config information.
    Each config info uses the exact character YAML filename as the value and
    the filename without suffix as the display name.

    Parameters:
    - config_alts_dir (str): The path to the config_alts directory.

    Returns:
    - list[dict]: A list of dicts containing config info:
        - filename: The actual config file name
        - name: The config file name without the .yaml/.yml suffix
    """
    config_files = []

    for root, _, files in os.walk(config_alts_dir):
        for file in files:
            if file.endswith(".yaml"):
                config_path = Path(root) / file
                conf_name = os.path.splitext(file)[0]
                character_name = conf_name
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f) or {}
                    character_config = config_data.get("character_config", {})
                    conf_name = character_config.get("conf_name") or conf_name
                    character_name = (
                        character_config.get("character_name")
                        or conf_name
                    )
                except Exception as e:
                    logger.warning(f"Failed to read character config {config_path}: {e}")

                config_files.append(
                    {
                        "filename": file,
                        "name": character_name,
                        "conf_name": conf_name,
                        "character_name": character_name,
                    }
                )
    logger.debug(f"Found config files: {config_files}")
    return config_files


def scan_bg_directory() -> list[str]:
    bg_files = []
    bg_dir = PROJECT_ROOT / "backgrounds"
    for root, _, files in os.walk(bg_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")):
                bg_files.append(os.path.relpath(os.path.join(root, file), bg_dir).replace("\\", "/"))
    return bg_files
