import json

import chardet
from loguru import logger


class AvatarModel:
    """Expression profile shared by the conversation pipeline and VRM frontend."""

    def __init__(
        self,
        avatar_model_name: str,
        model_dict_path: str = "avatar_model_dict.json",
    ) -> None:
        self.model_dict_path = model_dict_path
        self.avatar_model_name = avatar_model_name
        self.set_model(avatar_model_name)

    def set_model(self, model_name: str) -> None:
        self.model_info = self._lookup_model_info(model_name)
        self.emo_map = {
            key.lower(): value
            for key, value in self.model_info["emotionMap"].items()
        }
        self.emo_str = " ".join(f"[{key}]," for key in self.emo_map)

    def _load_file_content(self, file_path: str) -> str:
        for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii"]:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)["encoding"]
            if detected_encoding:
                return raw_data.decode(detected_encoding)
        except Exception as error:
            logger.error(f"Error detecting encoding for {file_path}: {error}")

        raise UnicodeError(f"Failed to decode {file_path} with any supported encoding")

    def _lookup_model_info(self, model_name: str) -> dict:
        self.avatar_model_name = model_name
        try:
            model_dict = json.loads(self._load_file_content(self.model_dict_path))
        except Exception as error:
            logger.critical(
                f"Unable to read avatar expression profile {self.model_dict_path}: {error}"
            )
            raise

        matched_model = next(
            (model for model in model_dict if model["name"] == model_name), None
        )
        if matched_model is None:
            raise KeyError(
                f"{model_name} not found in avatar expression profile {self.model_dict_path}."
            )
        logger.info("Avatar expression profile loaded.")
        return matched_model

    def extract_emotion(self, text: str) -> list:
        expression_list = []
        lowered = text.lower()
        index = 0
        while index < len(lowered):
            if lowered[index] != "[":
                index += 1
                continue
            for key, value in self.emo_map.items():
                tag = f"[{key}]"
                if lowered[index : index + len(tag)] == tag:
                    expression_list.append(value)
                    index += len(tag) - 1
                    break
            index += 1
        return expression_list

    def remove_emotion_keywords(self, text: str) -> str:
        lowered = text.lower()
        for key in self.emo_map:
            tag = f"[{key}]"
            while tag in lowered:
                start = lowered.find(tag)
                end = start + len(tag)
                text = text[:start] + text[end:]
                lowered = lowered[:start] + lowered[end:]
        return text
