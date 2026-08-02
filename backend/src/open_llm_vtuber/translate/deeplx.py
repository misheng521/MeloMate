import httpx
from loguru import logger
from .translate_interface import TranslateInterface


NETWORK_TIMEOUT = httpx.Timeout(15.0, connect=5.0, write=5.0, pool=5.0)
MAX_TRANSLATION_RESPONSE_BYTES = 1024 * 1024


class DeepLXTranslate(TranslateInterface):
    api_endpoint: str = "http://127.0.0.1:1188/v2/translate"
    target_lang: str = "JP"

    def __init__(self, api_endpoint: str, target_lang: str):
        super().__init__()
        self.api_endpoint = api_endpoint
        self.target_lang = target_lang

    # translate v2 endpoint from DeepLX
    def translate(self, text: str) -> str:
        self._consume_request_budget(text)
        try:
            data = {"text": [text], "target_lang": self.target_lang}
            with httpx.Client(
                timeout=NETWORK_TIMEOUT, follow_redirects=False
            ) as client:
                with client.stream("POST", self.api_endpoint, json=data) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_TRANSLATION_RESPONSE_BYTES:
                            raise ValueError("Translation response exceeds the size limit")
            payload = httpx.Response(200, content=bytes(body)).json()
            translations = payload.get("translations")
            if not isinstance(translations, list):
                raise ValueError("Translation response is missing translations")
            result = " ".join(
                item["text"]
                for item in translations
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
            if not result:
                raise ValueError("Translation service returned an empty result")
        except Exception as e:
            logger.error(f"DeepLX translation failed: {type(e).__name__}")
            raise

        return self._validate_output(result)
