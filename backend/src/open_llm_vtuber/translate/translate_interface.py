import abc
import asyncio
import threading
import time
from collections import deque


MAX_TRANSLATION_INPUT_CHARS = 6_000
MAX_TRANSLATION_OUTPUT_CHARS = 12_000
MAX_TRANSLATION_CALLS_PER_MINUTE = 60
MAX_TRANSLATION_CHARS_PER_MINUTE = 60_000
TRANSLATION_TASK_TIMEOUT_SECONDS = 20


class TranslateInterface(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self._translation_budget: deque[tuple[float, int]] = deque()
        self._translation_budget_lock = threading.Lock()

    def _consume_request_budget(self, text: str) -> None:
        if not isinstance(text, str) or not text:
            raise ValueError("Translation input must be a non-empty string")
        if len(text) > MAX_TRANSLATION_INPUT_CHARS:
            raise ValueError("Translation input exceeds the per-call character limit")

        now = time.monotonic()
        with self._translation_budget_lock:
            while self._translation_budget and now - self._translation_budget[0][0] >= 60:
                self._translation_budget.popleft()
            calls = len(self._translation_budget)
            characters = sum(item[1] for item in self._translation_budget)
            if calls >= MAX_TRANSLATION_CALLS_PER_MINUTE:
                raise RuntimeError("Translation request rate limit reached")
            if characters + len(text) > MAX_TRANSLATION_CHARS_PER_MINUTE:
                raise RuntimeError("Translation character budget reached")
            self._translation_budget.append((now, len(text)))

    @staticmethod
    def _validate_output(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Translation service returned a non-text result")
        if len(text) > MAX_TRANSLATION_OUTPUT_CHARS:
            raise ValueError("Translation result exceeds the character limit")
        return text

    async def async_translate(self, text: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(self.translate, text),
            timeout=TRANSLATION_TASK_TIMEOUT_SECONDS,
        )

    @abc.abstractmethod
    def translate(self, text: str) -> str:
        """
        Translate the input text to the target language."""
        raise NotImplementedError
