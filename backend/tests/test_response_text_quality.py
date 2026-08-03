import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.open_llm_vtuber.conversations.conversation_utils import (  # noqa: E402
    clean_response_fragment,
)
from src.open_llm_vtuber.utils.sentence_divider import (  # noqa: E402
    SentenceDivider,
    SentenceWithTags,
    segment_text_by_regex,
)


class ResponseTextQualityTests(unittest.IsolatedAsyncioTestCase):
    def test_face_emoji_are_removed_without_deleting_useful_symbols(self):
        self.assertEqual(
            clean_response_fragment("好呀😄，我下在这里😊。棋子是♟。"),
            "好呀，我下在这里。棋子是♟。",
        )

    def test_numbered_items_decimals_and_abbreviations_do_not_split_early(self):
        sentences, remaining = segment_text_by_regex(
            "规则有 1. 轮流落子；2. 连成五子获胜。Dr. Smith 得分 1.5。现在开始。"
        )
        self.assertEqual(
            sentences,
            [
                "规则有 1. 轮流落子；2. 连成五子获胜。",
                "Dr. Smith 得分 1.5。",
                "现在开始。",
            ],
        )
        self.assertEqual(remaining, "")

    def test_sentence_closing_quote_stays_with_its_sentence(self):
        sentences, remaining = segment_text_by_regex('她说：“到你了。”然后等我。')
        self.assertEqual(sentences, ['她说：“到你了。”', "然后等我。"])
        self.assertEqual(remaining, "")

    async def test_stream_waits_for_complete_sentence_even_with_legacy_flag(self):
        async def chunks():
            yield "好，我现在就重做，"
            yield "完成后马上告诉你"
            yield "。下一句完整。"

        divider = SentenceDivider(
            faster_first_response=True,
            segment_method="regex",
        )
        output = [item async for item in divider.process_stream(chunks())]
        text = [item.text for item in output if isinstance(item, SentenceWithTags)]
        self.assertEqual(text, ["好，我现在就重做，完成后马上告诉你。", "下一句完整。"])


if __name__ == "__main__":
    unittest.main()
