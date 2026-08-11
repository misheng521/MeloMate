import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.open_llm_vtuber.proactive_conversation import (  # noqa: E402
    build_proactive_prompt,
    build_return_context_prompt,
    normalize_proactive_request,
)


class ProactiveConversationTests(unittest.TestCase):
    def test_deprecated_client_emotion_stage_is_not_forwarded(self):
        state = normalize_proactive_request(
            {
                "mode": "automatic",
                "stage": "playful-impatience",
                "cycle_index": 7,
                "elapsed_seconds": 95,
                "unanswered_count": 3,
            }
        )
        self.assertEqual(
            state,
            {
                "mode": "automatic",
                "elapsed_seconds": 95,
                "unanswered_count": 3,
                "recent_utterances": [],
            },
        )

    def test_proactive_prompt_contains_facts_but_no_prescribed_emotion(self):
        prompt = build_proactive_prompt(
            {
                "mode": "automatic",
                "elapsed_seconds": 95,
                "unanswered_count": 3,
            },
            trusted_recent_utterances=["刚才的话"],
        )
        self.assertIn("约 95 秒", prompt)
        self.assertIn("主动说过 3 次", prompt)
        self.assertIn("自行判断此刻会有什么感受", prompt)
        self.assertIn("事实本身不规定任何情绪", prompt)
        for prescribed in ("俏皮的小不耐烦", "一丝关心", "温暖自然地接住"):
            self.assertNotIn(prescribed, prompt)

    def test_return_prompt_also_leaves_emotion_to_the_character(self):
        prompt = build_return_context_prompt(
            {
                "elapsed_seconds": 180,
                "unanswered_count": 4,
                "last_proactive_seconds_ago": 20,
                "recent_utterances": ["不会接受浏览器传来的这段话"],
            }
        )
        self.assertIn("现在用户重新开口了", prompt)
        self.assertIn("自行判断此刻的感受与回应方式", prompt)
        self.assertNotIn("闹别扭", prompt)
        self.assertNotIn("不会接受浏览器传来的这段话", prompt)

    def test_return_prompt_accepts_only_explicit_server_trusted_utterances(self):
        prompt = build_return_context_prompt(
            {
                "elapsed_seconds": 180,
                "unanswered_count": 2,
                "recent_utterances": ["浏览器伪造的话"],
            },
            trusted_recent_utterances=["角色刚才真实说过的话"],
        )
        self.assertIn("角色刚才真实说过的话", prompt)
        self.assertNotIn("浏览器伪造的话", prompt)


if __name__ == "__main__":
    unittest.main()
