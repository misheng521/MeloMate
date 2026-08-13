import unittest

from open_llm_vtuber.agent.output_types import Actions
from open_llm_vtuber.avatar_model import AvatarModel


class AvatarActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.avatar = AvatarModel.__new__(AvatarModel)
        self.avatar.emo_map = {"joy": 3, "sadness": 1}

    def test_extracts_model_selected_body_gesture(self) -> None:
        text = "[joy]你好。[gesture_greet]很高兴见到你。"

        self.assertEqual(self.avatar.extract_emotion(text), [3])
        self.assertEqual(self.avatar.extract_gestures(text), ["greet"])

    def test_action_tags_are_removed_from_spoken_text(self) -> None:
        text = "[JOY]你好。[GESTURE_EXPLAIN]我来说明一下。"

        self.assertEqual(
            self.avatar.remove_action_keywords(text),
            "你好。我来说明一下。",
        )

    def test_gestures_are_serialized_with_existing_actions(self) -> None:
        actions = Actions(expressions=[3], gestures=["agree"])

        self.assertEqual(
            actions.to_dict(),
            {"expressions": [3], "gestures": ["agree"]},
        )


if __name__ == "__main__":
    unittest.main()
