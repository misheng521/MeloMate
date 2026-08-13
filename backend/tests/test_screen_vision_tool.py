import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.open_llm_vtuber.agent.agents.basic_memory_agent import (  # noqa: E402
    BasicMemoryAgent,
    _screen_vision_tool_schema,
)
from src.open_llm_vtuber.conversations.conversation_utils import (  # noqa: E402
    augment_text_with_screen_context,
)
from src.open_llm_vtuber.mcpp.types import (  # noqa: E402
    ToolCallFunctionObject,
    ToolCallObject,
)


class FakeOpenAIToolLLM:
    def __init__(self, *, call_screen_tool: bool):
        self.call_screen_tool = call_screen_tool
        self.calls = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        if self.call_screen_tool and len(self.calls) == 1:
            yield [
                ToolCallObject(
                    id="screen-call-1",
                    function=ToolCallFunctionObject(
                        name="view_current_screen", arguments="{}"
                    ),
                )
            ]
            return
        yield "结合当前信息回答。"


def bare_agent(llm):
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._llm = llm
    agent._tool_executor = None
    agent._tool_manager = None
    agent._mcp_prompt_string = ""
    agent._json_detector = None
    agent.prompt_mode_flag = False
    return agent


class ScreenVisionToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_turn_never_injects_screen_failure_dialogue(self):
        user_text = "你看看这个说法是否合理"
        result = await augment_text_with_screen_context(
            user_text,
            images=None,
            screen_vision=None,
            force=False,
        )
        self.assertEqual(result, user_text)
        self.assertNotIn("API", result)
        self.assertNotIn("识图模型", result)

    async def test_forced_proactive_failure_stays_silent(self):
        user_text = "主动说一句"
        result = await augment_text_with_screen_context(
            user_text,
            images=None,
            screen_vision=None,
            force=True,
        )
        self.assertEqual(result, user_text)

    async def test_model_can_answer_without_calling_enabled_screen_tool(self):
        llm = FakeOpenAIToolLLM(call_screen_tool=False)
        agent = bare_agent(llm)
        calls = 0

        async def view_screen():
            nonlocal calls
            calls += 1
            return "游戏画面"

        output = [
            item
            async for item in agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "你看看这个观点"}],
                [_screen_vision_tool_schema("OpenAI")],
                "system",
                remember_turn=False,
                screen_vision_tool=view_screen,
            )
        ]
        self.assertEqual(output, ["结合当前信息回答。"])
        self.assertEqual(calls, 0)

    async def test_disabled_screen_does_not_offer_the_tool(self):
        llm = FakeOpenAIToolLLM(call_screen_tool=False)
        agent = bare_agent(llm)
        output = [
            item
            async for item in agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "你看看这个观点"}],
                [],
                "system",
                remember_turn=False,
                screen_vision_tool=None,
            )
        ]
        self.assertEqual(output, ["结合当前信息回答。"])
        self.assertEqual(llm.calls[0]["tools"], [])

    async def test_model_tool_call_receives_current_screen_result_once(self):
        llm = FakeOpenAIToolLLM(call_screen_tool=True)
        agent = bare_agent(llm)
        calls = 0

        async def view_screen():
            nonlocal calls
            calls += 1
            return "画面里角色正在面对一个红色 Boss。"

        output = [
            item
            async for item in agent._openai_tool_interaction_loop(
                [{"role": "user", "content": "我现在该怎么躲？"}],
                [_screen_vision_tool_schema("OpenAI")],
                "system",
                remember_turn=False,
                screen_vision_tool=view_screen,
            )
        ]
        self.assertEqual(output, ["结合当前信息回答。"])
        self.assertEqual(calls, 1)
        self.assertEqual(llm.calls[1]["tools"], [])
        tool_messages = [
            message
            for message in llm.calls[1]["messages"]
            if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("红色 Boss", tool_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
