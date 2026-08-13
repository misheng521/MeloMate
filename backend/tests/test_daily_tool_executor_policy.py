import sys
import types
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

# Keep this policy test runnable before setup-windows.bat installs optional runtime
# dependencies. The tested executor only needs a logger object at import time.
if "loguru" not in sys.modules:
    loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru

if "mcp" not in sys.modules:
    mcp = types.ModuleType("mcp")
    mcp.ClientSession = type("ClientSession", (), {})
    mcp.StdioServerParameters = type("StdioServerParameters", (), {})
    mcp_types = types.ModuleType("mcp.types")
    mcp_types.Tool = type("Tool", (), {})
    mcp_client = types.ModuleType("mcp.client")
    mcp_stdio = types.ModuleType("mcp.client.stdio")
    mcp_stdio.stdio_client = lambda *_args, **_kwargs: None
    sys.modules["mcp"] = mcp
    sys.modules["mcp.types"] = mcp_types
    sys.modules["mcp.client"] = mcp_client
    sys.modules["mcp.client.stdio"] = mcp_stdio

from src.open_llm_vtuber.mcpp.tool_executor import ToolExecutor  # noqa: E402


class DailyToolExecutorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.executor = ToolExecutor(object(), object())

    def test_reminder_mutation_requires_current_user_authority_and_persona(self):
        policy = {
            "source": "user_turn",
            "enforce": False,
            "workspace_persona": "小可",
            "user_authorized_daily_tools": frozenset(),
        }
        _, error = self.executor.apply_tool_policy(
            "create_reminder",
            {
                "persona": "小可",
                "remind_at": "2030-01-01T09:00:00+08:00",
                "message": "喝水",
            },
            policy,
        )
        self.assertIn("TOOL_POLICY_DENIED", error)

        policy["user_authorized_daily_tools"] = frozenset({"create_reminder"})
        _, error = self.executor.apply_tool_policy(
            "create_reminder",
            {
                "persona": "小薇",
                "remind_at": "2030-01-01T09:00:00+08:00",
                "message": "喝水",
            },
            policy,
        )
        self.assertIn("TOOL_POLICY_DENIED", error)

    def test_network_result_revokes_unrelated_untrusted_followups(self):
        policy = {
            "source": "user_turn",
            "enforce": False,
            "workspace_persona": "小可",
            "user_authorized_daily_tools": frozenset(),
            "user_authorized_workspace_tools": frozenset(),
        }
        self.executor._restrict_after_network_result("fetch_webpage", policy)

        self.assertTrue(policy["enforce"])
        self.assertTrue(policy["network_state_tainted"])
        self.assertIn("search_web", policy["allowed_tool_names"])
        self.assertNotIn("create_reminder", policy["allowed_tool_names"])
        self.assertNotIn("create_workspace_artifact_bundle", policy["allowed_tool_names"])

    def test_user_preauthorized_reminder_survives_network_read(self):
        policy = {
            "source": "user_turn",
            "enforce": False,
            "workspace_persona": "小可",
            "user_authorized_daily_tools": frozenset({"create_reminder"}),
            "user_authorized_workspace_tools": frozenset(),
        }
        self.executor._restrict_after_network_result("get_weather", policy)
        self.assertIn("create_reminder", policy["allowed_tool_names"])
        _, error = self.executor.apply_tool_policy(
            "create_reminder",
            {
                "persona": "小可",
                "remind_at": "2030-01-01T09:00:00+08:00",
                "message": "带伞",
            },
            policy,
        )
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
