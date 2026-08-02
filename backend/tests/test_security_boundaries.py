import os
import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

os.environ["MELOMATE_SESSION_TOKEN"] = "backend-security-test-token"
os.environ["MELOMATE_FRONTEND_ORIGIN"] = (
    "http://127.0.0.1:35178,http://localhost:35178"
)

from src.open_llm_vtuber import routes  # noqa: E402


class FakeSocket:
    def __init__(self, origin: str, protocol: str):
        self.headers = {
            "origin": origin,
            "sec-websocket-protocol": protocol,
        }


class SecurityBoundaryTests(unittest.TestCase):
    def test_websocket_requires_matching_origin_and_session_protocol(self):
        protocol = "melomate.session.backend-security-test-token"
        self.assertEqual(
            routes._authenticated_websocket_protocol(
                FakeSocket("http://127.0.0.1:35178", protocol)
            ),
            protocol,
        )
        self.assertIsNone(
            routes._authenticated_websocket_protocol(
                FakeSocket("https://attacker.invalid", protocol)
            )
        )
        self.assertIsNone(
            routes._authenticated_websocket_protocol(
                FakeSocket("http://127.0.0.1:35178", "")
            )
        )

    def test_unused_standalone_voice_routes_are_not_registered(self):
        router = routes.init_webtool_routes(object())
        registered = {(route.path, type(route).__name__) for route in router.routes}
        self.assertNotIn(("/asr", "APIRoute"), registered)
        self.assertNotIn(("/tts-ws", "APIWebSocketRoute"), registered)
        self.assertIn(("/live2d-models/info", "APIRoute"), registered)


if __name__ == "__main__":
    unittest.main()
