import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from open_llm_vtuber.secure_credentials import (  # noqa: E402
    CHAT_API_KEY,
    SCREEN_VISION_API_KEY,
    SecureCredentialError,
    SecureCredentialStore,
    validate_profile_id,
)
from open_llm_vtuber.websocket_handler import WebSocketHandler  # noqa: E402


PROFILE_ID = "test-profile-0123456789abcdef"


def protect(value: bytes) -> bytes:
    return b"protected:" + value[::-1]


def unprotect(value: bytes) -> bytes:
    if not value.startswith(b"protected:"):
        raise ValueError("invalid ciphertext")
    return value.removeprefix(b"protected:")[::-1]


class SecureCredentialStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "credentials-v1.json"
        self.store = SecureCredentialStore(
            path=self.path,
            protector=protect,
            unprotector=unprotect,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_secrets_round_trip_without_plaintext_on_disk(self):
        chat_key = "chat-super-secret-value"
        vision_key = "vision-super-secret-value"
        status = self.store.update(
            PROFILE_ID,
            secrets={
                CHAT_API_KEY: chat_key,
                SCREEN_VISION_API_KEY: vision_key,
            },
        )
        self.assertTrue(status[CHAT_API_KEY])
        self.assertTrue(status[SCREEN_VISION_API_KEY])
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn(chat_key, raw)
        self.assertNotIn(vision_key, raw)
        self.assertEqual(self.store.get(PROFILE_ID, CHAT_API_KEY), chat_key)
        self.assertEqual(self.store.get(PROFILE_ID, SCREEN_VISION_API_KEY), vision_key)

    def test_clear_removes_only_the_requested_credential(self):
        self.store.update(
            PROFILE_ID,
            secrets={CHAT_API_KEY: "chat-key", SCREEN_VISION_API_KEY: "vision-key"},
        )
        status = self.store.update(PROFILE_ID, clear={CHAT_API_KEY})
        self.assertFalse(status[CHAT_API_KEY])
        self.assertTrue(status[SCREEN_VISION_API_KEY])
        self.assertIsNone(self.store.get(PROFILE_ID, CHAT_API_KEY))
        self.assertEqual(
            self.store.get(PROFILE_ID, SCREEN_VISION_API_KEY), "vision-key"
        )

    def test_invalid_profile_and_credential_names_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_profile_id("../../outside")
        with self.assertRaises(ValueError):
            self.store.update(PROFILE_ID, secrets={"unknown": "secret"})

    def test_damaged_store_is_not_silently_overwritten(self):
        self.path.write_text("{damaged", encoding="utf-8")
        with self.assertRaises(SecureCredentialError):
            self.store.update(PROFILE_ID, secrets={CHAT_API_KEY: "new-secret"})
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{damaged")

    @unittest.skipUnless(sys.platform == "win32", "Windows DPAPI test")
    def test_real_dpapi_is_bound_and_not_plaintext(self):
        dpapi_path = Path(self.temporary.name) / "dpapi-credentials.json"
        store = SecureCredentialStore(path=dpapi_path)
        self.assertTrue(store.available)
        store.update(PROFILE_ID, secrets={CHAT_API_KEY: "dpapi-test-secret"})
        self.assertNotIn(
            "dpapi-test-secret", dpapi_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            store.get(PROFILE_ID, CHAT_API_KEY), "dpapi-test-secret"
        )
        document = json.loads(dpapi_path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], 1)


class FakeWebSocket:
    def __init__(self):
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class FakeContext:
    def __init__(self):
        self.screen_vision_api_key = ""
        self.applied: dict | None = None

    async def apply_client_api_config(self, **config) -> None:
        self.applied = config


class CredentialProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_saves_keys_without_returning_them_to_browser(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SecureCredentialStore(
                path=Path(temporary) / "credentials-v1.json",
                protector=protect,
                unprotector=unprotect,
            )
            handler = object.__new__(WebSocketHandler)
            handler.credential_store = store
            context = FakeContext()
            handler.client_contexts = {"client": context}
            websocket = FakeWebSocket()
            chat_key = "chat-key-that-must-not-return"
            vision_key = "vision-key-that-must-not-return"

            await handler._handle_client_api_config(
                websocket,
                "client",
                {
                    "type": "client-api-config",
                    "request_id": "request-1",
                    "credential_profile_id": PROFILE_ID,
                    "api_base_url": "https://api.example.test",
                    "model": "model-name",
                    "api_key": chat_key,
                    "screen_vision_api_key": vision_key,
                },
            )

            self.assertEqual(context.applied["api_key"], chat_key)
            self.assertEqual(context.screen_vision_api_key, vision_key)
            response = websocket.messages[-1]
            self.assertNotIn(chat_key, response)
            self.assertNotIn(vision_key, response)
            payload = json.loads(response)
            self.assertTrue(payload["success"])
            self.assertTrue(payload["chat_api_key_saved"])
            self.assertTrue(payload["screen_vision_api_key_saved"])


if __name__ == "__main__":
    unittest.main()
