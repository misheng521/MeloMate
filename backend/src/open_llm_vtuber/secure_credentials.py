"""Windows-user-bound storage for API credentials.

Only encrypted DPAPI blobs are written to disk.  Decrypted values are returned
to trusted backend callers and are never included in status responses.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Final, Iterable
from uuid import uuid4


CHAT_API_KEY: Final = "chat_api_key"
SCREEN_VISION_API_KEY: Final = "screen_vision_api_key"
CREDENTIAL_NAMES: Final = frozenset({CHAT_API_KEY, SCREEN_VISION_API_KEY})
MAX_SECRET_BYTES: Final = 16_384
MAX_STORE_BYTES: Final = 2 * 1024 * 1024
MAX_PROFILES: Final = 128
PROFILE_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{20,128}$")
DPAPI_ENTROPY: Final = b"MeloMate credential vault v1"


class SecureCredentialError(RuntimeError):
    """Raised when credentials cannot be stored or decrypted safely."""


def validate_profile_id(profile_id: str) -> str:
    normalized = str(profile_id or "").strip()
    if not PROFILE_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid credential profile identifier")
    return normalized


def _default_store_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app_data:
        raise SecureCredentialError(
            "Windows LOCALAPPDATA is unavailable; secure credential persistence is disabled"
        )
    return Path(local_app_data) / "MeloMate" / "credentials-v1.json"


class SecureCredentialStore:
    """Persist a small set of API keys as current-user DPAPI ciphertext."""

    def __init__(
        self,
        path: Path | None = None,
        protector: Callable[[bytes], bytes] | None = None,
        unprotector: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self._path = path
        self._protector = protector
        self._unprotector = unprotector
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        if self._protector is not None and self._unprotector is not None:
            return True
        if sys.platform != "win32" or not os.environ.get("LOCALAPPDATA", "").strip():
            return False
        try:
            import win32crypt  # noqa: F401
            import win32cryptcon  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def path(self) -> Path:
        return self._path or _default_store_path()

    @staticmethod
    def _validate_names(names: Iterable[str]) -> set[str]:
        normalized = {str(name) for name in names}
        unknown = normalized - CREDENTIAL_NAMES
        if unknown:
            raise ValueError("Unknown credential name")
        return normalized

    def _protect(self, value: bytes) -> bytes:
        if self._protector is not None:
            return self._protector(value)
        if not self.available:
            raise SecureCredentialError(
                "Secure credential persistence requires Windows DPAPI and pywin32"
            )
        import win32crypt
        import win32cryptcon

        return win32crypt.CryptProtectData(
            value,
            "MeloMate API credential",
            DPAPI_ENTROPY,
            None,
            None,
            win32cryptcon.CRYPTPROTECT_UI_FORBIDDEN,
        )

    def _unprotect(self, value: bytes) -> bytes:
        if self._unprotector is not None:
            return self._unprotector(value)
        if not self.available:
            raise SecureCredentialError(
                "Secure credential persistence requires Windows DPAPI and pywin32"
            )
        import win32crypt
        import win32cryptcon

        try:
            return win32crypt.CryptUnprotectData(
                value,
                DPAPI_ENTROPY,
                None,
                None,
                win32cryptcon.CRYPTPROTECT_UI_FORBIDDEN,
            )[1]
        except Exception as exc:
            raise SecureCredentialError(
                "Saved API credential cannot be decrypted by the current Windows user"
            ) from exc

    @staticmethod
    def _empty_document() -> dict:
        return {"version": 1, "profiles": {}}

    def _load_unlocked(self) -> dict:
        path = self.path
        if not path.exists():
            return self._empty_document()
        if path.is_symlink() or not path.is_file():
            raise SecureCredentialError("Secure credential store path is not a regular file")
        size = path.stat().st_size
        if size < 2 or size > MAX_STORE_BYTES:
            raise SecureCredentialError("Secure credential store has an invalid size")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SecureCredentialError("Secure credential store is damaged") from exc
        if not isinstance(document, dict) or document.get("version") != 1:
            raise SecureCredentialError("Secure credential store version is invalid")
        profiles = document.get("profiles")
        if not isinstance(profiles, dict) or len(profiles) > MAX_PROFILES:
            raise SecureCredentialError("Secure credential store profiles are invalid")
        for profile_id, profile in profiles.items():
            validate_profile_id(profile_id)
            if not isinstance(profile, dict) or set(profile) - CREDENTIAL_NAMES:
                raise SecureCredentialError("Secure credential profile is invalid")
            if not all(isinstance(value, str) for value in profile.values()):
                raise SecureCredentialError("Secure credential ciphertext is invalid")
        return document

    def _write_unlocked(self, document: dict) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise SecureCredentialError("Secure credential directory must not be a link")
        encoded = json.dumps(
            document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > MAX_STORE_BYTES:
            raise SecureCredentialError("Secure credential store exceeds its size limit")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as destination:
                destination.write(encoded)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def update(
        self,
        profile_id: str,
        secrets: dict[str, str] | None = None,
        clear: Iterable[str] = (),
    ) -> dict[str, bool]:
        """Atomically set and/or clear allowlisted credentials for one profile."""
        profile_id = validate_profile_id(profile_id)
        secrets = secrets or {}
        self._validate_names(secrets)
        clear_names = self._validate_names(clear)
        encrypted: dict[str, str] = {}
        for name, raw_value in secrets.items():
            value = str(raw_value or "").strip()
            encoded = value.encode("utf-8")
            if not encoded or len(encoded) > MAX_SECRET_BYTES:
                raise ValueError("API credential is empty or exceeds the size limit")
            encrypted[name] = base64.b64encode(self._protect(encoded)).decode("ascii")

        with self._lock:
            document = self._load_unlocked()
            profiles = document["profiles"]
            if profile_id not in profiles and len(profiles) >= MAX_PROFILES:
                raise SecureCredentialError("Secure credential profile limit reached")
            profile = dict(profiles.get(profile_id) or {})
            for name in clear_names:
                profile.pop(name, None)
            profile.update(encrypted)
            if profile:
                profiles[profile_id] = profile
            else:
                profiles.pop(profile_id, None)
            self._write_unlocked(document)
            return {name: name in profile for name in CREDENTIAL_NAMES}

    def get(self, profile_id: str, name: str) -> str | None:
        profile_id = validate_profile_id(profile_id)
        self._validate_names({name})
        with self._lock:
            document = self._load_unlocked()
            ciphertext = (document["profiles"].get(profile_id) or {}).get(name)
        if not ciphertext:
            return None
        try:
            protected = base64.b64decode(ciphertext, validate=True)
            plaintext = self._unprotect(protected)
            if not plaintext or len(plaintext) > MAX_SECRET_BYTES:
                raise SecureCredentialError("Saved API credential has an invalid size")
            return plaintext.decode("utf-8")
        except SecureCredentialError:
            raise
        except (ValueError, UnicodeError) as exc:
            raise SecureCredentialError("Saved API credential is damaged") from exc

    def status(self, profile_id: str) -> dict[str, bool]:
        """Return only validity booleans; never return ciphertext or plaintext."""
        profile_id = validate_profile_id(profile_id)
        result: dict[str, bool] = {}
        for name in CREDENTIAL_NAMES:
            try:
                result[name] = self.get(profile_id, name) is not None
            except SecureCredentialError:
                result[name] = False
        return result
