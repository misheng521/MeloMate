"""SQLite-backed reminder persistence shared by MCP tools and the chat runtime."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BACKEND_ROOT / "cache" / "reminders.sqlite3"
MAX_PENDING_REMINDERS = 100
MAX_MESSAGE_CHARS = 500
MAX_FUTURE_DAYS = 5 * 366
CLAIM_STALE_SECONDS = 5 * 60
_UTC_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.I)


def _db_path() -> Path:
    configured = os.getenv("MELOMATE_REMINDER_DB", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            persona TEXT NOT NULL,
            message TEXT NOT NULL,
            remind_at_utc TEXT NOT NULL,
            timezone_name TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            claim_token TEXT,
            claimed_at_utc TEXT,
            delivered_at_utc TEXT,
            cancelled_at_utc TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_reminders_due "
        "ON reminders(persona, status, remind_at_utc)"
    )
    return connection


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _persona(value: str) -> str:
    text = " ".join(str(value or "").split())[:128]
    if not text or any(ord(char) < 32 for char in text):
        raise ValueError("persona is required.")
    return text


def _message(value: str) -> str:
    text = " ".join(str(value or "").split())[:MAX_MESSAGE_CHARS]
    if not text:
        raise ValueError("reminder message is required.")
    return text


def resolve_timezone(value: str = "local") -> tuple[tzinfo, str]:
    requested = str(value or "local").strip()[:128]
    if requested.lower() in {"local", "system", "本地", "系统"}:
        offset = datetime.now().astimezone().utcoffset() or timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        absolute = abs(total_minutes)
        label = f"UTC{sign}{absolute // 60:02d}:{absolute % 60:02d}"
        return timezone(offset, label), label
    if requested.upper() in {"UTC", "Z"}:
        return timezone.utc, "UTC"
    match = _UTC_OFFSET.fullmatch(requested)
    if match:
        hours = int(match.group(2))
        minutes = int(match.group(3) or 0)
        if hours > 14 or minutes > 59 or (hours == 14 and minutes):
            raise ValueError("timezone UTC offset is out of range.")
        total = hours * 60 + minutes
        if match.group(1) == "-":
            total = -total
        label = f"UTC{match.group(1)}{hours:02d}:{minutes:02d}"
        return timezone(timedelta(minutes=total), label), label
    try:
        return ZoneInfo(requested), requested
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            "timezone must be local, UTC, a UTC offset, or an available IANA name."
        ) from exc


def current_time(timezone_name: str = "local") -> dict[str, Any]:
    zone, label = resolve_timezone(timezone_name)
    now = datetime.now(zone)
    return {
        "ok": True,
        "timezone": label,
        "iso8601": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "utc_offset": now.strftime("%z"),
        "unix_ms": int(now.timestamp() * 1000),
    }


def _parse_remind_at(value: str, timezone_name: str) -> tuple[datetime, str]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("remind_at is required as an ISO 8601 date-time.")
    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("remind_at must be an ISO 8601 date-time.") from exc
    zone, label = resolve_timezone(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    now = _utc_now()
    target = parsed.astimezone(timezone.utc)
    if target <= now:
        raise ValueError("remind_at must be in the future.")
    if target > now + timedelta(days=MAX_FUTURE_DAYS):
        raise ValueError("remind_at is too far in the future.")
    return target, label


def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    zone, label = resolve_timezone(str(row["timezone_name"]))
    target_utc = datetime.fromisoformat(str(row["remind_at_utc"]))
    return {
        "id": str(row["id"]),
        "persona": str(row["persona"]),
        "message": str(row["message"]),
        "remind_at": target_utc.astimezone(zone).isoformat(timespec="seconds"),
        "remind_at_utc": target_utc.astimezone(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "timezone": label,
        "status": str(row["status"]),
        "created_at_utc": str(row["created_at_utc"]),
    }


def create_reminder(
    persona: str,
    remind_at: str,
    message: str,
    timezone_name: str = "local",
) -> dict[str, Any]:
    owner = _persona(persona)
    clean_message = _message(message)
    target, label = _parse_remind_at(remind_at, timezone_name)
    reminder_id = uuid4().hex
    created = _utc_text(_utc_now())
    connection = _connect()
    try:
        pending = connection.execute(
            "SELECT COUNT(*) FROM reminders WHERE persona = ? AND status IN ('pending', 'delivering')",
            (owner,),
        ).fetchone()[0]
        if int(pending) >= MAX_PENDING_REMINDERS:
            raise ValueError("too many pending reminders for this persona.")
        connection.execute(
            """
            INSERT INTO reminders(
                id, persona, message, remind_at_utc, timezone_name,
                created_at_utc, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (reminder_id, owner, clean_message, _utc_text(target), label, created),
        )
        row = connection.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ).fetchone()
        connection.commit()
    finally:
        connection.close()
    result = _public_row(row)
    result.update(
        {
            "ok": True,
            "delivery": (
                "MeloMate will speak this reminder while the persona is connected; "
                "an overdue reminder is delivered on the next connection."
            ),
        }
    )
    return result


def list_reminders(persona: str, include_finished: bool = False) -> dict[str, Any]:
    owner = _persona(persona)
    query = "SELECT * FROM reminders WHERE persona = ?"
    params: list[Any] = [owner]
    if not include_finished:
        query += " AND status IN ('pending', 'delivering')"
    query += " ORDER BY remind_at_utc ASC LIMIT 200"
    connection = _connect()
    try:
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()
    return {
        "ok": True,
        "persona": owner,
        "reminders": [_public_row(row) for row in rows],
    }


def cancel_reminder(persona: str, reminder_id: str) -> dict[str, Any]:
    owner = _persona(persona)
    clean_id = str(reminder_id or "").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", clean_id):
        raise ValueError("reminder_id is invalid.")
    cancelled = _utc_text(_utc_now())
    connection = _connect()
    try:
        cursor = connection.execute(
            """
            UPDATE reminders
            SET status = 'cancelled', cancelled_at_utc = ?, claim_token = NULL,
                claimed_at_utc = NULL
            WHERE id = ? AND persona = ? AND status = 'pending'
            """,
            (cancelled, clean_id, owner),
        )
        changed = cursor.rowcount
        connection.commit()
    finally:
        connection.close()
    if changed != 1:
        return {
            "ok": False,
            "persona": owner,
            "id": clean_id,
            "message": "Pending reminder was not found or is already being delivered.",
        }
    return {"ok": True, "persona": owner, "id": clean_id, "status": "cancelled"}


def claim_due_reminders(persona: str, limit: int = 5) -> list[dict[str, Any]]:
    owner = _persona(persona)
    now = _utc_now()
    stale = _utc_text(now - timedelta(seconds=CLAIM_STALE_SECONDS))
    token = uuid4().hex
    bounded_limit = max(1, min(int(limit or 5), 20))
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE reminders
            SET status = 'pending', claim_token = NULL, claimed_at_utc = NULL
            WHERE status = 'delivering' AND claimed_at_utc < ?
            """,
            (stale,),
        )
        rows = connection.execute(
            """
            SELECT * FROM reminders
            WHERE persona = ? AND status = 'pending' AND remind_at_utc <= ?
            ORDER BY remind_at_utc ASC LIMIT ?
            """,
            (owner, _utc_text(now), bounded_limit),
        ).fetchall()
        ids = [str(row["id"]) for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f"UPDATE reminders SET status = 'delivering', claim_token = ?, "
                f"claimed_at_utc = ? WHERE id IN ({placeholders}) AND status = 'pending'",
                (token, _utc_text(now), *ids),
            )
        connection.commit()
    finally:
        connection.close()
    claimed = []
    for row in rows:
        item = _public_row(row)
        item["claim_token"] = token
        claimed.append(item)
    return claimed


def finish_delivery(reminder_id: str, claim_token: str, delivered: bool) -> bool:
    clean_id = str(reminder_id or "").strip()
    clean_token = str(claim_token or "").strip()
    if not clean_id or not clean_token:
        return False
    connection = _connect()
    try:
        if delivered:
            cursor = connection.execute(
                """
                UPDATE reminders SET status = 'delivered', delivered_at_utc = ?,
                    claim_token = NULL, claimed_at_utc = NULL
                WHERE id = ? AND status = 'delivering' AND claim_token = ?
                """,
                (_utc_text(_utc_now()), clean_id, clean_token),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE reminders SET status = 'pending', claim_token = NULL,
                    claimed_at_utc = NULL
                WHERE id = ? AND status = 'delivering' AND claim_token = ?
                """,
                (clean_id, clean_token),
            )
        changed = cursor.rowcount
        connection.commit()
    finally:
        connection.close()
    return changed == 1
