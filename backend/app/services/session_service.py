"""Session persistence: one JSON file per session under the configured sessions directory."""

import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.schemas.chat import ChatMessage
from app.schemas.session import SessionInfo
from app.utils.identifiers import generate_session_id

logger = logging.getLogger(__name__)

# Matches URL-safe tokens only (no dots or slashes), so an ID can never escape the directory.
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{20,128}")


class SessionError(Exception):
    """Base class for session storage errors."""


class SessionNotFoundError(SessionError):
    """No session exists for the given ID (including malformed IDs)."""


class SessionExpiredError(SessionError):
    """The session existed but has expired; its file has been removed."""


class SessionCorruptError(SessionError):
    """The session file exists but could not be parsed."""


class SessionData(BaseModel):
    """Internal stored representation of a session. Not an API schema."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage] = Field(default_factory=list)
    # Placeholder until schemas/complaint.py exists; will become a typed complaint model.
    complaint: dict[str, Any] | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _short(session_id: str) -> str:
    # The full ID is the access secret, so logs only ever carry a prefix.
    return session_id[:8]


class SessionService:
    def __init__(self, sessions_dir: Path, expiry_hours: int) -> None:
        self._dir = sessions_dir
        self._expiry = timedelta(hours=expiry_hours)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def create(self) -> SessionData:
        now = _now()
        session = SessionData(session_id=generate_session_id(), created_at=now, updated_at=now)
        self._write(session)
        logger.info("Created session %s", _short(session.session_id))
        return session

    def get(self, session_id: str) -> SessionData:
        """Load a session. Raises SessionNotFoundError, SessionExpiredError, or SessionCorruptError."""
        path = self._path(session_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise SessionNotFoundError("Session not found.") from None

        try:
            session = SessionData.model_validate_json(raw)
        except ValidationError:
            # No exception chaining: validation errors can echo stored complaint text.
            logger.error("Session file for %s is malformed", _short(session_id))
            raise SessionCorruptError("Stored session data is unreadable.") from None
        if session.session_id != session_id:
            logger.error("Session file for %s has a mismatched ID", _short(session_id))
            raise SessionCorruptError("Stored session data is unreadable.")

        if _now() >= self.expires_at(session):
            path.unlink(missing_ok=True)
            logger.info("Removed expired session %s", _short(session_id))
            raise SessionExpiredError("Session has expired.")
        return session

    def save(self, session: SessionData) -> SessionData:
        """Persist changes (refreshing updated_at, which also extends the expiry window)."""
        if not self._path(session.session_id).exists():
            raise SessionNotFoundError("Session not found.")
        
        session.updated_at = _now()
        self._write(session)
        return session

    def delete(self, session_id: str) -> None:
        """Delete a session's file. Works for expired or corrupt sessions too."""
        try:
            self._path(session_id).unlink()
        except FileNotFoundError:
            raise SessionNotFoundError("Session not found.") from None
        logger.info("Deleted session %s", _short(session_id))

    def expires_at(self, session: SessionData) -> datetime:
        """Sessions expire after a period of inactivity, measured from the last save."""
        return session.updated_at + self._expiry

    def to_info(self, session: SessionData) -> SessionInfo:
        return SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=self.expires_at(session),
        )

    def purge_expired(self) -> int:
        """Delete all expired session files and return the number removed."""
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                # get() automatically unlinks the file and raises SessionExpiredError if expired
                self.get(path.stem)
            except SessionExpiredError:
                removed += 1
            except SessionError:
                # Ignore corrupt or not-found files during the purge
                pass
        return removed

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise SessionNotFoundError("Session not found.")
        return self._dir / f"{session_id}.json"

    def _write(self, session: SessionData) -> None:
        """Write to a temp file in the same directory, then atomically replace the target."""
        target = self._path(session.session_id)
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(session.model_dump_json(indent=2))
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, target)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise