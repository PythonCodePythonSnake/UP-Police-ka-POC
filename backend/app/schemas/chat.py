"""Chat message schemas shared by session storage and API responses."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.session import SessionInfo


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    """One turn of the stored conversation."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=_utcnow, description="When the message was added (UTC).")


class SessionHistoryResponse(SessionInfo):
    """Session metadata plus the full message history, for restoring a conversation."""

    messages: list[ChatMessage]