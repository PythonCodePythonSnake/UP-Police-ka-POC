"""API schemas for session management."""

from datetime import datetime

from pydantic import BaseModel, Field


class SessionInfo(BaseModel):
    """Public metadata about a session. Used for session creation and lookup responses."""

    session_id: str = Field(description="Unique identifier for the conversation session.")
    created_at: datetime = Field(description="When the session was created (UTC).")
    updated_at: datetime = Field(description="When the session was last modified (UTC).")
    expires_at: datetime = Field(description="When the session expires if left inactive (UTC).")


class SessionCreateResponse(SessionInfo):
    """Response returned when a new session is started."""