"""Session management endpoints."""

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import SessionServiceDep
from app.schemas.chat import SessionHistoryResponse
from app.schemas.session import SessionCreateResponse
from app.services.session_service import (
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_ERROR_RESPONSES = {
    404: {"description": "Session not found."},
    410: {"description": "Session has expired."},
}


def _to_http_error(exc: SessionError) -> HTTPException:
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")
    if isinstance(exc, SessionExpiredError):
        return HTTPException(status.HTTP_410_GONE, "Session has expired. Please start a new one.")
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Session data could not be read.")


@router.post("", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
def create_session(service: SessionServiceDep) -> SessionCreateResponse:
    """Start a new conversation session."""
    session = service.create()
    return SessionCreateResponse(**service.to_info(session).model_dump())


@router.get("/{session_id}", response_model=SessionHistoryResponse, responses=_ERROR_RESPONSES)
def get_session(session_id: str, service: SessionServiceDep) -> SessionHistoryResponse:
    """Return session metadata and message history so a conversation can be restored."""
    try:
        session = service.get(session_id)
    except SessionError as exc:
        raise _to_http_error(exc) from None
    return SessionHistoryResponse(**service.to_info(session).model_dump(), messages=session.messages)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, responses={404: _ERROR_RESPONSES[404]})
def delete_session(session_id: str, service: SessionServiceDep) -> None:
    """Delete a session and all its stored data."""
    try:
        service.delete(session_id)
    except SessionError as exc:
        raise _to_http_error(exc) from None