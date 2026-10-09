"""Complaint state and draft endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.agent.orchestrator import session_lock
from app.agent.state_manager import ComplaintDetail
from app.api.dependencies import SessionServiceDep
from app.integrations.llm_client import LLMConfigurationError, LLMError, LLMTimeoutError
from app.schemas.session import SessionInfo
from app.services import complaint_service
from app.services.complaint_service import (
    MAX_ALLEGATION_LENGTH,
    MAX_ALLEGATIONS,
    MAX_DETAIL_LENGTH,
    ComplaintSnapshot,
    ComplaintValidationError,
)
from app.services.draft_service import ComplaintDraft, NothingToDraftError, generate_draft
from app.services.session_service import (
    SessionData,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/complaints", tags=["complaints"])

_SESSION_ERRORS = {
    404: {"description": "Session not found."},
    410: {"description": "Session has expired."},
}

Allegation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ALLEGATION_LENGTH)
]


class DetailUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: str = Field(min_length=1, max_length=MAX_DETAIL_LENGTH)
    uncertain: bool = Field(default=False, description="True if the user is approximate or unsure.")


class ComplaintUpdateRequest(BaseModel):
    """Partial update. Omitted fields are preserved."""

    model_config = ConfigDict(extra="forbid")

    details: dict[str, DetailUpdate | None] | None = Field(
        default=None,
        description="Field name -> new value, or null to remove the field. Unknown field names are rejected.",
    )
    allegations: list[Allegation] | None = Field(
        default=None,
        max_length=MAX_ALLEGATIONS,
        description="If given, replaces the whole allegations list.",
    )


class ComplaintResponse(ComplaintSnapshot):
    session: SessionInfo


def _load_session(service: SessionService, session_id: str) -> SessionData:
    try:
        return service.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.") from None
    except SessionExpiredError:
        raise HTTPException(
            status.HTTP_410_GONE, "Session has expired. Please start a new one."
        ) from None
    except SessionError:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Session data could not be read."
        ) from None


def _response(
    service: SessionService, session: SessionData, snapshot: ComplaintSnapshot
) -> ComplaintResponse:
    return ComplaintResponse(session=service.to_info(session), **snapshot.model_dump())


@router.get("/{session_id}", response_model=ComplaintResponse, responses=_SESSION_ERRORS)
def get_complaint(session_id: str, service: SessionServiceDep) -> ComplaintResponse:
    """Current structured complaint, with missing and uncertain information listed explicitly."""
    session = _load_session(service, session_id)
    return _response(service, session, complaint_service.get_snapshot(session))


@router.patch(
    "/{session_id}",
    response_model=ComplaintResponse,
    responses={**_SESSION_ERRORS, 422: {"description": "Unknown field or invalid value."}},
)
async def patch_complaint(
    session_id: str, request: ComplaintUpdateRequest, service: SessionServiceDep
) -> ComplaintResponse:
    """Correct or add complaint details. Conversation history is not modified."""
    details = (
        {
            name: ComplaintDetail(value=d.value, uncertain=d.uncertain) if d is not None else None
            for name, d in request.details.items()
        }
        if request.details is not None
        else None
    )
    # Same lock the chat endpoint holds, so an edit cannot be overwritten by an in-flight chat turn.
    async with session_lock(session_id):
        session = _load_session(service, session_id)
        try:
            snapshot = complaint_service.update_complaint(
                session, service, details=details, allegations=request.allegations
            )
        except ComplaintValidationError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    return _response(service, session, snapshot)


@router.post(
    "/{session_id}/draft",
    response_model=ComplaintDraft,
    responses={
        **_SESSION_ERRORS,
        409: {"description": "No complaint details collected yet."},
        502: {"description": "The language model failed or returned an unusable reply."},
        503: {"description": "The language model is not configured."},
        504: {"description": "The language model timed out."},
    },
)
async def create_draft(session_id: str, service: SessionServiceDep) -> ComplaintDraft:
    """Generate a draft for review from the latest saved details. Nothing is filed or sent."""
    session = _load_session(service, session_id)
    try:
        return await generate_draft(session)
    except NothingToDraftError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No complaint details have been collected yet. Describe the incident in chat first.",
        ) from None
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The language model is not configured."
        ) from None
    except LLMTimeoutError as exc:
        logger.error("LLM timeout: %s", exc)
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "The language model timed out. Please try again."
        ) from None
    except LLMError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The draft could not be generated. Please try again."
        ) from None