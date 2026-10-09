"""Chat endpoint."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.agent.orchestrator import handle_user_message, session_lock
from app.agent.state_manager import ComplaintState
from app.api.dependencies import SessionServiceDep
from app.integrations.llm_client import (
    LLMConfigurationError,
    LLMError,
    LLMTimeoutError,
)
from app.schemas.chat import ChatMessage
from app.schemas.session import SessionInfo
from app.services.session_service import SessionExpiredError, SessionNotFoundError, SessionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=5000, description="The user's message.")


class ChatResponse(BaseModel):
    session: SessionInfo
    reply: ChatMessage
    complaint: ComplaintState = Field(description="Details collected so far (user-reported, unverified).")
    missing_required_fields: list[str]
    required_details_collected: bool
    sources_available: bool = Field(
        description="False while knowledge retrieval is not configured; no sources were consulted."
    )


@router.post(
    "/{session_id}",
    response_model=ChatResponse,
    responses={
        404: {"description": "Session not found."},
        410: {"description": "Session has expired."},
        502: {"description": "The language model failed or returned an unusable reply."},
        503: {"description": "The language model is not configured."},
        504: {"description": "The language model timed out."},
    },
)
async def chat(session_id: str, request: ChatRequest, service: SessionServiceDep) -> ChatResponse:
    """Send a message in an existing session. On any error, nothing is saved and the user can resend."""
    async with session_lock(session_id):
        try:
            session = service.get(session_id)
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

        try:
            result = await handle_user_message(session, request.message, service)
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
                status.HTTP_502_BAD_GATEWAY,
                "The language model could not produce a reply. Please try again.",
            ) from None

    return ChatResponse(
        session=service.to_info(session),
        reply=result.reply,
        complaint=result.complaint,
        missing_required_fields=result.missing_required_fields,
        required_details_collected=not result.missing_required_fields,
        sources_available=result.sources_available,
    )