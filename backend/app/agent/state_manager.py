"""Helpers for conversation history and structured complaint state stored in a session."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatMessage
from app.services.session_service import SessionData

logger = logging.getLogger(__name__)

# Older turns are dropped from the LLM prompt; facts from them survive in the complaint state.
MAX_HISTORY_MESSAGES = 20

FIELD_DESCRIPTIONS: dict[str, str] = {
    "complainant_name": "Name of the person making the complaint",
    "complainant_contact": "Phone number or other contact details of the complainant",
    "complaint_type": "Short category in the user's own terms (e.g. theft, harassment, online fraud)",
    "incident_description": "What happened, as the user describes it",
    "incident_datetime": "When it happened, as the user gave it",
    "incident_location": "Where it happened",
    "accused_details": "Who the complaint is against, or identifying details, if known",
    "loss_or_harm": "Injury, damage, or financial loss suffered",
    "evidence_witnesses": "Evidence (messages, photos, receipts) or witnesses the user mentions",
}

# Which details count as "required" is a product decision; edit this tuple to change it.
REQUIRED_FIELDS: tuple[str, ...] = (
    "complainant_name",
    "complainant_contact",
    "incident_description",
    "incident_datetime",
    "incident_location",
)


class ComplaintDetail(BaseModel):
    value: str
    uncertain: bool = False  # the user was approximate or unsure


class ComplaintState(BaseModel):
    """Everything here is user-reported and unverified. Nothing is confirmed until a review step."""

    details: dict[str, ComplaintDetail] = Field(default_factory=dict)
    # Claims about other people's conduct, kept separate from neutral details.
    allegations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Conversation history
# --------------------------------------------------------------------------- #
def add_message(
    session: SessionData, role: Literal["user", "assistant"], content: str
) -> ChatMessage:
    message = ChatMessage(role=role, content=content)
    session.messages.append(message)
    return message


def get_history(session: SessionData, limit: int = MAX_HISTORY_MESSAGES) -> list[ChatMessage]:
    """Most recent messages, always starting with a user message (some providers require it)."""
    history = session.messages[-limit:]
    while history and history[0].role != "user":
        history = history[1:]
    return history


# --------------------------------------------------------------------------- #
# Complaint state (stored in session.complaint as a plain dict)
# --------------------------------------------------------------------------- #
def get_complaint_state(session: SessionData) -> ComplaintState:
    if not session.complaint:
        return ComplaintState()
    return ComplaintState.model_validate(session.complaint)


def set_complaint_state(session: SessionData, state: ComplaintState) -> None:
    session.complaint = state.model_dump()


def missing_required_fields(state: ComplaintState) -> list[str]:
    return [name for name in REQUIRED_FIELDS if name not in state.details]


def apply_update(
    state: ComplaintState,
    details: dict[str, ComplaintDetail | None],
    allegations_add: list[str],
    allegations_remove: list[str],
) -> None:
    """Merge one turn's changes into the state. A None detail removes that field."""
    for name, detail in details.items():
        if name not in FIELD_DESCRIPTIONS:
            logger.warning("Ignoring unknown complaint field from model: %.40s", name)
            continue
        if detail is None:
            state.details.pop(name, None)
            continue
        value = detail.value.strip()
        if value:
            state.details[name] = ComplaintDetail(value=value, uncertain=detail.uncertain)

    # Removals first, so a correction (remove old + add new) works in one turn.
    removed = {text.strip().casefold() for text in allegations_remove}
    state.allegations = [a for a in state.allegations if a.casefold() not in removed]
    existing = {a.casefold() for a in state.allegations}
    for text in allegations_add:
        text = text.strip()
        if text and text.casefold() not in existing:
            state.allegations.append(text)
            existing.add(text.casefold())