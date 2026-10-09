"""Business logic for reading and editing the structured complaint stored in a session."""

import logging

from pydantic import BaseModel

from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
    ComplaintDetail,
    ComplaintState,
    get_complaint_state,
    missing_required_fields,
    set_complaint_state,
)
from app.services.session_service import SessionData, SessionService

logger = logging.getLogger(__name__)

MAX_DETAIL_LENGTH = 2000
MAX_ALLEGATIONS = 50
MAX_ALLEGATION_LENGTH = 1000


class ComplaintValidationError(ValueError):
    """A requested complaint edit is invalid. Nothing was changed."""


class ComplaintSnapshot(BaseModel):
    """The complaint as the user should review it. Everything is user-reported and unverified;
    missing information is listed explicitly rather than filled in."""

    complaint: ComplaintState
    missing_required_fields: list[str]
    missing_optional_fields: list[str]
    uncertain_fields: list[str]
    required_details_collected: bool


def build_snapshot(state: ComplaintState) -> ComplaintSnapshot:
    missing_required = missing_required_fields(state)
    return ComplaintSnapshot(
        complaint=state,
        missing_required_fields=missing_required,
        missing_optional_fields=[
            name
            for name in FIELD_DESCRIPTIONS
            if name not in REQUIRED_FIELDS and name not in state.details
        ],
        uncertain_fields=[name for name, detail in state.details.items() if detail.uncertain],
        required_details_collected=not missing_required,
    )


def get_snapshot(session: SessionData) -> ComplaintSnapshot:
    return build_snapshot(get_complaint_state(session))


def update_complaint(
    session: SessionData,
    service: SessionService,
    details: dict[str, ComplaintDetail | None] | None = None,
    allegations: list[str] | None = None,
) -> ComplaintSnapshot:
    """Apply a partial edit and persist it.

    - details: field name -> new value; None removes that field. Omitted fields are untouched.
    - allegations: if given, replaces the whole list; if None, the list is untouched.
    Everything is validated before anything is changed. Conversation messages are never touched.
    """
    details = details or {}
    if not details and allegations is None:
        raise ComplaintValidationError("No changes supplied.")

    unknown = sorted(set(details) - set(FIELD_DESCRIPTIONS))
    if unknown:
        raise ComplaintValidationError(
            f"Unknown complaint field(s): {', '.join(name[:40] for name in unknown)}. "
            f"Allowed fields: {', '.join(FIELD_DESCRIPTIONS)}."
        )

    cleaned_details: dict[str, ComplaintDetail | None] = {}
    for name, detail in details.items():
        if detail is None:
            cleaned_details[name] = None
            continue
        value = detail.value.strip()
        if not value:
            raise ComplaintValidationError(f"Field '{name}' must not be empty; use null to remove it.")
        if len(value) > MAX_DETAIL_LENGTH:
            raise ComplaintValidationError(f"Field '{name}' exceeds {MAX_DETAIL_LENGTH} characters.")
        cleaned_details[name] = ComplaintDetail(value=value, uncertain=detail.uncertain)

    cleaned_allegations: list[str] | None = None
    if allegations is not None:
        if len(allegations) > MAX_ALLEGATIONS:
            raise ComplaintValidationError(f"At most {MAX_ALLEGATIONS} allegations are allowed.")
        cleaned_allegations = []
        seen: set[str] = set()
        for text in allegations:
            text = text.strip()
            if not text or len(text) > MAX_ALLEGATION_LENGTH:
                raise ComplaintValidationError(
                    f"Each allegation must be 1-{MAX_ALLEGATION_LENGTH} characters."
                )
            if text.casefold() not in seen:
                seen.add(text.casefold())
                cleaned_allegations.append(text)

    state = get_complaint_state(session)
    for name, detail in cleaned_details.items():
        if detail is None:
            state.details.pop(name, None)
        else:
            state.details[name] = detail
    if cleaned_allegations is not None:
        state.allegations = cleaned_allegations

    set_complaint_state(session, state)
    service.save(session)  # only session.complaint changed; messages are preserved
    logger.info(
        "Updated complaint for session %.8s (fields: %s, allegations replaced: %s)",
        session.session_id,
        ", ".join(sorted(cleaned_details)) or "none",
        cleaned_allegations is not None,
    )
    return build_snapshot(state)