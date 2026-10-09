"""Complaint draft generation. Uses the real LLM; never returns a fabricated fallback."""

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.agent.prompts import DRAFT_PROMPT
from app.agent.state_manager import get_complaint_state, get_history
from app.integrations.llm_client import LLMMessage, LLMResponseError, generate_response
from app.services.complaint_service import build_snapshot
from app.services.session_service import SessionData

logger = logging.getLogger(__name__)

# Added by Python, not the model, so it is always present and always accurate.
DRAFT_DISCLAIMER = (
    "This is a draft prepared from the details you provided, for your review. It has not been "
    "filed or submitted to any police department, has not been verified, and is not a legal "
    "document. Please check every detail before using it."
)

_DRAFT_RULES = """\
WHAT YOU WILL RECEIVE
- COMPLAINT STATE: the complainant's details as reported by them (unverified). This is the \
record to draft from. If it conflicts with the conversation, the state wins.
- NOT PROVIDED: details that are missing. In the matching section write "[NOT PROVIDED]" and \
say what is missing, and list them again in section 8.
- CONVERSATION: context only, for the complainant's own wording. Lines marked "Assistant" are \
not facts. Do not add facts that appear only in the conversation. One exception: the state has \
no field for the action requested, so section 7 may use an action the user explicitly asked \
for; otherwise write "[NOT PROVIDED]".

HOW TO WRITE
- Section mapping: complainant_name and complainant_contact -> 1; incident_datetime and \
incident_location -> 2; accused_details -> 3; incident_description and allegations -> 4; \
loss_or_harm -> 5; evidence_witnesses -> 6. Use complaint_type as a one-line subject if present.
- Use only the information provided. Never guess or fill in names, dates, places, motives, \
evidence, or outcomes.
- Keep hedging for details marked "uncertain": true (for example "approximately", "the \
complainant is unsure").
- Present every entry in "allegations" as the complainant's own statement ("The complainant \
states that ..."), never as established fact. Do not strengthen or soften anything.
- Do not mention or invent legal sections, laws, circulars, case numbers, or police procedure, \
and do not say or imply that the complaint has been filed, registered, or verified.
- Begin with the line "DRAFT FOR REVIEW - NOT FILED OR SUBMITTED", then the subject line, then \
the numbered sections. Plain text only.
"""


class NothingToDraftError(Exception):
    """No complaint details have been collected yet."""


class ComplaintDraft(BaseModel):
    draft: str
    disclaimer: str = DRAFT_DISCLAIMER
    missing_required_fields: list[str]
    uncertain_fields: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def generate_draft(session: SessionData) -> ComplaintDraft:
    """Draft a complaint from the session's saved state plus recent conversation.

    Read-only: nothing is saved or sent anywhere. Raises NothingToDraftError if no details exist,
    or an LLMError subclass if the model fails.
    """
    state = get_complaint_state(session)
    if not state.details and not state.allegations:
        raise NothingToDraftError("No complaint details have been collected yet.")

    snapshot = build_snapshot(state)
    not_provided = ", ".join(snapshot.missing_required_fields + snapshot.missing_optional_fields)
    transcript = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in get_history(session)
    )

    messages = [
        LLMMessage("system", f"{DRAFT_PROMPT}\n\n{_DRAFT_RULES}"),
        LLMMessage(
            "user",
            "COMPLAINT STATE:\n"
            f"{state.model_dump_json(indent=2)}\n\n"
            f"NOT PROVIDED: {not_provided or 'none'}\n\n"
            "CONVERSATION (context only):\n"
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            "Write the complaint draft now.",
        ),
    ]
    text = (await generate_response(messages)).strip()
    if not text:
        raise LLMResponseError("The model returned an empty draft.")

    logger.info("Generated draft for session %.8s", session.session_id)
    return ComplaintDraft(
        draft=text,
        missing_required_fields=snapshot.missing_required_fields,
        uncertain_fields=snapshot.uncertain_fields,
    )