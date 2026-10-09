"""Conversation orchestration: one LLM call per user turn; state and persistence handled in Python."""

import asyncio
import logging
from dataclasses import dataclass
from weakref import WeakValueDictionary

from pydantic import BaseModel, Field, ValidationError

from app.agent.prompts import GUIDANCE_PROMPT, INTAKE_PROMPT, SOURCES_UNAVAILABLE_NOTE
from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
    ComplaintDetail,
    ComplaintState,
    add_message,
    apply_update,
    get_complaint_state,
    get_history,
    missing_required_fields,
    set_complaint_state,
)
from app.integrations.llm_client import LLMMessage, LLMResponseError, generate_response
from app.schemas.chat import ChatMessage
from app.services.session_service import SessionData, SessionService

logger = logging.getLogger(__name__)

# Flip when retrieval_service is wired in. Retrieval is not implemented, so the model is told
# that no sources exist and must not cite any.
SOURCES_AVAILABLE = False

# This instruction block belongs with the other prompts and can move to prompts.py later.
_TURN_INSTRUCTIONS = """\
HOW TO RESPOND TO EACH MESSAGE
Reply with ONE JSON object and nothing else (no markdown fences), with exactly these keys:
{
  "reply": "<the message shown to the user>",
  "details": {"<field_name>": {"value": "<text>", "uncertain": false}},
  "allegations_add": ["<text>"],
  "allegations_remove": ["<text>"]
}

Complaint fields you may record in "details":
__FIELDS__

Rules for recording information:
- Record only what the user said in their latest message, in short neutral wording. Never add, \
guess, or infer facts, and never record your own suggestions or assumptions.
- "details" lists only fields that are new, corrected, or withdrawn in this message. Set a \
field to null to remove it when the user withdraws it. Use {} if nothing changed.
- Set "uncertain" to true when the user is approximate or unsure (for example "around 9 pm", \
"I think").
- "allegations_add": claims the user makes about another person's wrongdoing, written as \
"The complainant states that ...". "allegations_remove": exact existing entries the user \
withdraws or corrects. Use [] if none.
- If the user corrects earlier information, update the field and briefly acknowledge the change.

Rules for the reply:
- Write natural, plain conversational text. No JSON, markdown, or long lists of questions.
- Ask at most two or three questions, most important first, and only about required details \
still missing. Never ask for anything already in the complaint state.
- If the message is a greeting or unrelated to filing a complaint, answer briefly and politely, \
then invite the user to describe what happened. Record nothing.
- If the user asks about procedures, circulars, or legal provisions, say that official sources \
are not yet connected so you cannot cite any, and label any general comments as general \
information only.
- When no required details are missing, briefly summarise what has been recorded, make clear it \
is as reported by the user and has not been verified or submitted, and ask whether anything \
needs correcting or adding.
"""

_FIELDS_TEXT = "\n".join(
    f"- {name}{' (required)' if name in REQUIRED_FIELDS else ''}: {description}"
    for name, description in FIELD_DESCRIPTIONS.items()
)

_STATIC_SYSTEM_PROMPT = "\n\n".join(
    [
        INTAKE_PROMPT,
        GUIDANCE_PROMPT,
        SOURCES_UNAVAILABLE_NOTE,
        _TURN_INSTRUCTIONS.replace("__FIELDS__", _FIELDS_TEXT),
    ]
)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
class _LLMTurn(BaseModel):
    """The JSON object the model must return each turn."""

    reply: str
    details: dict[str, ComplaintDetail | None] = Field(default_factory=dict)
    allegations_add: list[str] = Field(default_factory=list)
    allegations_remove: list[str] = Field(default_factory=list)


@dataclass
class TurnResult:
    reply: ChatMessage
    complaint: ComplaintState
    missing_required_fields: list[str]
    sources_available: bool


# --------------------------------------------------------------------------- #
# Per-session lock
# --------------------------------------------------------------------------- #
# Prevents two simultaneous messages (for example a double-submit) on the same session from
# overwriting each other's saved state. Entries disappear once no request holds them.
_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def session_lock(session_id: str) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session_id] = lock
    return lock


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
def _state_context(state: ComplaintState) -> str:
    missing = ", ".join(missing_required_fields(state)) or "none"
    return (
        "CURRENT COMPLAINT STATE (reported by the user, not verified):\n"
        f"{state.model_dump_json(indent=2)}\n\n"
        f"Required details still missing: {missing}"
    )


def _parse_turn(raw: str) -> _LLMTurn:
    try:
        turn = _LLMTurn.model_validate_json(raw)
    except ValidationError:
        # No chaining: validation errors can echo model output containing complaint text.
        raise LLMResponseError("The model returned a reply in an unexpected format.") from None
    if not turn.reply.strip():
        raise LLMResponseError("The model returned an empty reply.")
    return turn


async def handle_user_message(
    session: SessionData, user_message: str, service: SessionService
) -> TurnResult:
    """Process one user turn.

    The session is only modified and saved after the LLM call succeeds, so a failed call
    leaves the stored conversation untouched and the user can safely resend the message.
    Callers should hold session_lock(session_id) around loading the session and calling this.
    """
    state = get_complaint_state(session)

    messages = [
        LLMMessage("system", _STATIC_SYSTEM_PROMPT),
        LLMMessage("system", _state_context(state)),
        *(LLMMessage(m.role, m.content) for m in get_history(session)),
        LLMMessage("user", user_message),
    ]
    turn = _parse_turn(await generate_response(messages, json_mode=True))

    add_message(session, "user", user_message)
    reply = add_message(session, "assistant", turn.reply.strip())
    apply_update(state, turn.details, turn.allegations_add, turn.allegations_remove)
    set_complaint_state(session, state)
    service.save(session)

    logger.info(
        "Processed turn for session %.8s (%d messages stored)", session.session_id, len(session.messages)
    )
    return TurnResult(
        reply=reply,
        complaint=state,
        missing_required_fields=missing_required_fields(state),
        sources_available=SOURCES_AVAILABLE,
    )