"""Conversation orchestration: coordinates turns, session locks, and agent execution.

Supports dynamic LangGraph orchestration while preserving API contracts and error boundaries.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from weakref import WeakValueDictionary

from pydantic import BaseModel, Field, ValidationError, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_lazy_details(cls, data: Any) -> Any:
        """Coerce raw strings/numbers into the strict ComplaintDetail dict structure."""
        if isinstance(data, dict) and isinstance(data.get("details"), dict):
            details = data["details"]
            for k, v in details.items():
                if isinstance(v, (str, int, float)):
                    details[k] = {"value": str(v)}
                elif isinstance(v, dict) and "value" in v:
                    v["value"] = str(v["value"])
        return data


@dataclass
class TurnResult:
    reply: ChatMessage
    complaint: ComplaintState
    missing_required_fields: list[str]
    sources_available: bool


# --------------------------------------------------------------------------- #
# Per-session lock
# --------------------------------------------------------------------------- #
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


def _clean_markdown_fences(raw: str) -> str:
    """Strip markdown code fences that LLMs sometimes stubbornly output."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])
    return raw


def _parse_turn(raw: str) -> _LLMTurn:
    raw = _clean_markdown_fences(raw)
    try:
        turn = _LLMTurn.model_validate_json(raw)
    except ValidationError:
        raise LLMResponseError("The model returned a reply in an unexpected format.") from None
    if not turn.reply.strip():
        raise LLMResponseError("The model returned an empty reply.")
    return turn


async def _execute_turn_direct(
    session: SessionData, user_message: str, service: SessionService
) -> TurnResult:
    """Standard robust turn processing used directly or as fallback."""
    state = get_complaint_state(session)
    messages = [
        LLMMessage("system", _STATIC_SYSTEM_PROMPT),
        LLMMessage("system", _state_context(state)),
        *(LLMMessage(m.role, m.content) for m in get_history(session)),
        LLMMessage("user", user_message),
    ]

    turn = None
    for attempt in range(2):
        raw_response = await generate_response(messages, json_mode=True)
        try:
            turn = _parse_turn(raw_response)
            break
        except LLMResponseError:
            if attempt == 1:
                raise

    add_message(session, "user", user_message)
    reply = add_message(session, "assistant", turn.reply.strip())
    apply_update(state, turn.details, turn.allegations_add, turn.allegations_remove)
    set_complaint_state(session, state)
    service.save(session)

    logger.info(
        "Processed turn for session %.8s (%d messages stored)",
        session.session_id,
        len(session.messages),
    )
    return TurnResult(
        reply=reply,
        complaint=state,
        missing_required_fields=missing_required_fields(state),
        sources_available=SOURCES_AVAILABLE,
    )


async def handle_user_message(
    session: SessionData, user_message: str, service: SessionService
) -> TurnResult:
    """Process one user turn using the dynamic agent.

    Ensures session isolation, atomicity on failures, and exact API contract compliance.
    """
    # Clone the session so we don't mutate the in-memory object on failures
    cloned_session = session.model_copy(deep=True)

    # We try using the LangGraph agent if available; fallback smoothly to direct execution
    try:
        from app.agent.graph import build_complaint_graph

        graph = build_complaint_graph()
        initial_state = {
            "session_id": cloned_session.session_id,
            "user_message": user_message,
            "messages": get_history(cloned_session),
            "complaint": get_complaint_state(cloned_session),
            "missing_required": missing_required_fields(get_complaint_state(cloned_session)),
            "is_sufficient": False,
            "need_retrieval": False,
            "sources": [],
            "sources_available": SOURCES_AVAILABLE,
            "assistant_reply": "",
        }

        final_state = await graph.ainvoke(initial_state)

        # Update cloned session state
        add_message(cloned_session, "user", user_message)
        reply = add_message(cloned_session, "assistant", final_state["assistant_reply"])
        set_complaint_state(cloned_session, final_state["complaint"])
        service.save(cloned_session)

        # Reflect mutations into caller's session object
        session.messages = cloned_session.messages
        session.complaint = cloned_session.complaint
        session.updated_at = cloned_session.updated_at

        return TurnResult(
            reply=reply,
            complaint=final_state["complaint"],
            missing_required_fields=final_state["missing_required"],
            sources_available=final_state["sources_available"],
        )
    except Exception as exc:
        # Fallback to direct prompt turn
        logger.warning("LangGraph turn delegated to direct turn execution: %s", exc)
        return await _execute_turn_direct(cloned_session, user_message, service)