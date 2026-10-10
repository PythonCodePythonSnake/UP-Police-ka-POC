"""Tests for services/draft_service.py. The LLM is replaced by the fake_llm fixture."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.prompts import DRAFT_PROMPT
from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
    ComplaintDetail,
    ComplaintState,
    add_message,
    get_complaint_state,
    set_complaint_state,
)
from app.integrations.llm_client import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.services import draft_service
from app.services.complaint_service import update_complaint
from app.services.draft_service import (
    DRAFT_DISCLAIMER,
    ComplaintDraft,
    NothingToDraftError,
    generate_draft,
)

NARRATIVE = "SECRET-NARRATIVE-TEXT"
OPTIONAL_FIELDS = [name for name in FIELD_DESCRIPTIONS if name not in REQUIRED_FIELDS]


def detail(value, uncertain=False) -> ComplaintDetail:
    return ComplaintDetail(value=value, uncertain=uncertain)


def seed(service, session, details=None, allegations=None, messages=()):
    set_complaint_state(
        session, ComplaintState(details=details or {}, allegations=allegations or [])
    )
    for role, content in messages:
        add_message(session, role, content)
    service.save(session)
    return service.get(session.session_id)


def draft_for(session) -> ComplaintDraft:
    return asyncio.run(generate_draft(session))


def user_prompt(fake_llm) -> str:
    return fake_llm.calls[0].messages[1].content


def state_part(prompt: str) -> dict:
    body = prompt.split("COMPLAINT STATE:\n", 1)[1].split("\n\nNOT PROVIDED:", 1)[0]
    return json.loads(body)


def transcript_lines(prompt: str) -> list[str]:
    body = prompt.split("<conversation>\n", 1)[1].split("\n</conversation>", 1)[0]
    return body.splitlines()


def norm(text: str) -> str:
    return " ".join(text.lower().split())


@pytest.fixture
def seeded(session_service):
    session = session_service.create()
    return seed(
        session_service,
        session,
        details={
            "complainant_name": detail("Asha Verma"),
            "incident_description": detail("My phone was snatched."),
            "incident_location": detail("Kanpur Central station", uncertain=True),
        },
        allegations=["The complainant states that two men took the phone."],
        messages=[
            ("user", "My phone was snatched at Kanpur Central."),
            ("assistant", "Do you remember the time?"),
            ("user", "Around 9 pm, I think."),
        ],
    )


# --------------------------------------------------------------------------- #
# When there is nothing to draft
# --------------------------------------------------------------------------- #
def test_a_new_session_cannot_be_drafted_and_the_llm_is_not_called(session_service, fake_llm):
    session = session_service.create()
    with pytest.raises(NothingToDraftError):
        draft_for(session)
    assert fake_llm.calls == []


def test_an_empty_complaint_state_cannot_be_drafted(session_service, fake_llm):
    session = seed(session_service, session_service.create())
    with pytest.raises(NothingToDraftError):
        draft_for(session)
    assert fake_llm.calls == []


def test_conversation_alone_is_not_enough_to_draft(session_service, fake_llm):
    session = seed(
        session_service,
        session_service.create(),
        messages=[("user", "hi"), ("assistant", "Hello, what happened?")],
    )
    with pytest.raises(NothingToDraftError):
        draft_for(session)
    assert fake_llm.calls == []


def test_allegations_alone_are_enough_to_draft(session_service, fake_llm):
    session = seed(
        session_service,
        session_service.create(),
        allegations=["The complainant states that a neighbour threatened them."],
    )
    fake_llm.queue("DRAFT FOR REVIEW - NOT FILED OR SUBMITTED")
    draft_for(session)
    assert "neighbour threatened" in user_prompt(fake_llm)


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #
def test_the_draft_is_the_models_text_stripped(seeded, fake_llm):
    fake_llm.queue("  DRAFT FOR REVIEW - NOT FILED OR SUBMITTED\n1. Complainant details...  \n")
    result = draft_for(seeded)
    assert result.draft == "DRAFT FOR REVIEW - NOT FILED OR SUBMITTED\n1. Complainant details..."


def test_the_disclaimer_is_always_attached_by_python(seeded, fake_llm):
    fake_llm.queue("A draft that says nothing about being unfiled.")
    result = draft_for(seeded)

    assert result.disclaimer == DRAFT_DISCLAIMER
    text = norm(result.disclaimer)
    assert "has not been filed or submitted" in text
    assert "not a legal document" in text


def test_missing_and_uncertain_fields_are_computed_by_python(seeded, fake_llm):
    fake_llm.queue("A draft.")
    result = draft_for(seeded)

    assert result.missing_required_fields == [
        name for name in REQUIRED_FIELDS if name not in get_complaint_state(seeded).details
    ]
    assert result.uncertain_fields == ["incident_location"]


def test_generated_at_is_timezone_aware_and_current(seeded, fake_llm):
    fake_llm.queue("A draft.")
    result = draft_for(seeded)
    assert result.generated_at.tzinfo is not None
    assert abs(datetime.now(timezone.utc) - result.generated_at) < timedelta(seconds=5)


def test_the_result_survives_a_json_roundtrip(seeded, fake_llm):
    fake_llm.queue("A draft.")
    result = draft_for(seeded)
    assert ComplaintDraft.model_validate_json(result.model_dump_json()) == result


# --------------------------------------------------------------------------- #
# What the LLM receives
# --------------------------------------------------------------------------- #
def test_one_plain_text_llm_call_with_the_drafting_prompt(seeded, fake_llm):
    fake_llm.queue("A draft.")
    draft_for(seeded)

    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call.json_mode is False
    assert [m.role for m in call.messages] == ["system", "user"]
    assert call.messages[0].content == f"{DRAFT_PROMPT}\n\n{draft_service._DRAFT_RULES}"
    assert call.messages[1].content.endswith("Write the complaint draft now.")


@pytest.mark.parametrize(
    "phrase",
    [
        "the state wins",
        "are not facts",
        "do not mention or invent legal sections",
        "draft for review - not filed or submitted",
        "[not provided]",
        "uncertain",
    ],
)
def test_the_system_prompt_carries_the_drafting_guardrails(seeded, fake_llm, phrase):
    fake_llm.queue("A draft.")
    draft_for(seeded)
    assert phrase in norm(fake_llm.calls[0].messages[0].content)


def test_the_prompt_contains_exactly_the_saved_state(seeded, fake_llm):
    fake_llm.queue("A draft.")
    draft_for(seeded)
    assert state_part(user_prompt(fake_llm)) == get_complaint_state(seeded).model_dump()


def test_uncertainty_is_visible_to_the_model(seeded, fake_llm):
    fake_llm.queue("A draft.")
    draft_for(seeded)
    location = state_part(user_prompt(fake_llm))["details"]["incident_location"]
    assert location == {"value": "Kanpur Central station", "uncertain": True}


def test_not_provided_lists_every_missing_field_in_order(session_service, fake_llm):
    session = seed(
        session_service,
        session_service.create(),
        details={"complainant_name": detail("Asha")},
    )
    fake_llm.queue("A draft.")
    draft_for(session)

    expected = [n for n in REQUIRED_FIELDS if n != "complainant_name"] + OPTIONAL_FIELDS
    line = next(l for l in user_prompt(fake_llm).splitlines() if l.startswith("NOT PROVIDED:"))
    assert line == "NOT PROVIDED: " + ", ".join(expected)


def test_not_provided_says_none_when_every_field_is_present(session_service, fake_llm):
    session = seed(
        session_service,
        session_service.create(),
        details={name: detail("x") for name in FIELD_DESCRIPTIONS},
    )
    fake_llm.queue("A draft.")
    draft_for(session)
    assert "NOT PROVIDED: none" in user_prompt(fake_llm)


def test_the_conversation_is_included_as_labelled_context(seeded, fake_llm):
    fake_llm.queue("A draft.")
    draft_for(seeded)
    assert transcript_lines(user_prompt(fake_llm)) == [
        "User: My phone was snatched at Kanpur Central.",
        "Assistant: Do you remember the time?",
        "User: Around 9 pm, I think.",
    ]


def test_only_the_recent_conversation_window_is_included(session_service, fake_llm):
    session = session_service.create()
    messages = [
        ("user" if i % 2 == 0 else "assistant", f"seed message {i}") for i in range(30)
    ]
    session = seed(
        session_service,
        session,
        details={"complainant_name": detail("Asha")},
        messages=messages,
    )
    fake_llm.queue("A draft.")
    draft_for(session)

    lines = transcript_lines(user_prompt(fake_llm))
    assert len(lines) == 20
    assert lines[0] == "User: seed message 10"
    assert "seed message 9" not in user_prompt(fake_llm)


def test_the_latest_saved_state_is_what_gets_drafted(seeded, session_service, fake_llm):
    update_complaint(
        seeded, session_service, details={"incident_location": detail("Platform 3")}
    )
    reloaded = session_service.get(seeded.session_id)
    fake_llm.queue("A draft.")
    draft_for(reloaded)

    details = state_part(user_prompt(fake_llm))["details"]
    assert details["incident_location"] == {"value": "Platform 3", "uncertain": False}
    assert "Kanpur Central station" not in json.dumps(details)


# --------------------------------------------------------------------------- #
# Read-only behaviour
# --------------------------------------------------------------------------- #
def test_drafting_does_not_modify_or_save_the_session(seeded, session_service, fake_llm, clock):
    before = session_service.get(seeded.session_id)
    clock.advance(hours=2)
    fake_llm.queue("A draft.")
    draft_for(before)

    assert session_service.get(seeded.session_id) == before


# --------------------------------------------------------------------------- #
# Failures: no fabricated draft
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("provider down", status_code=500),
        LLMTimeoutError("too slow"),
        LLMConfigurationError("no key"),
        LLMResponseError("blocked"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_llm_errors_propagate_and_no_draft_is_returned(seeded, fake_llm, error):
    fake_llm.queue(error)
    with pytest.raises(type(error)) as exc_info:
        draft_for(seeded)
    assert exc_info.value is error


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_an_empty_model_reply_is_an_error_not_an_empty_draft(seeded, fake_llm, blank):
    fake_llm.queue(blank)
    with pytest.raises(LLMResponseError, match="empty draft"):
        draft_for(seeded)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_logs_never_contain_the_narrative_or_the_full_session_id(
    session_service, fake_llm, caplog
):
    session = seed(
        session_service,
        session_service.create(),
        details={"incident_description": detail(NARRATIVE)},
        messages=[("user", f"user says {NARRATIVE}")],
    )
    fake_llm.queue(f"Draft mentioning {NARRATIVE}")
    with caplog.at_level(logging.DEBUG):
        draft_for(session)

    assert caplog.text
    assert NARRATIVE not in caplog.text
    assert session.session_id not in caplog.text