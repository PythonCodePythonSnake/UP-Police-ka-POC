"""Tests for agent/orchestrator.py. The LLM is replaced by the fake_llm fixture."""

import asyncio
import gc
import json
import logging
import os

import pytest

from app.agent import orchestrator
from app.agent.orchestrator import TurnResult, handle_user_message, session_lock
from app.agent.state_manager import (
    MAX_HISTORY_MESSAGES,
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

NARRATIVE = "SECRET-NARRATIVE-TEXT"


def run_turn(service, session, text="My phone was snatched near the bus stop."):
    return asyncio.run(handle_user_message(session, text, service))


def stored(service, session):
    return service.get(session.session_id)


def seed_conversation(service, session, count):
    for i in range(count):
        add_message(session, "user" if i % 2 == 0 else "assistant", f"seed message {i}")
    service.save(session)


def failing_replace(*args, **kwargs):
    raise OSError("simulated disk failure")


async def locked_turn(service, session_id, text):
    """What the chat route does: hold the session lock while loading, calling the LLM, saving."""
    async with session_lock(session_id):
        return await handle_user_message(service.get(session_id), text, service)


@pytest.fixture
def session(session_service):
    return session_service.create()


# --------------------------------------------------------------------------- #
# What the LLM receives
# --------------------------------------------------------------------------- #
def test_llm_receives_the_prompt_the_state_and_the_new_message(session_service, session, fake_llm):
    fake_llm.queue(fake_llm.turn("Where did it happen?"))
    run_turn(session_service, session, "My phone was snatched.")

    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call.json_mode is True
    assert [m.role for m in call.messages] == ["system", "system", "user"]
    assert call.messages[0].content == orchestrator._STATIC_SYSTEM_PROMPT
    assert "CURRENT COMPLAINT STATE" in call.messages[1].content
    for name in REQUIRED_FIELDS:
        assert name in call.messages[1].content
    assert call.messages[2].content == "My phone was snatched."


def test_second_turn_sends_the_first_exchange_and_the_recorded_details(
    session_service, session, fake_llm
):
    fake_llm.queue(
        fake_llm.turn(
            "What is your name?",
            details={"incident_description": fake_llm.detail("Phone snatched")},
        ),
        fake_llm.turn("Thanks."),
    )
    first = run_turn(session_service, session, "My phone was snatched.")
    run_turn(session_service, stored(session_service, session), "It was at the bus stop.")

    messages = fake_llm.calls[1].messages
    assert [m.role for m in messages] == ["system", "system", "user", "assistant", "user"]
    assert messages[2].content == "My phone was snatched."
    assert messages[3].content == first.reply.content == "What is your name?"
    assert messages[4].content == "It was at the bus stop."

    context = messages[1].content
    assert "Phone snatched" in context
    missing_line = next(
        line for line in context.splitlines() if line.startswith("Required details still missing:")
    )
    assert "incident_description" not in missing_line
    assert "complainant_name" in missing_line


def test_only_the_most_recent_messages_are_sent_but_older_facts_stay_in_the_state(
    session_service, session, fake_llm
):
    set_complaint_state(
        session, ComplaintState(details={"complainant_name": ComplaintDetail(value="Asha Verma")})
    )
    seed_conversation(session_service, session, 30)
    before = stored(session_service, session)

    fake_llm.queue(fake_llm.turn("Noted."))
    run_turn(session_service, before, "latest message")

    messages = fake_llm.calls[0].messages
    history = messages[2:-1]
    assert len(history) == MAX_HISTORY_MESSAGES
    assert [m.content for m in history] == [m.content for m in before.messages[10:]]
    assert messages[-1].content == "latest message"
    assert "Asha Verma" in messages[1].content  # the fact outlives the dropped messages
    for i in range(10):
        assert not any(f"seed message {i}" == m.content for m in messages)


def test_the_new_message_is_sent_and_stored_exactly_once(session_service, session, fake_llm):
    fake_llm.queue(fake_llm.turn("Ok."))
    run_turn(session_service, session, "unique text 12345")

    sent = [m for m in fake_llm.calls[0].messages if m.content == "unique text 12345"]
    kept = [m for m in stored(session_service, session).messages if m.content == "unique text 12345"]
    assert len(sent) == 1
    assert len(kept) == 1


def test_turns_in_different_sessions_do_not_leak_into_each_other(session_service, fake_llm):
    first = session_service.create()
    second = session_service.create()
    fake_llm.queue(
        fake_llm.turn("a", details={"complainant_name": fake_llm.detail("Asha")}),
        fake_llm.turn("b"),
    )
    run_turn(session_service, first, "message for A")
    run_turn(session_service, second, "message for B")

    second_call = fake_llm.calls[1]
    assert not any("message for A" in m.content for m in second_call.messages)
    assert not any("Asha" in m.content for m in second_call.messages)
    assert [m.content for m in stored(session_service, second).messages] == ["message for B", "b"]


# --------------------------------------------------------------------------- #
# What a successful turn stores and returns
# --------------------------------------------------------------------------- #
def test_a_turn_stores_both_messages_and_returns_the_reply(session_service, session, fake_llm):
    fake_llm.queue(fake_llm.turn("  Where did it happen?  "))
    result = run_turn(session_service, session, "My phone was snatched.")

    assert isinstance(result, TurnResult)
    assert result.reply.role == "assistant"
    assert result.reply.content == "Where did it happen?"

    reloaded = stored(session_service, session)
    assert [(m.role, m.content) for m in reloaded.messages] == [
        ("user", "My phone was snatched."),
        ("assistant", "Where did it happen?"),
    ]
    assert reloaded.messages[1] == result.reply
    assert all(m.timestamp.tzinfo is not None for m in reloaded.messages)


def test_details_and_allegations_from_the_model_are_recorded_and_persisted(
    session_service, session, fake_llm
):
    fake_llm.queue(
        fake_llm.turn(
            "Noted.",
            details={
                "complainant_name": fake_llm.detail("Asha Verma"),
                "incident_datetime": fake_llm.detail("around 9 pm yesterday", uncertain=True),
            },
            allegations_add=["The complainant states that two men took the phone."],
        )
    )
    result = run_turn(session_service, session)

    assert result.complaint.details["complainant_name"] == ComplaintDetail(value="Asha Verma")
    assert result.complaint.details["incident_datetime"].uncertain is True
    assert result.complaint.allegations == ["The complainant states that two men took the phone."]
    assert get_complaint_state(stored(session_service, session)) == result.complaint


def test_everything_survives_a_restart(session_service, session, make_service, fake_llm):
    fake_llm.queue(
        fake_llm.turn("Noted.", details={"complainant_name": fake_llm.detail("Asha")})
    )
    result = run_turn(session_service, session, "I am Asha.")

    reloaded = make_service().get(session.session_id)
    assert [m.content for m in reloaded.messages] == ["I am Asha.", "Noted."]
    assert get_complaint_state(reloaded) == result.complaint


def test_a_greeting_is_stored_but_records_nothing(session_service, session, fake_llm):
    fake_llm.queue(fake_llm.turn("Hello! Please tell me what happened."))
    result = run_turn(session_service, session, "hi")

    assert result.complaint == ComplaintState()
    assert len(stored(session_service, session).messages) == 2


def test_missing_fields_and_sources_flag_come_from_python_not_the_model(
    session_service, session, fake_llm
):
    fake_llm.queue(fake_llm.turn("Your complaint is complete and has been filed."))
    result = run_turn(session_service, session)

    assert result.missing_required_fields == list(REQUIRED_FIELDS)
    assert result.sources_available is False


def test_missing_fields_clear_when_every_required_detail_is_recorded(
    session_service, session, fake_llm
):
    details = {name: fake_llm.detail("x") for name in REQUIRED_FIELDS}
    fake_llm.queue(fake_llm.turn("Is there anything else?", details=details))
    result = run_turn(session_service, session)
    assert result.missing_required_fields == []


def test_corrections_overwrite_and_remove_earlier_details_and_keep_the_rest(
    session_service, session, fake_llm
):
    fake_llm.queue(
        fake_llm.turn(
            "Noted.",
            details={
                "complainant_name": fake_llm.detail("Asha"),
                "incident_location": fake_llm.detail("Platform 3"),
                "loss_or_harm": fake_llm.detail("Phone"),
            },
            allegations_add=["Claim A"],
        ),
        fake_llm.turn(
            "Updated.",
            details={
                "incident_location": fake_llm.detail("Platform 5"),
                "loss_or_harm": None,
            },
            allegations_add=["Claim B"],
            allegations_remove=["Claim A"],
        ),
    )
    run_turn(session_service, session, "first")
    result = run_turn(session_service, stored(session_service, session), "correction")

    assert result.complaint.details["incident_location"].value == "Platform 5"
    assert "loss_or_harm" not in result.complaint.details
    assert result.complaint.details["complainant_name"].value == "Asha"
    assert result.complaint.allegations == ["Claim B"]


def test_unknown_and_blank_fields_from_the_model_are_ignored(session_service, session, fake_llm):
    fake_llm.queue(
        fake_llm.turn(
            "Ok.",
            details={
                "made_up_field": fake_llm.detail("x"),
                "complainant_name": fake_llm.detail("   "),
                "incident_location": fake_llm.detail("Station"),
            },
        )
    )
    result = run_turn(session_service, session)
    assert set(result.complaint.details) == {"incident_location"}


def test_a_turn_refreshes_updated_at(session_service, session, fake_llm, clock):
    fake_llm.queue(fake_llm.turn("Ok."))
    clock.advance(hours=1)
    run_turn(session_service, session)
    assert stored(session_service, session).updated_at == clock.now


# --------------------------------------------------------------------------- #
# Failures: nothing is saved, and no reply is made up
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("provider down", status_code=500),
        LLMTimeoutError("too slow"),
        LLMConfigurationError("no key"),
        LLMResponseError("empty reply"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_llm_errors_propagate_and_leave_the_session_untouched(
    session_service, session, fake_llm, clock, error
):
    before = stored(session_service, session)
    clock.advance(hours=1)
    fake_llm.queue(error)

    with pytest.raises(type(error)) as exc_info:
        run_turn(session_service, session, "a complaint message")

    assert exc_info.value is error
    after = stored(session_service, session)
    assert after.messages == []
    assert after.complaint is None
    assert after.updated_at == before.updated_at
    assert session.messages == []  # the in-memory session is untouched as well


def test_resending_after_a_failure_stores_exactly_one_exchange(session_service, session, fake_llm):
    fake_llm.queue(LLMTimeoutError("slow"), fake_llm.turn("Thanks."))
    with pytest.raises(LLMTimeoutError):
        run_turn(session_service, session, "my message")

    run_turn(session_service, stored(session_service, session), "my message")
    assert [(m.role, m.content) for m in stored(session_service, session).messages] == [
        ("user", "my message"),
        ("assistant", "Thanks."),
    ]


BAD_OUTPUTS = {
    "not_json": "Sorry, I cannot help with that.",
    "empty_string": "",
    "json_array": "[]",
    "missing_reply": json.dumps({"details": {}}),
    "blank_reply": json.dumps({"reply": "   "}),
    "reply_wrong_type": json.dumps({"reply": 42}),
    "details_wrong_type": json.dumps({"reply": "ok", "details": "none"}),
    "allegations_wrong_type": json.dumps({"reply": "ok", "allegations_add": "a claim"}),
}


@pytest.mark.parametrize("name", list(BAD_OUTPUTS))
def test_unusable_model_output_is_a_response_error_and_saves_nothing(
    session_service, session, fake_llm, name
):
    # Queued twice so the test stays valid if a single retry is added later.
    fake_llm.queue(BAD_OUTPUTS[name], BAD_OUTPUTS[name])

    with pytest.raises(LLMResponseError):
        run_turn(session_service, session)

    after = stored(session_service, session)
    assert after.messages == []
    assert after.complaint is None


def test_model_output_is_never_echoed_in_the_error(session_service, session, fake_llm):
    bad = f"{NARRATIVE} this is not json"
    fake_llm.queue(bad, bad)

    with pytest.raises(LLMResponseError) as exc_info:
        run_turn(session_service, session)

    assert NARRATIVE not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_a_failed_save_after_a_successful_llm_call_changes_nothing_on_disk(
    session_service, session, fake_llm, monkeypatch
):
    fake_llm.queue(
        fake_llm.turn("Ok.", details={"complainant_name": fake_llm.detail("Asha")})
    )
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            run_turn(session_service, session)

    after = stored(session_service, session)
    assert after.messages == []
    assert after.complaint is None


# --------------------------------------------------------------------------- #
# Known issue 4: strict parsing of the model's JSON makes a turn brittle.
# These currently FAIL (the turn raises LLMResponseError), which is the point.
# --------------------------------------------------------------------------- #
def test_a_plain_string_detail_value_is_accepted(session_service, session, fake_llm):
    """KNOWN ISSUE 4: models often write "field": "text" instead of {"value": "text"}."""
    fake_llm.queue(
        json.dumps({"reply": "Noted.", "details": {"complainant_name": "Asha Verma"}})
    )
    result = run_turn(session_service, session)
    assert result.complaint.details["complainant_name"].value == "Asha Verma"


def test_a_numeric_detail_value_is_accepted_as_text(session_service, session, fake_llm):
    """KNOWN ISSUE 4: a phone number is easily emitted as a JSON number."""
    fake_llm.queue(
        json.dumps(
            {"reply": "Noted.", "details": {"complainant_contact": {"value": 9876543210}}}
        )
    )
    result = run_turn(session_service, session)
    assert result.complaint.details["complainant_contact"].value == "9876543210"


@pytest.mark.parametrize("fence", ["```json", "```"])
def test_json_wrapped_in_a_code_fence_is_accepted(session_service, session, fake_llm, fence):
    """KNOWN ISSUE 4: some models wrap JSON in a markdown fence even in JSON mode."""
    payload = fake_llm.turn("Noted.", details={"complainant_name": fake_llm.detail("Asha")})
    fake_llm.queue(f"{fence}\n{payload}\n```")
    result = run_turn(session_service, session)
    assert result.reply.content == "Noted."
    assert result.complaint.details["complainant_name"].value == "Asha"


def test_one_malformed_reply_is_retried_once(session_service, session, fake_llm):
    """KNOWN ISSUE 4: a single bad generation should not fail the whole turn."""
    fake_llm.queue(
        "this is not json at all",
        fake_llm.turn("Noted.", details={"complainant_name": fake_llm.detail("Asha")}),
    )
    result = run_turn(session_service, session)

    assert result.reply.content == "Noted."
    assert len(fake_llm.calls) == 2
    assert [m.role for m in stored(session_service, session).messages] == ["user", "assistant"]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_logs_never_contain_the_narrative_or_the_full_session_id(
    session_service, session, fake_llm, caplog
):
    fake_llm.queue(
        fake_llm.turn(
            f"reply about {NARRATIVE}",
            details={"incident_description": fake_llm.detail(NARRATIVE)},
            allegations_add=[NARRATIVE],
        )
    )
    with caplog.at_level(logging.DEBUG):
        run_turn(session_service, session, f"user says {NARRATIVE}")

    assert caplog.text
    assert NARRATIVE not in caplog.text
    assert session.session_id not in caplog.text


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_session_lock_is_shared_per_session_and_distinct_across_sessions():
    first = session_lock("a" * 30)
    again = session_lock("a" * 30)
    other = session_lock("b" * 30)
    assert isinstance(first, asyncio.Lock)
    assert first is again
    assert first is not other


def test_unused_locks_are_dropped_from_the_registry():
    key = "z" * 30
    lock = session_lock(key)
    assert key in orchestrator._locks
    del lock
    gc.collect()
    assert key not in orchestrator._locks


def test_simultaneous_turns_on_one_session_are_serialised_and_none_is_lost(
    session_service, session, fake_llm
):
    def slow(reply):
        async def respond(messages):
            await asyncio.sleep(0.05)  # long enough that unserialised turns would interleave
            return fake_llm.turn(reply)

        return respond

    fake_llm.queue(slow("reply 1"), slow("reply 2"))

    async def main():
        return await asyncio.gather(
            locked_turn(session_service, session.session_id, "first message"),
            locked_turn(session_service, session.session_id, "second message"),
        )

    asyncio.run(main())

    final = stored(session_service, session)
    assert [m.role for m in final.messages] == ["user", "assistant", "user", "assistant"]
    assert sorted(m.content for m in final.messages if m.role == "user") == [
        "first message",
        "second message",
    ]
    assert sorted(m.content for m in final.messages if m.role == "assistant") == [
        "reply 1",
        "reply 2",
    ]
    # The second LLM call must have seen the first exchange: 2 system + 2 history + 1 new.
    assert len(fake_llm.calls[0].messages) == 3
    assert len(fake_llm.calls[1].messages) == 5


def test_turns_on_different_sessions_do_not_block_each_other(session_service, fake_llm):
    first = session_service.create()
    second = session_service.create()

    async def main():
        gate = asyncio.Event()

        async def waits_for_the_other_turn(messages):
            await asyncio.wait_for(gate.wait(), timeout=2)  # times out if turns block each other
            return fake_llm.turn("a")

        def releases_the_other_turn(messages):
            gate.set()
            return fake_llm.turn("b")

        fake_llm.queue(waits_for_the_other_turn, releases_the_other_turn)
        await asyncio.wait_for(
            asyncio.gather(
                locked_turn(session_service, first.session_id, "message A"),
                locked_turn(session_service, second.session_id, "message B"),
            ),
            timeout=5,
        )

    asyncio.run(main())
    assert len(stored(session_service, first).messages) == 2
    assert len(stored(session_service, second).messages) == 2


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def test_state_context_embeds_the_state_as_json_and_lists_missing_fields():
    state = ComplaintState(details={"complainant_name": ComplaintDetail(value="Asha")})
    context = orchestrator._state_context(state)

    body = context.split("\n", 1)[1].rsplit("\n\nRequired details still missing:", 1)[0]
    assert json.loads(body) == state.model_dump()
    missing = context.rsplit("Required details still missing:", 1)[1]
    assert "complainant_name" not in missing
    for name in REQUIRED_FIELDS[1:]:
        assert name in missing


def test_state_context_says_none_when_nothing_required_is_missing():
    state = ComplaintState(details={name: ComplaintDetail(value="x") for name in REQUIRED_FIELDS})
    assert orchestrator._state_context(state).endswith("Required details still missing: none")


def test_parse_turn_accepts_a_minimal_reply_and_ignores_extra_keys():
    turn = orchestrator._parse_turn(json.dumps({"reply": "Hi", "unexpected": 1}))
    assert turn.reply == "Hi"
    assert turn.details == {}
    assert turn.allegations_add == []
    assert turn.allegations_remove == []


def test_parse_turn_reads_details_including_removals():
    raw = json.dumps(
        {
            "reply": "Ok",
            "details": {
                "complainant_name": {"value": "Asha", "uncertain": True},
                "loss_or_harm": None,
            },
            "allegations_add": ["Claim"],
            "allegations_remove": ["Old claim"],
        }
    )
    turn = orchestrator._parse_turn(raw)
    assert turn.details["complainant_name"] == ComplaintDetail(value="Asha", uncertain=True)
    assert turn.details["loss_or_harm"] is None
    assert turn.allegations_add == ["Claim"]
    assert turn.allegations_remove == ["Old claim"]