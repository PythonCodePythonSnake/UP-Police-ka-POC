"""Tests for agent/state_manager.py."""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    MAX_HISTORY_MESSAGES,
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
from app.schemas.chat import ChatMessage
from app.services.session_service import SessionData


def make_session(messages=None, complaint=None) -> SessionData:
    now = datetime.now(timezone.utc)
    return SessionData(
        session_id="s" * 30,
        created_at=now,
        updated_at=now,
        messages=messages or [],
        complaint=complaint,
    )


def conversation(roles):
    return [ChatMessage(role=role, content=f"{role} message {i}") for i, role in enumerate(roles)]


def alternating(count):
    return conversation(["user" if i % 2 == 0 else "assistant" for i in range(count)])


def detail(value, uncertain=False) -> ComplaintDetail:
    return ComplaintDetail(value=value, uncertain=uncertain)


def update(state, details=None, add=None, remove=None):
    apply_update(state, details or {}, add or [], remove or [])


# --------------------------------------------------------------------------- #
# Field configuration
# --------------------------------------------------------------------------- #
def test_required_fields_are_all_known_fields():
    assert set(REQUIRED_FIELDS) <= set(FIELD_DESCRIPTIONS)
    assert len(set(REQUIRED_FIELDS)) == len(REQUIRED_FIELDS)


def test_every_field_has_a_description():
    for name, description in FIELD_DESCRIPTIONS.items():
        assert isinstance(description, str) and description.strip(), name


def test_new_states_do_not_share_mutable_defaults():
    first, second = ComplaintState(), ComplaintState()
    first.details["complainant_name"] = detail("Asha")
    first.allegations.append("a claim")
    assert second.details == {}
    assert second.allegations == []


# --------------------------------------------------------------------------- #
# add_message
# --------------------------------------------------------------------------- #
def test_add_message_appends_and_returns_the_message():
    session = make_session()
    message = add_message(session, "user", "My phone was stolen.")
    assert session.messages == [message]
    assert message.role == "user"
    assert message.content == "My phone was stolen."


def test_add_message_timestamps_are_timezone_aware_and_current():
    message = add_message(make_session(), "assistant", "When did it happen?")
    assert message.timestamp.tzinfo is not None
    assert abs(datetime.now(timezone.utc) - message.timestamp) < timedelta(seconds=5)


def test_add_message_preserves_order_and_existing_messages():
    session = make_session(alternating(2))
    existing = list(session.messages)
    add_message(session, "user", "third")
    add_message(session, "assistant", "fourth")
    assert session.messages[:2] == existing
    assert [m.content for m in session.messages[2:]] == ["third", "fourth"]


def test_add_message_rejects_unsupported_roles_without_appending():
    session = make_session()
    with pytest.raises(ValidationError):
        add_message(session, "system", "not stored")
    assert session.messages == []


# --------------------------------------------------------------------------- #
# get_history
# --------------------------------------------------------------------------- #
def test_history_of_an_empty_session_is_empty():
    assert get_history(make_session()) == []


def test_history_returns_everything_when_under_the_limit():
    session = make_session(alternating(6))
    assert get_history(session) == session.messages


def test_history_default_limit_is_the_module_constant():
    assert MAX_HISTORY_MESSAGES == 20
    session = make_session(alternating(30))
    assert get_history(session) == session.messages[-20:]


def test_history_never_starts_with_an_assistant_message():
    session = make_session(alternating(31))  # the last 20 would start with an assistant message
    history = get_history(session)
    assert history[0].role == "user"
    assert history == session.messages[12:]
    assert len(history) == 19


def test_history_respects_a_custom_limit_and_trims_a_leading_assistant_message():
    session = make_session(alternating(10))
    assert get_history(session, limit=4) == session.messages[6:]
    assert get_history(session, limit=3) == session.messages[8:]  # leading assistant dropped


def test_history_of_only_assistant_messages_is_empty():
    session = make_session(conversation(["assistant"] * 3))
    assert get_history(session) == []


def test_history_is_a_copy_and_does_not_modify_the_session():
    session = make_session(alternating(6))
    original = list(session.messages)
    history = get_history(session)
    history.clear()
    assert session.messages == original


def test_history_with_a_zero_limit_is_empty():
    """MINOR KNOWN ISSUE: messages[-0:] is the whole list, so limit=0 returns everything."""
    session = make_session(alternating(6))
    assert get_history(session, limit=0) == []


# --------------------------------------------------------------------------- #
# get/set complaint state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stored", [None, {}])
def test_session_without_complaint_data_gives_an_empty_state(stored):
    state = get_complaint_state(make_session(complaint=stored))
    assert state.details == {}
    assert state.allegations == []


def test_stored_dict_is_loaded_into_a_typed_state():
    stored = {
        "details": {"complainant_name": {"value": "Asha", "uncertain": False}},
        "allegations": ["The complainant states that X."],
    }
    state = get_complaint_state(make_session(complaint=stored))
    assert state.details["complainant_name"] == detail("Asha")
    assert state.allegations == ["The complainant states that X."]


def test_unreadable_stored_complaint_raises_instead_of_resetting():
    session = make_session(complaint={"details": {"complainant_name": "just a string"}})
    with pytest.raises(ValidationError):
        get_complaint_state(session)


def test_set_then_get_roundtrips_and_stores_plain_json_data():
    session = make_session()
    state = ComplaintState(
        details={"incident_location": detail("bus stop", uncertain=True)},
        allegations=["The complainant states that two men took the phone."],
    )
    set_complaint_state(session, state)

    assert isinstance(session.complaint, dict)
    json.dumps(session.complaint)  # must be JSON-serialisable for session storage
    assert get_complaint_state(session) == state


def test_loaded_state_is_independent_of_the_stored_dict():
    session = make_session()
    set_complaint_state(session, ComplaintState(details={"complainant_name": detail("Asha")}))
    snapshot = json.loads(json.dumps(session.complaint))

    state = get_complaint_state(session)
    state.details["complainant_name"] = detail("Changed")
    state.allegations.append("new")

    assert session.complaint == snapshot  # nothing changes until set_complaint_state is called


# --------------------------------------------------------------------------- #
# missing_required_fields
# --------------------------------------------------------------------------- #
def test_all_required_fields_are_missing_from_an_empty_state():
    assert missing_required_fields(ComplaintState()) == list(REQUIRED_FIELDS)


def test_missing_fields_shrink_as_details_arrive_and_keep_their_order():
    state = ComplaintState(
        details={"incident_description": detail("x"), "complainant_name": detail("y")}
    )
    assert missing_required_fields(state) == [
        name for name in REQUIRED_FIELDS if name not in ("incident_description", "complainant_name")
    ]


def test_optional_fields_do_not_count_toward_required():
    optional = [n for n in FIELD_DESCRIPTIONS if n not in REQUIRED_FIELDS]
    state = ComplaintState(details={name: detail("x") for name in optional})
    assert missing_required_fields(state) == list(REQUIRED_FIELDS)


def test_nothing_is_missing_when_every_required_field_is_present():
    state = ComplaintState(details={name: detail("x") for name in REQUIRED_FIELDS})
    assert missing_required_fields(state) == []


# --------------------------------------------------------------------------- #
# apply_update: details
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(FIELD_DESCRIPTIONS))
def test_every_known_field_can_be_set(name):
    state = ComplaintState()
    update(state, {name: detail("some value")})
    assert state.details[name].value == "some value"


def test_value_is_stripped():
    state = ComplaintState()
    update(state, {"complainant_name": detail("  Asha Verma \n")})
    assert state.details["complainant_name"].value == "Asha Verma"


def test_new_value_replaces_the_old_one_and_updates_uncertainty():
    state = ComplaintState(details={"incident_datetime": detail("around 9 pm", uncertain=True)})
    update(state, {"incident_datetime": detail("9:15 pm")})
    assert state.details["incident_datetime"] == detail("9:15 pm", uncertain=False)


def test_uncertain_flag_is_kept():
    state = ComplaintState()
    update(state, {"incident_location": detail("near the station", uncertain=True)})
    assert state.details["incident_location"].uncertain is True


def test_none_removes_a_field_and_leaves_others_alone():
    state = ComplaintState(
        details={"complainant_name": detail("Asha"), "incident_location": detail("station")}
    )
    update(state, {"incident_location": None})
    assert set(state.details) == {"complainant_name"}


def test_removing_a_field_that_is_not_set_is_harmless():
    state = ComplaintState(details={"complainant_name": detail("Asha")})
    update(state, {"loss_or_harm": None})
    assert set(state.details) == {"complainant_name"}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_values_are_ignored_and_never_overwrite_existing_ones(blank):
    state = ComplaintState(details={"complainant_name": detail("Asha")})
    update(state, {"complainant_name": detail(blank), "loss_or_harm": detail(blank)})
    assert state.details == {"complainant_name": detail("Asha")}


def test_unknown_fields_are_ignored_and_logged_without_the_full_name(caplog):
    state = ComplaintState()
    bogus = "made_up_field_" + "x" * 100
    with caplog.at_level(logging.WARNING):
        update(state, {bogus: detail("value"), "complainant_name": detail("Asha")})

    assert set(state.details) == {"complainant_name"}
    assert "Ignoring unknown complaint field" in caplog.text
    assert bogus not in caplog.text


# --------------------------------------------------------------------------- #
# apply_update: allegations
# --------------------------------------------------------------------------- #
def test_allegations_are_added_stripped_and_deduplicated_case_insensitively():
    state = ComplaintState()
    update(state, add=["  The complainant states A.  ", "the complainant states a.", "B", "", "  "])
    assert state.allegations == ["The complainant states A.", "B"]


def test_adding_an_existing_allegation_does_not_duplicate_it():
    state = ComplaintState(allegations=["Claim one"])
    update(state, add=["claim ONE", "Claim two"])
    assert state.allegations == ["Claim one", "Claim two"]


def test_removal_is_case_insensitive_and_ignores_surrounding_spaces():
    state = ComplaintState(allegations=["The complainant states X.", "Other claim"])
    update(state, remove=["  the complainant STATES x.  "])
    assert state.allegations == ["Other claim"]


def test_removing_an_unknown_allegation_changes_nothing():
    state = ComplaintState(allegations=["Claim one"])
    update(state, remove=["never recorded"])
    assert state.allegations == ["Claim one"]


def test_a_correction_can_remove_the_old_claim_and_add_the_new_one_in_one_turn():
    state = ComplaintState(allegations=["The complainant states two men took the phone."])
    update(
        state,
        add=["The complainant states one man took the phone."],
        remove=["The complainant states two men took the phone."],
    )
    assert state.allegations == ["The complainant states one man took the phone."]


def test_removing_and_adding_the_same_claim_keeps_it():
    state = ComplaintState(allegations=["Claim"])
    update(state, add=["Claim"], remove=["Claim"])
    assert state.allegations == ["Claim"]


def test_an_update_with_no_changes_leaves_the_state_untouched():
    state = ComplaintState(
        details={"complainant_name": detail("Asha")}, allegations=["Claim one"]
    )
    before = state.model_dump()
    update(state)
    assert state.model_dump() == before


def test_details_and_allegations_can_change_in_the_same_update():
    state = ComplaintState()
    update(state, {"complainant_name": detail("Asha")}, add=["Claim one"])
    assert state.details["complainant_name"].value == "Asha"
    assert state.allegations == ["Claim one"]