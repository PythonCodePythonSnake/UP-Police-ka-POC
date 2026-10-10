"""Tests for services/complaint_service.py."""

import pytest

from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
    ComplaintDetail,
    ComplaintState,
    get_complaint_state,
    set_complaint_state,
)
from app.schemas.chat import ChatMessage
from app.services.complaint_service import (
    MAX_ALLEGATION_LENGTH,
    MAX_ALLEGATIONS,
    MAX_DETAIL_LENGTH,
    ComplaintSnapshot,
    ComplaintValidationError,
    build_snapshot,
    get_snapshot,
    update_complaint,
)
import os

OPTIONAL_FIELDS = [name for name in FIELD_DESCRIPTIONS if name not in REQUIRED_FIELDS]


def detail(value, uncertain=False) -> ComplaintDetail:
    return ComplaintDetail(value=value, uncertain=uncertain)


def seed(session, service, details=None, allegations=None):
    set_complaint_state(
        session, ComplaintState(details=details or {}, allegations=allegations or [])
    )
    service.save(session)


def stored_state(service, session) -> ComplaintState:
    return get_complaint_state(service.get(session.session_id))


@pytest.fixture
def session(session_service):
    return session_service.create()


@pytest.fixture
def seeded(session, session_service):
    seed(
        session,
        session_service,
        details={
            "complainant_name": detail("Asha Verma"),
            "incident_description": detail("My phone was snatched."),
            "incident_location": detail("Kanpur Central station", uncertain=True),
        },
        allegations=["The complainant states that two men took the phone."],
    )
    return session


def assert_rejected_and_unchanged(service, session, clock, **kwargs) -> ComplaintValidationError:
    before = service.get(session.session_id)
    clock.advance(minutes=5)
    with pytest.raises(ComplaintValidationError) as exc_info:
        update_complaint(session, service, **kwargs)
    after = service.get(session.session_id)
    assert after.complaint == before.complaint
    assert after.updated_at == before.updated_at  # nothing was saved
    return exc_info.value


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def test_snapshot_of_a_new_session_lists_everything_as_missing(session):
    snapshot = get_snapshot(session)
    assert snapshot.complaint == ComplaintState()
    assert snapshot.missing_required_fields == list(REQUIRED_FIELDS)
    assert snapshot.missing_optional_fields == OPTIONAL_FIELDS
    assert snapshot.uncertain_fields == []
    assert snapshot.required_details_collected is False


def test_snapshot_is_complete_once_every_required_field_is_present():
    state = ComplaintState(details={name: detail("x") for name in REQUIRED_FIELDS})
    snapshot = build_snapshot(state)
    assert snapshot.missing_required_fields == []
    assert snapshot.required_details_collected is True
    assert snapshot.missing_optional_fields == OPTIONAL_FIELDS


def test_snapshot_reports_uncertain_fields_and_provided_optional_fields():
    state = ComplaintState(
        details={
            "incident_datetime": detail("around 9 pm", uncertain=True),
            "complaint_type": detail("theft"),
            "incident_location": detail("station", uncertain=True),
            "complainant_name": detail("Asha"),
        }
    )
    snapshot = build_snapshot(state)
    assert snapshot.uncertain_fields == ["incident_datetime", "incident_location"]
    assert "complaint_type" not in snapshot.missing_optional_fields
    assert "accused_details" in snapshot.missing_optional_fields


def test_snapshot_survives_json_roundtrip(seeded):
    snapshot = get_snapshot(seeded)
    assert ComplaintSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_snapshot_after_a_restart_matches(seeded, make_service):
    restarted = make_service()
    reloaded = restarted.get(seeded.session_id)
    assert get_snapshot(reloaded) == get_snapshot(seeded)


# --------------------------------------------------------------------------- #
# Updates
# --------------------------------------------------------------------------- #
def test_update_adds_a_detail_and_persists_it(session, session_service):
    snapshot = update_complaint(
        session, session_service, details={"complainant_name": detail("Asha Verma")}
    )
    assert snapshot.complaint.details["complainant_name"].value == "Asha Verma"
    assert stored_state(session_service, session).details["complainant_name"] == detail("Asha Verma")


def test_update_works_on_a_session_whose_complaint_is_none(session, session_service):
    assert session.complaint is None
    update_complaint(session, session_service, details={"loss_or_harm": detail("Phone, about 20000 rupees")})
    assert "loss_or_harm" in stored_state(session_service, session).details


def test_partial_update_preserves_untouched_details_and_allegations(seeded, session_service):
    before = stored_state(session_service, seeded)
    update_complaint(
        seeded, session_service, details={"incident_location": detail("Platform 3", uncertain=False)}
    )
    after = stored_state(session_service, seeded)

    assert after.details["incident_location"] == detail("Platform 3")
    assert after.details["complainant_name"] == before.details["complainant_name"]
    assert after.details["incident_description"] == before.details["incident_description"]
    assert set(after.details) == set(before.details)
    assert after.allegations == before.allegations


def test_update_does_not_add_anything_that_was_not_supplied(seeded, session_service):
    update_complaint(seeded, session_service, details={"complaint_type": detail("theft")})
    after = stored_state(session_service, seeded)
    assert set(after.details) == {
        "complainant_name",
        "incident_description",
        "incident_location",
        "complaint_type",
    }


def test_values_are_stripped_before_saving(session, session_service):
    update_complaint(session, session_service, details={"complainant_name": detail("  Asha  \n")})
    assert stored_state(session_service, session).details["complainant_name"].value == "Asha"


def test_null_removes_a_field_and_keeps_the_rest(seeded, session_service):
    update_complaint(seeded, session_service, details={"incident_location": None})
    after = stored_state(session_service, seeded)
    assert "incident_location" not in after.details
    assert "complainant_name" in after.details


def test_removing_a_field_that_was_never_set_is_accepted(session, session_service):
    snapshot = update_complaint(session, session_service, details={"loss_or_harm": None})
    assert snapshot.complaint.details == {}


def test_resending_a_value_without_uncertain_marks_it_certain(seeded, session_service):
    assert stored_state(session_service, seeded).details["incident_location"].uncertain is True
    update_complaint(
        seeded, session_service, details={"incident_location": detail("Kanpur Central station")}
    )
    assert stored_state(session_service, seeded).details["incident_location"].uncertain is False


def test_a_value_can_be_marked_uncertain(session, session_service):
    update_complaint(
        session, session_service, details={"incident_datetime": detail("last night", uncertain=True)}
    )
    assert stored_state(session_service, session).details["incident_datetime"].uncertain is True


def test_omitting_allegations_preserves_them(seeded, session_service):
    update_complaint(seeded, session_service, details={"complaint_type": detail("theft")})
    assert stored_state(session_service, seeded).allegations == [
        "The complainant states that two men took the phone."
    ]


def test_supplied_allegations_replace_the_whole_list(seeded, session_service):
    update_complaint(seeded, session_service, allegations=["The complainant states it was one man."])
    assert stored_state(session_service, seeded).allegations == [
        "The complainant states it was one man."
    ]


def test_an_empty_allegation_list_clears_them(seeded, session_service):
    update_complaint(seeded, session_service, allegations=[])
    after = stored_state(session_service, seeded)
    assert after.allegations == []
    assert "complainant_name" in after.details  # details untouched


def test_allegations_are_stripped_and_deduplicated(session, session_service):
    update_complaint(
        session,
        session_service,
        allegations=["  Claim one ", "claim ONE", "Claim two"],
    )
    assert stored_state(session_service, session).allegations == ["Claim one", "Claim two"]


def test_details_and_allegations_can_be_updated_together(session, session_service):
    update_complaint(
        session,
        session_service,
        details={"complainant_name": detail("Asha")},
        allegations=["Claim one"],
    )
    after = stored_state(session_service, session)
    assert after.details["complainant_name"].value == "Asha"
    assert after.allegations == ["Claim one"]


def test_editing_never_touches_the_conversation(seeded, session_service):
    seeded.messages.append(ChatMessage(role="user", content="My phone was snatched."))
    seeded.messages.append(ChatMessage(role="assistant", content="When did it happen?"))
    session_service.save(seeded)
    before = session_service.get(seeded.session_id).messages

    update_complaint(seeded, session_service, details={"complainant_name": detail("Someone Else")})
    update_complaint(seeded, session_service, allegations=[])

    assert session_service.get(seeded.session_id).messages == before


def test_editing_counts_as_activity_and_extends_the_session(session, session_service, clock):
    clock.advance(hours=20)
    update_complaint(session, session_service, details={"complainant_name": detail("Asha")})
    clock.advance(hours=20)  # 40h since creation, 20h since the edit
    assert session_service.get(session.session_id).session_id == session.session_id


def test_edits_to_one_session_do_not_affect_another(session_service):
    first = session_service.create()
    second = session_service.create()
    update_complaint(first, session_service, details={"complainant_name": detail("Asha")})
    assert stored_state(session_service, second) == ComplaintState()


def test_updates_survive_a_restart(session, session_service, make_service):
    update_complaint(session, session_service, details={"complainant_name": detail("Asha")})
    restarted = make_service()
    reloaded = restarted.get(session.session_id)
    assert get_snapshot(reloaded).complaint.details["complainant_name"].value == "Asha"


def test_returned_snapshot_matches_what_was_saved(seeded, session_service):
    snapshot = update_complaint(seeded, session_service, details={"complaint_type": detail("theft")})
    assert snapshot == get_snapshot(session_service.get(seeded.session_id))


def test_required_details_collected_flips_when_the_last_required_field_arrives(
    session, session_service
):
    names = list(REQUIRED_FIELDS)
    for name in names[:-1]:
        snapshot = update_complaint(session, session_service, details={name: detail("x")})
        assert snapshot.required_details_collected is False
    snapshot = update_complaint(session, session_service, details={names[-1]: detail("x")})
    assert snapshot.required_details_collected is True
    assert snapshot.missing_required_fields == []


def test_boundary_lengths_are_accepted(session, session_service):
    update_complaint(
        session,
        session_service,
        details={"incident_description": detail("d" * MAX_DETAIL_LENGTH)},
        allegations=["a" * MAX_ALLEGATION_LENGTH],
    )
    after = stored_state(session_service, session)
    assert len(after.details["incident_description"].value) == MAX_DETAIL_LENGTH
    assert len(after.allegations[0]) == MAX_ALLEGATION_LENGTH


def test_the_maximum_number_of_allegations_is_accepted(session, session_service):
    claims = [f"Claim {i}" for i in range(MAX_ALLEGATIONS)]
    update_complaint(session, session_service, allegations=claims)
    assert stored_state(session_service, session).allegations == claims


def test_a_failed_save_leaves_the_stored_complaint_unchanged(seeded, session_service, monkeypatch):
    before = stored_state(session_service, seeded)

    def failing_replace(*args, **kwargs):
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            update_complaint(
                seeded, session_service, details={"complainant_name": detail("Changed")}
            )

    assert stored_state(session_service, seeded) == before


# --------------------------------------------------------------------------- #
# Rejections (each must leave the stored session untouched)
# --------------------------------------------------------------------------- #
def test_validation_error_is_a_value_error():
    assert issubclass(ComplaintValidationError, ValueError)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"details": None, "allegations": None}, {"details": {}}, {"details": {}, "allegations": None}],
)
def test_an_update_with_no_changes_is_rejected(seeded, session_service, clock, kwargs):
    error = assert_rejected_and_unchanged(session_service, seeded, clock, **kwargs)
    assert "no changes" in str(error).lower()


def test_unknown_field_is_rejected_and_the_message_lists_the_allowed_fields(
    seeded, session_service, clock
):
    error = assert_rejected_and_unchanged(
        session_service, seeded, clock, details={"made_up_field": detail("value")}
    )
    message = str(error)
    assert "made_up_field" in message
    for name in FIELD_DESCRIPTIONS:
        assert name in message


def test_unknown_field_is_rejected_even_when_the_value_is_null(seeded, session_service, clock):
    assert_rejected_and_unchanged(session_service, seeded, clock, details={"made_up_field": None})


def test_one_unknown_field_blocks_the_valid_changes_in_the_same_request(
    seeded, session_service, clock
):
    assert_rejected_and_unchanged(
        session_service,
        seeded,
        clock,
        details={"complainant_name": detail("Changed"), "made_up_field": detail("value")},
        allegations=["Should not be saved"],
    )


def test_a_very_long_unknown_field_name_is_truncated_in_the_error(seeded, session_service, clock):
    error = assert_rejected_and_unchanged(
        session_service, seeded, clock, details={"x" * 100: detail("value")}
    )
    assert "x" * 40 in str(error)
    assert "x" * 41 not in str(error)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_values_are_rejected(seeded, session_service, clock, blank):
    error = assert_rejected_and_unchanged(
        session_service, seeded, clock, details={"complainant_name": detail(blank)}
    )
    assert "complainant_name" in str(error)


def test_oversized_detail_is_rejected(seeded, session_service, clock):
    assert_rejected_and_unchanged(
        session_service,
        seeded,
        clock,
        details={"incident_description": detail("d" * (MAX_DETAIL_LENGTH + 1))},
    )


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_allegations_are_rejected(seeded, session_service, clock, bad):
    assert_rejected_and_unchanged(session_service, seeded, clock, allegations=["Fine claim", bad])


def test_oversized_allegation_is_rejected(seeded, session_service, clock):
    assert_rejected_and_unchanged(
        session_service, seeded, clock, allegations=["a" * (MAX_ALLEGATION_LENGTH + 1)]
    )


def test_too_many_allegations_are_rejected(seeded, session_service, clock):
    claims = [f"Claim {i}" for i in range(MAX_ALLEGATIONS + 1)]
    assert_rejected_and_unchanged(session_service, seeded, clock, allegations=claims)


def test_an_invalid_allegation_blocks_valid_detail_changes_in_the_same_request(
    seeded, session_service, clock
):
    assert_rejected_and_unchanged(
        session_service,
        seeded,
        clock,
        details={"complainant_name": detail("Changed")},
        allegations=[""],
    )