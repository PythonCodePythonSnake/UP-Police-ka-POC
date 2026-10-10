"""Tests for services/session_service.py."""

import json
import logging
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.chat import ChatMessage
from app.schemas.session import SessionInfo
from app.services.session_service import (
    SessionCorruptError,
    SessionData,
    SessionExpiredError,
    SessionNotFoundError,
)
from app.utils.identifiers import generate_session_id

UNKNOWN_ID = "A" * 43


def path_of(directory, session_id):
    return directory / f"{session_id}.json"


def sample_messages():
    return [
        ChatMessage(role="user", content="My phone was snatched near the bus stop."),
        ChatMessage(role="assistant", content="I'm sorry to hear that. When did it happen?"),
    ]


def failing_replace(*args, **kwargs):
    raise OSError("simulated crash during write")


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
def test_constructor_creates_a_missing_directory(make_service, tmp_path):
    target = tmp_path / "a" / "b" / "sessions"
    make_service(directory=target)
    assert target.is_dir()


def test_create_returns_an_empty_timezone_aware_session(session_service, clock):
    session = session_service.create()
    assert session.messages == []
    assert session.complaint is None
    assert session.created_at == session.updated_at == clock.now
    assert session.created_at.tzinfo is not None


def test_create_writes_one_json_file_named_by_id(session_service, sessions_dir):
    session = session_service.create()
    assert [p.name for p in sessions_dir.iterdir()] == [f"{session.session_id}.json"]
    data = json.loads(path_of(sessions_dir, session.session_id).read_text(encoding="utf-8"))
    assert {"session_id", "created_at", "updated_at", "messages", "complaint"} <= set(data)
    assert data["session_id"] == session.session_id
    assert data["messages"] == []
    assert data["complaint"] is None


def test_create_generates_distinct_ids_and_files(session_service, sessions_dir):
    ids = {session_service.create().session_id for _ in range(20)}
    assert len(ids) == 20
    assert len(list(sessions_dir.glob("*.json"))) == 20


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_session_files_are_not_readable_by_other_users(session_service, sessions_dir):
    session = session_service.create()
    mode = stat.S_IMODE(path_of(sessions_dir, session.session_id).stat().st_mode)
    assert mode & 0o077 == 0


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_get_returns_saved_messages_and_complaint(session_service):
    session = session_service.create()
    session.messages.extend(sample_messages())
    session.complaint = {
        "details": {"incident_location": {"value": "bus stop", "uncertain": True}},
        "allegations": ["The complainant states that two men took the phone."],
    }
    session_service.save(session)

    loaded = session_service.get(session.session_id)
    assert loaded.messages == session.messages
    assert loaded.complaint == session.complaint
    assert loaded.created_at == session.created_at
    assert loaded.created_at.tzinfo is not None


def test_state_survives_a_restart(make_service):
    first = make_service()
    session = first.create()
    session.messages.extend(sample_messages())
    session.complaint = {"details": {}, "allegations": ["x"]}
    first.save(session)

    restarted = make_service()  # new instance on the same directory
    loaded = restarted.get(session.session_id)
    assert loaded.messages == session.messages
    assert loaded.complaint == session.complaint


def test_message_order_is_preserved(session_service):
    session = session_service.create()
    contents = [f"message {i}" for i in range(10)]
    session.messages.extend(
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=text)
        for i, text in enumerate(contents)
    )
    session_service.save(session)
    assert [m.content for m in session_service.get(session.session_id).messages] == contents


def test_non_ascii_content_roundtrips(session_service):
    text = "मेरा फ़ोन चोरी हो गया 📱 — café"
    session = session_service.create()
    session.messages.append(ChatMessage(role="user", content=text))
    session_service.save(session)
    assert session_service.get(session.session_id).messages[0].content == text


def test_save_returns_the_session_and_refreshes_only_updated_at(session_service, clock):
    session = session_service.create()
    created = session.created_at
    clock.advance(hours=1)

    assert session_service.save(session) is session
    loaded = session_service.get(session.session_id)
    assert loaded.created_at == created
    assert loaded.updated_at == clock.now


def test_sessions_are_isolated_from_each_other(session_service):
    first = session_service.create()
    second = session_service.create()
    first.messages.append(ChatMessage(role="user", content="only in the first"))
    first.complaint = {"details": {}, "allegations": ["first"]}
    session_service.save(first)

    other = session_service.get(second.session_id)
    assert other.messages == []
    assert other.complaint is None

    session_service.delete(first.session_id)
    assert session_service.get(second.session_id).session_id == second.session_id


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #
def test_session_is_valid_just_before_the_deadline(session_service, clock):
    session = session_service.create()
    clock.advance(hours=23, minutes=59, seconds=59)
    assert session_service.get(session.session_id).session_id == session.session_id


def test_session_expires_exactly_at_the_deadline_and_its_file_is_removed(
    session_service, sessions_dir, clock
):
    session = session_service.create()
    clock.advance(hours=24)

    with pytest.raises(SessionExpiredError):
        session_service.get(session.session_id)
    assert not path_of(sessions_dir, session.session_id).exists()
    with pytest.raises(SessionNotFoundError):
        session_service.get(session.session_id)


def test_saving_extends_the_expiry_window(session_service, clock):
    session = session_service.create()
    clock.advance(hours=20)
    session_service.save(session)

    clock.advance(hours=20)  # 40h since creation, 20h since the save
    assert session_service.get(session.session_id).session_id == session.session_id

    clock.advance(hours=5)  # 25h since the save
    with pytest.raises(SessionExpiredError):
        session_service.get(session.session_id)


def test_reading_does_not_extend_the_expiry_window(session_service, clock):
    session = session_service.create()
    clock.advance(hours=20)
    session_service.get(session.session_id)

    clock.advance(hours=5)
    with pytest.raises(SessionExpiredError):
        session_service.get(session.session_id)


def test_expiry_hours_is_configurable(make_service, clock):
    service = make_service(expiry_hours=1)
    session = service.create()
    clock.advance(minutes=59)
    service.get(session.session_id)
    clock.advance(minutes=1)
    with pytest.raises(SessionExpiredError):
        service.get(session.session_id)


def test_expires_at_and_to_info(session_service, clock):
    session = session_service.create()
    clock.advance(hours=2)
    session_service.save(session)

    assert session_service.expires_at(session) == session.updated_at + timedelta(hours=24)
    info = session_service.to_info(session)
    assert isinstance(info, SessionInfo)
    assert info.session_id == session.session_id
    assert info.created_at == session.created_at
    assert info.updated_at == session.updated_at
    assert info.expires_at == session.updated_at + timedelta(hours=24)
    assert set(info.model_dump()) == {"session_id", "created_at", "updated_at", "expires_at"}


def test_purge_expired_removes_abandoned_sessions(session_service, sessions_dir, clock):
    """KNOWN ISSUE: expiry is only enforced on access, so abandoned complaints stay on disk.

    Contract for the fix: SessionService.purge_expired() -> int deletes every expired session
    file, leaves live ones alone, and returns how many it removed.
    """
    session_service.create()
    session_service.create()
    clock.advance(hours=25)
    fresh = session_service.create()

    removed = session_service.purge_expired()

    assert removed == 2
    assert [p.name for p in sessions_dir.glob("*.json")] == [f"{fresh.session_id}.json"]


def test_purge_expired_with_nothing_expired_removes_nothing(session_service, sessions_dir, clock):
    """KNOWN ISSUE: purge_expired() does not exist yet (see the test above)."""
    session_service.create()
    clock.advance(hours=1)
    assert session_service.purge_expired() == 0
    assert len(list(sessions_dir.glob("*.json"))) == 1


# --------------------------------------------------------------------------- #
# Missing sessions and unsafe IDs
# --------------------------------------------------------------------------- #
def test_unknown_valid_looking_id_is_not_found(session_service):
    with pytest.raises(SessionNotFoundError):
        session_service.get(UNKNOWN_ID)


INVALID_IDS = [
    "",
    "short",
    "../secret",
    "..\\secret",
    "a/b",
    "/etc/passwd",
    "../" + "s" * 25,
    "s" * 25 + "/x",
    "x" * 129,
    "s" * 30 + ".json",
    "has space " * 3,
    "%2e%2e%2f" + "s" * 20,
    "ünïcode-ünïcode-ünïcode",
    "a" * 25 + "\x00",
]


@pytest.mark.parametrize("bad_id", INVALID_IDS)
def test_malformed_ids_are_reported_as_not_found(session_service, bad_id):
    with pytest.raises(SessionNotFoundError):
        session_service.get(bad_id)
    with pytest.raises(SessionNotFoundError):
        session_service.delete(bad_id)


def test_traversal_cannot_reach_files_outside_the_sessions_directory(
    session_service, tmp_path
):
    victim_id = "s" * 25
    victim = tmp_path / f"{victim_id}.json"  # sits next to the sessions directory
    now = datetime.now(timezone.utc)
    victim.write_text(
        SessionData(session_id=victim_id, created_at=now, updated_at=now).model_dump_json(),
        encoding="utf-8",
    )

    traversal_id = f"../{victim_id}"
    with pytest.raises(SessionNotFoundError):
        session_service.get(traversal_id)
    with pytest.raises(SessionNotFoundError):
        session_service.delete(traversal_id)
    assert victim.exists()


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
def test_delete_removes_the_file_and_the_session(session_service, sessions_dir):
    session = session_service.create()
    session_service.delete(session.session_id)
    assert not path_of(sessions_dir, session.session_id).exists()
    with pytest.raises(SessionNotFoundError):
        session_service.get(session.session_id)


def test_deleting_an_unknown_session_is_not_found(session_service):
    with pytest.raises(SessionNotFoundError):
        session_service.delete(UNKNOWN_ID)


def test_deleting_twice_fails_the_second_time(session_service):
    session = session_service.create()
    session_service.delete(session.session_id)
    with pytest.raises(SessionNotFoundError):
        session_service.delete(session.session_id)


def test_save_after_delete_does_not_resurrect_the_session(session_service, sessions_dir):
    """KNOWN ISSUE: save() recreates a deleted session.

    DELETE does not take the per-session lock, so a chat turn that is waiting on the LLM
    when the user deletes the session would write the whole conversation back to disk.
    Expected: save() raises SessionNotFoundError and writes nothing.
    """
    session = session_service.create()
    session_service.delete(session.session_id)

    with pytest.raises(SessionNotFoundError):
        session_service.save(session)
    assert not path_of(sessions_dir, session.session_id).exists()


# --------------------------------------------------------------------------- #
# Corrupt files
# --------------------------------------------------------------------------- #
CORRUPT_CONTENTS = [
    b"",
    b"{not json",
    b"[]",
    b'{"session_id": 1}',
    b"\xff\xfe\x00\x01",
    b'{"session_id": "x", "created_at": "not a date"}',
]


@pytest.mark.parametrize("content", CORRUPT_CONTENTS)
def test_corrupt_file_is_reported_and_kept_for_inspection(session_service, sessions_dir, content):
    session_id = generate_session_id()
    path = path_of(sessions_dir, session_id)
    path.write_bytes(content)

    with pytest.raises(SessionCorruptError):
        session_service.get(session_id)
    assert path.read_bytes() == content


def test_file_whose_stored_id_differs_from_its_name_is_corrupt(session_service, sessions_dir):
    real = session_service.create()
    other_id = generate_session_id()
    path_of(sessions_dir, other_id).write_bytes(path_of(sessions_dir, real.session_id).read_bytes())

    with pytest.raises(SessionCorruptError):
        session_service.get(other_id)


def test_corrupt_session_can_still_be_deleted(session_service, sessions_dir):
    session_id = generate_session_id()
    path = path_of(sessions_dir, session_id)
    path.write_bytes(b"{broken")

    session_service.delete(session_id)
    assert not path.exists()


def test_corruption_error_does_not_leak_stored_content(session_service, sessions_dir, caplog):
    session_id = generate_session_id()
    secret = "SECRET-NARRATIVE-TEXT"
    broken = {
        "session_id": session_id,
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-01-01T12:00:00Z",
        "messages": [{"role": secret, "content": "x"}],  # invalid role echoes the value
    }
    path_of(sessions_dir, session_id).write_text(json.dumps(broken), encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SessionCorruptError) as exc_info:
            session_service.get(session_id)

    error = exc_info.value
    assert secret not in str(error)
    assert secret not in caplog.text
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def test_failed_save_keeps_the_previous_version(session_service, sessions_dir, monkeypatch):
    session = session_service.create()
    session.messages.append(ChatMessage(role="user", content="original"))
    session_service.save(session)

    session.messages.append(ChatMessage(role="assistant", content="never persisted"))
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            session_service.save(session)

    loaded = session_service.get(session.session_id)
    assert [m.content for m in loaded.messages] == ["original"]
    assert [p.suffix for p in sessions_dir.iterdir()] == [".json"]  # no stray .tmp files


def test_failed_create_leaves_no_files(session_service, sessions_dir, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            session_service.create()
    assert list(sessions_dir.iterdir()) == []


def test_repeated_saves_leave_only_the_json_file(session_service, sessions_dir):
    session = session_service.create()
    for i in range(5):
        session.messages.append(ChatMessage(role="user", content=f"message {i}"))
        session_service.save(session)
    assert [p.name for p in sessions_dir.iterdir()] == [f"{session.session_id}.json"]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_logs_never_contain_a_full_session_id(session_service, sessions_dir, clock, caplog):
    with caplog.at_level(logging.DEBUG):
        first = session_service.create()
        session_service.save(first)
        session_service.get(first.session_id)
        session_service.delete(first.session_id)

        second = session_service.create()
        clock.advance(hours=25)
        with pytest.raises(SessionExpiredError):
            session_service.get(second.session_id)

        corrupt_id = generate_session_id()
        path_of(sessions_dir, corrupt_id).write_bytes(b"{broken")
        with pytest.raises(SessionCorruptError):
            session_service.get(corrupt_id)

    assert caplog.text  # something was logged, so the assertions below mean something
    for session_id in (first.session_id, second.session_id, corrupt_id):
        assert session_id not in caplog.text