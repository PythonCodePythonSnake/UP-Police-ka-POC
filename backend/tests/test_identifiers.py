"""Tests for utils/identifiers.py."""

import re
from urllib.parse import quote

from app.utils import identifiers
from app.utils.identifiers import generate_session_id


def test_returns_url_and_filename_safe_string():
    session_id = generate_session_id()
    assert isinstance(session_id, str)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", session_id)
    assert quote(session_id, safe="") == session_id


def test_is_long_enough_for_256_bits():
    # base64url carries 6 bits per character, so 256 bits needs at least 43 characters.
    assert len(generate_session_id()) >= 43


def test_ids_are_unique_across_many_calls():
    ids = {generate_session_id() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_delegates_to_secrets_with_at_least_32_bytes(monkeypatch):
    calls = []

    def fake_token_urlsafe(nbytes=None):
        calls.append(nbytes)
        return "x" * 43

    monkeypatch.setattr(identifiers.secrets, "token_urlsafe", fake_token_urlsafe)
    assert generate_session_id() == "x" * 43
    assert calls and calls[0] is not None and calls[0] >= 32


def test_does_not_use_non_cryptographic_randomness():
    assert not hasattr(identifiers, "random")
    assert not hasattr(identifiers, "uuid")