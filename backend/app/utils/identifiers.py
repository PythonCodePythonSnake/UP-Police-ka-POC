"""Identifier generation."""

import secrets


def generate_session_id() -> str:
    """Return an unpredictable, URL-safe session ID (256 bits of randomness)."""
    return secrets.token_urlsafe(32)