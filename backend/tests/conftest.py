"""Shared fixtures. Unit tests never see real API keys or the real storage directory."""

import inspect
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.config import settings
from app.services.session_service import SessionService

START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: calls real external services; opt in with RUN_LIVE_TESTS=1"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live test; set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_settings(request, monkeypatch, tmp_path):
    """Keep real credentials and the real storage directory out of every non-live test."""
    if request.node.get_closest_marker("live"):
        return
    for name in ("llm_api_key", "pinecone_api_key", "pinecone_index_name", "pinecone_namespace"):
        monkeypatch.setattr(settings, name, None)
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")


class FakeClock:
    """Replaces session_service._now so expiry can be tested without sleeping."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("app.services.session_service._now", fake)
    return fake


@pytest.fixture
def sessions_dir(tmp_path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def make_service(sessions_dir):
    """Factory, so a test can build a second service on the same directory to simulate a restart."""

    def _make(expiry_hours: int = 24, directory: Path | None = None) -> SessionService:
        return SessionService(directory or sessions_dir, expiry_hours)

    return _make


@pytest.fixture
def session_service(make_service) -> SessionService:
    return make_service()


# --------------------------------------------------------------------------- #
# Fake LLM
# --------------------------------------------------------------------------- #
@dataclass
class LLMCall:
    messages: list
    json_mode: bool
    temperature: float | None


class FakeLLM:
    """Replaces generate_response. Queue one item per expected call; each item is one of:

    - a string: returned as the model's reply
    - an exception instance: raised
    - a callable taking the messages (sync or async) and returning the reply string

    An unexpected call (nothing queued) raises AssertionError. Every call is recorded.
    """

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []
        self._queue: deque = deque()

    def queue(self, *items):
        self._queue.extend(items)
        return self

    async def __call__(self, messages, *, json_mode=False, temperature=None):
        self.calls.append(LLMCall(list(messages), json_mode, temperature))
        if not self._queue:
            raise AssertionError("FakeLLM received a call but no reply was queued")
        item = self._queue.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(list(messages))
            if inspect.isawaitable(item):
                item = await item
        return item

    @staticmethod
    def turn(reply="Understood.", details=None, allegations_add=None, allegations_remove=None) -> str:
        """Build the JSON reply the orchestrator expects from the model."""
        return json.dumps(
            {
                "reply": reply,
                "details": details or {},
                "allegations_add": allegations_add or [],
                "allegations_remove": allegations_remove or [],
            }
        )

    @staticmethod
    def detail(value, uncertain: bool = False) -> dict:
        return {"value": value, "uncertain": uncertain}


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr("app.agent.orchestrator.generate_response", fake)
    monkeypatch.setattr("app.services.draft_service.generate_response", fake)
    return fake