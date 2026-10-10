"""Tests for integrations/llm_client.py.

No provider is contacted: HTTP goes through httpx.MockTransport, and settings are patched so a
real key from .env is never used. The tests are plain sync functions that call asyncio.run.
"""

import asyncio
import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.integrations import llm_client
from app.integrations.llm_client import (
    LLMConfigurationError,
    LLMError,
    LLMMessage,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    generate_response,
)

OPENAI_HOST = "api.openai.com"
GEMINI_HOST = "generativelanguage.googleapis.com"
SECRET_TEXT = "SECRET-NARRATIVE-TEXT"
API_KEY = "test-api-key-123"


def openai_reply(text):
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ]
    }


def gemini_reply(text):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ]
    }


def user(text):
    return LLMMessage("user", text)


def call(*messages, **kwargs):
    return asyncio.run(generate_response(list(messages), **kwargs))


class MockProvider:
    """Stands in for a provider's HTTP API and records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = lambda request: httpx.Response(
            500, json={"error": {"message": "mock provider not configured for this test"}}
        )
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def handle(self, handler) -> None:
        self._handler = handler

    def reply_with(self, payload, status: int = 200) -> None:
        self._handler = lambda request: httpx.Response(status, json=payload)

    def fail_with(self, exc: Exception) -> None:
        def raiser(request):
            raise exc

        self._handler = raiser

    @property
    def last_json(self):
        return json.loads(self.requests[-1].content)


@pytest.fixture
def provider(monkeypatch):
    mock = MockProvider()
    monkeypatch.setattr(llm_client, "_client", mock.client)
    yield mock
    asyncio.run(mock.client.aclose())


@pytest.fixture
def configure(monkeypatch):
    def _configure(name="openai", model="test-model", api_key=API_KEY):
        monkeypatch.setattr(settings, "llm_provider", name)
        monkeypatch.setattr(settings, "llm_model", model)
        monkeypatch.setattr(settings, "llm_api_key", SecretStr(api_key) if api_key else None)

    return _configure


@pytest.fixture(autouse=True)
def default_configuration(configure):
    configure()


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
def test_openai_returns_the_reply_text(provider):
    provider.reply_with(openai_reply("Hello there"))
    assert call(user("Hi")) == "Hello there"


def test_openai_request_shape(provider, configure):
    configure("openai", model="gpt-test")
    provider.reply_with(openai_reply("ok"))
    call(
        LLMMessage("system", "Be brief."),
        LLMMessage("user", "Hello"),
        LLMMessage("assistant", "Hi"),
        user("Next question"),
    )

    request = provider.requests[0]
    assert request.method == "POST"
    assert request.url.host == OPENAI_HOST
    assert str(request.url) == llm_client.OPENAI_URL
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert "x-goog-api-key" not in request.headers
    assert request.headers["content-type"] == "application/json"

    body = provider.last_json
    assert body["model"] == "gpt-test"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Next question"},
    ]
    assert "temperature" not in body
    assert "response_format" not in body


def test_openai_json_mode_requests_a_json_object(provider):
    provider.reply_with(openai_reply("{}"))
    call(user("Reply with JSON"), json_mode=True)
    assert provider.last_json["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("temperature", [0.0, 0.2, 1.0])
def test_openai_temperature_is_sent_when_given_including_zero(provider, temperature):
    provider.reply_with(openai_reply("ok"))
    call(user("Hi"), temperature=temperature)
    assert provider.last_json["temperature"] == temperature


def test_openai_api_key_is_only_in_the_authorization_header(provider):
    provider.reply_with(openai_reply("ok"))
    call(user("Hi"))
    request = provider.requests[0]
    assert API_KEY not in str(request.url)
    assert API_KEY not in request.content.decode()


def test_openai_malformed_structures_are_response_errors(provider):
    for payload in ({}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]}, []):
        provider.reply_with(payload)
        with pytest.raises(LLMResponseError):
            call(user("Hi"))


@pytest.mark.parametrize("content", [None, ""])
def test_openai_empty_content_is_a_response_error(provider, content):
    provider.reply_with(openai_reply(content))
    with pytest.raises(LLMResponseError):
        call(user("Hi"))


def test_openai_concurrent_calls_share_one_client_without_crosstalk(provider):
    def echo(request):
        body = json.loads(request.content)
        return httpx.Response(200, json=openai_reply(body["messages"][-1]["content"]))

    provider.handle(echo)

    async def many():
        return await asyncio.gather(*(generate_response([user(f"msg-{i}")]) for i in range(5)))

    assert asyncio.run(many()) == [f"msg-{i}" for i in range(5)]
    assert len(provider.requests) == 5


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def test_gemini_returns_the_reply_text(provider, configure):
    configure("gemini")
    provider.reply_with(gemini_reply("Hello there"))
    assert call(user("Hi")) == "Hello there"


def test_gemini_request_shape_and_key_placement(provider, configure):
    configure("gemini", model="gemini-test")
    provider.reply_with(gemini_reply("ok"))
    call(user("Hi"))

    request = provider.requests[0]
    expected_url = llm_client.GEMINI_URL.format(model="gemini-test")
    assert request.method == "POST"
    assert str(request.url) == expected_url
    assert request.url.host == GEMINI_HOST
    assert request.headers["x-goog-api-key"] == API_KEY
    assert "authorization" not in request.headers
    assert API_KEY not in str(request.url)
    assert API_KEY not in request.content.decode()


def test_gemini_maps_roles_and_merges_system_messages(provider, configure):
    configure("gemini")
    provider.reply_with(gemini_reply("ok"))
    call(
        LLMMessage("system", "Rule A"),
        LLMMessage("system", "Rule B"),
        LLMMessage("user", "Hello"),
        LLMMessage("assistant", "Hi"),
        LLMMessage("user", "Next"),
    )
    body = provider.last_json
    assert body["system_instruction"] == {"parts": [{"text": "Rule A\n\nRule B"}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi"}]},
        {"role": "user", "parts": [{"text": "Next"}]},
    ]


def test_gemini_without_system_messages_omits_system_instruction(provider, configure):
    configure("gemini")
    provider.reply_with(gemini_reply("ok"))
    call(user("Hi"))
    assert "system_instruction" not in provider.last_json


def test_gemini_strips_a_models_prefix_from_the_model_name(provider, configure):
    configure("gemini", model="models/gemini-test")
    provider.reply_with(gemini_reply("ok"))
    call(user("Hi"))
    assert provider.requests[0].url.path.endswith("/models/gemini-test:generateContent")
    assert "models/models" not in str(provider.requests[0].url)


def test_gemini_without_options_sends_no_generation_config(provider, configure):
    configure("gemini")
    provider.reply_with(gemini_reply("ok"))
    call(user("Hi"))
    assert "generationConfig" not in provider.last_json


def test_gemini_json_mode_and_temperature(provider, configure):
    configure("gemini")
    provider.reply_with(gemini_reply("{}"))
    call(user("Hi"), json_mode=True, temperature=0.0)
    assert provider.last_json["generationConfig"] == {
        "temperature": 0.0,
        "responseMimeType": "application/json",
    }


def test_gemini_joins_multiple_text_parts(provider, configure):
    configure("gemini")
    provider.reply_with(
        {"candidates": [{"content": {"parts": [{"text": "Hello, "}, {"text": "world"}]}}]}
    )
    assert call(user("Hi")) == "Hello, world"


def test_gemini_requires_a_non_system_message_and_makes_no_request(provider, configure):
    configure("gemini")
    with pytest.raises(ValueError):
        call(LLMMessage("system", "Only instructions"))
    assert provider.requests == []


def test_gemini_blocked_prompt_reports_the_reason(provider, configure):
    configure("gemini")
    provider.reply_with({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMResponseError, match="SAFETY"):
        call(user("Hi"))


def test_gemini_no_candidates_is_a_response_error(provider, configure):
    configure("gemini")
    provider.reply_with({})
    with pytest.raises(LLMResponseError):
        call(user("Hi"))


def test_gemini_candidate_without_text_reports_the_finish_reason(provider, configure):
    configure("gemini")
    provider.reply_with({"candidates": [{"finishReason": "SAFETY"}]})
    with pytest.raises(LLMResponseError, match="SAFETY"):
        call(user("Hi"))


# --------------------------------------------------------------------------- #
# Provider selection and configuration
# --------------------------------------------------------------------------- #
def test_switching_provider_needs_only_configuration(provider, configure):
    def by_host(request):
        if request.url.host == OPENAI_HOST:
            return httpx.Response(200, json=openai_reply("from openai"))
        return httpx.Response(200, json=gemini_reply("from gemini"))

    provider.handle(by_host)

    configure("openai")
    assert call(user("Hi")) == "from openai"
    configure("gemini")
    assert call(user("Hi")) == "from gemini"
    assert [r.url.host for r in provider.requests] == [OPENAI_HOST, GEMINI_HOST]


def test_provider_name_is_case_and_whitespace_insensitive(provider, configure):
    configure("  GeMiNi ")
    provider.reply_with(gemini_reply("ok"))
    assert call(user("Hi")) == "ok"
    assert provider.requests[0].url.host == GEMINI_HOST


def test_unsupported_provider_is_a_configuration_error_and_makes_no_request(provider, configure):
    configure("mystery-ai")
    with pytest.raises(LLMConfigurationError) as exc_info:
        call(user("Hi"))
    message = str(exc_info.value)
    assert "mystery-ai" in message
    assert "openai" in message and "gemini" in message
    assert provider.requests == []


@pytest.mark.parametrize("name", ["openai", "gemini"])
def test_missing_api_key_is_a_configuration_error_and_makes_no_request(provider, configure, name):
    configure(name, api_key=None)
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        call(user("Hi"))
    assert provider.requests == []


def test_empty_message_list_is_rejected_before_any_request(provider):
    with pytest.raises(ValueError):
        asyncio.run(generate_response([]))
    assert provider.requests == []


def test_all_llm_exceptions_share_a_base_class():
    for error in (LLMConfigurationError, LLMTimeoutError, LLMProviderError, LLMResponseError):
        assert issubclass(error, LLMError)


# --------------------------------------------------------------------------- #
# HTTP and network failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_http_errors_become_provider_errors_with_the_status_code(provider, status):
    provider.reply_with({"error": {"message": "Something went wrong upstream"}}, status)
    with pytest.raises(LLMProviderError) as exc_info:
        call(user("Hi"))
    error = exc_info.value
    assert error.status_code == status
    assert str(status) in str(error)
    assert "Something went wrong upstream" in str(error)
    assert "openai" in str(error)


def test_gemini_http_errors_name_the_provider(provider, configure):
    configure("gemini")
    provider.reply_with({"error": {"code": 400, "message": "API key not valid"}}, 400)
    with pytest.raises(LLMProviderError, match="gemini") as exc_info:
        call(user("Hi"))
    assert exc_info.value.status_code == 400


def test_long_provider_error_messages_are_truncated(provider):
    provider.reply_with({"error": {"message": "E" * 1000}}, 400)
    with pytest.raises(LLMProviderError) as exc_info:
        call(user("Hi"))
    assert "E" * 300 in str(exc_info.value)
    assert "E" * 301 not in str(exc_info.value)


def test_non_json_error_bodies_are_included_but_bounded(provider):
    provider.handle(lambda request: httpx.Response(502, text="Bad gateway " + "x" * 5000))
    with pytest.raises(LLMProviderError) as exc_info:
        call(user("Hi"))
    assert "Bad gateway" in str(exc_info.value)
    assert len(str(exc_info.value)) < 300


def test_error_json_without_an_error_message_falls_back_to_the_body(provider):
    provider.reply_with({"detail": "nope"}, 400)
    with pytest.raises(LLMProviderError, match="nope"):
        call(user("Hi"))


@pytest.mark.parametrize(
    "timeout_error",
    [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout],
)
def test_timeouts_become_timeout_errors_with_the_configured_duration(
    provider, monkeypatch, timeout_error
):
    monkeypatch.setattr(settings, "llm_timeout_seconds", 2.5)
    provider.fail_with(timeout_error("too slow"))
    with pytest.raises(LLMTimeoutError, match=r"2\.5s"):
        call(user("Hi"))


def test_connection_failures_are_provider_errors_without_a_status(provider):
    provider.fail_with(httpx.ConnectError("name resolution failed"))
    with pytest.raises(LLMProviderError, match="ConnectError") as exc_info:
        call(user("Hi"))
    assert exc_info.value.status_code is None
    assert not isinstance(exc_info.value, LLMTimeoutError)


def test_a_non_json_success_body_is_a_response_error(provider):
    provider.handle(lambda request: httpx.Response(200, content=b"<html>not json</html>"))
    with pytest.raises(LLMResponseError):
        call(user("Hi"))


# --------------------------------------------------------------------------- #
# Privacy: neither errors nor logs may contain message text or the API key
# --------------------------------------------------------------------------- #
FAILURES = {
    "http_401": lambda p: p.reply_with({"error": {"message": "bad key"}}, 401),
    "http_500": lambda p: p.reply_with({"error": {"message": "boom"}}, 500),
    "timeout": lambda p: p.fail_with(httpx.ReadTimeout("slow")),
    "connect": lambda p: p.fail_with(httpx.ConnectError("down")),
    "non_json": lambda p: p.handle(lambda request: httpx.Response(200, content=b"<html>")),
    "empty": lambda p: p.reply_with(openai_reply("")),
}


@pytest.mark.parametrize("failure", list(FAILURES))
def test_errors_and_logs_never_contain_message_text_or_the_api_key(provider, caplog, failure):
    FAILURES[failure](provider)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError) as exc_info:
            call(user(SECRET_TEXT))
    for text in (str(exc_info.value), caplog.text):
        assert SECRET_TEXT not in text
        assert API_KEY not in text


def test_successful_calls_do_not_log_message_text_or_the_api_key(provider, caplog):
    provider.reply_with(openai_reply("ok"))
    with caplog.at_level(logging.DEBUG):
        call(user(SECRET_TEXT))
    assert caplog.text  # the request was logged (provider, model, message count)
    assert SECRET_TEXT not in caplog.text
    assert API_KEY not in caplog.text


# --------------------------------------------------------------------------- #
# Shared client lifecycle
# --------------------------------------------------------------------------- #
def test_the_shared_client_is_created_once_with_the_configured_timeout(monkeypatch):
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(settings, "llm_timeout_seconds", 7.5)
    try:
        first = llm_client._get_client()
        second = llm_client._get_client()
        assert first is second
        assert first.timeout == httpx.Timeout(7.5)
    finally:
        asyncio.run(llm_client.close_llm_client())


def test_close_llm_client_closes_and_resets_the_client(provider):
    asyncio.run(llm_client.close_llm_client())
    assert provider.client.is_closed
    assert llm_client._client is None


def test_close_llm_client_is_safe_to_call_repeatedly(monkeypatch):
    monkeypatch.setattr(llm_client, "_client", None)
    asyncio.run(llm_client.close_llm_client())
    asyncio.run(llm_client.close_llm_client())
    assert llm_client._client is None