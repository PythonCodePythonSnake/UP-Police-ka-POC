"""LLM integration: one async interface over OpenAI and Google Gemini (REST via httpx)."""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """Base class for all LLM integration errors."""


class LLMConfigurationError(LLMError):
    """Provider is unsupported or credentials are missing."""


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMProviderError(LLMError):
    """The provider was unreachable or returned an HTTP error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """The provider replied, but with an empty, blocked, or malformed response."""


# --------------------------------------------------------------------------- #
# Shared HTTP client
# --------------------------------------------------------------------------- #
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
    return _client


async def close_llm_client() -> None:
    """Close the shared HTTP client. Call from the app's shutdown/lifespan handler."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:300]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


async def _post_json(
    provider: str, url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        response = await _get_client().post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"{provider} did not respond within {settings.llm_timeout_seconds:g}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMProviderError(f"Could not reach {provider}: {type(exc).__name__}.") from exc

    if response.is_error:
        logger.warning("%s returned HTTP %s", provider, response.status_code)
        raise LLMProviderError(
            f"{provider} returned HTTP {response.status_code}: {_error_detail(response)}",
            status_code=response.status_code,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise LLMResponseError(f"{provider} returned a non-JSON response.") from exc


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
async def _generate_openai(
    messages: list[LLMMessage], api_key: str, json_mode: bool, temperature: float | None
) -> str:
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        # OpenAI requires the word "JSON" to appear somewhere in the messages.
        payload["response_format"] = {"type": "json_object"}

    data = await _post_json(
        "openai", OPENAI_URL, {"Authorization": f"Bearer {api_key}"}, payload
    )
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError("openai response had an unexpected structure.") from exc
    if not text:
        raise LLMResponseError("openai returned an empty response.")
    return text


async def _generate_gemini(
    messages: list[LLMMessage], api_key: str, json_mode: bool, temperature: float | None
) -> str:
    system_text = "\n\n".join(m.content for m in messages if m.role == "system")
    payload: dict[str, Any] = {
        "contents": [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
    }
    if not payload["contents"]:
        raise ValueError("Gemini requires at least one user or assistant message.")
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}

    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    if generation_config:
        payload["generationConfig"] = generation_config

    url = GEMINI_URL.format(model=settings.llm_model.removeprefix("models/"))
    data = await _post_json("gemini", url, {"x-goog-api-key": api_key}, payload)

    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        detail = f" (blocked: {reason})" if reason else ""
        raise LLMResponseError(f"gemini returned no candidates{detail}.")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        finish = candidates[0].get("finishReason", "unknown")
        raise LLMResponseError(f"gemini returned no text (finish reason: {finish}).")
    return text


_PROVIDERS = {
    "openai": _generate_openai,
    "gemini": _generate_gemini,
}


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
async def generate_response(
    messages: list[LLMMessage],
    *,
    json_mode: bool = False,
    temperature: float | None = None,
) -> str:
    """Send messages to the configured provider and return the reply text.

    json_mode asks the provider for a JSON-only reply (for later structured extraction);
    the caller is responsible for parsing and validating the returned string.
    temperature is omitted from the request unless given, since some models reject it.

    Raises LLMConfigurationError, LLMTimeoutError, LLMProviderError, or LLMResponseError.
    """
    if not messages:
        raise ValueError("messages must not be empty.")

    provider = settings.llm_provider.strip().lower()
    generator = _PROVIDERS.get(provider)
    if generator is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Supported: {supported}."
        )
    if settings.llm_api_key is None:
        raise LLMConfigurationError("LLM_API_KEY is not set.")

    logger.debug("LLM request: provider=%s model=%s messages=%d", provider, settings.llm_model, len(messages))
    return await generator(
        messages, settings.llm_api_key.get_secret_value(), json_mode, temperature
    )