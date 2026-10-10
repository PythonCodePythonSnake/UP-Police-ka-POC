"""Tests for agent/prompts.py and the system prompt the orchestrator assembles from it."""

import re

import pytest

from app.agent import prompts
from app.agent.orchestrator import _STATIC_SYSTEM_PROMPT
from app.agent.state_manager import FIELD_DESCRIPTIONS, REQUIRED_FIELDS

PROMPT_NAMES = ["INTAKE_PROMPT", "GUIDANCE_PROMPT", "SOURCES_UNAVAILABLE_NOTE", "DRAFT_PROMPT"]


def text_of(name: str) -> str:
    return getattr(prompts, name)


def norm(text: str) -> str:
    """Lowercase and collapse whitespace so line wrapping never breaks a phrase check."""
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------- #
# Every prompt
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_is_a_substantial_string(name):
    value = text_of(name)
    assert isinstance(value, str)
    assert len(value.strip()) > 100


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_has_no_leftover_template_or_escape_artifacts(name):
    value = text_of(name)
    assert "{" not in value and "}" not in value
    assert "\\" not in value


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_is_provider_neutral(name):
    assert not re.search(r"\b(openai|chatgpt|gpt|gemini|claude|anthropic|google)\b", norm(text_of(name)))


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_does_not_cite_specific_statutes_or_numbered_provisions(name):
    value = text_of(name)
    assert not re.search(r"\b(IPC|BNS|BNSS|CrPC)\b", value)
    assert not re.search(r"\b(section|article)\s+\d+", norm(value))


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "phrase",
    [
        "never ask again",
        "never invent facts",
        "two or three",
        "emergency",
        "have not filed",
        "nothing you say means a complaint has been accepted",
    ],
)
def test_intake_prompt_contains_its_key_rules(phrase):
    assert phrase in norm(prompts.INTAKE_PROMPT)


# --------------------------------------------------------------------------- #
# Guidance and the unavailable-sources note
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "phrase",
    [
        "never invent or guess circulars",
        "cite only what appears in the provided sources",
        "if no source material is provided",
        "general information",
        "qualified legal professional",
    ],
)
def test_guidance_prompt_contains_its_key_rules(phrase):
    assert phrase in norm(prompts.GUIDANCE_PROMPT)


@pytest.mark.parametrize(
    "phrase", ["do not cite any circular", "could not be consulted", "general only"]
)
def test_sources_unavailable_note_forbids_citations(phrase):
    assert phrase in norm(prompts.SOURCES_UNAVAILABLE_NOTE)


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "phrase",
    [
        "only the complaint details",
        "it has not been submitted",
        "do not invent or infer any fact",
        "[not provided]",
        "the complainant states that",
    ],
)
def test_draft_prompt_contains_its_key_rules(phrase):
    assert phrase in norm(prompts.DRAFT_PROMPT)


def test_draft_prompt_defines_eight_numbered_sections_in_order():
    assert re.findall(r"^(\d)\. ", prompts.DRAFT_PROMPT, re.MULTILINE) == list("12345678")

    headings = [
        "complainant details",
        "date, time, and place of incident",
        "person(s) complained against",
        "description of events",
        "loss, harm, or damage",
        "evidence and witnesses",
        "action requested",
        "missing information",
    ]
    text = norm(prompts.DRAFT_PROMPT)
    positions = [text.find(heading) for heading in headings]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


# --------------------------------------------------------------------------- #
# The system prompt the orchestrator builds from these pieces
# --------------------------------------------------------------------------- #
def test_assembled_prompt_includes_every_building_block_verbatim():
    for name in ("INTAKE_PROMPT", "GUIDANCE_PROMPT", "SOURCES_UNAVAILABLE_NOTE"):
        assert text_of(name) in _STATIC_SYSTEM_PROMPT


def test_assembled_prompt_mentions_json_which_openai_json_mode_requires():
    assert "JSON" in _STATIC_SYSTEM_PROMPT


def test_assembled_prompt_has_no_unfilled_placeholder():
    assert "__FIELDS__" not in _STATIC_SYSTEM_PROMPT


def test_assembled_prompt_lists_every_field_and_marks_only_required_ones():
    for name in FIELD_DESCRIPTIONS:
        required_line = f"- {name} (required):"
        optional_line = f"- {name}:"
        if name in REQUIRED_FIELDS:
            assert required_line in _STATIC_SYSTEM_PROMPT, name
        else:
            assert optional_line in _STATIC_SYSTEM_PROMPT, name
            assert required_line not in _STATIC_SYSTEM_PROMPT, name


def test_assembled_prompt_describes_the_expected_reply_keys():
    for key in ('"reply"', '"details"', '"allegations_add"', '"allegations_remove"'):
        assert key in _STATIC_SYSTEM_PROMPT