"""Tests for the deterministic, non-API logic: argument parsing, prompt
construction, the model registry, and JSONL round-trip.
"""

from __future__ import annotations

import pytest

from critic_evals.critique import CRITIQUE_PROMPTS, build_prompt
from critic_evals.llm.models import MODELS, resolve
from critic_evals.schema import ArgumentItem, CritiqueRecord, Usage
from critic_evals.transcripts import parse_argument, read_jsonl, write_jsonl

SAMPLE_MD = """# Why do estimates overrun?

## Argument
Estimates overrun because of the planning fallacy.
They also overrun because of scope creep.
"""


def test_parse_argument_splits_question_and_body():
    item = parse_argument(SAMPLE_MD, item_id="estimates")
    assert item.id == "estimates"
    assert item.question == "Why do estimates overrun?"
    assert item.argument.startswith(
        "Estimates overrun because of the planning fallacy."
    )
    assert "scope creep" in item.argument
    # the heading lines themselves are not part of the argument body
    assert "## Argument" not in item.argument


def test_parse_argument_without_argument_heading_uses_body_after_question():
    text = "# Q?\n\nBody line one.\nBody line two.\n"
    item = parse_argument(text, item_id="x")
    assert item.question == "Q?"
    assert item.argument == "Body line one.\nBody line two."


def test_parse_argument_rejects_empty_body():
    with pytest.raises(ValueError):
        parse_argument("# Just a question\n", item_id="x")


def test_build_prompt_fills_question_and_argument():
    item = ArgumentItem(id="x", question="Why?", argument="Because.")
    system, prompt = build_prompt(item, variant="natural")
    assert "critic" in system.lower()
    assert "Why?" in prompt
    assert "Because." in prompt


def test_build_prompt_every_variant_accepts_fields():
    item = ArgumentItem(id="x", question="Q", argument="A")
    for variant in CRITIQUE_PROMPTS:
        _, prompt = build_prompt(item, variant=variant)
        assert "Q" in prompt and "A" in prompt


def test_build_prompt_unknown_variant_raises():
    item = ArgumentItem(id="x", question="Q", argument="A")
    with pytest.raises(KeyError):
        build_prompt(item, variant="does-not-exist")


def test_model_registry_ids_and_effort_flags():
    assert MODELS["opus-4.8"].model_id == "claude-opus-4-8"
    assert MODELS["sonnet-4.6"].model_id == "claude-sonnet-4-6"
    assert MODELS["haiku-4.5"].model_id == "claude-haiku-4-5"
    # Haiku rejects `effort`; the registry must record that so callers can branch.
    assert MODELS["haiku-4.5"].supports_effort is False
    assert MODELS["opus-4.8"].supports_effort is True


def test_resolve_unknown_label_raises():
    with pytest.raises(KeyError):
        resolve(["opus-4.8", "nope"])


def test_jsonl_roundtrip_preserves_records(tmp_path):
    records = [
        CritiqueRecord(
            item_id="estimates",
            model="opus-4.8",
            model_id="claude-opus-4-8",
            prompt_variant="natural",
            sample=0,
            system="sys",
            prompt="prompt",
            response="a critique",
            success=True,
            stop_reason="end_turn",
            usage=Usage(input_tokens=120, output_tokens=300),
            request_id="req_abc",
            timestamp="2026-06-30T00:00:00+00:00",
        ),
        CritiqueRecord(
            item_id="estimates",
            model="haiku-4.5",
            model_id="claude-haiku-4-5",
            prompt_variant="natural",
            sample=1,
            system="sys",
            prompt="prompt",
            response="",
            success=False,
            stop_reason="error",
            usage=None,
            request_id=None,
            timestamp="2026-06-30T00:00:01+00:00",
            error="boom",
        ),
    ]
    path = write_jsonl(records, tmp_path / "t.jsonl")
    loaded = read_jsonl(path)
    assert loaded == records  # frozen dataclasses compare by value
