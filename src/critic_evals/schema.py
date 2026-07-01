"""Small shared data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArgumentItem:
    """A question and an argument answering it — the unit a model critiques.

    `id` is a short stable handle (e.g. the source filename stem) that joins a
    critique back to the argument it targets across transcripts.
    """

    id: str
    question: str
    argument: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for a single model call."""

    input_tokens: int
    output_tokens: int
