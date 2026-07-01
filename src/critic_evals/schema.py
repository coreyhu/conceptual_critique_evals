"""Typed data structures shared across the platform.

Plain `dataclasses` describing the inputs the platform reasons about (an
argument to be critiqued) and the transcript rows it emits (one critique
elicited from one model).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict, cast


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


class UsageDict(TypedDict):
    """Serialized form of `Usage`."""

    input_tokens: int
    output_tokens: int


class CritiqueRow(TypedDict):
    """Serialized form of `CritiqueRecord` — one JSON object per transcript line."""

    item_id: str
    model: str
    model_id: str
    sample: int
    system: str
    prompt: str
    response: str
    success: bool
    stop_reason: str | None
    usage: UsageDict | None
    request_id: str | None
    timestamp: str
    error: str | None


@dataclass(frozen=True, slots=True)
class CritiqueRecord:
    """One critique elicited from one model — a single transcript row.

    Carries the full prompt and response plus enough metadata to reproduce and
    audit the call. `error` is populated instead of `response` when the call
    failed, so failures stay visible in the transcript rather than being dropped.
    """

    item_id: str
    model: str  # short label, e.g. "opus-4.8"
    model_id: str  # resolved Anthropic API id, e.g. "claude-opus-4-8"
    sample: int
    system: str
    prompt: str
    response: str
    success: bool
    stop_reason: str | None
    usage: Usage | None
    request_id: str | None
    timestamp: str  # ISO-8601 UTC
    error: str | None = None

    def to_dict(self) -> CritiqueRow:
        """Serialize to a JSON-ready row (nested `Usage` flattened recursively)."""
        return cast(CritiqueRow, asdict(self))

    @classmethod
    def from_dict(cls, data: CritiqueRow) -> CritiqueRecord:
        """Rebuild a record from a transcript row (inverse of `to_dict`)."""
        usage = data["usage"]
        return cls(
            item_id=data["item_id"],
            model=data["model"],
            model_id=data["model_id"],
            sample=data["sample"],
            system=data["system"],
            prompt=data["prompt"],
            response=data["response"],
            success=data.get("success", data.get("error") is None),
            stop_reason=data["stop_reason"],
            usage=(
                Usage(usage["input_tokens"], usage["output_tokens"])
                if usage is not None
                else None
            ),
            request_id=data["request_id"],
            timestamp=data["timestamp"],
            error=data["error"],
        )
