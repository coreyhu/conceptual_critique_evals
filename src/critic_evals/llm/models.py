from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    label: str  # short, human/file-friendly
    model_id: str  # Anthropic API id
    supports_effort: bool = False  # Haiku 4.5 rejects `effort`; Opus/Sonnet accept it


MODELS: dict[str, ModelSpec] = {
    "opus-4.8": ModelSpec("opus-4.8", "claude-opus-4-8", supports_effort=True),
    "sonnet-4.6": ModelSpec("sonnet-4.6", "claude-sonnet-4-6", supports_effort=True),
    "haiku-4.5": ModelSpec("haiku-4.5", "claude-haiku-4-5", supports_effort=False),
}

# Ordered strongest -> weakest: the ordering the eval signal must recover.
DEFAULT_MODELS: tuple[str, ...] = ("opus-4.8", "sonnet-4.6", "haiku-4.5")


def resolve(labels: Iterable[str]) -> list[ModelSpec]:
    """Resolve label strings to :class:`ModelSpec`, with a clear error on typos."""
    specs: list[ModelSpec] = []
    for label in labels:
        try:
            specs.append(MODELS[label])
        except KeyError:
            known = ", ".join(MODELS)
            raise KeyError(f"unknown model label {label!r}; known: {known}") from None
    return specs
