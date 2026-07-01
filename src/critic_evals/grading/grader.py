"""Base interfaces for axis and composite graders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from critic_evals.llm.client import AnthropicClient

_SYSTEM = "You are a rigorous, adversarial referee of argument critiques. Judge substance, not eloquence."

Reference = dict[str, object]


@dataclass(frozen=True, slots=True)
class GraderScore:
    score: float  # 0-1 (a single axis: its normalized level; the composite: the final score)
    dimensions: dict[str, float]
    raw: dict[str, object]


def level_score(parsed: dict[str, object], hi: int) -> float:
    """Normalize a discrete rubric level in [0, hi] to [0, 1] (the LLM emits the raw level)."""
    raw = parsed.get("level", 0)
    n = int(raw) if isinstance(raw, (int, float)) else 0
    return max(0, min(hi, n)) / hi


class BaseGrader(ABC):
    """Anything that scores a critique into a [0,1] GraderScore against a reference."""

    __slots__ = ()

    name: str

    @abstractmethod
    async def grade(
        self,
        client: AnthropicClient,
        *,
        model_id: str,
        argument: str,
        reference: Reference,
        critique: str,
    ) -> GraderScore:
        """Score `critique` in [0,1]."""


class LLMGrader(BaseGrader):
    """A single-aspect judge: one LLM call. Subclasses supply prompt, schema, and extract."""

    __slots__ = ()

    max_tokens: int

    @abstractmethod
    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        """Render this axis's user prompt for one critique."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, object]:
        """JSON schema constraining the model's structured output."""

    @abstractmethod
    def extract(self, parsed: dict[str, object]) -> float:
        """Turn the parsed response into this axis's [0,1] score."""

    async def grade(
        self,
        client: AnthropicClient,
        *,
        model_id: str,
        argument: str,
        reference: Reference,
        critique: str,
    ) -> GraderScore:
        out = (
            await client.complete(
                model_id=model_id,
                system=_SYSTEM,
                prompt=self.build_prompt(
                    argument=argument, reference=reference, critique=critique
                ),
                schema=self.schema,
                max_tokens=self.max_tokens,
            )
        ).parsed or {}
        return GraderScore(score=self.extract(out), dimensions={}, raw=out)
