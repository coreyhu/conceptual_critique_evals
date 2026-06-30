"""Grader variants compared in the methodology experiment.

A grader maps (argument, key, critique) to a quality score in [0,1] so variants are
comparable. `BaseGrader` owns the shared mechanics (build a prompt, call the model,
wrap the result); each concrete grader supplies only the three pieces that differ —
the prompt, the output schema, and how a parsed response becomes a score. Holistic
graders emit one number; rubric graders score five weighted dimensions. Constructor
toggles place each named grader on a ladder that adds one mechanic at a time: the
reference key (holistic_nokey -> holistic_key) and claim decomposition
(rubric_nodecomp -> rubric_decomp).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from critic_evals.llm.client import AnthropicClient

WEIGHTS: dict[str, float] = {
    "centrality": 0.30,
    "mechanism_depth": 0.25,
    "precision": 0.20,
    "faithfulness": 0.15,
    "calibration": 0.10,
}

_SYSTEM = "You grade the quality of critiques of arguments. Grade substance, not eloquence."

_DIMENSION_GUIDE = """Score each dimension from 0.0 to 1.0:
- centrality: did it reach the load-bearing flaw? tier1 ~1.0, tier2 ~0.5, textbook ~0.25, none 0.
- mechanism_depth: does it explain WHY the central flaw is fatal, engaging the argument's specific move, rather than merely naming it?
- precision: 1.0 if it asserts no false/unsupported flaws; drop sharply for each manufactured or keyed-false flaw it asserts.
- faithfulness: are its characterizations of the argument accurate (no misreading, strawman, or fabricated quotation)?
- calibration: correct critical stance for the argument's soundness; penalize endorsing a real flaw, reflexively tearing down a sound argument, or inflating a minor point into a fatal one.
Grade SUBSTANCE, not eloquence: a fluent, confident critique that manufactures a flaw, misreads, inflates severity, or only recites the textbook critique must score LOW."""

_HOLISTIC_SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": ["score", "justification"],
    "properties": {
        "score": {"type": "number", "description": "overall critique quality, 0-100"},
        "justification": {"type": "string"},
    },
}

_DIM_PROPS: dict[str, object] = {k: {"type": "number"} for k in WEIGHTS}

_RUBRIC_SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": [*WEIGHTS, "justification"],
    "properties": {**_DIM_PROPS, "justification": {"type": "string"}},
}

_RUBRIC_DECOMP_SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": ["claims", *WEIGHTS, "justification"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["claim", "maps_to", "faithful"],
                "properties": {
                    "claim": {"type": "string"},
                    "maps_to": {"type": "string", "enum": [
                        "tier1", "tier2", "textbook", "false_flaw", "novel_valid", "other"]},
                    "faithful": {"type": "boolean"},
                },
            },
        },
        **_DIM_PROPS,
        "justification": {"type": "string"},
    },
}


@dataclass(frozen=True, slots=True)
class GraderScore:
    score: float  # 0-1
    dimensions: dict[str, float]  # empty for holistic variants
    raw: dict[str, object]


def _clamp(x: object, hi: float) -> float:
    return max(0.0, min(hi, float(x)))  # type: ignore[arg-type]


def _aggregate(dims: dict[str, float]) -> float:
    return sum(WEIGHTS[k] * dims[k] for k in WEIGHTS)


class BaseGrader(ABC):
    """A grading strategy: scores one critique into a [0,1] GraderScore.

    Subclasses supply the three varying pieces — `build_prompt`, `schema`, and
    `extract` — and `grade` runs the shared mechanics that wrap them.
    """

    __slots__ = ()

    name: str
    max_tokens: int

    @abstractmethod
    def build_prompt(self, *, argument: str, key: str, critique: str) -> str:
        """Render the grader's user prompt for one critique."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, object]:
        """JSON schema constraining the model's structured output."""

    @abstractmethod
    def extract(self, parsed: dict[str, object]) -> tuple[float, dict[str, float]]:
        """Turn the parsed response into a [0,1] score and its per-dimension scores."""

    async def grade(
        self,
        client: AnthropicClient,
        *,
        model_id: str,
        argument: str,
        key: str,
        critique: str,
    ) -> GraderScore:
        """Score `critique` with this grader. Returns a [0,1] score (+ dims)."""
        prompt = self.build_prompt(argument=argument, key=key, critique=critique)
        out = (await client.complete(
            model_id=model_id, system=_SYSTEM, prompt=prompt,
            schema=self.schema, max_tokens=self.max_tokens,
        )).parsed or {}
        score, dimensions = self.extract(out)
        return GraderScore(score=score, dimensions=dimensions, raw=out)


@dataclass(frozen=True, slots=True)
class HolisticGrader(BaseGrader):
    """Emits a single 0-100 quality score, normalized to [0,1]; no dimensions."""

    name: str
    use_key: bool
    max_tokens: int = 1200

    def build_prompt(self, *, argument: str, key: str, critique: str) -> str:
        key_block = (
            f"\n\nREFERENCE KEY (load-bearing flaw, valid flaws, and known FALSE flaws):\n{key}"
            if self.use_key else ""
        )
        return (
            "Rate the overall QUALITY of this critique of an argument on a 0-100 scale. "
            "Judge substance, not eloquence."
            f"{key_block}\n\nARGUMENT:\n{argument}\n\nCRITIQUE:\n{critique}"
        )

    @property
    def schema(self) -> dict[str, object]:
        return _HOLISTIC_SCHEMA

    def extract(self, parsed: dict[str, object]) -> tuple[float, dict[str, float]]:
        return _clamp(parsed.get("score", 0), 100.0) / 100.0, {}


@dataclass(frozen=True, slots=True)
class RubricGrader(BaseGrader):
    """Scores five dimensions against the reference key and aggregates with WEIGHTS.

    When `decompose`, the model first maps each critique claim to a key element before
    scoring, using the wider schema that captures that decomposition.
    """

    name: str
    decompose: bool
    max_tokens: int = 2500

    def build_prompt(self, *, argument: str, key: str, critique: str) -> str:
        lead = (
            "First decompose the critique into its main claims, mapping each to a key element "
            "(tier1/tier2/textbook/false_flaw/novel_valid/other) and whether it is faithful to the text. Then "
            if self.decompose else ""
        )
        return (
            f"{lead}score the critique on the rubric below.\n\n{_DIMENSION_GUIDE}\n\n"
            f"ARGUMENT:\n{argument}\n\nREFERENCE KEY:\n{key}\n\nCRITIQUE:\n{critique}"
        )

    @property
    def schema(self) -> dict[str, object]:
        return _RUBRIC_DECOMP_SCHEMA if self.decompose else _RUBRIC_SCHEMA

    def extract(self, parsed: dict[str, object]) -> tuple[float, dict[str, float]]:
        dims = {k: _clamp(parsed.get(k, 0), 1.0) for k in WEIGHTS}
        return _aggregate(dims), dims


GRADERS: dict[str, BaseGrader] = {
    g.name: g
    for g in (
        HolisticGrader(name="holistic_nokey", use_key=False),
        HolisticGrader(name="holistic_key", use_key=True),
        RubricGrader(name="rubric_nodecomp", decompose=False),
        RubricGrader(name="rubric_decomp", decompose=True),
    )
}

GRADER_NAMES: tuple[str, ...] = tuple(GRADERS)
