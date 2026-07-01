"""Centrality — a graded axis. Is the critique's primary finding a load-bearing flaw?

The floor a critique must clear: did it attack a flaw that is central to the argument's
conclusion, rather than a peripheral technicality? This replaces coverage-style recall, which
wrongly punishes depth — a critique that goes narrow-and-deep on the single most serious flaw, or
exposes one *more* serious than the reference lists, is excellent and must not lose to a shallow
critique that touches more flaws. The reference's load-bearing gaps are a GUIDE for what counts as
central, not a checklist: a flaw deeper than the list still scores 2, and missing lesser gaps is
never penalized. Reference-light (uses the gap list only to calibrate 'central').

Levels (normalized level/2):
  0 — the primary finding is a peripheral technicality or nitpick; not load-bearing.
  1 — a real, relevant flaw, but not the most load-bearing one.
  2 — a genuinely load-bearing flaw central to the conclusion (a listed gap, or a more serious
      flaw the list missed).
"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level", "evidence"],
    "properties": {
        "level": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0 the primary finding is peripheral / a nitpick / not load-bearing / 1 a real relevant flaw but not the most load-bearing / 2 a genuinely load-bearing flaw central to the conclusion (a listed gap, OR a more serious flaw the list missed)",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's primary finding and why it is (or isn't) central to the conclusion",
        },
    },
}

_PROMPT = """CENTRALITY. Rate whether the critique's PRIMARY finding targets a genuinely LOAD-BEARING flaw — one central to the argument's conclusion. As a GUIDE, the argument's known load-bearing gaps are:
{key}
A critique may also expose a flaw MORE serious than these — credit that too; do NOT require it to match the list, and do NOT penalize it for missing lesser gaps.
- 0: the primary finding is a peripheral technicality or nitpick, not load-bearing.
- 1: a real, relevant flaw, but not the most load-bearing one.
- 2: a genuinely load-bearing flaw central to the conclusion (a listed gap, or a more serious flaw the list missed).

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class CentralityGrader(LLMGrader):
    name: str = "centrality"
    max_tokens: int = 1000

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        gaps = reference.get("load_bearing_gaps", ())
        listed = (
            "\n".join(f"({i + 1}) {g}" for i, g in enumerate(gaps))
            if isinstance(gaps, (list, tuple))
            else "(none provided)"
        )
        return _PROMPT.format(key=listed, argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
