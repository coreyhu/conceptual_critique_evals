"""Calibration — the graded axis for SOUND arguments. Did the critique correctly recognize it?

On a sound argument there is no flaw to find, so centrality/derivation/decisiveness do not apply.
The skill is calibration, and it fails in two mirror directions: over-critical (manufacturing
flaws / tearing it down) and under-critical (a vacuous rubber-stamp that never engages). A great
critique affirms it, engages the reasoning, and adds real value. Reference-free.

Levels (normalized level/2):
  0 — miscalibrated, either direction: over-critical (manufactures flaws, tears it down, overstates
      a minor caveat as decisive) OR under-critical (a vacuous rubber-stamp — 'looks fine' — that
      never engages the reasoning).
  1 — endorse: correctly affirms it AND engages the reasoning, at most minor caveats.
  2 — adds value: affirms it AND adds a real, proportionate boundary condition, limitation, or
      stronger reformulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level", "operative_quote"],
    "properties": {
        "level": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0 miscalibrated either way — over-critical (manufactures flaws / tears it down / overstates a minor caveat) OR under-critical (a vacuous rubber-stamp that never engages) / 1 endorse (affirms AND engages, minor caveats) / 2 adds value (affirms AND adds a real boundary condition or stronger reformulation)",
        },
        "operative_quote": {"type": "string"},
    },
}

_PROMPT = """CALIBRATION. The ARGUMENT is fundamentally SOUND. Rate whether the CRITIQUE correctly and substantively RECOGNIZES this.
- 0: miscalibrated in EITHER direction — over-critical (manufactures flaws, reflexively tears it down, or overstates a minor caveat as decisive) OR under-critical (a vacuous rubber-stamp — 'looks fine' — that never engages the reasoning).
- 1: endorse — correctly affirms it AND engages the reasoning, with at most minor caveats.
- 2: adds value — affirms it AND adds a real, proportionate boundary condition, limitation, or stronger reformulation.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class CalibrationGrader(LLMGrader):
    name: str = "calibration"
    max_tokens: int = 1000

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
