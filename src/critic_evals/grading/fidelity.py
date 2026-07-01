"""Fidelity — a gate. The critique's central objection must fairly represent the argument.

A gate because a false central target disqualifies a critique regardless of its other merits, and
it fails in two mirror directions: claiming what the text does NOT say (fabrication) and ignoring
what the text DOES say (attacking an objection the argument already pre-empts via a qualifier or
concession — a false flaw). Over-stating the *severity* of a REAL flaw is not a fidelity failure
(decisiveness grades that). Reads the argument + the reference's concessions (the pre-empted list).

Levels (normalized level/2 → the composite caps the whole score at level 0):
  0 — the central objection FABRICATES (a quote/contradiction not in the text) OR attacks a point
      the argument already PRE-EMPTS (a false flaw).
  1 — the central objection holds, but a PERIPHERAL claim fabricates or attacks a pre-empted point.
  2 — fully faithful: every attribution is accurate to the text and none is already pre-empted.
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
            "description": "0 central objection is fabricated (quote/contradiction not in the text) OR attacks a point the argument pre-empts (a false flaw) / 1 central holds but a peripheral claim fabricates or attacks a pre-empted point / 2 fully faithful and none pre-empted",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's central attribution, and whether the text supports it and has not already pre-empted it",
        },
    },
}

_PROMPT = """FIDELITY. Rate whether the critique's CENTRAL objection fairly represents the ARGUMENT — it must neither fabricate nor attack a point the argument already pre-empts. The argument's own concessions / qualifiers (objections it PRE-EMPTS) are:
{concessions}
- 0: the central objection FAILS on either count — it is FABRICATED (a quote/contradiction not in the text) OR it attacks a point the argument already PRE-EMPTS via one of the qualifiers above (a false flaw).
- 1: the central objection holds and is not pre-empted, but a PERIPHERAL claim fabricates or attacks a pre-empted point.
- 2: fully faithful — every attribution is accurate to the text and none is already pre-empted.
A REAL, un-pre-empted flaw whose severity is over-stated still scores 2 on fidelity (severity is graded elsewhere); only FABRICATION or attacking a PRE-EMPTED point scores 0.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class FidelityGrader(LLMGrader):
    name: str = "fidelity"
    max_tokens: int = 1000

    def build_prompt(self, *, argument: str, reference: Reference, critique: str) -> str:
        conc = reference.get("concessions", ())
        listed = "\n".join(f"- {c}" for c in conc) if isinstance(conc, (list, tuple)) and conc else "(none listed)"
        return _PROMPT.format(concessions=listed, argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
