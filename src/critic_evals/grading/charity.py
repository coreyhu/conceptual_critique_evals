"""Charity — a gate. The critique must engage the argument's strongest form, not a weakman.

A gate because winning against a version the author does not hold is not a real critique. The
strongest reading comes from the reference (`key`). Reference input, but only the steelman — not
a flaw list.

Levels (normalized level/2 → the composite caps the whole score at level 0):
  0 — the central objection attacks a WEAKMAN: a version the argument does not hold.
  1 — engages a real reading, but not the strongest form.
  2 — engages the argument's STRONGEST form.
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
            "description": "0 central objection attacks a weakman / 1 engages a real but not strongest form / 2 engages the strongest form",
        },
        "evidence": {
            "type": "string",
            "description": "which form of the argument the critique's central objection targets",
        },
    },
}

_PROMPT = """CHARITY. The argument's strongest form is:
{key}

Rate whether the CRITIQUE's central objection engages that strongest form.
- 0: it attacks a WEAKMAN — a version the argument does not actually hold.
- 1: it engages a real reading, but not the strongest form.
- 2: it engages the argument's STRONGEST form.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class CharityGrader(LLMGrader):
    name: str = "charity"
    max_tokens: int = 900

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(
            key=str(reference.get("strongest_reading", "")),
            argument=argument,
            critique=critique,
        )

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
