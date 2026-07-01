"""Decisiveness — a graded axis. Does the critique's finding convert to damage, or sit inert?

The great-vs-good line is often the *verdict*, not the finding: two critiques reach the same flaw
but one calls it fatal (the conclusion/remedy fails as a matter of KIND) and the other calls it a
fixable gap ("too thin", "underdeveloped"). It scores CORRECT severity in both directions —
judging the *actual* damage, not the critique's confidence — so it catches under-calling a fatal
flaw as fixable AND over-calling a minor flaw as fatal. Reference-free.

Levels (normalized level/2):
  0 — inert: the critique under-calls it as a patchable gap, OR over-calls a minor flaw as fatal
      when it does not actually damage the conclusion.
  1 — partial: a real non-decisive dent, or a genuinely fatal flaw the critique hedges or demotes.
  2 — fatal: the flaw genuinely makes the conclusion fail as a matter of kind, and the critique
      correctly treats it so.
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
            "description": "0 inert (the critique under-calls it as 'too thin'/'underdeveloped', OR over-calls a minor flaw as 'fatal' when it does not actually damage the conclusion) / 1 partial (a real non-decisive dent, or a fatal flaw hedged/demoted) / 2 fatal (the flaw genuinely makes the conclusion fail as a matter of kind, and the critique correctly treats it so)",
        },
        "operative_quote": {
            "type": "string",
            "description": "the critique's operative verdict on its primary finding",
        },
    },
}

_PROMPT = """DECISIVENESS. Focus on the critique's PRIMARY finding. Rate whether that finding TRULY damages the argument — whether, granting it, the conclusion or its remedy fails as a matter of KIND, not degree. Judge the ACTUAL damage, not the critique's confidence: correct severity in BOTH directions is what scores.
- 0: inert — the finding does not undermine the conclusion, EITHER because the critique itself treats it as a patchable gap ('too thin', 'underdeveloped', 'somewhat circular'), OR because it OVER-calls a minor point as 'fatal' when the flaw does not actually damage the conclusion.
- 1: partial — a real but non-decisive dent, or a genuinely fatal flaw the critique hedges or ranks below a lesser point.
- 2: fatal — the flaw genuinely makes the conclusion fail as a matter of kind, AND the critique correctly treats it so.
A critique that merely DECLARES a flaw fatal earns 2 only if it truly is; over-calling a minor flaw as fatal is a 0-1, never a 2. Quote the operative verdict.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class DecisivenessGrader(LLMGrader):
    name: str = "decisiveness"
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
