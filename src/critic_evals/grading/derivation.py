"""Derivation — a graded axis, four tiers. Where does the critique's flaw get its force?

An excellent critique derives the flaw from the argument's OWN commitments and turns them against
its conclusion (immanent); a weaker one imports an external law/frame and recites it (lazy import)
or asserts the flaw. The strip test decides lazy-vs-necessary: would the flaw still stand if the
imported premise were removed? Four tiers because the strip test genuinely separates them.
Reference-free: reads the argument + critique.

Levels (normalized level/3):
  0 — asserted: claimed, not derived; no support from the text or a clear principle.
  1 — lazy import: recites an external law/frame (e.g. 'Goodhart's law') as a stand-in for
      engagement; the flaw leans on the import.
  2 — necessary import: relies on an external premise the argument cannot reject, deployed to a
      specific point; survives the strip test.
  3 — immanent: turns the argument's own commitments/examples against its conclusion.
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
            "enum": [0, 1, 2, 3],
            "description": "0 asserted (not derived) / 1 lazy import (recites an external law as a stand-in) / 2 necessary import (external premise the argument cannot reject, survives the strip test) / 3 immanent (turns the argument's own commitments against its conclusion)",
        },
        "operative_quote": {
            "type": "string",
            "description": "what the critique's central flaw rests on",
        },
    },
}

_PROMPT = """DERIVATION. Focus on the critique's PRIMARY flaw-claim. Rate where its force comes from.
- 0: asserted — claimed, not derived; no support from the text or a clear principle.
- 1: lazy import — recites an external law/frame as a stand-in for engagement.
- 2: necessary import — relies on an external premise the argument cannot reject, deployed to a specific point.
- 3: immanent — turns the argument's OWN commitments/examples against its conclusion.
STRIP TEST: would the flaw still stand if the imported premise were removed? Survives → 2/3; collapses → 0/1.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class DerivationGrader(LLMGrader):
    name: str = "derivation"
    max_tokens: int = 1000

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 3)
