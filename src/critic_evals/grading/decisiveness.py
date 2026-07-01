"""Decisiveness axis: how much damage does the main objection establish?"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level", "operative_quote", "why_not_higher"],
    "properties": {
        "level": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0 flags a point or overclaims fatality without damage / 1 materially weakens but repairable or hedged / 2 passes the repair test: conclusion must change kind, not merely add support",
        },
        "operative_quote": {
            "type": "string",
            "description": "the critique's operative verdict on its primary finding",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain what prevents fatality under the repair test; if level 2, state why repair would require changing the conclusion's kind",
        },
    },
}

_PROMPT = """DECISIVENESS. Focus on the critique's most load-bearing objection — its primary line of attack (the same one the other axes judge); if the critique makes several points, evaluate the one it treats as most important. Rate how far the critique CONVERTS that finding into damage. Judge the ACTUAL damage established, not the critique's confidence.

Use the REPAIR TEST before assigning the top level: if the author could plausibly repair the problem by adding evidence, a qualification, or more detail while keeping the same kind of conclusion, the critique is not fatal. Top-level fatality requires showing that the conclusion or remedy must materially change kind.

- 0: miss — the critique only flags a point ('this is assumed', 'too thin', 'underdeveloped') without showing damage, OR over-claims 'fatal' without establishing it.
- 1: good — the critique materially weakens the conclusion, but the same conclusion might survive with more evidence, a narrower scope, or a filled-in mechanism; also use this for a fatal flaw that is hedged or demoted.
- 2: excellent — the critique rigorously establishes that, granting the flaw, the conclusion fails as a matter of kind and must retreat to a materially different thesis or remedy.
A bare 'fatal' claim without the work is a 0, never a top score. Quote the operative verdict.

COMMON FALSE TOP-SCORES:
- The critique says "fatal" or "undermines the argument" but only shows that more evidence is needed. Score 1 at most.
- The critique identifies a real structural issue but frames it as "underdeveloped," "too thin," or "needs more detail." If the critique itself treats the flaw as repairable, do not score 2.
- The critique damages a broad version of the conclusion, but a narrower version would survive with the same basic thesis. Score 1 unless it shows even the repaired/narrowed thesis changes kind.

EXAMPLES — a rising ladder of critiques of the unrelated argument "Downtown shops are losing customers because the suburban mall offers free parking and downtown charges for it; so make downtown parking free and shoppers return":
- 0 (miss): "The argument leans on the assumption that the parking fee is what drove shoppers to the mall, which it never really establishes." (correctly names the assumption, but leaves it as a flagged concern)
- 1 (good): "...the remedy needs evidence that parking costs are large enough relative to selection and comfort to matter. Without that evidence, the proposal is under-supported." (serious, but repairable with causal evidence)
- 2 (excellent): "...the remedy works ONLY if parking is the binding cause; the argument's own story frames shoppers as total-cost comparers, so parking is one variable among selection, prices, travel time, and comfort. The policy must retreat from 'free parking will bring shoppers back' to 'parking is one possible lever to test'." (passes the repair test: the conclusion changes kind)

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
