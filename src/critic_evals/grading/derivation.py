"""Derivation axis: where does the critique's main objection get its force?"""

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
            "description": "0 asserted or lazy imported frame / 1 necessary external premise applied specifically / 2 immanent: turns the argument's own commitments/examples against its conclusion",
        },
        "operative_quote": {
            "type": "string",
            "description": "what the critique's central flaw rests on",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain whether the critique relies on assertion, a label, or an external premise; if level 2, name the argument-owned commitment doing the work",
        },
    },
}

_PROMPT = """DERIVATION. Focus on the critique's most load-bearing objection — its primary line of attack (the same one the other axes judge); if the critique makes several points, evaluate the one it treats as most important. Rate where its force comes from.
- 0: miss — asserted or lazy import. It names a flaw without derivation, or recites an external law/frame ('correlation isn't causation', 'Goodhart's law') as a stand-in.
- 1: good — necessary import. It deploys an external premise the argument cannot reject, to a specific point; it survives the STRIP TEST (remove the imported premise and the flaw still stands).
- 2: excellent — immanent. It turns the argument's OWN commitments, examples, distinctions, or concessions against its conclusion, using only what the argument already grants.

Do not award level 2 just because the critique gives a vivid counterexample. If the objection's force depends on adding a new empirical example, outside literature, or a broad external theory, max level 1. Level 2 requires that the damaging premise comes from the argument itself.

COMMON FALSE TOP-SCORES:
- The critique gives a concrete external counterexample that is correct and damaging. That is usually level 1, not 2, unless the argument itself already grants the premise the example relies on.
- The critique uses a named framework ("Goodhart's law", "correlation is not causation", "incentives") and then paraphrases the framework. If the argument-specific derivation disappears when the label is removed, score 0.
- The critique sounds immanent because it quotes the argument, but the quote only identifies the target. Level 2 requires the quoted commitment to generate the objection.

EXAMPLES — a rising ladder of critiques of the unrelated argument "Downtown shops are losing customers because the suburban mall offers free parking and downtown charges for it; so make downtown parking free and shoppers return":
- 0 (miss): "The claim that parking is what drove shoppers away just isn't convincing." Or: "This is a textbook correlation-isn't-causation error." (bare denial or generic label)
- 1 (good): "Malls also win on selection, anchor stores, and indoor comfort — well-documented drivers of where people shop — so parking cannot be assumed the operative difference." (an external premise the argument can't reject; survives the strip test)
- 2 (excellent): "The argument's own premise — that shoppers defected to save on parking — casts them as cost-comparison shoppers, who by that same logic also weigh the mall's lower prices and larger selection; so its own model of the shopper makes parking one factor among several, not the decisive cause." (turns the argument's own commitment against its conclusion)

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
        return level_score(parsed, 2)
