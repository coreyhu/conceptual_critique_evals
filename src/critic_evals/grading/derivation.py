"""Derivation axis: where does the critique's main point get its force — asserted, imported, or
immanent (grounded in the argument's own reasoning)?

Depth-oriented, not flaw-oriented: for a flaw, immanent means turning the argument's own commitments
against its conclusion; for a SOUND argument, immanent means showing why the conclusion genuinely
follows from those commitments (engaging the actual inferential structure, not generic praise). So
the axis rewards depth of engagement on both without routing.
"""

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
            "description": "0 asserted or a recited label / 1 grounded in a necessary external premise or in quoted text that only identifies the target / 2 immanent: an argument-owned commitment generates the critique's point, turning it against the conclusion or explaining why the conclusion follows",
        },
        "operative_quote": {
            "type": "string",
            "description": "what the critique's central point rests on",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, whether the critique relies on assertion, a label, or an external premise; if level 2, name the argument-owned commitment doing the work",
        },
    },
}

_PROMPT = """DERIVATION. Focus on the critique's primary line of engagement — the point it treats as most important (the same one the other axes judge). Rate where its force comes from — how deeply it is grounded in the argument's own reasoning. This applies whether the point is a flaw or an affirmation of a sound argument.
- 0: miss — asserted or lazy import. A bare claim (a flaw named without derivation, or generic praise like 'well reasoned') or a recited external label ('a slippery slope', 'an appeal to authority') used as a stand-in for engagement.
- 1: good — necessary import or text-supported diagnosis. It grounds the point in an external premise or corroboration the argument cannot reject, applied to a specific point; OR it quotes/paraphrases the argument accurately but uses the text mainly to identify the target rather than to generate the objection. It survives the STRIP TEST (remove the imported premise and the point still stands).
- 2: excellent — immanent. The argument's OWN commitments, examples, or distinctions GENERATE the point: for a flaw, the critique turns those commitments against the conclusion; for a SOUND argument, it shows why the conclusion genuinely follows from them (engaging the actual inferential structure, not generic praise). The point should be hard to make without the argument's specific reasoning.

Do not award level 2 for a vivid external counterexample, for confident praise, or for merely quoting the right sentence. If the force depends on adding a new empirical example, outside literature, or a broad external theory, max level 1. Level 2 requires the operative premise to come from the argument itself AND to do the work of the critique.

COMMON FALSE TOP-SCORES:
- A concrete external counterexample that is correct and damaging is usually level 1, not 2, unless the argument already grants the premise it relies on.
- A named framework ('a slippery slope', 'begging the question') then paraphrased. If the argument-specific derivation disappears when the label is removed, score 0.
- Quoting the argument only to identify the target, or restating its conclusion as praise. Level 2 requires the quoted commitment to GENERATE the point (the objection, or the reason the conclusion holds).
- Correctly saying "the argument itself says X" but then applying a generic objection to X. Score 1 unless X itself creates the contradiction, limitation, or support.

EXAMPLES — a rising ladder of critiques of the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall":
- 0 (miss): "The claim that parking drove shoppers away just isn't convincing." Or: "This is just a textbook hasty generalization." (bare denial or generic label)
- 1 (good): "Malls also win on selection, anchor stores, and comfort — well-documented drivers of where people shop — so parking cannot be assumed the operative difference." (external premise the argument can't reject; survives the strip test)
- 2 (excellent): "The argument's OWN premise — that shoppers defected to save on parking — casts them as cost-comparison shoppers, who by that logic also weigh the mall's lower prices and selection; so its own model makes parking one factor among several." (turns the argument's own commitment against its conclusion)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0 (miss): "Sounds right, well argued." (generic praise, no derivation)
- 2 (excellent): "The argument's own framing — a claim about the population AVERAGE — is precisely what makes the outlier harmless: an average is defined over the whole distribution, so a single long-lived case is exactly the kind of point it already absorbs. The conclusion follows from the argument's own terms." (shows why the conclusion holds from the argument's own commitments)

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class DerivationGrader(LLMGrader):
    name: str = "derivation"

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
