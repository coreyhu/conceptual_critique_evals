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
            "description": "0 asserted, generic, or label-reciting / 1 specifically grounded in the argument's text or in a necessary external premise / 2 exceptional: an argument-owned commitment generates a deeper diagnosis and rules out a nearby shallower derivation",
        },
        "operative_quote": {
            "type": "string",
            "description": "what the critique's central point rests on",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain what deeper argument-generated move is missing; if level 2, name the argument-owned commitment and how it elevates the critique",
        },
    },
}

_PROMPT = """DERIVATION. Focus on the critique's primary line of engagement — the point it treats as most important (the same one the other axes judge). Rate where its force comes from — how deeply it is grounded in the argument's own reasoning. This applies whether the point is a flaw or an affirmation of a sound argument.
- 0: miss — asserted or lazy import. A bare claim (a flaw named without derivation, or generic praise like 'well reasoned') or a recited external label ('a slippery slope', 'an appeal to authority') used as a stand-in for engagement.
- 1: solid — specifically grounded. The critique accurately uses the argument's text, examples, distinctions, or a necessary external premise to support its point. It is not generic and not merely a label. This includes ordinary immanent objections that correctly turn one stated premise against another.
- 2: excellent — argument-generated insight. The argument's OWN commitments do more than support the point; they unlock a deeper diagnosis that materially improves the critique AND distinguish it from a nearby shallower derivation. For a flaw, the critique shows how the argument's own best move creates the problem; for a SOUND argument, it shows why the argument survives a tempting objection from its own inferential structure. The point should be hard to see without tracing the argument's specific machinery.

Do not award level 2 for a vivid external counterexample, for confident praise, for merely quoting the right sentence, or for a correct ordinary immanent objection. If the force depends on adding a new empirical example, outside literature, or a broad external theory, max level 1. Level 2 requires the argument-owned premise to create a non-obvious diagnosis, not just support a correct point, and the critique must make clear why a shallower derivation would miss the real mechanism.

COMMON FALSE TOP-SCORES:
- A concrete external counterexample that is correct and damaging is level 1, not 2, unless the argument's own commitments make the counterexample diagnose the argument's structure.
- A named framework ('a slippery slope', 'begging the question') then paraphrased. If the argument-specific derivation disappears when the label is removed, score 0.
- Quoting the argument only to identify the target, or restating its conclusion as praise. Level 2 requires the quoted commitment to GENERATE a deeper diagnosis.
- Correctly saying "the argument itself says X" and deriving a valid objection from X is usually level 1. Score 2 only if X explains why the argument seemed plausible and where its machinery breaks.
- A critique that uses the argument's own text well but does not contrast that mechanism with a plausible shallower account is usually level 1.

EXAMPLES — a rising ladder of critiques of the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall":
- 0 (miss): "The claim that parking drove shoppers away just isn't convincing." Or: "This is just a textbook hasty generalization." (bare denial or generic label)
- 1 (solid): "The argument assumes shoppers defected to save on parking; that premise itself leaves open that other mall advantages, like price and selection, may also be decisive." (grounded in the argument, but ordinary)
- 2 (excellent): "The argument's OWN premise — that shoppers defected to save on parking — casts them as cost-comparison shoppers, who by that logic also weigh the mall's lower prices and selection; so its own model makes parking one factor among several rather than the binding constraint. The argument's best causal picture generates the objection." (turns the best internal model into the diagnosis)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0 (miss): "Sounds right, well argued." (generic praise, no derivation)
- 1 (solid): "The argument's own framing is a population-average claim, so a single long-lived case does not refute it." (grounded in the argument's terms)
- 2 (excellent): "The argument's own framing — a claim about the population AVERAGE — is precisely what makes the outlier harmless: an average is defined over the whole distribution, so a single long-lived case is exactly the kind of point it already absorbs. That also explains why the tempting anecdote-based objection fails rather than merely being irrelevant." (shows why the conclusion holds and why the tempting critique fails)

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
