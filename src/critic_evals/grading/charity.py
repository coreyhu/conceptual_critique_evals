"""Charity gate: does the critique survive the argument's strongest form?"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level", "evidence", "why_not_higher"],
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
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain the stronger form or concession the critique fails to grant; if level 2, explain why the objection survives the strongest form",
        },
    },
}

_PROMPT = """CHARITY. The argument's strongest form is:
{key}

The argument's important concessions / pre-empted easy objections are:
{concessions}

Rate whether the CRITIQUE's central objection engages that strongest form. Top charity is not merely "no strawman." A good critique attacks a real reading; an excellent critique first preserves what makes the argument tempting, including its concessions, and then shows why the objection still works.
- 0: it attacks a WEAKMAN — a version the argument does not actually hold.
- 1: it engages a real reading, but not the strongest form.
- 2: it engages the argument's STRONGEST form: the critique's central objection still works after granting the argument's main concessions and modest formulation.

COMMON FALSE TOP-SCORES:
- The critique says "even granting..." but grants only a weak concession, not the strongest form listed above. Score 1.
- The critique accurately attacks a sentence in the argument, but ignores a nearby qualification or concession that makes the argument more modest. Score 1 unless the objection still works after that qualification.
- The critique rebuts an extreme version ("always", "only", "inevitably") when the argument only needs a marginal or probabilistic claim. Score 0 or 1 depending on whether that extreme reading is actually present.

EXAMPLES — for the unrelated argument "make downtown parking free to win shoppers back from the free-parking mall", whose strongest form is "parking cost is a real friction that, at the margin, tips some price-sensitive shoppers toward the mall" (a gate: 0 disqualifies):
- 0 (weakman): "The argument assumes shoppers pick stores SOLELY by parking and would never pay a dollar to shop where they prefer." (a caricature the argument never commits to)
- 1 (real, not strongest): "It treats 'free parking brings them back' as an all-or-nothing prediction and rebuts that" — engages a real reading, but not the modest marginal-friction version the argument actually needs.
- 2 (strongest): "Even granting parking is a genuine friction at the margin, the fee would have to be a large enough share of the mall's total advantage to reverse the trend — which is never shown." (engages the strongest form)

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
        concessions = reference.get("concessions", ())
        listed = (
            "\n".join(f"({i + 1}) {c}" for i, c in enumerate(concessions))
            if isinstance(concessions, (list, tuple)) and concessions
            else "(none provided)"
        )
        return _PROMPT.format(
            key=str(reference.get("strongest_reading", "")),
            concessions=listed,
            argument=argument,
            critique=critique,
        )

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
