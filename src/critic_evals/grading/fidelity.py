"""Fidelity gate: does the critique fabricate what the argument says?"""

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
            "description": "0 central objection fabricated (a quote/contradiction not in the text) / 1 central faithful but a peripheral claim misquotes or manufactures / 2 fully faithful (a fair paraphrase/inference counts), or the critique endorses (no objection)",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's central attribution and whether the text supports it",
        },
    },
}

_PROMPT = """FIDELITY. Rate whether the critique's CENTRAL objection is FABRICATED. Distinguish a fair characterization or inference about what the argument says (faithful) from citing something the text does not contain.
- 0: the central objection FABRICATES — it cites a quote, or a contradiction/tension, that the text does NOT actually contain.
- 1: the central objection is faithful, but a PERIPHERAL claim misquotes or manufactures.
- 2: fully faithful — every attribution is accurate to the text (a fair paraphrase or inference about the argument's own reasoning counts as faithful); OR the critique makes no objection (it endorses), so there is nothing to fabricate.
A REAL flaw whose severity is over-stated still scores 2 (severity is graded elsewhere); only FABRICATION of the central target scores 0.

EXAMPLES — for the unrelated argument "Downtown shops are losing customers because the suburban mall offers free parking and downtown charges for it; so make downtown parking free and shoppers return" (a gate: 0 disqualifies):
- 0 (fabricated): "The argument insists parking is the ONLY reason downtown struggles and dismisses online shopping outright." (both attributions are invented — the text says neither)
- 1 (peripheral slip): the central objection faithfully targets the single-cause leap, but tacks on "it even concedes the fee is trivially small" — a detail the text never states.
- 2 (faithful): "The argument treats the parking fee as the sole cause of the customer loss, when the mall differs in many other ways." (an accurate paraphrase of what the argument actually assumes) — or the critique simply endorses, so there is nothing to fabricate.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class FidelityGrader(LLMGrader):
    name: str = "fidelity"
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
