"""Centrality axis: does the critique target the argument's dominant bottleneck?"""

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
            "description": "0 peripheral or relevant-but-not-central / 1 load-bearing flaw but not shown to dominate / 2 dominant bottleneck that the conclusion most depends on",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's primary finding and why it is (or isn't) central to the conclusion",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain the missing condition; if level 2, say why the dominant-bottleneck test is met",
        },
    },
}

_PROMPT = """CENTRALITY. Focus on the critique's most load-bearing objection — its primary line of attack (the same one the other axes judge); if the critique makes several points, evaluate the one it treats as most important. Rate how LOAD-BEARING that objection is — how close to what the conclusion rests on. As a GUIDE, the argument's known load-bearing gaps are:
{key}
A critique may expose a flaw MORE serious than these — credit that too; do NOT require it to match the list. But do not award the top level just for naming a listed gap. Top centrality requires correct weighting: the critique must show why this objection is the dominant bottleneck, not merely one valid concern.
- 0: miss — peripheral, merely relevant, or not what the conclusion rests on.
- 1: good — a genuinely load-bearing flaw central to the conclusion, but the critique does not show it is the dominant bottleneck or ranks it alongside weaker points.
- 2: excellent — the dominant bottleneck: the critique identifies the objection the conclusion most depends on, correctly ranks it, and explains why easier objections are subordinate or merely coordinate.

COMMON FALSE TOP-SCORES:
- A critique names a listed load-bearing gap, but does not show why that gap is the bottleneck rather than one important concern among several. Score 1, not 2.
- A critique lists many serious flaws. Breadth does not imply top centrality unless it ranks the main flaw and explains why the rest are subordinate.
- A critique uses broad language like "the core issue" or "the real problem" but never justifies the weighting. Do not infer dominance from confident wording.

EXAMPLES — a rising ladder of critiques of the unrelated argument "Downtown shops are losing customers because the suburban mall offers free parking and downtown charges for it; so make downtown parking free and shoppers return":
- 0 (miss): "The argument never gives figures for the customer loss or tax revenue." (real observations, but not the assumption the conclusion stands on)
- 1 (good): "The whole case assumes the parking fee — not the mall's selection, anchor stores, or comfort — is what drives shoppers away." (load-bearing, but still just names the missing alternative causes)
- 2 (excellent): "The conclusion works only if parking is the binding constraint. Tax costs, exact fee size, and implementation details are secondary; if selection or comfort dominate shopper choice, free parking does not recover the trend at all. So causal identification of the binding constraint is the bottleneck the whole remedy rests on." (central and correctly weighted)

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class CentralityGrader(LLMGrader):
    name: str = "centrality"

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        gaps = reference.get("load_bearing_gaps", ())
        listed = (
            "\n".join(f"({i + 1}) {g}" for i, g in enumerate(gaps))
            if isinstance(gaps, (list, tuple))
            else "(none provided)"
        )
        return _PROMPT.format(key=listed, argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
