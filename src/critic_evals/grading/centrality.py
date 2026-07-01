"""Centrality axis: does the critique engage the argument's most load-bearing element?

Element-oriented, not flaw-oriented: for a FLAWED argument the load-bearing element is the decisive
flaw; for a SOUND argument it is the central reasoning the conclusion rests on, and a critique that
engages THAT (to affirm, sharpen, or bound it) is central, while one that fixates on a peripheral
aside is not. So the axis works on both without routing.
"""

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
            "description": "0 engages a peripheral / merely-relevant element / 1 engages a genuinely load-bearing element but not shown to be the one the conclusion most depends on / 2 engages the dominant element the conclusion most depends on (a decisive flaw, or the central reasoning of a sound argument), correctly weighted",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's primary point and why the element it engages is (or isn't) central to the conclusion",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain the missing condition; if level 2, say why this is the element the conclusion most depends on",
        },
    },
}

_PROMPT = """CENTRALITY. Focus on the critique's primary line of engagement — the point it treats as most important (the same one the other axes judge). Rate how LOAD-BEARING the element it engages is — how close to what the conclusion actually rests on.

This is NOT about finding a flaw. For a FLAWED argument the load-bearing element is the decisive flaw. For a SOUND argument it is the central reasoning the conclusion depends on — a critique that engages THAT (to affirm, sharpen, or bound it) is central; one that fixates on a peripheral aside is not.

As a GUIDE, the argument's known load-bearing gaps are:
{key}
(If none are listed, treat the argument as having no decisive flaw — credit the critique for engaging the central reasoning the conclusion stands on, NOT for manufacturing a gap.) A critique may engage an element MORE central than the list; credit that too. Do not award the top level just for naming a listed gap, or for generic praise — top centrality requires correct weighting: the critique must show why this element is the one the conclusion most depends on.
- 0: miss — engages a peripheral or merely-relevant element, not what the conclusion rests on.
- 1: good — engages a genuinely load-bearing element, but does not show it is the dominant one, or ranks it alongside lesser points.
- 2: excellent — engages the dominant element the conclusion most depends on (a decisive flaw, or the central reasoning of a sound argument), correctly weighted, with lesser points shown to be subordinate.

COMMON FALSE TOP-SCORES:
- Names a listed gap (or, for a sound argument, merely restates the conclusion) without showing why it is the load-bearing element rather than one concern among several. Score 1.
- Lists many points. Breadth does not imply centrality unless it ranks the main one and explains why the rest are subordinate.
- Uses broad language ("the core issue", "the real problem") but never justifies the weighting. Do not infer dominance from confident wording.

EXAMPLES — a rising ladder of critiques of the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall":
- 0 (miss): "It never gives figures for the customer loss or tax revenue." (real observations, but not what the conclusion stands on)
- 1 (good): "The whole case assumes the parking fee — not selection or comfort — is what drives shoppers away." (load-bearing, but just names the missing alternatives)
- 2 (excellent): "The conclusion works only if parking is the binding constraint; fee size and implementation are secondary — if selection or comfort dominate, free parking recovers nothing. Causal identification of the binding constraint is the bottleneck the whole remedy rests on." (central, correctly weighted)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0 (miss): "It doesn't define 'life expectancy' precisely." (a peripheral aside, not what the conclusion rests on)
- 2 (excellent): "The conclusion rests entirely on the distinction between a population-average claim and a single individual outcome — the critique engages exactly that load-bearing point, and correctly, since one outlier is precisely what an average absorbs." (engages the central reasoning the conclusion depends on)

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
            if isinstance(gaps, (list, tuple)) and gaps
            else "(none provided)"
        )
        return _PROMPT.format(key=listed, argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
