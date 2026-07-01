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
            "description": "0 engages a peripheral / merely-relevant element / 1 correctly identifies the dominant load-bearing element / 2 exceptional: chooses the revision-relevant fulcrum among plausible alternatives and explicitly rejects a nearby tempting but less central target",
        },
        "evidence": {
            "type": "string",
            "description": "the critique's primary point and why the element it engages is (or isn't) central to the conclusion",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain what judgment call is missing; if level 2, say how the critique's choice of target elevates the whole critique",
        },
    },
}

_PROMPT = """CENTRALITY. Focus on the critique's primary line of engagement — the point it treats as most important (the same one the other axes judge). Rate how LOAD-BEARING the element it engages is — how close to what the conclusion actually rests on.

This is NOT about finding a flaw. For a FLAWED argument the load-bearing element is the decisive flaw. For a SOUND argument it is the central reasoning the conclusion depends on — a critique that engages THAT (to affirm, sharpen, or bound it) is central; one that fixates on a peripheral aside is not.

As a GUIDE, the argument's known load-bearing gaps are:
{key}
(If none are listed, treat the argument as having no decisive flaw — credit the critique for engaging the central reasoning the conclusion stands on, NOT for manufacturing a gap.) A critique may engage an element MORE central than the list; credit that too. Do not award the top level just for naming a listed gap, or for generic praise — top centrality requires comparative weighting: the critique must show why this element is the one the conclusion most depends on, not just one important issue.
- 0: miss — engages a peripheral or merely-relevant element, not what the conclusion rests on.
- 1: solid — engages the dominant element the conclusion most depends on (a decisive flaw, or the central reasoning of a sound argument), and at least roughly weights it above lesser points. This is good ordinary critique.
- 2: excellent — makes a centrality judgment that elevates the critique overall: it chooses the exact revision-relevant fulcrum among plausible alternatives, explicitly distinguishes it from at least one tempting-but-less-central critique, and shows how the rest of the critique should be organized around it. This should be rare.

COMMON FALSE TOP-SCORES:
- Names a listed gap (or, for a sound argument, merely restates the conclusion) without showing why it is load-bearing. Score 1 at most.
- Lists many points. Breadth does not imply centrality unless it ranks the main one and explains why the rest are subordinate.
- Uses broad language ("the core issue", "the real problem") but never justifies the weighting. Do not infer dominance from confident wording.
- Finds and weights the right flaw but does not explicitly reject a nearby tempting target. Score 1, not 2.
- Treats the dominant flaw as one item in a list, even if the item is correct. Score 1 unless the critique uses that judgment to reorganize the whole assessment.
- Says other issues are "secondary" only in passing. Level 2 requires the critique to show WHY a plausible alternative target would lead revision in the wrong direction or would leave the main argument standing.

EXAMPLES — a rising ladder of critiques of the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall":
- 0 (miss): "It never gives figures for the customer loss or tax revenue." (real observations, but not what the conclusion stands on)
- 1 (solid): "The conclusion works only if parking is the binding constraint; fee size and implementation are secondary — if selection or comfort dominate, free parking recovers nothing. Causal identification of the binding constraint is the bottleneck the whole remedy rests on." (central, correctly weighted)
- 2 (excellent): "The revision should not start with more parking data; it should first prove parking is the binding constraint rather than one friction among several. Otherwise the argument's best modest version ('parking matters at the margin') survives but its policy conclusion does not. Selection/comfort/prices are not separate objections — they are the live alternatives that make binding-constraint identification the fulcrum." (chooses and defends the revision-relevant center)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0 (miss): "It doesn't define 'life expectancy' precisely." (a peripheral aside, not what the conclusion rests on)
- 1 (solid): "The conclusion rests entirely on the distinction between a population-average claim and a single individual outcome — the critique engages exactly that load-bearing point, and correctly, since one outlier is precisely what an average absorbs." (engages the central reasoning the conclusion depends on)
- 2 (excellent): "The critique correctly makes the average-vs-outlier distinction the fulcrum and also shows why tempting side objections — sample size, genetic immunity, or precise definition of 'life expectancy' — are secondary: none challenge the logical compatibility between a lower average and an individual outlier. That judgment tells the author what must be preserved in revision." (centrality judgment improves the critique)

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
