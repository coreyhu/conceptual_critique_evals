"""Calibration axis: on sound arguments, does the critique endorse expertly?"""

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
            "description": "0 miscalibrated or vacuous / 1 correct engaged endorsement / 2 expert calibration: defuses tempting invalid objection and adds precise preserving boundary",
        },
        "operative_quote": {"type": "string"},
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, explain whether the critique lacks engagement, lacks a real boundary, or fails to defuse a tempting invalid objection; if level 2, identify the invalid objection defused",
        },
    },
}

_PROMPT = """CALIBRATION. The ARGUMENT is fundamentally SOUND. Rate whether the CRITIQUE correctly and substantively RECOGNIZES this.

The top level is not for generic praise plus a generic caveat. A great calibrated critique should show why tempting objections fail, then add a precise boundary condition or sharpening that preserves the argument's core claim.

- 0: miss — miscalibrated in either direction: over-critical (manufactures flaws, reflexively tears it down, or overstates a minor caveat as decisive) OR under-critical (a vacuous rubber-stamp that never engages the reasoning).
- 1: good — correctly affirms it AND engages the reasoning, possibly adding a real, proportionate boundary condition, limitation, or stronger reformulation.
- 2: excellent — affirms it, engages the reasoning, defuses at least one tempting but invalid objection, and adds a precise boundary condition or sharpening that does not weaken the original point.

COMMON FALSE TOP-SCORES:
- The critique correctly endorses the argument and adds a generic caveat ("more evidence is always useful"). Score 1, not 2.
- The critique adds a boundary condition that actually weakens or muddles the sound argument rather than preserving it. Do not score 2.
- The critique praises the argument but does not identify any tempting objection and explain why that objection fails. Score 1 at most.

EXAMPLES — for the unrelated SOUND argument "One smoker who lived to 95 does not refute that smoking lowers life expectancy: that is a claim about population averages, which tolerate outliers":
- 0 (miss): "This is flawed — it waves away real evidence that some people are simply immune to smoking's harms." (manufactures a flaw) — or "Yep, seems right." (vacuous)
- 1 (good): "Correct — a single long-lived smoker is fully consistent with smoking lowering the average; the argument rightly separates the individual case from the population distribution. The point generalizes to other statistic-versus-anecdote disputes." (affirms, engages, and adds a useful boundary)
- 2 (excellent): "Correct. A tempting objection is that the long-lived smoker is still evidence, but one anecdote has tiny evidential weight against a population-level claim and is exactly the kind of outlier the claim predicts. The argument should be framed as 'does not materially update the population estimate,' not 'has literally zero evidential relevance'." (defuses the tempting objection and sharpens the boundary)

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class CalibrationGrader(LLMGrader):
    name: str = "calibration"

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
