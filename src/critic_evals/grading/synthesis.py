"""Synthesis axis: does the critique integrate correct points into a deeper diagnosis?

This is the holistic good-vs-excellent axis. It should not re-grade whether the critique is
faithful, central, or proportionate; the other axes do that. Instead, it asks whether those
parts compose into a diagnosis that changes the reader's understanding of the argument.
"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level", "diagnosis", "why_not_higher"],
    "properties": {
        "level": {
            "type": "integer",
            "enum": [0, 1, 2, 3],
            "description": "0 checklist-like or merely collects correct observations / 1 partially integrates the critique into a useful sharpening or boundary / 2 gives a genuinely structural diagnosis / 3 exceptional: structural diagnosis plus best-version reconstruction, tempting-near-miss discrimination, and materially revision-changing insight",
        },
        "diagnosis": {
            "type": "string",
            "description": "the critique's integrated diagnosis, if any; otherwise describe the collection of points it makes",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, what prevents it from being a structural diagnosis; if level 2, explain the hidden premise, equivocation, structural tension, boundary, or reconstruction it exposes",
        },
    },
}

_PROMPT = """SYNTHESIS. Assume the other axes separately judge fidelity, charity, centrality, derivation, and proportionality. Your job is narrower: rate whether the critique's correct points COMPOSE into a deeper diagnosis of the argument, rather than a checklist of observations.

The argument's strongest reading is:
{strongest}

Known load-bearing gaps, if any:
{gaps}

Rate the CRITIQUE:
- 0: competent but unsynthesized. It lists correct observations, names the main flaw, or gives a right verdict, but does not integrate them into an explanatory diagnosis. It would help revision locally, but it does not change how one understands the argument.
- 1: partially synthesized. It sharpens the main issue, adds a useful boundary condition, or explains one important interaction among points, but stops short of a deeper structural account.
- 2: strong synthesis. It explains why the argument is tempting AND exactly where it succeeds or fails by exposing a hidden premise, equivocation, structural tension, paradigm mismatch, decisive boundary condition, or stronger reconstruction. For a sound argument, it explains why tempting objections fail and what boundary keeps the argument sound.
- 3: exceptional synthesis. It does everything in level 2, and also makes a materially revision-changing move that a merely strong critique would miss: it reconstructs the best version of the argument before diagnosing it, distinguishes the right critique from a nearby tempting-but-wrong critique, and leaves a clear path for how the argument would need to change. This should be rare; if the critique is merely very good, score 2.

COMMON FALSE TOP-SCORES:
- The critique uses phrases like "the deeper issue" or "the hidden premise" without demonstrating the move. Score 0 or 1.
- The critique has many sections or many correct points. Breadth is not synthesis unless the points are integrated into one diagnosis.
- The critique is eloquent or confident. Judge the explanatory move, not style.
- The critique merely repeats a known load-bearing gap. Level 2 requires showing the structure behind it; level 3 requires going beyond that into best-version reconstruction, near-miss discrimination, and revision-changing guidance.
- A critique with a real structural diagnosis but no discrimination from tempting wrong critiques is level 2, not level 3.

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class SynthesisGrader(LLMGrader):
    name: str = "synthesis"

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        gaps = reference.get("load_bearing_gaps", ())
        listed = (
            "\n".join(f"({i + 1}) {g}" for i, g in enumerate(gaps))
            if isinstance(gaps, (list, tuple)) and gaps
            else "(none provided)"
        )
        return _PROMPT.format(
            strongest=str(reference.get("strongest_reading", "(none provided)")),
            gaps=listed,
            argument=argument,
            critique=critique,
        )

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 3)
