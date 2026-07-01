"""Proportionality axis: does the critique's verdict match the argument's actual merit?

The single severity axis — it replaces decisiveness (which graded the flawed-argument path) and
calibration (the sound-argument path). Both measured the same two failures, over-severity and
under-severity, just against different arguments, so ONE axis spans the whole spectrum and the eval
no longer routes on a binary soundness classification. The judge first assesses the argument's own
merit, then rates whether the critique's verdict fits it — so a borderline argument (a sound claim
with a fixable coda) is graded on the fit of the verdict rather than forced into a sound/flawed
bucket. Reference-free.

Levels (normalized level/2):
  0 — miscalibrated: OVER-calls (manufactures a flaw in sound reasoning, or declares a minor/fixable
      flaw fatal) OR UNDER-calls (misses or downplays a genuinely decisive flaw, or rubber-stamps
      without engaging the reasoning).
  1 — proportionate: the verdict matches the argument's actual merit — affirms what is sound, names
      what is genuinely wrong at the right severity, and engages the reasoning.
  2 — proportionate and illuminating: as (1), and it establishes the verdict rigorously (shows why a
      fatal flaw truly defeats the conclusion, or why the argument survives a tempting objection)
      and/or adds a real, proportionate boundary condition, limitation, or sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass

from critic_evals.grading.grader import LLMGrader, Reference, level_score

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["argument_merit", "critique_verdict", "level", "why_not_higher"],
    "properties": {
        "argument_merit": {
            "type": "string",
            "description": "YOUR OWN assessment first: is the argument's central reasoning sound, or does it contain a decisive flaw — and if so, how severe (a fatal defeater vs a fixable gap)?",
        },
        "critique_verdict": {
            "type": "string",
            "description": "the critique's operative verdict on the argument's merit",
        },
        "level": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0 miscalibrated (over-calls a sound/fixable point as fatal, OR under-calls/misses a decisive flaw, OR rubber-stamps) / 1 proportionate (verdict matches the argument's actual merit, engages the reasoning) / 2 proportionate AND illuminating (establishes the verdict rigorously and/or adds a real boundary or sharpening)",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, what holds it there — a miscalibration, or a proportionate-but-thin verdict that establishes nothing and adds no value",
        },
    },
}

_PROMPT = """PROPORTIONALITY. Rate whether the CRITIQUE's verdict matches the ARGUMENT's actual merit — neither over-stating nor under-stating what is wrong (or right) with it.

STEP 1 — Judge the ARGUMENT on its own, from the text: is its central reasoning sound, or does it contain a decisive flaw? If flawed, how severe — a FATAL defeater (granting the flaw, the conclusion fails as a matter of kind) or a FIXABLE gap (the central claim survives with more support, a narrower scope, or a different remedy)? A sound argument can still have an imperfect fix or minor gaps; that does NOT make it fatally flawed.

STEP 2 — Read the CRITIQUE's operative verdict and rate whether it FITS that merit:
- 0: miscalibrated — it OVER-calls (manufactures a flaw in sound reasoning, or declares a minor/fixable flaw fatal) OR UNDER-calls (misses or downplays a genuinely decisive flaw, or rubber-stamps without engaging the reasoning).
- 1: proportionate — the verdict matches the argument's actual merit: it affirms what is sound and names what is genuinely wrong, at the right severity, and engages the reasoning.
- 2: proportionate and illuminating — as level 1, and it ESTABLISHES the verdict rigorously (shows why a fatal flaw truly defeats the conclusion, or why the argument survives a tempting objection) and/or adds a real, proportionate boundary condition, limitation, or sharpening.

Judge the ACTUAL fit, not the critique's confidence: a critique that merely SOUNDS decisive while over- or under-calling is a 0.

EXAMPLES — on the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall" (whose real flaw is assuming parking, not selection/comfort, is the binding cause):
- 0: "This argument is worthless — it never even defines 'customer,' so the whole thing collapses." (over-calls a trivial gap as fatal) / "Seems reasonable to me." (under-calls — rubber-stamps a flawed argument)
- 1: "Its real weakness is assuming parking is the decisive factor when the mall also wins on selection and comfort, so free parking may not recover the trend." (right flaw, right severity)
- 2: "...and by the argument's OWN picture of a cost-comparison shopper, price and selection matter too, so parking is one factor among several — the remedy works only if parking is the binding constraint, which is never shown." (establishes the flaw from the argument's own commitments)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0: "This is flawed — it ignores that some people are genetically immune to smoking's harms." (manufactures a flaw in sound reasoning)
- 1: "Correct — a single outlier can't refute a population-average claim; the argument rightly separates the individual from the distribution." (affirms sound reasoning and engages it)
- 2: "Correct, and it can be sharpened: the anecdote would bite only if smoking made early death impossible, which no one claims — so the point generalizes to any statistic-vs-anecdote dispute." (affirms and adds real value)

ARGUMENT:
{argument}

CRITIQUE:
{critique}"""


@dataclass(frozen=True, slots=True)
class ProportionalityGrader(LLMGrader):
    name: str = "proportionality"

    def build_prompt(
        self, *, argument: str, reference: Reference, critique: str
    ) -> str:
        return _PROMPT.format(argument=argument, critique=critique)

    @property
    def schema(self) -> dict[str, object]:
        return _SCHEMA

    def extract(self, parsed: dict[str, object]) -> float:
        return level_score(parsed, 2)
