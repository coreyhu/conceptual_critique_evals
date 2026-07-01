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
  1 — proportionate: the verdict matches the argument's actual merit at ordinary critique quality.
  2 — excellent judgment: the critique's severity call materially improves the critique by naming the
      nearest plausible over-call or under-call, rejecting it, preserving the strongest surviving
      claim, and explaining the exact boundary between fatal, fixable, partial, and sound.
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
            "description": "0 miscalibrated / 1 proportionate ordinary verdict / 2 excellent severity judgment that names and rejects the nearest tempting over/under-call, preserves the strongest surviving claim, and draws the revision-relevant boundary",
        },
        "why_not_higher": {
            "type": "string",
            "description": "if not level 2, what judgment call is missing; if level 2, explain how the severity judgment elevates the critique",
        },
    },
}

_PROMPT = """PROPORTIONALITY. Rate whether the CRITIQUE's verdict matches the ARGUMENT's actual merit — neither over-stating nor under-stating what is wrong (or right) with it.

STEP 1 — Judge the ARGUMENT on its own, from the text: is its central reasoning sound, or does it contain a decisive flaw? If flawed, how severe — a FATAL defeater (granting the flaw, the conclusion fails as a matter of kind) or a FIXABLE gap (the central claim survives with more support, a narrower scope, or a different remedy)? A sound argument can still have an imperfect fix or minor gaps; that does NOT make it fatally flawed.

STEP 2 — Read the CRITIQUE's operative verdict and rate whether it FITS that merit:
- 0: miscalibrated — it OVER-calls (manufactures a flaw in sound reasoning, or declares a minor/fixable flaw fatal) OR UNDER-calls (misses or downplays a genuinely decisive flaw, or rubber-stamps without engaging the reasoning).
- 1: proportionate — the verdict matches the argument's actual merit: it affirms what is sound and names what is genuinely wrong, at the right severity, and engages the reasoning. This is good ordinary calibration.
- 2: excellent judgment — the severity call elevates the critique. It ESTABLISHES why this verdict follows rather than a weaker or stronger one, explicitly names or clearly reconstructs the nearest tempting over-call or under-call, rejects that wrong severity call, preserves the strongest surviving version of the argument, and draws a revision-relevant boundary (fatal vs fixable, partial vs total, sound vs sound-with-caveat).

Judge the ACTUAL fit, not the critique's confidence: a critique that merely SOUNDS decisive while over- or under-calling is a 0.
Do not award level 2 merely for a correct verdict, a fair caveat, a preserved weaker claim, or a good boundary condition in isolation. A critique that says "mostly right but incomplete" or "flawed in this way" without comparing that verdict to a nearby wrong severity call is level 1.

COMMON FALSE TOP-SCORES:
- The critique says "fatal to the conclusion but not the whole argument" without identifying the plausible weaker verdict it is rejecting. Score 1.
- The critique preserves a modest surviving claim but never explains why the flaw is fatal rather than fixable, or fixable rather than fatal. Score 1.
- The critique correctly endorses a sound argument and adds caveats, but does not explain why a tempting objection fails. Score 1.
- The critique uses calibrated words ("partially", "mostly", "fatal", "fixable") without adjudicating the nearby boundary. Score 1.

EXAMPLES — on the unrelated FLAWED argument "make downtown parking free to win shoppers back from the free-parking mall" (whose real flaw is assuming parking, not selection/comfort, is the binding cause):
- 0: "This argument is worthless — it never even defines 'customer,' so the whole thing collapses." (over-calls a trivial gap as fatal) / "Seems reasonable to me." (under-calls — rubber-stamps a flawed argument)
- 1: "Its real weakness is assuming parking is the decisive factor when the mall also wins on selection and comfort, so free parking may not recover the trend." (right flaw, right severity)
- 2: "The modest claim that parking is one friction survives; the policy conclusion does not. The tempting under-call is 'parking matters, so the recommendation is probably fine'; the tempting over-call is 'parking is irrelevant.' Both are wrong. By the argument's OWN picture of a cost-comparison shopper, price and selection matter too, so the fix is not 'prove parking matters' but 'prove it is the binding constraint.' That is why the flaw is fatal to the recommendation but not to the narrower causal observation." (adjudicates exact severity against nearby wrong calls)

EXAMPLES — on the unrelated SOUND argument "one smoker who lived to 95 doesn't refute that smoking lowers life-expectancy averages, which tolerate outliers":
- 0: "This is flawed — it ignores that some people are genetically immune to smoking's harms." (manufactures a flaw in sound reasoning)
- 1: "Correct — a single outlier can't refute a population-average claim; the argument rightly separates the individual from the distribution." (affirms sound reasoning and engages it)
- 2: "Correct, and the tempting over-call is to say the anecdote shows the smoking-risk argument ignores individual variation. It does not: the anecdote would bite only against an impossibility claim, not an average-risk claim. So the argument is sound as stated while still leaving room for other evidence about the size of the average effect." (affirms soundness, rejects the tempting over-call, and draws the boundary)

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
