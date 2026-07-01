"""Composite critique grader built from focused axis judges.

Two GATES cap the score (keyed on the central claim, so a peripheral lapse never zeroes an otherwise
strong critique): Fidelity (no fabrication) and Charity (engages the strongest form). Three GRADED
axes then score the critique on ONE path — no sound/flawed routing: Centrality (targets the
load-bearing element), Derivation (where its force comes from), and Proportionality (is the verdict's
severity proportionate to the argument's actual merit — this subsumes the old decisiveness/calibration
split and grades borderline arguments continuously instead of forcing a binary soundness route).

Aggregation (each axis returns its rubric level normalized to [0,1]):
    core  = mean(centrality, derivation, proportionality)
    SCORE = [fidelity>0] x [charity>0] x core

The core is a mean so a single baseline axis costs points without annihilating the score; the gates
are the only hard caps. Only Charity (strongest reading) and Centrality (gap-list guide) read the
reference, and only as a guide; the rest work from the argument + critique alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from critic_evals.grading.centrality import CentralityGrader
from critic_evals.grading.charity import CharityGrader
from critic_evals.grading.derivation import DerivationGrader
from critic_evals.grading.fidelity import FidelityGrader
from critic_evals.grading.grader import BaseGrader, GraderScore, Reference
from critic_evals.grading.proportionality import ProportionalityGrader
from critic_evals.llm.client import AnthropicClient

_FIDELITY = FidelityGrader()
_CHARITY = CharityGrader()
_CENTRALITY = CentralityGrader()
_DERIVATION = DerivationGrader()
_PROPORTIONALITY = ProportionalityGrader()


class CompositeGrader(BaseGrader):
    """Runs the axis graders concurrently and combines them into a [0,1] score."""

    name: str = "composite"

    async def grade(
        self,
        client: AnthropicClient,
        *,
        model_id: str,
        argument: str,
        reference: Reference,
        critique: str,
    ) -> GraderScore:
        has_strongest = bool(reference.get("strongest_reading"))
        core_axes = (_CENTRALITY, _DERIVATION, _PROPORTIONALITY)
        if has_strongest:
            fid_g, cha_g, cen_g, deriv_g, prop_g = await _grade_axes(
                (_FIDELITY, _CHARITY, *core_axes),
                client, model_id, argument, reference, critique,
            )
        else:
            fid_g, cen_g, deriv_g, prop_g = await _grade_axes(
                (_FIDELITY, *core_axes),
                client, model_id, argument, reference, critique,
            )
            cha_g = _skipped_charity_score()

        fid, cha = fid_g.score, cha_g.score
        cen, deriv, prop = cen_g.score, deriv_g.score, prop_g.score
        core = (cen + deriv + prop) / 3  # mean: a baseline axis costs points, never zeroes
        gate = (1.0 if fid > 0 else 0.0) * (1.0 if cha > 0 else 0.0)
        return GraderScore(
            score=gate * core,
            dimensions={
                "fidelity": fid,
                "charity": cha,
                "core": core,
                "centrality": cen,
                "derivation": deriv,
                "proportionality": prop,
            },
            raw={
                "fidelity": fid_g.raw,
                "charity": cha_g.raw,
                "centrality": cen_g.raw,
                "derivation": deriv_g.raw,
                "proportionality": prop_g.raw,
            },
        )


async def _grade_axes(
    graders: Sequence[BaseGrader],
    client: AnthropicClient,
    model_id: str,
    argument: str,
    reference: Reference,
    critique: str,
) -> tuple[GraderScore, ...]:
    return tuple(
        await asyncio.gather(
            *(
                grader.grade(
                    client,
                    model_id=model_id,
                    argument=argument,
                    reference=reference,
                    critique=critique,
                )
                for grader in graders
            )
        )
    )


def _skipped_charity_score() -> GraderScore:
    return GraderScore(
        score=1.0,
        dimensions={},
        raw={"skipped": "no strongest_reading in reference"},
    )
