"""Composite critique grader built from focused axis judges.

Fidelity and Charity form a smooth validity factor. Four graded axes then score the critique on ONE
path — no sound/flawed routing: Centrality (targets the load-bearing element), Derivation (where its
force comes from), Proportionality (is the verdict's severity proportionate to the argument's actual
merit), and Synthesis (do the correct points compose into a deeper diagnosis).

Aggregation (each axis returns its rubric level normalized to [0,1]):
    validity  = min(fidelity, charity)
    core      = geometric_mean(centrality, derivation, proportionality, synthesis)
    SCORE     = validity x core

This is intentionally smooth rather than a hard cap: a solid but unsynthesized critique is penalized
without a threshold discontinuity, while only critiques with top synthesis can reach the ceiling.
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
from critic_evals.grading.synthesis import SynthesisGrader
from critic_evals.llm.client import AnthropicClient

_FIDELITY = FidelityGrader()
_CHARITY = CharityGrader()
_CENTRALITY = CentralityGrader()
_DERIVATION = DerivationGrader()
_PROPORTIONALITY = ProportionalityGrader()
_SYNTHESIS = SynthesisGrader()


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
        core_axes = (_CENTRALITY, _DERIVATION, _PROPORTIONALITY, _SYNTHESIS)
        if has_strongest:
            fid_g, cha_g, cen_g, deriv_g, prop_g, syn_g = await _grade_axes(
                (_FIDELITY, _CHARITY, *core_axes),
                client,
                model_id,
                argument,
                reference,
                critique,
            )
        else:
            fid_g, cen_g, deriv_g, prop_g, syn_g = await _grade_axes(
                (_FIDELITY, *core_axes),
                client,
                model_id,
                argument,
                reference,
                critique,
            )
            cha_g = _skipped_charity_score()

        fid, cha = fid_g.score, cha_g.score
        cen, deriv, prop, syn = cen_g.score, deriv_g.score, prop_g.score, syn_g.score
        validity = min(fid, cha)
        core = _geometric_mean(cen, deriv, prop, syn)
        return GraderScore(
            score=validity * core,
            dimensions={
                "fidelity": fid,
                "charity": cha,
                "validity": validity,
                "core": core,
                "centrality": cen,
                "derivation": deriv,
                "proportionality": prop,
                "synthesis": syn,
            },
            raw={
                "fidelity": fid_g.raw,
                "charity": cha_g.raw,
                "centrality": cen_g.raw,
                "derivation": deriv_g.raw,
                "proportionality": prop_g.raw,
                "synthesis": syn_g.raw,
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


def _geometric_mean(*scores: float) -> float:
    product = 1.0
    for score in scores:
        product *= max(0.0, min(1.0, score))
    return product ** (1 / len(scores))
