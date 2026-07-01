"""Composite critique grader built from focused axis judges."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from critic_evals.grading.calibration import CalibrationGrader
from critic_evals.grading.centrality import CentralityGrader
from critic_evals.grading.charity import CharityGrader
from critic_evals.grading.decisiveness import DecisivenessGrader
from critic_evals.grading.derivation import DerivationGrader
from critic_evals.grading.fidelity import FidelityGrader
from critic_evals.grading.grader import BaseGrader, GraderScore, Reference
from critic_evals.llm.client import AnthropicClient

_FIDELITY = FidelityGrader()
_CHARITY = CharityGrader()
_DECISIVENESS = DecisivenessGrader()
_DERIVATION = DerivationGrader()
_CENTRALITY = CentralityGrader()
_CALIBRATION = CalibrationGrader()


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

        if str(reference.get("soundness")) == "sound":
            if has_strongest:
                fid_g, cha_g, cal_g = await _grade_axes(
                    (_FIDELITY, _CHARITY, _CALIBRATION),
                    client,
                    model_id,
                    argument,
                    reference,
                    critique,
                )
            else:
                fid_g, cal_g = await _grade_axes(
                    (_FIDELITY, _CALIBRATION),
                    client,
                    model_id,
                    argument,
                    reference,
                    critique,
                )
                cha_g = _skipped_charity_score()
            fid, cha, cal = fid_g.score, cha_g.score, cal_g.score
            core, dims = cal, {"calibration": cal}
            raw: dict[str, object] = {
                "fidelity": fid_g.raw,
                "charity": cha_g.raw,
                "calibration": cal_g.raw,
            }
        else:
            if has_strongest:
                fid_g, cha_g, decis_g, deriv_g, cen_g = await _grade_axes(
                    (_FIDELITY, _CHARITY, _DECISIVENESS, _DERIVATION, _CENTRALITY),
                    client,
                    model_id,
                    argument,
                    reference,
                    critique,
                )
            else:
                fid_g, decis_g, deriv_g, cen_g = await _grade_axes(
                    (_FIDELITY, _DECISIVENESS, _DERIVATION, _CENTRALITY),
                    client,
                    model_id,
                    argument,
                    reference,
                    critique,
                )
                cha_g = _skipped_charity_score()
            fid, cha = fid_g.score, cha_g.score
            decis, deriv, cen = decis_g.score, deriv_g.score, cen_g.score
            core = (
                decis + deriv + cen
            ) / 3  # mean: a baseline axis costs points, never zeroes
            dims = {"decisiveness": decis, "derivation": deriv, "centrality": cen}
            raw = {
                "fidelity": fid_g.raw,
                "charity": cha_g.raw,
                "decisiveness": decis_g.raw,
                "derivation": deriv_g.raw,
                "centrality": cen_g.raw,
            }

        gate = (1.0 if fid > 0 else 0.0) * (1.0 if cha > 0 else 0.0)
        return GraderScore(
            score=gate * core,
            dimensions={"fidelity": fid, "charity": cha, "core": core, **dims},
            raw=raw,
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
