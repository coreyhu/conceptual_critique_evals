"""The critique grader: a composite of single-aspect judges (one per file in this package).

Two GATES cap the score (keyed on the central claim, so a peripheral lapse never zeroes an
otherwise-strong critique): Fidelity and Charity. Three GRADED axes score the critique only if the
gates pass: Centrality (uses the gap-list as a guide), Derivation and Decisiveness (reference-free).
Sound arguments take the Calibration path instead of the graded core.

Aggregation (each axis returns its rubric level normalized to [0,1]):
    core  = decisiveness x derivation x centrality   (flawed)   |   calibration   (sound)
    SCORE = [fidelity>0] x [charity>0] x core        (a gate at level 0 caps the whole score)

Only Centrality (gap-list guide) and Charity (strongest reading) read the reference, and only as
a guide; the rest work from the argument + critique alone, so the eval degrades gracefully when
the reference is thin (key-free / RL).
"""

from __future__ import annotations

import asyncio

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
        async def axis(g: BaseGrader) -> float:
            return (
                await g.grade(
                    client,
                    model_id=model_id,
                    argument=argument,
                    reference=reference,
                    critique=critique,
                )
            ).score

        async def passthrough() -> (
            float
        ):  # a skipped axis: gate passes / recall is neutral
            return 1.0

        has_strongest = bool(reference.get("strongest_reading"))
        cha_t = axis(_CHARITY) if has_strongest else passthrough()

        if str(reference.get("soundness")) == "sound":
            fid, cha, cal = await asyncio.gather(
                axis(_FIDELITY), cha_t, axis(_CALIBRATION)
            )
            core, dims = cal, {"calibration": cal}
        else:
            fid, cha, decis, deriv, cen = await asyncio.gather(
                axis(_FIDELITY),
                cha_t,
                axis(_DECISIVENESS),
                axis(_DERIVATION),
                axis(_CENTRALITY),
            )
            core = (
                decis * deriv * cen
            )  # quality (decisiveness x derivation) x centrality
            dims = {"decisiveness": decis, "derivation": deriv, "centrality": cen}

        gate = (1.0 if fid > 0 else 0.0) * (1.0 if cha > 0 else 0.0)
        return GraderScore(
            score=gate * core,
            dimensions={"fidelity": fid, "charity": cha, "core": core, **dims},
            raw={},
        )
