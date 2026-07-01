"""Registry for composite and standalone axis graders."""

from __future__ import annotations

from critic_evals.grading.centrality import CentralityGrader
from critic_evals.grading.charity import CharityGrader
from critic_evals.grading.composite import CompositeGrader
from critic_evals.grading.derivation import DerivationGrader
from critic_evals.grading.fidelity import FidelityGrader
from critic_evals.grading.grader import BaseGrader
from critic_evals.grading.proportionality import ProportionalityGrader
from critic_evals.grading.synthesis import SynthesisGrader

_GRADERS: dict[str, BaseGrader] = {
    g.name: g
    for g in (
        CompositeGrader(),
        FidelityGrader(),
        CharityGrader(),
        CentralityGrader(),
        DerivationGrader(),
        ProportionalityGrader(),
        SynthesisGrader(),
    )
}

GRADERS: tuple[str, ...] = tuple(_GRADERS)  # registered names, for CLI choices/help


def get_grader(name: str) -> BaseGrader:
    """Resolve a grader name to its shared instance; raise `KeyError` on an unknown name."""
    try:
        return _GRADERS[name]
    except KeyError:
        raise KeyError(
            f"unknown grader {name!r}; choose from {', '.join(GRADERS)}"
        ) from None
