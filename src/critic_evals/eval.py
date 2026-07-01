"""The evaluation scoring how well a model critiques arguments.

`evaluate(config)` grades `config.model_label`'s frozen dataset rows with the chosen
grader and aggregates to a single score in [0,1]. It also returns per-critique
transcript records (prompt + response + grade + metadata).

Aggregation: case score = mean over the model's graded critiques for that case; overall =
mean over cases. The case is the unit; multiple critiques per case reduce per-critique
noise but each case counts once.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from critic_evals.dataset import EvalDatasetRow, load_eval_dataset
from critic_evals.grading.grader import BaseGrader
from critic_evals.grading.registry import get_grader
from critic_evals.llm.client import AnthropicClient
from critic_evals.references import load_reference

DEFAULT_GRADER = "composite"
DEFAULT_GRADER_MODEL = (
    "claude-opus-4-8"  # the validity check needs a strong, adversarial grader
)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """The knobs that define a grading run — serialized with its trajectory.

    The critiques are fixed (the committed dataset fixture); everything here is a
    grading choice applied to them. `model_label` selects which model's frozen
    critiques to score.
    """

    model_label: str
    grader: str = DEFAULT_GRADER
    grader_model_id: str = DEFAULT_GRADER_MODEL
    k_grade: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One graded critique — a transcript row."""

    model: str
    model_id: str
    item_id: str
    sample: int
    critique_prompt: str
    critique: str
    grader: str
    grader_model_id: str
    score: float
    dimensions: dict[str, float]
    grader_raw: dict[str, object]
    input_tokens: int
    output_tokens: int
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ItemResult:
    item_id: str
    score: float
    sample_scores: list[float]


@dataclass(frozen=True, slots=True)
class EvalResult:
    config: EvalConfig
    score: float  # overall [0,1]
    items: list[ItemResult]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _grade_critique(
    row: EvalDatasetRow,
    *,
    grader_impl: BaseGrader,
    config: EvalConfig,
    client: AnthropicClient,
    semaphore: asyncio.Semaphore,
) -> EvalRecord:
    argument = f"{row.question}\n\n{row.argument}"
    reference = load_reference(row.item_id)
    async with semaphore:
        scores, dims_runs, raws = [], [], []
        for _ in range(config.k_grade):
            gs = await grader_impl.grade(
                client,
                model_id=config.grader_model_id,
                argument=argument,
                reference=reference,
                critique=row.critique,
            )
            scores.append(gs.score)
            if gs.dimensions:
                dims_runs.append(gs.dimensions)
            raws.append(gs.raw)
    return EvalRecord(
        model=row.model,
        model_id=row.model_id,
        item_id=row.item_id,
        sample=row.sample,
        critique_prompt=row.critique_prompt,
        critique=row.critique,
        grader=config.grader,
        grader_model_id=config.grader_model_id,
        score=sum(scores) / len(scores),
        dimensions=_average_dimensions(dims_runs),
        grader_raw=raws[-1],
        input_tokens=row.usage.input_tokens if row.usage else 0,
        output_tokens=row.usage.output_tokens if row.usage else 0,
        timestamp=_utcnow_iso(),
    )


def _average_dimensions(dims_runs: list[dict[str, float]]) -> dict[str, float]:
    if not dims_runs:
        return {}
    return {k: sum(d[k] for d in dims_runs) / len(dims_runs) for k in dims_runs[0]}


async def evaluate(
    config: EvalConfig,
    *,
    concurrency: int = 8,
    client: AnthropicClient | None = None,
) -> tuple[EvalResult, list[EvalRecord]]:
    """Grade `config.model_label`'s frozen critiques; return score in [0,1] + records."""
    eval_inputs = load_eval_dataset(model_label=config.model_label)
    grader_impl = get_grader(config.grader)
    owns_client = client is None
    client = client or AnthropicClient()
    sem = asyncio.Semaphore(concurrency)

    records = await asyncio.gather(
        *(
            _grade_critique(
                eval_input,
                grader_impl=grader_impl,
                config=config,
                client=client,
                semaphore=sem,
            )
            for eval_input in eval_inputs
        )
    )

    item_ids = list(dict.fromkeys(r.item_id for r in records))  # first-seen order
    item_results: list[ItemResult] = []
    for iid in item_ids:
        sample_scores = [r.score for r in records if r.item_id == iid]
        item_results.append(
            ItemResult(
                item_id=iid,
                score=sum(sample_scores) / len(sample_scores),
                sample_scores=sample_scores,
            )
        )
    overall = sum(r.score for r in item_results) / len(item_results)

    if owns_client:
        await client.aclose()
    return EvalResult(config=config, score=overall, items=item_results), list(records)
