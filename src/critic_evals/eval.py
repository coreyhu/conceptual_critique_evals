"""The evaluation itself: `LLM -> [0,1]` scoring how well a model critiques arguments.

`evaluate(config)` elicits critiques from `config.model_label` for each argument in the
dataset, grades each with the chosen grader, and aggregates to a single score in [0,1].
It also returns per-critique transcript records (prompt + response + grade + metadata),
and `write_eval_jsonl` serializes the config as a header line ahead of those records so a
trajectory file fully describes the run that produced it.

Aggregation: item score = mean over the model's sampled critiques; overall = mean over
items. The item is the unit; sampling reduces per-critique noise but each item counts once.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from critic_evals.critique import DEFAULT_VARIANT, build_prompt
from critic_evals.dataset import ITEMS, EvalItem
from critic_evals.grading.preprocess import load_reference
from critic_evals.grading.registry import get_grader
from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import MODELS
from critic_evals.transcripts import load_argument

DEFAULT_GRADER = "composite"
DEFAULT_GRADER_MODEL = (
    "claude-opus-4-8"  # the validity check needs a strong, adversarial grader
)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """The knobs that define what a run measures — serialized with its trajectory.

    Everything here changes the score's meaning: which model is judged, how it's
    prompted, how many samples, how it's graded. Execution-only knobs (concurrency,
    client) stay out — they affect speed, not the result. The dataset is fixed
    (critic_evals.dataset), so it isn't a per-run knob.
    """

    model_label: str
    grader: str = DEFAULT_GRADER
    grader_model_id: str = DEFAULT_GRADER_MODEL
    n_samples: int = 3
    k_grade: int = 1
    use_key: bool = True
    variant: str = DEFAULT_VARIANT

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One graded critique — a transcript row."""

    model: str
    model_id: str
    item_id: str
    soundness: str
    sample: int
    critique_prompt: str
    critique: str
    grader: str
    grader_model_id: str
    used_key: bool
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


async def evaluate(
    config: EvalConfig,
    *,
    concurrency: int = 8,
    client: AnthropicClient | None = None,
) -> tuple[EvalResult, list[EvalRecord]]:
    """Score `config.model_label` in [0,1] and return per-critique transcript records."""
    spec = MODELS[config.model_label]
    grader_impl = get_grader(config.grader)
    owns_client = client is None
    client = client or AnthropicClient()
    sem = asyncio.Semaphore(concurrency)

    async def eval_one(item: EvalItem, sample: int) -> EvalRecord:
        arg_item = load_argument(item.argument_path)
        argument = f"{arg_item.question}\n\n{arg_item.argument}"
        system, prompt = build_prompt(arg_item, config.variant)
        async with sem:
            crit = await client.complete(
                model_id=spec.model_id, system=system, prompt=prompt, max_tokens=4096
            )
            reference = load_reference(item.id) if config.use_key else {}
            scores, dims_runs, raws = [], [], []
            for _ in range(config.k_grade):
                gs = await grader_impl.grade(
                    client,
                    model_id=config.grader_model_id,
                    argument=argument,
                    reference=reference,
                    critique=crit.text,
                )
                scores.append(gs.score)
                if gs.dimensions:
                    dims_runs.append(gs.dimensions)
                raws.append(gs.raw)
        dims = (
            {k: sum(d[k] for d in dims_runs) / len(dims_runs) for k in dims_runs[0]}
            if dims_runs
            else {}
        )
        return EvalRecord(
            model=spec.label,
            model_id=crit.model_id,  # id the API actually served, not just what we requested
            item_id=item.id,
            soundness=item.soundness,
            sample=sample,
            critique_prompt=prompt,
            critique=crit.text,
            grader=config.grader,
            grader_model_id=config.grader_model_id,
            used_key=config.use_key,
            score=sum(scores) / len(scores),
            dimensions=dims,
            grader_raw=raws[-1],
            input_tokens=crit.input_tokens,
            output_tokens=crit.output_tokens,
            timestamp=_utcnow_iso(),
        )

    records = await asyncio.gather(
        *(eval_one(item, s) for item in ITEMS for s in range(config.n_samples))
    )

    item_results: list[ItemResult] = []
    for item in ITEMS:
        sample_scores = [r.score for r in records if r.item_id == item.id]
        item_results.append(
            ItemResult(
                item_id=item.id,
                score=sum(sample_scores) / len(sample_scores),
                sample_scores=sample_scores,
            )
        )
    overall = sum(r.score for r in item_results) / len(item_results)

    if owns_client:
        await client.aclose()
    return EvalResult(config=config, score=overall, items=item_results), list(records)


async def score_model(
    config: EvalConfig, *, client: AnthropicClient | None = None
) -> float:
    """The bare `LLM -> [0,1]` signature: a config in, a scalar out."""
    result, _ = await evaluate(config, client=client)
    return result.score


def write_eval_jsonl(
    records: Iterable[EvalRecord], path: str | Path, *, config: EvalConfig
) -> Path:
    """Write the trajectory: a config header line, then one EvalRecord per line.

    The header carries `kind: config` so it survives files being concatenated —
    a reader picks configs and records apart by that tag, not by line position.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"kind": "config", **config.to_dict()}, ensure_ascii=False)
            + "\n"
        )
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_eval_jsonl(path: str | Path) -> tuple[EvalConfig, list[EvalRecord]]:
    """Read one trajectory back into its config and records — the inverse of the writer.

    A trajectory describes exactly one run: exactly one `kind: config` header, then its
    records. Zero headers or more than one means the file is malformed for this contract
    (e.g. two runs merged), so raise rather than guess which config the records belong to.
    """
    path = Path(path)
    config: EvalConfig | None = None
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.pop("kind", None) == "config":
                if config is not None:
                    raise ValueError(f"{path}: multiple config headers (merged runs?)")
                config = EvalConfig(**obj)
            else:
                records.append(EvalRecord(**obj))
    if config is None:
        raise ValueError(f"{path}: no config header (kind=config) found")
    return config, records
