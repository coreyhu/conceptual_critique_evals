"""Build the frozen eval dataset from source argument files.

uv scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.dataset import EVAL_DATASET_PATH, EvalDatasetRow, write_eval_dataset
from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import DEFAULT_MODELS, ModelSpec, resolve
from critic_evals.schema import ArgumentItem, Usage

ROOT = Path(__file__).resolve().parents[1]
ARGS_DIR = ROOT / "arguments"
_HEADING = re.compile(r"^#\s+(.*)$")
_ARGUMENT_HEADING = re.compile(r"^#+\s*argument\b.*$", re.IGNORECASE)

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful, intellectually honest critic of conceptual reasoning arguments. "
    "Engage with the specific argument in front of you on its own terms."
)

CRITIQUE_PROMPT = (
    "You are given a question and an argument that attempts to answer it.\n\n"
    "Assess how well the argument actually supports its conclusion, and explain your reasoning. "
    "Engage with the argument as written.\n\n"
    "Question:\n{question}\n\n"
    "Argument:\n{argument}"
)


@dataclass(frozen=True, slots=True)
class SourceItem:
    id: str
    argument_path: Path


SOURCE_ITEMS: tuple[SourceItem, ...] = (
    SourceItem("organization_metrics", ARGS_DIR / "organization_metrics.txt"),
    SourceItem("cancer_screening", ARGS_DIR / "cancer_screening.txt"),
    SourceItem("model_causal_structure", ARGS_DIR / "model_causal_structure.txt"),
    SourceItem("science_funding_returns", ARGS_DIR / "science_funding_returns.txt"),
)


def select_source_items(ids: Iterable[str] | None = None) -> tuple[SourceItem, ...]:
    if ids is None:
        return SOURCE_ITEMS
    by_id = {item.id: item for item in SOURCE_ITEMS}
    ids = list(ids)
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        raise ValueError(f"unknown item id(s): {', '.join(missing)}")
    return tuple(by_id[item_id] for item_id in ids)


def parse_argument(text: str, item_id: str) -> ArgumentItem:
    lines = text.splitlines()
    question = ""
    question_idx = -1
    for i, line in enumerate(lines):
        match = _HEADING.match(line)
        if match:
            question = match.group(1).strip()
            question_idx = i
            break

    arg_idx = next(
        (i for i, line in enumerate(lines) if _ARGUMENT_HEADING.match(line)), -1
    )
    start = arg_idx + 1 if arg_idx != -1 else question_idx + 1
    argument = "\n".join(lines[start:]).strip()
    if not argument:
        raise ValueError(f"no argument body found in {item_id!r}")
    return ArgumentItem(id=item_id, question=question, argument=argument)


def load_argument(path: str | Path, item_id: str | None = None) -> ArgumentItem:
    path = Path(path)
    return parse_argument(path.read_text(encoding="utf-8"), item_id or path.stem)


def build_prompt(item: ArgumentItem) -> tuple[str, str]:
    return CRITIQUE_SYSTEM_PROMPT, CRITIQUE_PROMPT.format(
        question=item.question, argument=item.argument
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _build_row(
    item: SourceItem,
    sample: int,
    spec: ModelSpec,
    *,
    client: AnthropicClient,
    arguments: dict[str, ArgumentItem],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> EvalDatasetRow:
    argument = arguments[item.id]
    system, prompt = build_prompt(argument)
    async with semaphore:
        completion = await client.complete(
            model_id=spec.model_id,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    return EvalDatasetRow(
        id=f"{item.id}__{spec.label}__sample_{sample}",
        item_id=item.id,
        question=argument.question,
        argument=argument.argument,
        model=spec.label,
        model_id=completion.model_id,
        sample=sample,
        critique_prompt=prompt,
        critique=completion.text,
        stop_reason=completion.stop_reason,
        usage=Usage(completion.input_tokens, completion.output_tokens),
        request_id=completion.request_id,
        timestamp=_utcnow_iso(),
    )


async def build_eval_dataset(
    items: Sequence[SourceItem],
    model_labels: Sequence[str],
    *,
    samples: int,
    max_tokens: int = 4096,
    concurrency: int = 5,
    client: AnthropicClient | None = None,
) -> list[EvalDatasetRow]:
    arguments = {item.id: load_argument(item.argument_path, item.id) for item in items}
    specs = resolve(model_labels)
    owns_client = client is None
    client = client or AnthropicClient()
    sem = asyncio.Semaphore(concurrency)
    try:
        return await asyncio.gather(
            *(
                _build_row(
                    item,
                    sample,
                    spec,
                    client=client,
                    arguments=arguments,
                    semaphore=sem,
                    max_tokens=max_tokens,
                )
                for item in items
                for spec in specs
                for sample in range(samples)
            )
        )
    finally:
        if owns_client:
            await client.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="argument item ids to include; defaults to all source items",
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--out", default=str(EVAL_DATASET_PATH))
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    items = select_source_items(args.items)
    print(
        f"building dataset: {len(items)} items x {args.models} x {args.samples} samples"
    )
    rows = await build_eval_dataset(
        items,
        args.models,
        samples=args.samples,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    path = write_eval_dataset(rows, args.out)
    print(f"wrote {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
