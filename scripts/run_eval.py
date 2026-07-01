"""Run the critique eval across models and write transcripts — the LLM -> [0,1] benchmark.

Usage:
    uv run python scripts/run_eval.py                      # all three models, default grader
    uv run python scripts/run_eval.py --models opus-4.8
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.eval import (
    DEFAULT_GRADER,
    DEFAULT_GRADER_MODEL,
    EvalConfig,
    evaluate,
    write_eval_jsonl,
)
from critic_evals.grading.registry import GRADERS
from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import DEFAULT_MODELS

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--grader", default=DEFAULT_GRADER, choices=GRADERS)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    p.add_argument("--no-key", action="store_true", help="grade key-free (RL-style)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="concurrent critiques in flight; the composite fans out ~5 axis calls each, "
        "so peak API load is ~5x this — keep modest to avoid 529 overloads",
    )
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    out_dir = ROOT / "transcripts" / "eval"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(
        f"grader={args.grader} (verifier {args.grader_model}), "
        f"key={'off' if args.no_key else 'on'}\n"
    )
    print(f"{'model':<12}{'score':>8}   per-item")
    async with AnthropicClient() as client:
        for model in args.models:
            config = EvalConfig(
                model_label=model,
                grader=args.grader,
                grader_model_id=args.grader_model,
                use_key=not args.no_key,
            )
            result, records = await evaluate(
                config, client=client, concurrency=args.concurrency
            )
            path = write_eval_jsonl(
                records, out_dir / f"{ts}_{model}_{args.grader}.jsonl", config=config
            )
            per_item = "  ".join(
                f"{i.item_id.split('_')[0]}={i.score:.2f}" for i in result.items
            )
            print(f"{model:<12}{result.score:>8.3f}   {per_item}   -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
