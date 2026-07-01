"""Build the committed critique fixture the eval grades against.

Usage:
    uv run python scripts/build_dataset.py                 # all models x 3 samples
    uv run python scripts/build_dataset.py --models opus-4.8 --samples 5

Requires ANTHROPIC_API_KEY (.env). Aborts without writing if any critique fails.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.dataset import (
    CRITIQUES_PATH,
    build_critiques,
    get_items,
    utcnow_iso,
    write_critiques,
)
from critic_evals.llm.models import DEFAULT_MODELS

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument(
        "--samples", type=int, default=3, help="critiques per item per model"
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--out", default=str(CRITIQUES_PATH))
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    items = get_items()

    print(
        f"building fixture: {len(items)} items x {args.models} x {args.samples} samples"
    )
    records = await build_critiques(
        items,
        args.models,
        samples=args.samples,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    path = write_critiques(
        records,
        args.out,
        models=args.models,
        samples=args.samples,
        item_ids=[it.id for it in items],
        built_at=utcnow_iso(),
    )
    print(f"wrote {len(records)} critiques -> {path}")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
