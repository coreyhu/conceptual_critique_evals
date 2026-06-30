"""Script for generating critiques for qualitative study.

Usage:
    uv run python scripts/generate_critiques.py --argument argument.txt --samples 3

Requires ANTHROPIC_API_KEY (.env)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.critique import (
    CRITIQUE_PROMPTS,
    DEFAULT_VARIANT,
    build_prompt,
    elicit_critique,
)
from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import DEFAULT_MODELS, ModelSpec, resolve
from critic_evals.schema import ArgumentItem, CritiqueRecord, Usage
from critic_evals.transcripts import load_argument, write_jsonl


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def generate(
    item: ArgumentItem,
    model_labels: list[str],
    *,
    samples: int,
    variant: str,
    max_tokens: int,
    concurrency: int,
) -> list[CritiqueRecord]:
    """Generate `samples` critiques from each model concurrently.

    A failed call becomes an error record (rather than vanishing) so the
    transcript faithfully reflects what happened.
    """
    specs = resolve(model_labels)
    system, prompt = build_prompt(item, variant)
    semaphore = asyncio.Semaphore(concurrency)

    async with AnthropicClient() as client:

        async def one(spec: ModelSpec, sample: int) -> CritiqueRecord:
            async with semaphore:
                try:
                    return await elicit_critique(
                        client,
                        spec,
                        item,
                        variant=variant,
                        sample=sample,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:  # record the failure, keep going
                    return CritiqueRecord(
                        item_id=item.id,
                        model=spec.label,
                        model_id=spec.model_id,
                        prompt_variant=variant,
                        sample=sample,
                        system=system,
                        prompt=prompt,
                        response="",
                        success=False,
                        stop_reason="error",
                        usage=Usage(0, 0),
                        request_id=None,
                        timestamp=_utcnow_iso(),
                        error=repr(exc),
                    )

        tasks = [one(spec, sample) for spec in specs for sample in range(samples)]
        return await asyncio.gather(*tasks)


def _summarize(records: list[CritiqueRecord]) -> None:
    """Print a short per-model summary + a preview of one critique each."""
    by_model: dict[str, list[CritiqueRecord]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)

    for model, recs in by_model.items():
        errors = sum(1 for r in recs if r.error)
        out_tokens = sum(r.usage.output_tokens for r in recs if r.usage)
        print(
            f"\n{model}  ({len(recs)} samples, {errors} errors, "
            f"{out_tokens} output tokens)"
        )
        preview = next((r for r in recs if not r.error), recs[0])
        if preview.error:
            print(f"  [error] {preview.error}")
        else:
            snippet = " ".join(preview.response.split())[:280]
            print(f"  sample {preview.sample}: {snippet}…")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--argument",
        default="arguments/organization_metrics.txt",
        help="path to the markdown argument file",
    )
    parser.add_argument("--samples", type=int, default=3, help="critiques per model")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="model labels to run"
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=list(CRITIQUE_PROMPTS),
        help="critique prompt variant",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--out",
        default=None,
        help="output JSONL path (default: transcripts/<ts>_<item>.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Put it in .env or export it.",
            file=sys.stderr,
        )
        return 1

    args = parse_args(argv)
    item = load_argument(args.argument)
    print(f'Argument: {item.id} — "{item.question}"')
    print(
        f"Models: {', '.join(args.models)} x {args.samples} samples "
        f"(variant={args.variant})"
    )

    records = asyncio.run(
        generate(
            item,
            args.models,
            samples=args.samples,
            variant=args.variant,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )

    out = args.out or (
        Path("transcripts")
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{item.id}.jsonl"
    )
    path = write_jsonl(records, out)
    _summarize(records)
    print(f"\nWrote {len(records)} records -> {path}")
    return 0


if __name__ == "__main__":
    main()
