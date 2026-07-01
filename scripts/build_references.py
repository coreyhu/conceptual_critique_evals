"""Precompute the auto-reference for each argument and commit it to references/.

Preprocessing is non-deterministic (LLM decomposition), so we run it ONCE per argument and
cache the result. The eval then loads the committed reference — making the eval reproducible
across runs — and the reference JSON is human-inspectable, so a reviewer can verify the analysis
found the real flaws without any hand-authored key. Re-run this to refresh (e.g. new arguments).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.dataset import EvalDatasetRow, load_eval_dataset
from critic_evals.llm.client import AnthropicClient

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "references"
MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a logician mapping an argument from the text alone. Be rigorous and "
    "charitable: engage the strongest reading, and neither invent gaps nor excuse real ones."
)

_DECOMPOSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["strongest_reading", "concessions"],
    "properties": {
        "strongest_reading": {"type": "string"},
        "concessions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "objections the argument already pre-empts; include a short quote inline when useful",
        },
    },
}

_GAPS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["load_bearing_gaps"],
    "properties": {
        "load_bearing_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "central, conclusion-damaging reasoning gaps; empty if there is no serious gap",
        },
    },
}

_DECOMPOSE_PROMPT = (
    "Decompose this ARGUMENT from the text alone. Give the argument's STRONGEST charitable "
    "reading: the central idea a good critique must engage rather than a weak phrasing. Then "
    "list its CONCESSIONS: objections it already pre-empts via qualifiers, scope limits, or "
    "explicit caveats.\n\nARGUMENT:\n{argument}"
)

_GAPS_PROMPT = (
    "Identify only the ARGUMENT's load-bearing reasoning gaps: objections that sit in the main "
    "chain to the conclusion and would materially damage that conclusion if accepted. Consider "
    "unstated premises, invalid steps, equivocations, internal tensions, fix-vs-diagnosis "
    "mismatches, and inductive overreach. Exclude peripheral nitpicks and objections the argument "
    "already pre-empts. If the argument is basically sound, return an empty list.\n\n"
    "ARGUMENT:\n{argument}"
)


def _items(only: str | None) -> list[EvalDatasetRow]:
    by_id: dict[str, EvalDatasetRow] = {}
    for row in load_eval_dataset():
        by_id.setdefault(row.item_id, row)
    if only is not None:
        if only not in by_id:
            raise ValueError(f"unknown item id: {only}")
        return [by_id[only]]
    return list(by_id.values())


async def preprocess_argument(
    client: AnthropicClient,
    *,
    model_id: str,
    argument: str,
) -> dict[str, object]:
    dec = (
        await client.complete(
            model_id=model_id,
            system=_SYSTEM,
            prompt=_DECOMPOSE_PROMPT.format(argument=argument),
            schema=_DECOMPOSE_SCHEMA,
            max_tokens=1800,
        )
    ).parsed or {}
    gap_obj = (
        await client.complete(
            model_id=model_id,
            system=_SYSTEM,
            prompt=_GAPS_PROMPT.format(argument=argument),
            schema=_GAPS_SCHEMA,
            max_tokens=1400,
        )
    ).parsed or {}

    concessions = dec.get("concessions", [])
    gaps = gap_obj.get("load_bearing_gaps", [])
    return {
        "strongest_reading": str(dec.get("strongest_reading", "")),
        "load_bearing_gaps": [str(gap) for gap in gaps if gap]
        if isinstance(gaps, list)
        else [],
        "concessions": [str(c) for c in concessions if c]
        if isinstance(concessions, list)
        else [],
    }


async def main(only: str | None) -> None:
    load_dotenv(ROOT / ".env")
    REF_DIR.mkdir(exist_ok=True)
    async with AnthropicClient() as client:
        for row in _items(only):
            reference = await preprocess_argument(
                client, model_id=MODEL, argument=f"{row.question}\n\n{row.argument}"
            )
            gaps = reference["load_bearing_gaps"]
            (REF_DIR / f"{row.item_id}.json").write_text(
                json.dumps(
                    {"argument_id": row.item_id, **reference},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"  {row.item_id:<26} {len(gaps)} load-bearing gap(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="rebuild a single argument by id")
    asyncio.run(main(ap.parse_args().only))
