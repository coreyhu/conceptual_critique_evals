"""Dataset items, frozen critique fixtures, and fixture construction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from critic_evals.critique import generate_critique
from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import ModelSpec, resolve
from critic_evals.schema import ArgumentItem, CritiqueRecord
from critic_evals.transcripts import load_argument

_ARGS = Path(__file__).resolve().parents[2] / "arguments"
CRITIQUES_PATH = Path(__file__).resolve().parents[2] / "dataset" / "critiques.jsonl"


@dataclass(frozen=True, slots=True)
class DatasetItem:
    id: str
    argument_path: Path
    soundness: str  # 'flawed' | 'sound'


ITEMS: tuple[DatasetItem, ...] = (
    DatasetItem("organization_metrics", _ARGS / "organization_metrics.txt", "flawed"),
    DatasetItem("cancer_screening", _ARGS / "cancer_screening.txt", "sound"),
    DatasetItem(
        "model_causal_structure", _ARGS / "model_causal_structure.txt", "flawed"
    ),
    DatasetItem(
        "science_funding_returns", _ARGS / "science_funding_returns.txt", "flawed"
    ),
)


def get_items(ids: Iterable[str] | None = None) -> tuple[DatasetItem, ...]:
    """Resolve item ids; `None` returns the full dataset in declared order."""
    if ids is None:
        return ITEMS
    by_id = {item.id: item for item in ITEMS}
    ids = list(ids)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"unknown item id(s): {', '.join(missing)}")
    return tuple(by_id[i] for i in ids)


def load_item_argument(item: DatasetItem) -> ArgumentItem:
    """Load an item's promptable argument text."""
    return load_argument(item.argument_path)


async def _build_critique(
    item: DatasetItem,
    sample: int,
    spec: ModelSpec,
    *,
    client: AnthropicClient,
    arguments: dict[str, ArgumentItem],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> CritiqueRecord:
    async with semaphore:
        return await generate_critique(
            client,
            spec,
            arguments[item.id],
            sample=sample,
            max_tokens=max_tokens,
        )


async def build_critiques(
    items: Sequence[DatasetItem],
    model_labels: Sequence[str],
    *,
    samples: int,
    max_tokens: int = 4096,
    concurrency: int = 5,
    client: AnthropicClient | None = None,
) -> list[CritiqueRecord]:
    """Elicit `samples` critiques for each item/model pair.

    A failed call propagates so callers do not persist an incomplete fixture.
    """
    arguments = {item.id: load_item_argument(item) for item in items}
    specs = resolve(model_labels)
    owns_client = client is None
    client = client or AnthropicClient()
    sem = asyncio.Semaphore(concurrency)

    try:
        return await asyncio.gather(
            *(
                _build_critique(
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


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_critiques(
    records: Iterable[CritiqueRecord],
    path: str | Path,
    *,
    models: Sequence[str],
    samples: int,
    item_ids: Sequence[str],
    built_at: str,
) -> Path:
    """Write the committed critique fixture: a manifest header, then one record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "manifest",
        "models": list(models),
        "samples": samples,
        "item_ids": list(item_ids),
        "built_at": built_at,
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_critiques(
    path: str | Path = CRITIQUES_PATH,
) -> tuple[dict[str, object], tuple[CritiqueRecord, ...]]:
    """Read the fixture into its manifest and critique records. Raises if the header is missing."""
    manifest: dict[str, object] | None = None
    records: list[CritiqueRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "manifest":
                obj.pop("kind")
                manifest = obj
            else:
                records.append(CritiqueRecord.from_dict(obj))
    if manifest is None:
        raise ValueError(f"{path}: no manifest header (kind=manifest) found")
    return manifest, tuple(records)


def get_critiques(
    model_label: str, *, path: str | Path = CRITIQUES_PATH
) -> tuple[CritiqueRecord, ...]:
    """Return one model's frozen critiques from the fixture; raise if it has none."""
    _, records = load_critiques(path)
    picked = tuple(r for r in records if r.model == model_label)
    if not picked:
        raise ValueError(
            f"no critiques for model {model_label!r} in {path}; build the fixture first"
        )
    return picked
