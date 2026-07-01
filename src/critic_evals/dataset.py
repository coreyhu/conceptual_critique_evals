"""Frozen eval dataset rows and JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from critic_evals.schema import Usage

EVAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "dataset" / "eval_dataset.jsonl"
)


@dataclass(frozen=True, slots=True)
class EvalDatasetRow:
    id: str
    item_id: str
    question: str
    argument: str
    model: str
    model_id: str
    sample: int
    critique_prompt: str
    critique: str
    stop_reason: str | None
    usage: Usage | None
    request_id: str | None
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EvalDatasetRow:
        raw_usage = data.get("usage")
        usage = (
            Usage(int(raw_usage["input_tokens"]), int(raw_usage["output_tokens"]))
            if isinstance(raw_usage, dict)
            else None
        )

        def s(key: str) -> str:
            return str(data[key])

        def opt(key: str) -> str | None:
            v = data.get(key)
            return None if v is None else str(v)

        default_id = f"{s('item_id')}__{s('model')}__sample_{s('sample')}"
        return cls(
            id=str(data["id"]) if data.get("id") is not None else default_id,
            item_id=s("item_id"),
            question=s("question"),
            argument=s("argument"),
            model=s("model"),
            model_id=s("model_id"),
            sample=int(str(data["sample"])),
            critique_prompt=s("critique_prompt"),
            critique=s("critique"),
            stop_reason=opt("stop_reason"),
            usage=usage,
            request_id=opt("request_id"),
            timestamp=s("timestamp"),
        )


def write_eval_dataset(
    rows: Iterable[EvalDatasetRow], path: str | Path = EVAL_DATASET_PATH
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_eval_dataset(
    path: str | Path = EVAL_DATASET_PATH,
    *,
    model_label: str | None = None,
) -> tuple[EvalDatasetRow, ...]:
    rows: list[EvalDatasetRow] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(EvalDatasetRow.from_dict(json.loads(line)))
    if model_label is None:
        return tuple(rows)
    filtered = tuple(row for row in rows if row.model == model_label)
    if not filtered:
        raise ValueError(
            f"no dataset rows for model {model_label!r} in {path}; build the dataset first"
        )
    return filtered
