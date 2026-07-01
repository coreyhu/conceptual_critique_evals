"""Read and write eval run transcripts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from critic_evals.eval import EvalConfig, EvalRecord

_CONFIG_FIELDS = set(EvalConfig.__dataclass_fields__)
_RECORD_FIELDS = set(EvalRecord.__dataclass_fields__)


def write_eval_jsonl(
    records: Iterable[EvalRecord], path: str | Path, *, config: EvalConfig
) -> Path:
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
                config = EvalConfig(
                    **{k: v for k, v in obj.items() if k in _CONFIG_FIELDS}
                )
            else:
                records.append(
                    EvalRecord(**{k: v for k, v in obj.items() if k in _RECORD_FIELDS})
                )
    if config is None:
        raise ValueError(f"{path}: no config header (kind=config) found")
    return config, records
