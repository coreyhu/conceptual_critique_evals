"""Loading argument items and reading/writing JSONL transcripts.

A transcript is one JSON object per line, each a serialized `CritiqueRecord`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from critic_evals.schema import ArgumentItem, CritiqueRecord, CritiqueRow

_HEADING = re.compile(r"^#\s+(.*)$")
_ARGUMENT_HEADING = re.compile(r"^#+\s*argument\b.*$", re.IGNORECASE)


def parse_argument(text: str, item_id: str) -> ArgumentItem:
    """Parse a markdown argument file into an `ArgumentItem`.

    Expected shape (lenient): a top-level `# question` heading, then the body,
    optionally under an `## Argument` heading. If no `Argument` heading is
    present, everything after the question heading is treated as the argument.
    """
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
    """Load and parse an argument file; `item_id` defaults to the filename stem."""
    path = Path(path)
    return parse_argument(path.read_text(encoding="utf-8"), item_id or path.stem)


def write_jsonl(records: Iterable[CritiqueRecord], path: str | Path) -> Path:
    """Write records as JSONL, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[CritiqueRecord]:
    """Read a transcript back into `CritiqueRecord` objects."""
    path = Path(path)
    records: list[CritiqueRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CritiqueRecord.from_dict(json.loads(line)))
    return records


def read_jsonl_dicts(path: str | Path) -> list[CritiqueRow]:
    """Read a transcript as raw rows (handy for ad-hoc analysis)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_rows(path: str | Path) -> list[dict[str, object]]:
    """Read any JSONL file into plain dicts (for schemas without a typed reader)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
