"""Load committed reference metadata."""

from __future__ import annotations

import json
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "references"


def load_reference(item_id: str, ref_dir: Path = REF_DIR) -> dict[str, object]:
    return json.loads((ref_dir / f"{item_id}.json").read_text(encoding="utf-8"))
