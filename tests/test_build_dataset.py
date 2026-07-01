from __future__ import annotations

from scripts.build_dataset import parse_args


def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.samples == 3
    assert ns.out.endswith("eval_dataset.jsonl")


def test_parse_args_overrides():
    ns = parse_args(["--models", "opus-4.8", "--samples", "5"])
    assert ns.models == ["opus-4.8"]
    assert ns.samples == 5
