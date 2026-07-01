"""Measure LLM-judge reproducibility by re-grading fixed critiques.

This isolates grader variance from model-generation variance: each case uses the
same argument, reference key, and critique text on every repetition.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from critic_evals.dataset import load_eval_dataset
from critic_evals.grading.composite import CompositeGrader
from critic_evals.grading.grader import Reference
from critic_evals.llm.client import AnthropicClient
from critic_evals.references import load_reference
from critic_evals.transcripts import read_eval_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_grader import (
    ENDORSE,
    EXPOSES,
    MANUFACTURED,
    MODEL,
    REACHES,
    TEARDOWN,
)

OUT_JSON = ROOT / "results" / "reproducibility" / "latest.json"
OUT_PLOT = ROOT / "results" / "report_plots" / "07_grader_reproducibility.png"
TRANSCRIPT_DIR = ROOT / "transcripts" / "eval"


ALL_CASES = [
    {
        "id": "metrics_great",
        "label": "metrics\ngreat",
        "item_id": "organization_metrics",
        "critique": EXPOSES,
    },
    {
        "id": "metrics_good",
        "label": "metrics\ngood",
        "item_id": "organization_metrics",
        "critique": REACHES,
    },
    {
        "id": "metrics_false",
        "label": "metrics\nfalse flaw",
        "item_id": "organization_metrics",
        "critique": MANUFACTURED,
    },
    {
        "id": "screening_endorse",
        "label": "screening\nendorse",
        "item_id": "cancer_screening",
        "critique": ENDORSE,
    },
    {
        "id": "screening_teardown",
        "label": "screening\nteardown",
        "item_id": "cancer_screening",
        "critique": TEARDOWN,
    },
    {
        "id": "live_prediction_high",
        "label": "prediction\nhigh",
        "item_id": "model_causal_structure",
        "model_label": "opus-4.8",
        "sample": 2,
    },
    {
        "id": "live_metrics_mid",
        "label": "metrics\nmid",
        "item_id": "organization_metrics",
        "model_label": "sonnet-4.6",
        "sample": 2,
    },
]


def _arg_text(item_id: str) -> str:
    for row in load_eval_dataset():
        if row.item_id == item_id:
            return f"{row.question}\n\n{row.argument}"
    raise ValueError(f"unknown item id: {item_id}")


async def _grade_case(
    grader: CompositeGrader,
    client: AnthropicClient,
    *,
    argument: str,
    ref: Reference,
    critique: str,
) -> dict[str, object]:
    gs = await grader.grade(
        client, model_id=MODEL, argument=argument, reference=ref, critique=critique
    )
    return {
        "score": gs.score,
        "dimensions": gs.dimensions,
        "raw": gs.raw,
    }


def _select_cases(case_ids: list[str] | None) -> list[dict[str, str]]:
    if not case_ids:
        case_ids = ["live_prediction_high", "live_metrics_mid", "metrics_false"]
    by_id = {case["id"]: case for case in ALL_CASES}
    unknown = [case_id for case_id in case_ids if case_id not in by_id]
    if unknown:
        raise SystemExit(f"unknown case id(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in case_ids]


def _latest_critique(model_label: str, item_id: str, sample: int) -> str:
    paths = sorted(
        TRANSCRIPT_DIR.glob(f"*_{model_label}_composite.jsonl"), reverse=True
    )
    for path in paths:
        try:
            config, records = read_eval_jsonl(path)
        except TypeError, ValueError:
            continue
        if config.model_label != model_label or config.grader != "composite":
            continue
        for record in records:
            if record.item_id == item_id and record.sample == sample:
                return record.critique
    raise ValueError(
        f"no latest critique found for {model_label} {item_id} sample {sample}"
    )


def _case_critique(case: dict[str, object]) -> str:
    if "critique" in case:
        return str(case["critique"])
    return _latest_critique(
        str(case["model_label"]),
        str(case["item_id"]),
        int(case["sample"]),
    )


async def run_probe(k: int, *, case_ids: list[str] | None = None) -> dict[str, object]:
    load_dotenv(ROOT / ".env")
    grader = CompositeGrader()
    cases = _select_cases(case_ids)
    async with AnthropicClient() as client:
        case_results = []
        for case in cases:
            argument = _arg_text(case["item_id"])
            ref = load_reference(case["item_id"])
            critique = _case_critique(case)
            reps = []
            for repetition in range(k):
                print(
                    f"grading {case['id']} repeat {repetition + 1}/{k}",
                    flush=True,
                )
                result = await _grade_case(
                    grader,
                    client,
                    argument=argument,
                    ref=ref,
                    critique=critique,
                )
                result["repetition"] = repetition
                reps.append(result)
            scores = np.array([rep["score"] for rep in reps], dtype=float)
            case_results.append(
                {
                    "id": case["id"],
                    "label": case["label"],
                    "item_id": case["item_id"],
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "repetitions": reps,
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grader": "composite",
        "grader_model_id": MODEL,
        "k": k,
        "case_ids": [case["id"] for case in cases],
        "cases": case_results,
    }


def plot_probe(data: dict[str, object], path: Path = OUT_PLOT) -> None:
    cases = data["cases"]
    labels = [case["label"] for case in cases]
    means = np.array([case["mean"] for case in cases], dtype=float)
    stds = np.array([case["std"] for case in cases], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(cases))
    ax.bar(x, means, color="#466b8f", alpha=0.82, label="mean score")
    ax.errorbar(
        x, means, yerr=stds, fmt="none", ecolor="#222222", capsize=5, label="std"
    )
    for i, case in enumerate(cases):
        scores = [rep["score"] for rep in case["repetitions"]]
        jitter = np.linspace(-0.12, 0.12, len(scores)) if len(scores) > 1 else [0]
        ax.scatter(
            x[i] + jitter,
            scores,
            color="#f2efe9",
            edgecolor="#222222",
            linewidth=0.7,
            s=34,
            zorder=3,
        )
        ax.text(
            x[i],
            min(1.03, means[i] + stds[i] + 0.05),
            f"std {stds[i]:.2f}",
            ha="center",
            fontsize=9,
        )

    ax.set_title("Repeated grading of fixed critiques")
    ax.set_ylabel("Composite score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_probe(data: dict[str, object], path: Path = OUT_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-k", type=int, default=3, help="number of repeated grades per case"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="case ids to run; defaults to metrics_great metrics_good metrics_false",
    )
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-plot", type=Path, default=OUT_PLOT)
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = await run_probe(args.k, case_ids=args.cases)
    write_probe(data, args.out_json)
    plot_probe(data, args.out_plot)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_plot}")
    for case in data["cases"]:
        print(
            f"{case['id']:<20} mean={case['mean']:.3f} std={case['std']:.3f} "
            f"range=[{case['min']:.3f}, {case['max']:.3f}]"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
