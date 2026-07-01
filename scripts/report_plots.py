"""Generate report-ready plots for the argument-critique eval.

The script is offline: it reads existing eval trajectories plus probe outputs and
writes PNGs under results/report_plots/.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from critic_evals.transcripts import read_eval_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_JSON = ROOT / "results" / "grader_experiment" / "results.json"
REPRO_JSON = ROOT / "results" / "reproducibility" / "latest.json"
STYLE_SUBSTANCE_JSON = ROOT / "results" / "style_substance_composite" / "latest.json"
DEFAULT_EVAL_DIR = ROOT / "transcripts" / "eval"
OUT_DIR = ROOT / "results" / "report_plots"

MODEL_ORDER = ["opus-4.8", "sonnet-4.6", "haiku-4.5"]
MODEL_COLORS = {
    "opus-4.8": "#2f5f98",
    "sonnet-4.6": "#4f9d69",
    "haiku-4.5": "#c07b3a",
}


def _read_experiment() -> dict[str, object]:
    return json.loads(EXPERIMENT_JSON.read_text(encoding="utf-8"))


def _score(records: list[object]) -> float:
    by_item: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_item[record.item_id].append(record.score)
    return float(np.mean([np.mean(scores) for scores in by_item.values()]))


def _item_scores(records: list[object]) -> dict[str, float]:
    by_item: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_item[record.item_id].append(record.score)
    return {item: float(np.mean(scores)) for item, scores in by_item.items()}


def _latest_composite_records(eval_dir: Path) -> dict[str, list[object]]:
    """Return the latest composite transcript per model label."""
    latest: dict[str, tuple[str, Path, list[object]]] = {}
    pattern = re.compile(r"^(?P<ts>\d{8}T\d{6}Z)_(?P<model>.+)_composite\.jsonl$")
    for path in sorted(eval_dir.glob("*_composite.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            config, records = read_eval_jsonl(path)
        except (TypeError, ValueError):
            continue
        if config.grader != "composite":
            continue
        model = config.model_label
        ts = match.group("ts")
        if model not in latest or ts > latest[model][0]:
            latest[model] = (ts, path, records)
    missing = [model for model in MODEL_ORDER if model not in latest]
    if missing:
        raise SystemExit(f"Missing composite transcripts for: {', '.join(missing)}")
    return {model: latest[model][2] for model in MODEL_ORDER}


def _cluster_bootstrap_ci(
    item_to_scores: dict[str, list[float]],
    *,
    n: int = 20_000,
    alpha: float = 0.10,
) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    items = list(item_to_scores)
    samples = []
    for _ in range(n):
        chosen = rng.choice(items, size=len(items), replace=True)
        item_means = []
        for item in chosen:
            scores = item_to_scores[item]
            picked = rng.choice(scores, size=len(scores), replace=True)
            item_means.append(float(np.mean(picked)))
        samples.append(float(np.mean(item_means)))
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_style_substance(_data: dict[str, object], out_dir: Path) -> None:
    if not STYLE_SUBSTANCE_JSON.exists():
        raise SystemExit(
            "Missing current composite style/substance results. Run "
            "`uv run python scripts/style_substance_composite.py` first."
        )
    data = json.loads(STYLE_SUBSTANCE_JSON.read_text(encoding="utf-8"))
    groups = [
        ("style_preserving", "style-preserving edits", "#9aa4b2"),
        ("substance_changing", "substance-changing edits", "#335c67"),
    ]
    deltas = {
        kind: [
            variant["abs_delta"]
            for variant in data["variants"]
            if variant["kind"] == kind
        ]
        for kind, _label, _color in groups
    }

    x = np.arange(len(groups))
    means = [float(np.mean(deltas[kind])) for kind, _label, _color in groups]
    colors = [color for _kind, _label, color in groups]

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(x, means, color=colors, alpha=0.88)
    for i, (kind, _label, _color) in enumerate(groups):
        points = deltas[kind]
        jitter = np.linspace(-0.08, 0.08, len(points)) if len(points) > 1 else [0]
        ax.scatter(
            x[i] + jitter,
            points,
            color="#f2efe9",
            edgecolor="#222222",
            linewidth=0.7,
            s=42,
            zorder=3,
        )
        ax.text(x[i], means[i] + 0.04, f"mean {means[i]:.2f}", ha="center", fontsize=9)
    ax.set_title("Score changes under critique edits")
    ax.set_ylabel("Absolute score change vs. baseline")
    ax.set_xticks(x, [label for _kind, label, _color in groups])
    ax.set_ylim(0, max(means + [max(values) for values in deltas.values()]) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir / "01_style_vs_substance.png")


def plot_atomic_dims(data: dict[str, object], out_dir: Path) -> None:
    graded = data["graded"]
    rows = [
        ("remove_central", "centrality", "centrality"),
        ("depth_strip", "mechanism depth", "mechanism_depth"),
        ("inject_false_flaw", "precision", "precision"),
        ("fabricate_quote", "faithfulness", "faithfulness"),
        ("severity_inflation", "calibration", "calibration"),
    ]
    dims = ["centrality", "mechanism_depth", "precision", "faithfulness", "calibration"]
    matrix = []
    baseline = graded["great:baseline"]["rubric_nodecomp"]["dimensions"]
    for suffix, _label, _target in rows:
        dims_after = graded[f"great:{suffix}"]["rubric_nodecomp"]["dimensions"]
        matrix.append([baseline[dim] - dims_after[dim] for dim in dims])
    matrix_arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    image = ax.imshow(
        matrix_arr, cmap="YlOrRd", vmin=0, vmax=max(0.5, matrix_arr.max())
    )
    ax.set_title("Atomic mutation score changes")
    ax.set_xticks(np.arange(len(dims)), [d.replace("_", " ") for d in dims])
    ax.tick_params(axis="x", labelrotation=28)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_yticks(np.arange(len(rows)), [label for _suffix, label, _target in rows])
    for y in range(matrix_arr.shape[0]):
        for x in range(matrix_arr.shape[1]):
            ax.text(
                x, y, f"{matrix_arr[y, x]:.2f}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(image, ax=ax, label="Score drop vs. great baseline")
    _save(fig, out_dir / "02_atomic_dimension_response.png")


def plot_e2e(
    records_by_model: dict[str, list[object]], out_dir: Path
) -> dict[str, object]:
    item_order = list(_item_scores(records_by_model[MODEL_ORDER[0]]))
    per_model_items = {
        model: _item_scores(records_by_model[model]) for model in MODEL_ORDER
    }
    matrix = np.array(
        [[per_model_items[model][item] for item in item_order] for model in MODEL_ORDER]
    )

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set_title("End-to-end item scores")
    ax.set_xticks(
        np.arange(len(item_order)), [item.replace("_", "\n") for item in item_order]
    )
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Item score")
    _save(fig, out_dir / "03_e2e_item_heatmap.png")

    scores = []
    lowers = []
    uppers = []
    for model in MODEL_ORDER:
        by_item: dict[str, list[float]] = defaultdict(list)
        for record in records_by_model[model]:
            by_item[record.item_id].append(record.score)
        score = _score(records_by_model[model])
        lo, hi = _cluster_bootstrap_ci(by_item)
        scores.append(score)
        lowers.append(score - lo)
        uppers.append(hi - score)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    ax.bar(MODEL_ORDER, scores, color=colors)
    ax.errorbar(
        MODEL_ORDER, scores, yerr=[lowers, uppers], fmt="none", ecolor="#222", capsize=4
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall score")
    ax.set_title("End-to-end model scores")
    ax.grid(axis="y", alpha=0.25)
    for i, score in enumerate(scores):
        ax.text(i, score + 0.03, f"{score:.2f}", ha="center", fontsize=9)
    _save(fig, out_dir / "04_e2e_model_scores.png")

    return {
        model: {
            "score": scores[i],
            "ci90": [scores[i] - lowers[i], scores[i] + uppers[i]],
            "items": per_model_items[model],
        }
        for i, model in enumerate(MODEL_ORDER)
    }


def plot_dimensions(records_by_model: dict[str, list[object]], out_dir: Path) -> None:
    dims = [
        "fidelity",
        "charity",
        "centrality",
        "derivation",
        "proportionality",
        "synthesis",
    ]
    x = np.arange(len(dims))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, model in zip([-width, 0, width], MODEL_ORDER, strict=True):
        means = [
            float(
                np.mean([record.dimensions[dim] for record in records_by_model[model]])
            )
            for dim in dims
        ]
        ax.bar(x + offset, means, width, label=model, color=MODEL_COLORS[model])
    ax.set_title("Axis scores by model")
    ax.set_ylabel("Mean axis score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [dim.replace("_", "\n") for dim in dims])
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir / "05_axis_breakdown_by_model.png")


def plot_length(records_by_model: dict[str, list[object]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for model in MODEL_ORDER:
        records = records_by_model[model]
        ax.scatter(
            [record.output_tokens for record in records],
            [record.score for record in records],
            label=model,
            alpha=0.78,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_title("Score by critique length")
    ax.set_xlabel("Critique output tokens")
    ax.set_ylabel("Composite score")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out_dir / "06_score_vs_length.png")


def plot_reproducibility(out_dir: Path) -> None:
    if not REPRO_JSON.exists():
        return
    data = json.loads(REPRO_JSON.read_text(encoding="utf-8"))
    cases = data["cases"]
    labels = [case["label"] for case in cases]
    means = np.array([case["mean"] for case in cases], dtype=float)
    stds = np.array([case["std"] for case in cases], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(cases))
    ax.bar(x, means, color="#466b8f", alpha=0.82, label="mean")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#222222", capsize=5, label="std")
    for i, case in enumerate(cases):
        scores = [rep["score"] for rep in case["repetitions"]]
        jitter = np.linspace(-0.10, 0.10, len(scores)) if len(scores) > 1 else [0]
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
    _save(fig, out_dir / "07_grader_reproducibility.png")


def write_summary(summary: dict[str, object], out_dir: Path) -> None:
    source = summary.pop("_source")
    lines = [
        "# Report plot summary",
        "",
        f"Generated from `{source}`.",
        "",
        "| model | score | 90% CI | item scores |",
        "|---|---:|---:|---|",
    ]
    for model in MODEL_ORDER:
        row = summary[model]
        ci = row["ci90"]
        items = " / ".join(
            f"{item}={score:.2f}" for item, score in row["items"].items()
        )
        lines.append(
            f"| {model} | {row['score']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {items} |"
        )
    lines.append("")
    lines.append("Plots:")
    for name in sorted(out_dir.glob("*.png")):
        lines.append(f"- `{name.name}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="directory containing *_composite.jsonl eval trajectories",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.eval_dir
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    data = _read_experiment()
    records_by_model = _latest_composite_records(eval_dir)

    plot_style_substance(data, args.out_dir)
    plot_atomic_dims(data, args.out_dir)
    summary = plot_e2e(records_by_model, args.out_dir)
    summary["_source"] = str(eval_dir.relative_to(ROOT) if eval_dir.is_relative_to(ROOT) else eval_dir)
    plot_dimensions(records_by_model, args.out_dir)
    plot_length(records_by_model, args.out_dir)
    plot_reproducibility(args.out_dir)
    write_summary(summary, args.out_dir)
    print(f"Wrote report plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
