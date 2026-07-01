# critic_evals

`critic_evals` is an evaluation benchmark for measuring how well LLMs critique conceptual arguments.
It freezes model-written critiques for a curated set of arguments, grades those critiques with a
multi-axis LLM judge, and aggregates the result to a single `LLM -> [0,1]` score.

The eval is designed to reward substance over presentation. A critique should not score well just
because it is confident, polished, or long; it should score well when it engages the argument
charitably, identifies the load-bearing issue, explains why that issue matters, and calibrates the
severity of its verdict.

## Overview

The default composite score is:

```text
score = validity * core
validity = min(fidelity, charity)
core = geometric_mean(centrality, derivation, proportionality, synthesis)
```

Each axis returns an ordinal level normalized into `[0, 1]`.

- `fidelity`: avoids fabricating claims, quotes, contradictions, or flaws.
- `charity`: engages the strongest reading rather than weakmanning the argument.
- `centrality`: targets the dominant load-bearing element.
- `derivation`: explains where the critique's force comes from in the argument's own reasoning.
- `proportionality`: matches the verdict's severity to the argument's actual merit.
- `synthesis`: composes correct observations into a deeper diagnosis rather than a checklist.

The geometric mean makes quality conjunctive: a critique cannot wash out weak grounding,
weak calibration, or shallow synthesis by being strong on another axis.

## Quickstart

Install dependencies:

```bash
uv sync
```

Set your Anthropic API key for scripts that call models:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...
```

Run the offline tests:

```bash
uv run pytest -q
```

Run the default benchmark over the committed dataset:

```bash
uv run python scripts/run_eval.py
```

Run one model only:

```bash
uv run python scripts/run_eval.py --models opus-4.8
```

Eval transcripts are written to `transcripts/eval/`.

## Current Results

The table below was generated from `eval_trajectories/` with
`uv run python scripts/report_plots.py --eval-dir eval_trajectories`.

| model | score | 90% CI | metrics / screening / prediction / funding |
|---|---:|---:|---|
| Opus 4.8 | 0.809 | [0.680, 0.904] | 0.86 / 0.57 / 0.90 / 0.90 |
| Sonnet 4.6 | 0.654 | [0.537, 0.776] | 0.69 / 0.54 / 0.86 / 0.54 |
| Haiku 4.5 | 0.481 | [0.264, 0.685] | 0.35 / 0.15 / 0.76 / 0.66 |

The ordering is monotonic in this run, but the item set is small, so the confidence intervals are
broad.

## Common Workflows

Rebuild the frozen critique dataset from `arguments/`:

```bash
uv run python scripts/build_dataset.py
```

Rebuild the cached argument references used by the keyed validity checks:

```bash
uv run python scripts/build_references.py
```

Probe grader reproducibility by repeatedly grading fixed critiques:

```bash
uv run python scripts/grader_reproducibility.py -k 3
```

Run style-vs-substance edit probes for the current composite grader:

```bash
uv run python scripts/style_substance_composite.py
```

Generate report plots from selected eval trajectories and probe outputs:

```bash
uv run python scripts/report_plots.py --eval-dir eval_trajectories
```

Export dataset critiques to readable Markdown:

```bash
uv run python scripts/export_critiques_markdown.py dataset/eval_dataset.jsonl --out exports/critiques
```

Optional/legacy checks:

```bash
uv run python scripts/validate_grader.py
uv run python scripts/variance.py --show-singles
```

