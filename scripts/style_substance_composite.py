"""Run style/substance edit probes against the current composite grader only."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from critic_evals.dataset import load_eval_dataset
from critic_evals.grading.composite import CompositeGrader
from critic_evals.llm.client import AnthropicClient
from critic_evals.references import load_reference
from critic_evals.transcripts import read_eval_jsonl

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_DIR = ROOT / "transcripts" / "eval"
OUT_JSON = ROOT / "results" / "style_substance_composite" / "latest.json"
MODEL = "claude-opus-4-8"


STYLE_FORMATTED = """Main assessment: the argument proves less than it claims. It is right that out-of-sample success rules out mere memorization and supports the presence of a stable statistical relationship. It is wrong to treat that stability as evidence that the model has found causal structure.

The missing category is stable non-causal association. If a common cause drives both a feature and an outcome, the feature can predict the outcome across new people, later time periods, and arbitrarily large held-out test sets. That relationship is not a sample artifact, but it is still not a causal lever.

This is why the argument's proposed escalation does not solve the problem. New individuals, future periods, and larger held-out sets usually test the same observational data-generating process. They do not test what happens under intervention, which is the evidence needed to distinguish causation from stable confounding.

The burden-shift therefore fails. The skeptic does not need to explain how a coincidental model keeps succeeding. A confounded model can keep succeeding indefinitely, so predictive success licenses stable correlation, not a presumption of causal levers."""


STYLE_POLISHED = """The argument is strongest when it says predictive success is evidence of a stable relationship rather than memorized noise. That point is correct. Its overreach is the next step: it moves from stable generalization to causal structure.

The missing distinction is between stability and causality. A common cause can make one variable predict another across held-out individuals, future periods, and larger test sets while neither variable is a causal lever for the other. Those forms of validation show persistence within an observational distribution; they do not supply the intervention-style evidence needed to distinguish causation from stable confounding.

So the final burden-shift is too strong. The alternative to causal structure is not only coincidence. Stable non-causal association is a third possibility, and it fully explains sustained out-of-sample success."""


SUBSTANCE_REMOVE_CENTRAL = """The argument is broadly persuasive. Out-of-sample performance is exactly the right test for whether a model has learned something real rather than noise, and repeated success across new individuals and later time periods should substantially raise confidence that the model reflects the structure of the system.

The argument could be more careful about edge cases. Some correlations may be more stable than expected, and predictive accuracy alone is not a mathematical proof of causation. But the author already acknowledges that prediction does not prove causation outright. As a practical matter, the burden-shifting claim is reasonable: if a model keeps succeeding under increasingly demanding tests, skepticism needs a concrete explanation rather than a generic warning about correlation."""


SUBSTANCE_FALSE_FLAW = """The argument fails because it says prediction proves causation outright. That is an obviously invalid inference: no amount of predictive success can logically prove a causal relationship. Since the argument rests on this proof claim, the whole position collapses.

It also treats any out-of-sample test as equivalent to an intervention. But the text never explains randomized assignment, instrumental variables, or any credible causal identification strategy. Without those, the argument has no basis for using causal language at all."""


SUBSTANCE_DEPTH_STRIP = """The argument is partly right but too strong. Out-of-sample success shows that the model found some pattern that generalizes beyond the training data. That is useful evidence.

But generalization is not the same thing as causation. A model can predict well for non-causal reasons, so the conclusion should be weaker. The argument would be better if it said prediction is suggestive evidence rather than creating a presumption of real causal levers."""


def _arg_text(item_id: str) -> str:
    for row in load_eval_dataset():
        if row.item_id == item_id:
            return f"{row.question}\n\n{row.argument}"
    raise ValueError(f"unknown item id: {item_id}")


def _latest_live_critique(model_label: str, item_id: str, sample: int) -> str:
    paths = sorted(TRANSCRIPT_DIR.glob(f"*_{model_label}_composite.jsonl"), reverse=True)
    for path in paths:
        try:
            config, records = read_eval_jsonl(path)
        except (TypeError, ValueError):
            continue
        if config.model_label != model_label or config.grader != "composite":
            continue
        for record in records:
            if record.item_id == item_id and record.sample == sample:
                return record.critique
    raise ValueError(f"no composite critique found for {model_label} {item_id} sample {sample}")


async def _grade(
    grader: CompositeGrader,
    client: AnthropicClient,
    *,
    argument: str,
    reference: dict[str, object],
    critique: str,
) -> dict[str, object]:
    score = await grader.grade(
        client,
        model_id=MODEL,
        argument=argument,
        reference=reference,
        critique=critique,
    )
    return {
        "score": score.score,
        "dimensions": score.dimensions,
        "raw": score.raw,
    }


async def run_probe() -> dict[str, object]:
    load_dotenv(ROOT / ".env")
    item_id = "model_causal_structure"
    argument = _arg_text(item_id)
    reference = load_reference(item_id)
    baseline = _latest_live_critique("opus-4.8", item_id, 2)
    variants = [
        {
            "id": "style_formatted",
            "kind": "style_preserving",
            "label": "formatted",
            "critique": STYLE_FORMATTED,
        },
        {
            "id": "style_polished",
            "kind": "style_preserving",
            "label": "polished",
            "critique": STYLE_POLISHED,
        },
        {
            "id": "remove_central",
            "kind": "substance_changing",
            "label": "remove central flaw",
            "critique": SUBSTANCE_REMOVE_CENTRAL,
        },
        {
            "id": "false_flaw",
            "kind": "substance_changing",
            "label": "false flaw",
            "critique": SUBSTANCE_FALSE_FLAW,
        },
        {
            "id": "depth_strip",
            "kind": "substance_changing",
            "label": "depth strip",
            "critique": SUBSTANCE_DEPTH_STRIP,
        },
    ]

    grader = CompositeGrader()
    async with AnthropicClient() as client:
        print("grading baseline", flush=True)
        baseline_grade = await _grade(
            grader, client, argument=argument, reference=reference, critique=baseline
        )
        graded_variants = []
        for variant in variants:
            print(f"grading {variant['id']}", flush=True)
            grade = await _grade(
                grader,
                client,
                argument=argument,
                reference=reference,
                critique=variant["critique"],
            )
            graded_variants.append(
                {
                    **{k: v for k, v in variant.items() if k != "critique"},
                    **grade,
                    "abs_delta": abs(grade["score"] - baseline_grade["score"]),
                    "delta": grade["score"] - baseline_grade["score"],
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grader": "composite",
        "grader_model_id": MODEL,
        "item_id": item_id,
        "baseline": {
            "id": "live_prediction_high",
            "source": "latest opus-4.8 composite model_causal_structure sample 2",
            **baseline_grade,
        },
        "variants": graded_variants,
    }


def write_result(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = await run_probe()
    write_result(data, args.out_json)
    print(f"Wrote {args.out_json}")
    print(f"baseline score={data['baseline']['score']:.3f}")
    for variant in data["variants"]:
        print(
            f"{variant['id']:<18} kind={variant['kind']:<18} "
            f"score={variant['score']:.3f} delta={variant['delta']:+.3f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
