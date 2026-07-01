"""Tests for the deterministic, non-API logic: argument parsing, prompt
construction, the model registry, and JSONL round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from critic_evals.critique import build_prompt
from critic_evals.grading.derivation import DerivationGrader
from critic_evals.grading.fidelity import FidelityGrader
from critic_evals.grading.grader import level_score
from critic_evals.grading.centrality import CentralityGrader
from critic_evals.llm.models import MODELS, resolve
from critic_evals.schema import ArgumentItem, CritiqueRecord, Usage
from critic_evals.transcripts import parse_argument, read_jsonl, write_jsonl

SAMPLE_MD = """# Why do estimates overrun?

## Argument
Estimates overrun because of the planning fallacy.
They also overrun because of scope creep.
"""


def test_parse_argument_splits_question_and_body():
    item = parse_argument(SAMPLE_MD, item_id="estimates")
    assert item.id == "estimates"
    assert item.question == "Why do estimates overrun?"
    assert item.argument.startswith(
        "Estimates overrun because of the planning fallacy."
    )
    assert "scope creep" in item.argument
    # the heading lines themselves are not part of the argument body
    assert "## Argument" not in item.argument


def test_parse_argument_without_argument_heading_uses_body_after_question():
    text = "# Q?\n\nBody line one.\nBody line two.\n"
    item = parse_argument(text, item_id="x")
    assert item.question == "Q?"
    assert item.argument == "Body line one.\nBody line two."


def test_parse_argument_rejects_empty_body():
    with pytest.raises(ValueError):
        parse_argument("# Just a question\n", item_id="x")


def test_build_prompt_fills_question_and_argument():
    item = ArgumentItem(id="x", question="Why?", argument="Because.")
    system, prompt = build_prompt(item)
    assert "critic" in system.lower()
    assert "Why?" in prompt
    assert "Because." in prompt


def test_model_registry_ids_and_effort_flags():
    assert MODELS["opus-4.8"].model_id == "claude-opus-4-8"
    assert MODELS["sonnet-4.6"].model_id == "claude-sonnet-4-6"
    assert MODELS["haiku-4.5"].model_id == "claude-haiku-4-5"
    # Haiku rejects `effort`; the registry must record that so callers can branch.
    assert MODELS["haiku-4.5"].supports_effort is False
    assert MODELS["opus-4.8"].supports_effort is True


def test_resolve_unknown_label_raises():
    with pytest.raises(KeyError):
        resolve(["opus-4.8", "nope"])


def test_jsonl_roundtrip_preserves_records(tmp_path):
    records = [
        CritiqueRecord(
            item_id="estimates",
            model="opus-4.8",
            model_id="claude-opus-4-8",
            sample=0,
            system="sys",
            prompt="prompt",
            response="a critique",
            success=True,
            stop_reason="end_turn",
            usage=Usage(input_tokens=120, output_tokens=300),
            request_id="req_abc",
            timestamp="2026-06-30T00:00:00+00:00",
        ),
        CritiqueRecord(
            item_id="estimates",
            model="haiku-4.5",
            model_id="claude-haiku-4-5",
            sample=1,
            system="sys",
            prompt="prompt",
            response="",
            success=False,
            stop_reason="error",
            usage=None,
            request_id=None,
            timestamp="2026-06-30T00:00:01+00:00",
            error="boom",
        ),
    ]
    path = write_jsonl(records, tmp_path / "t.jsonl")
    loaded = read_jsonl(path)
    assert loaded == records  # frozen dataclasses compare by value


def _sample_eval_record(sample: int = 0):
    from critic_evals.eval import EvalRecord

    return EvalRecord(
        model="opus-4.8",
        model_id="claude-opus-4-8",
        item_id="estimates",
        soundness="unsound",
        sample=sample,
        critique_prompt="p",
        critique="c",
        grader="composite",
        grader_model_id="claude-opus-4-8",
        used_key=True,
        score=0.5,
        dimensions={"centrality": 1.0},
        grader_raw={"level": 2},
        input_tokens=10,
        output_tokens=20,
        timestamp="2026-06-30T00:00:00+00:00",
    )


def test_grader_registry_resolves_names_and_rejects_unknown():
    from critic_evals.grading.composite import CompositeGrader
    from critic_evals.grading.registry import GRADERS, get_grader

    assert set(GRADERS) == {
        "composite",
        "fidelity",
        "charity",
        "centrality",
        "derivation",
        "decisiveness",
        "calibration",
    }
    assert isinstance(get_grader("composite"), CompositeGrader)
    assert isinstance(get_grader("centrality"), CentralityGrader)
    assert get_grader("centrality").name == "centrality"
    with pytest.raises(KeyError):
        get_grader("does-not-exist")


def test_get_items_selects_by_id_in_requested_order_and_rejects_unknown():
    from critic_evals.dataset import ITEMS, get_items

    assert get_items() == ITEMS  # None -> whole dataset in declared order
    picked = get_items(["cancer_screening", "organization_metrics"])
    assert [it.id for it in picked] == ["cancer_screening", "organization_metrics"]
    # a one-shot generator is materialized, not exhausted before the lookup
    assert get_items(i for i in ["cancer_screening"])[0].id == "cancer_screening"
    with pytest.raises(ValueError):
        get_items(["cancer_screening", "nope"])


def test_eval_jsonl_writes_config_header(tmp_path):
    from critic_evals.eval import EvalConfig, write_eval_jsonl

    config = EvalConfig(model_label="opus-4.8", k_grade=1)
    path = write_eval_jsonl(
        [_sample_eval_record()], tmp_path / "eval.jsonl", config=config
    )
    lines = path.read_text(encoding="utf-8").splitlines()

    header = json.loads(lines[0])
    assert header["kind"] == "config"
    assert header["model_label"] == "opus-4.8"
    # k_grade lives nowhere in EvalRecord — the header is what captures it
    assert header["k_grade"] == 1

    rows = [json.loads(line) for line in lines[1:]]
    assert len(rows) == 1
    assert rows[0]["item_id"] == "estimates"
    assert (
        "kind" not in rows[0]
    )  # records stay untagged; only the header carries `kind`


def test_eval_jsonl_roundtrip_recovers_config_and_records(tmp_path):
    from critic_evals.eval import EvalConfig, read_eval_jsonl, write_eval_jsonl

    config = EvalConfig(model_label="opus-4.8", k_grade=2)
    records = [_sample_eval_record(0), _sample_eval_record(1)]
    path = write_eval_jsonl(records, tmp_path / "e.jsonl", config=config)

    got_config, got_records = read_eval_jsonl(path)
    assert got_config == config  # frozen dataclasses compare by value
    assert got_records == records


def test_read_eval_jsonl_rejects_missing_and_duplicate_headers(tmp_path):
    from critic_evals.eval import EvalConfig, read_eval_jsonl, write_eval_jsonl

    no_header = tmp_path / "no_header.jsonl"
    no_header.write_text(
        json.dumps(_sample_eval_record().to_dict()) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_eval_jsonl(no_header)

    # two runs concatenated: which config do the records belong to? — refuse to guess
    merged = tmp_path / "merged.jsonl"
    write_eval_jsonl(
        [_sample_eval_record()], merged, config=EvalConfig(model_label="a")
    )
    with merged.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"kind": "config", **EvalConfig(model_label="b").to_dict()})
            + "\n"
        )
    with pytest.raises(ValueError):
        read_eval_jsonl(merged)


def test_level_score_normalizes_and_clamps():
    assert level_score({"level": 0}, 2) == 0.0
    assert level_score({"level": 1}, 2) == 0.5
    assert level_score({"level": 2}, 2) == 1.0
    assert level_score({"level": 3}, 3) == 1.0
    assert level_score({"level": 9}, 2) == 1.0  # clamps into range
    assert level_score({}, 2) == 0.0  # missing -> 0


def test_axis_graders_extract_their_rubric_level():
    # each axis is one dimension: extract returns the normalized level as a float
    assert FidelityGrader().extract({"level": 0}) == 0.0
    assert CentralityGrader().extract({"level": 2}) == 1.0
    assert CentralityGrader().extract({"level": 1}) == 0.5
    assert CentralityGrader().extract({"level": 0}) == 0.0
    assert DerivationGrader().extract({"level": 2}) == 1.0
    assert DerivationGrader().extract({"level": 1}) == 0.5


def test_composite_preserves_axis_raw_outputs(monkeypatch):
    import asyncio
    import critic_evals.grading.composite as comp
    from critic_evals.grading.grader import BaseGrader, GraderScore

    class DummyGrader(BaseGrader):
        def __init__(self, name: str, score: float):
            self.name = name
            self._score = score

        async def grade(self, client, *, model_id, argument, reference, critique):
            return GraderScore(
                score=self._score,
                dimensions={},
                raw={"axis": self.name, "why_not_higher": "test diagnostic"},
            )

    monkeypatch.setattr(comp, "_FIDELITY", DummyGrader("fidelity", 1.0))
    monkeypatch.setattr(comp, "_CHARITY", DummyGrader("charity", 1.0))
    monkeypatch.setattr(comp, "_DECISIVENESS", DummyGrader("decisiveness", 1.0))
    monkeypatch.setattr(comp, "_DERIVATION", DummyGrader("derivation", 0.5))
    monkeypatch.setattr(comp, "_CENTRALITY", DummyGrader("centrality", 0.5))

    gs = asyncio.run(
        comp.CompositeGrader().grade(
            client=None,
            model_id="grader",
            argument="arg",
            reference={"soundness": "flawed", "strongest_reading": "steelman"},
            critique="crit",
        )
    )

    assert gs.score == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert gs.raw["derivation"]["why_not_higher"] == "test diagnostic"
    assert gs.raw["centrality"]["axis"] == "centrality"


class _FakeCompletion:
    model_id = "claude-opus-4-8"
    text = "a critique"
    stop_reason = "end_turn"
    input_tokens = 10
    output_tokens = 20
    request_id = "req_x"


class _FakeClient:
    """Stands in for AnthropicClient in generation tests."""

    def __init__(self, *, fail: bool = False):
        self._fail = fail

    async def complete(self, *, model_id, system, prompt, max_tokens=4096, schema=None):
        if self._fail:
            raise RuntimeError("boom")
        return _FakeCompletion()

    async def aclose(self):
        pass


async def test_build_dataset_fans_out_arguments_models_samples():
    from critic_evals.dataset import DatasetItem, build_critiques

    item = DatasetItem("x", Path("unused.txt"), "flawed")

    async def fake_generate_critique(client, spec, arg, *, sample, max_tokens=4096):
        return _critique_record(spec.label, arg.id, sample)

    import critic_evals.dataset as dataset_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        dataset_mod, "load_item_argument", lambda _: ArgumentItem("x", "Q", "A")
    )
    monkeypatch.setattr(dataset_mod, "generate_critique", fake_generate_critique)
    try:
        records = await build_critiques(
            [item], ["opus-4.8"], samples=2, client=_FakeClient()
        )
    finally:
        monkeypatch.undo()

    assert len(records) == 2
    assert {r.sample for r in records} == {0, 1}
    assert all(
        r.item_id == "x" and r.model == "opus-4.8" and r.success for r in records
    )


async def test_build_dataset_aborts_on_failure():
    import pytest as _pytest

    from critic_evals.dataset import DatasetItem, build_critiques

    item = DatasetItem("x", Path("unused.txt"), "flawed")

    import critic_evals.dataset as dataset_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        dataset_mod, "load_item_argument", lambda _: ArgumentItem("x", "Q", "A")
    )
    with _pytest.raises(RuntimeError):
        try:
            await build_critiques(
                [item], ["opus-4.8"], samples=1, client=_FakeClient(fail=True)
            )
        finally:
            monkeypatch.undo()


def _critique_record(model: str, item_id: str, sample: int):
    from critic_evals.schema import CritiqueRecord, Usage

    return CritiqueRecord(
        item_id=item_id,
        model=model,
        model_id=f"claude-{model.replace('.', '-')}",
        sample=sample,
        system="sys",
        prompt="p",
        response="a critique",
        success=True,
        stop_reason="end_turn",
        usage=Usage(10, 20),
        request_id="req",
        timestamp="2026-06-30T00:00:00+00:00",
    )


def test_critiques_fixture_roundtrip_and_filter(tmp_path):
    from critic_evals.dataset import get_critiques, load_critiques, write_critiques

    records = [
        _critique_record("opus-4.8", "cancer_screening", 0),
        _critique_record("opus-4.8", "cancer_screening", 1),
        _critique_record("haiku-4.5", "cancer_screening", 0),
    ]
    path = write_critiques(
        records,
        tmp_path / "critiques.jsonl",
        models=["opus-4.8", "haiku-4.5"],
        samples=2,
        item_ids=["cancer_screening"],
        built_at="2026-06-30T00:00:00+00:00",
    )

    manifest, loaded = load_critiques(path)
    assert manifest["models"] == ["opus-4.8", "haiku-4.5"]
    assert manifest["samples"] == 2
    assert loaded == tuple(records)  # frozen dataclasses compare by value

    picked = get_critiques("opus-4.8", path=path)
    assert len(picked) == 2 and all(r.model == "opus-4.8" for r in picked)

    with pytest.raises(ValueError):
        get_critiques("not-a-model", path=path)


async def test_evaluate_grades_frozen_critiques_without_calling_models(monkeypatch):
    from critic_evals import eval as evalmod
    from critic_evals.eval import EvalConfig, evaluate
    from critic_evals.grading.grader import GraderScore

    frozen = [
        _critique_record("opus-4.8", "cancer_screening", 0),
        _critique_record("opus-4.8", "cancer_screening", 1),
    ]

    class _StubGrader:
        async def grade(self, client, *, model_id, argument, reference, critique):
            return GraderScore(score=0.5, dimensions={}, raw={})

    monkeypatch.setattr(evalmod, "get_critiques", lambda label: tuple(frozen))
    monkeypatch.setattr(evalmod, "get_grader", lambda name: _StubGrader())

    config = EvalConfig(model_label="opus-4.8", use_key=False)
    result, records = await evaluate(config, client=object())

    assert len(records) == 2
    assert all(r.score == 0.5 for r in records)
    assert result.score == 0.5
    assert [it.item_id for it in result.items] == ["cancer_screening"]
    # populated from the frozen record, not a fresh call:
    assert records[0].critique == "a critique"
    assert records[0].model_id == "claude-opus-4-8"
