from __future__ import annotations

import json
from pathlib import Path

from training.eval_report import build_report, format_table
from training.records import Record


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_build_report_pairs_only_questions_both_models_scored(tmp_path):
    recs = [Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject="s") for i in range(3)]
    write(tmp_path / "bench" / "eval_medqa.jsonl", [r.to_dict() for r in recs])
    ev = tmp_path / "eval"
    for role in ("base", "tuned"):
        (ev / role).mkdir(parents=True)
        (ev / role / "meta.json").write_text(json.dumps({"seed": 1, "limit": 0}))
    write(ev / "base" / "letter__medqa.jsonl", [{"id": "q0", "letter": "B"},
                                                {"id": "q1", "letter": "A"}])
    write(ev / "tuned" / "letter__medqa.jsonl", [{"id": "q0", "letter": "A"},
                                                 {"id": "q1", "letter": "A"},
                                                 {"id": "q2", "letter": "A"}])
    write(ev / "base" / "reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A", "forced": i == 0, "closed": i != 0,
            "new_tokens": 10 * (i + 1)} for i in range(3)])
    write(ev / "tuned" / "reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A", "forced": False, "closed": True,
            "new_tokens": 5} for i in range(3)])

    report = build_report(ev, tmp_path / "bench")
    letter = report["stages"]["letter:medqa"]
    assert (letter["planned"], letter["scored"]) == (3, 2)
    assert (letter["base_accuracy"], letter["tuned_accuracy"]) == (0.5, 1.0)
    reasoning = report["stages"]["reasoning:medqa"]
    assert reasoning["base_counts"] == {"forced": 1, "closed": 2, "median_new_tokens": 20}
    assert set(report["pooled"]) == {"letter", "reasoning"}
    assert "letter:medmcqa" not in report["stages"]
    assert "letter:medqa" in format_table(report)


def test_a_stage_only_one_model_reached_is_listed_not_scored(tmp_path):
    recs = [Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject="s") for i in range(3)]
    write(tmp_path / "bench" / "eval_medqa.jsonl", [r.to_dict() for r in recs])
    ev = tmp_path / "eval"
    for role in ("base", "tuned"):
        (ev / role).mkdir(parents=True)
        (ev / role / "meta.json").write_text(json.dumps({"seed": 1, "limit": 0}))
    write(ev / "base" / "reasoning__medqa.jsonl", [{"id": "q0", "letter": "A"}])
    write(ev / "tuned" / "reasoning__medqa.jsonl", [])
    report = build_report(ev, tmp_path / "bench")
    assert "reasoning:medqa" not in report["stages"]
    assert report["unpaired"] == ["reasoning:medqa"]
    assert report["pooled"] == {}


def _bench(tmp_path, n=4):
    recs = [Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject="s") for i in range(n)]
    write(tmp_path / "bench" / "eval_medqa.jsonl", [r.to_dict() for r in recs])
    return tmp_path / "bench"


def test_read_rows_keeps_the_first_sample(tmp_path):
    from training.eval_report import read_rows, read_samples

    path = tmp_path / "r.jsonl"
    write(path, [{"id": "q0", "letter": "B", "sample": 1}, {"id": "q0", "letter": "A", "sample": 0},
                 {"id": "q1", "letter": "A"}])
    assert read_rows(path)["q0"]["letter"] == "A" and read_rows(path)["q1"]["letter"] == "A"
    assert [r["letter"] for r in read_samples(path)["q0"]] == ["A", "B"]


def test_build_comparison_pairs_each_model_with_the_baseline(tmp_path):
    from training.eval_report import build_comparison, export_rows, format_comparison

    bench = _bench(tmp_path)
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    # base scored over two sessions: letters in s1, reasoning in s2
    write(s1 / "base" / "letter__medqa.jsonl", [{"id": f"q{i}", "letter": "A"} for i in range(4)])
    write(s2 / "base" / "reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A" if i < 2 else "B", "sample": s, "think_tokens": 100 * (i + 1),
            "new_tokens": 100 * (i + 1) + 5, "forced": False, "closed": True}
           for s in (0, 1) for i in range(4)])
    write(s1 / "run3" / "letter__medqa.jsonl", [{"id": f"q{i}", "letter": "B"} for i in range(3)])
    write(s1 / "run3" / "reasoning__medqa.jsonl",
          [{"id": "q0", "letter": "B", "sample": 0, "think_tokens": 10},
           {"id": "q0", "letter": "A", "sample": 1, "think_tokens": 10},
           {"id": "q1", "letter": "A", "sample": 0, "think_tokens": 10},
           {"id": "q1", "letter": "A", "sample": 1, "think_tokens": 10}])
    models = {"base": [s1 / "base", s2 / "base"], "run3": [s1 / "run3"]}
    rep = build_comparison(models, bench)

    letter = rep["stages"]["letter:medqa"]
    assert letter["models"]["base"]["accuracy"] == 1.0 and letter["models"]["run3"]["n"] == 3
    assert letter["common"]["n"] == 3 and letter["common"]["accuracy"]["run3"] == 0.0
    assert letter["vs_baseline"]["run3"]["n"] == 3

    reasoning = rep["stages"]["reasoning:medqa"]
    base = reasoning["models"]["base"]
    assert base["accuracy"] == 0.5 and base["think_tokens"]["median"] == 200
    assert base["any_correct"] == {"samples": 2, "n": 4, "accuracy": 0.5,
                                   "first_sample_accuracy": 0.5}
    run3 = reasoning["models"]["run3"]
    # first sample 1/2 right; at least one of two samples right on both
    assert run3["accuracy"] == 0.5 and run3["any_correct"]["accuracy"] == 1.0
    cmp = reasoning["vs_baseline"]["run3"]
    assert cmp["n"] == 2 and (cmp["base_accuracy"], cmp["tuned_accuracy"]) == (1.0, 0.5)
    assert "reasoning:medqa" in format_comparison(rep)

    rows = list(export_rows(models, bench))
    assert len(rows) == 4 + 8 + 3 + 4
    one = next(r for r in rows if r["model"] == "run3" and r["mode"] == "reasoning")
    assert set(one) >= {"question_id", "gold_answer", "prediction", "thinking_length",
                        "correct", "model", "mode", "sample"}
    assert one["prediction"] == "B" and one["correct"] is False and one["thinking_length"] == 10


def test_a_stage_in_two_directories_of_one_model_stops(tmp_path):
    import pytest

    from training.eval_report import build_comparison

    bench = _bench(tmp_path)
    for d in ("a", "b"):
        write(tmp_path / d / "letter__medqa.jsonl", [{"id": "q0", "letter": "A"}])
    with pytest.raises(SystemExit):
        build_comparison({"base": [tmp_path / "a", tmp_path / "b"]}, bench)


def test_parse_models():
    import pytest

    from training.eval_report import parse_models

    assert parse_models(["base=x,y", "run3=z"]) == {"base": [Path("x"), Path("y")],
                                                    "run3": [Path("z")]}
    for bad in (["base"], ["base=x", "base=y"], ["=x"]):
        with pytest.raises(SystemExit):
            parse_models(bad)


def test_discover_models_groups_worker_directories_by_name(tmp_path):
    import pytest

    from training.eval_report import discover_models

    for d in ("s1/gpu0/base", "s1/gpu1/base", "s1/gpu1/run3", "s2/gpu1/run3_4bit"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "meta.json").write_text("{}")
    found = discover_models([tmp_path / "s1", tmp_path / "s2"])
    assert {k: [p.relative_to(tmp_path).as_posix() for p in v] for k, v in found.items()} == {
        "base": ["s1/gpu0/base", "s1/gpu1/base"], "run3": ["s1/gpu1/run3"],
        "run3_4bit": ["s2/gpu1/run3_4bit"]}
    with pytest.raises(SystemExit):
        discover_models([tmp_path / "nothing"])
