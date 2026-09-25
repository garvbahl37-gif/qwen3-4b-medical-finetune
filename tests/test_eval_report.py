from __future__ import annotations

import json

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
