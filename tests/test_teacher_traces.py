from __future__ import annotations

import json

from training.records import Record
from training.self_traces import build_e2
from training.teacher_traces import choose_teacher, merge_sets, pilot_stats, weak_questions


def q(i: int, answer: str = "B") -> Record:
    return Record(id=f"medqatrain-{i}", source="medqa", kind="mcq", question=f"Q{i}?",
                  options={"A": "x", "B": "y", "C": "z"}, answer=answer, rationale=None,
                  response=None, subject=None)


def row(i, letter="B", sample=0, closed=True, forced=False, think="I reason.", tokens=40):
    text = (f"<think>\n{think}\n</think>\n\nAnswer: {letter}\n\nBecause." if closed
            else f"<think>\n{think}")
    return {"id": f"medqatrain-{i}", "letter": letter, "closed": closed, "forced": forced,
            "think_tokens": tokens, "sample": sample, "text": text}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_weak_questions_are_those_the_base_never_got_right():
    pool = {f"medqatrain-{i}": q(i) for i in range(4)}
    gen = {"medqatrain-0": [row(0)],                                   # right
           "medqatrain-1": [row(1, letter="A")],                       # wrong
           "medqatrain-2": [row(2, closed=False, forced=True)],        # never finished
           "medqatrain-3": [row(3, letter="A"), row(3, sample=1)]}     # right once
    weak, report = weak_questions(gen, pool)
    assert [r.id for r in weak] == ["medqatrain-1", "medqatrain-2"]
    assert report == {"answered": 4, "weak": 2, "by_source": {"medqa": 2}}


def test_pilot_picks_the_teacher_that_yields_most_verified_traces_per_hour(tmp_path):
    weak = {f"medqatrain-{i}": q(i) for i in range(10)}
    for name, right, seconds in (("t8b", 3, 600), ("t14b", 6, 1500), ("t30b", 6, 900)):
        d = tmp_path / "gpu0" / name
        write(d / "reasoning__weak.jsonl",
              [row(i, letter="B" if i < right else "A") for i in range(10)])
        (d / "meta.json").write_text(json.dumps(
            {"stages": {"reasoning:weak": {"scored": 10, "seconds": seconds}}}))
    stats = pilot_stats([tmp_path], weak)
    assert stats["t14b"] == {"answered": 10, "rescued": 6, "rescue_rate": 0.6,
                             "seconds": 1500, "rescued_per_hour": 14.4}
    assert choose_teacher(stats) == "t30b"          # same rescue rate as 14B, faster
    assert choose_teacher(stats, min_rescue=0.7) is None
    assert choose_teacher({}) is None


def test_teacher_traces_keep_up_to_two_distinct_correct_samples():
    gen = {"medqatrain-0": [row(0, sample=s, think=f"path {s}") for s in range(3)],
           "medqatrain-1": [row(1, letter="A", sample=s) for s in range(3)]}
    pool = {f"medqatrain-{i}": q(i) for i in range(2)}
    recs, report = build_e2(gen, pool, measure=lambda r: 100, max_think=3500, cap=4096,
                            direct_fraction=0.0, seed=1, max_per_question=2,
                            prefix="e3t", source="e3_teacher")
    assert len(recs) == 2 and {r.source for r in recs} == {"e3_teacher"}
    assert len({r.reasoning for r in recs}) == 2
    assert sorted(r.id for r in recs) == ["e3t-medqatrain-0-0", "e3t-medqatrain-0-1"]
    assert report["questions_with_a_kept_trace"] == 1


def test_merge_sets_concatenates_and_takes_the_longest_max_seq(tmp_path):
    for name, n, max_seq in (("e2", 3, 3968), ("e3t", 2, 4032)):
        d = tmp_path / name
        write(d / "train.jsonl", [q(i).to_dict() | {"id": f"{name}-{i}"} for i in range(n)])
        (d / "e2_report.json").write_text(json.dumps({"max_seq": max_seq, "rows": n}))
    report = merge_sets([tmp_path / "e2", tmp_path / "e3t"], tmp_path / "e3", seed=0)
    rows = (tmp_path / "e3" / "train.jsonl").read_text().splitlines()
    assert len(rows) == 5 and report["max_seq"] == 4032 and report["rows"] == 5
    assert report["from"] == {"e2": 3, "e3t": 2}
    assert json.loads((tmp_path / "e3" / "e2_report.json").read_text())["max_seq"] == 4032
