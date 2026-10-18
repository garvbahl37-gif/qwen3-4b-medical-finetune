from __future__ import annotations

from training.records import Record
from training.self_traces import build_e2, split_completion


def q(i: int) -> Record:
    return Record(id=f"medqatrain-{i}", source="medqa", kind="mcq", question=f"Q{i}?",
                  options={"A": "Aspirin", "B": "Heparin", "C": "Warfarin", "D": "None"},
                  answer="B", rationale=None, response=None, subject="step1")


def row(i, sample, *, letter="B", closed=True, forced=False, think="I think it is heparin.",
        final="Answer: B\n\nHeparin acts at once. It is given IV.", think_tokens=50):
    text = f"<think>\n{think}\n</think>\n\n{final}" if closed else f"<think>\n{think}"
    return {"id": f"medqatrain-{i}", "letter": letter, "closed": closed, "forced": forced,
            "think_tokens": think_tokens, "new_tokens": think_tokens + 10, "sample": sample,
            "text": text}


def test_split_completion_separates_thinking_from_the_answer():
    thinking, final = split_completion("<think>\nstep one\nstep two\n</think>\n\nAnswer: B\n\nWhy.")
    assert thinking == "step one\nstep two" and final == "Answer: B\n\nWhy."
    assert split_completion("<think>\nnever closed") == (None, None)


def test_only_finished_correct_unforced_traces_are_kept():
    rows = {"medqatrain-0": [row(0, 0), row(0, 1, letter="C")],
            "medqatrain-1": [row(1, 0, closed=False, forced=True), row(1, 1, letter="A")],
            "medqatrain-2": [row(2, 0, think="<think> nested </think> twice")],
            "medqatrain-3": [row(3, 0, think_tokens=5000)]}
    recs, report = build_e2(rows, {f"medqatrain-{i}": q(i) for i in range(4)},
                            measure=lambda r: 100, max_think=3500, cap=4096,
                            direct_fraction=0.0, seed=1)
    assert [r.id for r in recs] == ["e2-medqatrain-0"]
    kept = recs[0]
    assert kept.reasoning == "I think it is heparin." and kept.answer == "B"
    assert kept.rationale == "Heparin acts at once. It is given IV."
    assert report["rejected"] == {"wrong_letter": 2, "unfinished_or_forced": 1,
                                  "nested_think": 1, "thinking_too_long": 1}
    assert report["questions_with_a_kept_trace"] == 1
    assert report["pass_rate_histogram"] == {"0/1": 2, "0/2": 1, "1/2": 1}


def test_direct_rows_make_up_the_requested_share_and_lengths_are_capped():
    rows = {f"medqatrain-{i}": [row(i, 0)] for i in range(8)}
    pool = {f"medqatrain-{i}": q(i) for i in range(8)}
    recs, report = build_e2(rows, pool, measure=lambda r: 100, max_think=3500, cap=4096,
                            direct_fraction=0.2, seed=1)
    direct = [r for r in recs if r.reasoning is None]
    assert len(direct) == 2 and len(recs) == 10
    assert all(r.id.startswith("e2d-") and r.answer == "B" for r in direct)
    recs, report = build_e2(rows, pool, measure=lambda r: 5000 if r.reasoning else 100,
                            max_think=3500, cap=4096, direct_fraction=0.2, seed=1)
    assert all(r.reasoning is None for r in recs) and report["dropped_over_cap"] == 8


def test_choice_among_correct_samples_is_seeded_not_shortest():
    rows = {"medqatrain-0": [row(0, s, think=f"trace {s}", think_tokens=10 * (s + 1))
                             for s in range(4)]}
    pool = {"medqatrain-0": q(0)}
    picks = {build_e2(rows, pool, measure=lambda r: 100, max_think=3500, cap=4096,
                      direct_fraction=0.0, seed=s)[0][0].reasoning for s in range(12)}
    assert len(picks) > 1
