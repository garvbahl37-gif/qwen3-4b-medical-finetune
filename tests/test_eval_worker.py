from __future__ import annotations

import pytest

from training.eval_worker import STAGES, fits_before, parse_stages, pick_letter, stage_order
from training.records import Record


def recs(n: int) -> list[Record]:
    return [Record(id=f"q{i}", source="s", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject=None) for i in range(n)]


def test_letter_stages_keep_every_question_in_file_order():
    assert [r.id for r in stage_order(recs(5), "letter", "medqa", 1)] == [f"q{i}" for i in range(5)]


def test_reasoning_order_is_a_seeded_shuffle_cut_to_size():
    a = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    b = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    assert [r.id for r in a] == [r.id for r in b] and len(a) == 1000
    assert [r.id for r in a] != [f"q{i}" for i in range(1000)]
    assert len(stage_order(recs(1273), "reasoning", "medqa", 7)) == 1273


def test_fits_before():
    assert fits_before(100.0, 50.0, 50.0) and not fits_before(100.0, 50.0, 51.0)


def test_parse_stages():
    assert parse_stages("letter:medqa,reasoning:pubmedqa") == [("letter", "medqa"),
                                                               ("reasoning", "pubmedqa")]
    assert parse_stages("") == list(STAGES)
    with pytest.raises(SystemExit):
        parse_stages("letter:nope")


def test_pick_letter_restricts_to_the_questions_options_and_breaks_ties_early():
    row = {10: 1.0, 11: 5.0, 12: 5.0, 13: 9.0}
    ids = {"A": 10, "B": 11, "C": 12, "D": 13}
    assert pick_letter(row, ids, ["A", "B", "C"]) == "B"
    assert pick_letter(row, ids, ["A", "B", "C", "D"]) == "D"
    with pytest.raises(FloatingPointError):
        pick_letter({10: float("nan"), 11: 0.0}, ids, ["A", "B"])
