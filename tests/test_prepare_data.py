from __future__ import annotations

from training.prepare_data import decontaminate, normalise_question
from training.records import Record


def mk(rid: str, question: str) -> Record:
    return Record(id=rid, source="medmcqa", kind="mcq", question=question,
                  options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer="A",
                  rationale="x" * 100, response=None, subject=None)


def test_normalise_question_ignores_case_punctuation_and_spacing():
    a = normalise_question("Which  vitamin, deficiency causes SCURVY?")
    b = normalise_question("which vitamin deficiency causes scurvy")
    assert a == b


def test_normalise_question_still_separates_genuinely_different_questions():
    assert normalise_question("causes of scurvy") != normalise_question("causes of rickets")


def test_decontaminate_removes_training_rows_that_appear_in_a_holdout():
    train = [mk("1", "Causes of scurvy?"), mk("2", "Causes of rickets?")]
    holdout = [mk("h", "causes of scurvy")]
    kept, removed = decontaminate(train, holdout)
    assert removed == 1
    assert [r.id for r in kept] == ["2"]


def test_decontaminate_is_a_no_op_when_nothing_overlaps():
    train = [mk("1", "a question"), mk("2", "another question")]
    kept, removed = decontaminate(train, [mk("h", "unrelated")])
    assert removed == 0 and len(kept) == 2


def test_decontaminate_removes_duplicates_within_the_training_set_too():
    train = [mk("1", "Causes of scurvy?"), mk("2", "causes of  scurvy")]
    kept, removed = decontaminate(train, [])
    assert len(kept) == 1 and removed == 1
