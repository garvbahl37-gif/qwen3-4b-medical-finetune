from __future__ import annotations

import pytest

from training.records import Record
from training.sources import (
    load,
    normalise_chatdoctor,
    normalise_medical_o1,
    normalise_medmcqa,
    normalise_medqa,
)

GOOD_MEDMCQA = {
    "id": "x1",
    "question": "Chronic urethral obstruction leads to which change?",
    "opa": "Hyperplasia", "opb": "Hypertrophy", "opc": "Atrophy", "opd": "Dysplasia",
    "cop": 2,
    "choice_type": "single",
    "exp": "Chronic obstruction causes hydronephrosis, which over time thins and "
           "atrophies the renal parenchyma through sustained back pressure.",
    "subject_name": "Anatomy",
}


def test_medmcqa_maps_cop_index_to_the_right_letter():
    rec = normalise_medmcqa(GOOD_MEDMCQA, 0)
    assert rec is not None
    assert rec.answer == "C"           # cop=2 -> opc -> "Atrophy"
    assert rec.options["C"] == "Atrophy"
    assert rec.kind == "mcq"
    assert rec.subject == "Anatomy"


def test_medmcqa_drops_multi_choice_rows():
    raw = dict(GOOD_MEDMCQA, choice_type="multi")
    assert normalise_medmcqa(raw, 0) is None


def test_medmcqa_drops_rows_with_no_usable_explanation():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp=None), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp="   "), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp="Correct."), 0) is None


def test_medmcqa_keeps_unexplained_rows_when_the_rationale_is_not_required():
    # Evaluation only needs question, options and answer. Filtering the
    # benchmark by explanation-availability would change what it measures.
    raw = dict(GOOD_MEDMCQA, exp=None)
    assert normalise_medmcqa(raw, 0) is None
    rec = normalise_medmcqa(raw, 0, require_rationale=False)
    assert rec is not None and rec.answer == "C" and rec.rationale == ""


def test_medmcqa_drops_rows_with_an_out_of_range_answer_index():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, cop=-1), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, cop=9), 0) is None


def test_medmcqa_drops_rows_with_a_blank_option():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, opb=""), 0) is None


def test_medmcqa_keeps_multi_choice_rows_when_single_choice_is_not_required():
    # Training drops these as noisy. Evaluation must keep them: scoring a
    # filtered subset of MedMCQA validation would not be comparable to any
    # published MedMCQA number.
    raw = dict(GOOD_MEDMCQA, choice_type="multi")
    assert normalise_medmcqa(raw, 0) is None
    rec = normalise_medmcqa(raw, 0, require_single_choice=False)
    assert rec is not None and rec.answer == "C"


def test_medqa_uses_its_own_answer_idx():
    raw = {
        "question": "A 23-year-old pregnant woman presents with dysuria.",
        "options": {"A": "Ampicillin", "B": "Ceftriaxone",
                    "C": "Doxycycline", "D": "Nitrofurantoin"},
        "answer": "Nitrofurantoin",
        "answer_idx": "D",
        "meta_info": "step2&3",
    }
    rec = normalise_medqa(raw, 0)
    assert rec is not None and rec.answer == "D"
    assert rec.options["D"] == "Nitrofurantoin"
    assert rec.rationale is None


def test_medqa_drops_rows_that_are_not_four_option():
    raw = {"question": "q", "options": {"A": "a", "B": "b"},
           "answer": "a", "answer_idx": "A", "meta_info": ""}
    assert normalise_medqa(raw, 0) is None


def test_medical_o1_keeps_the_chain_of_thought_as_the_rationale():
    raw = {
        "Question": "What cardiac abnormality explains a paradoxical embolism?",
        "Complex_CoT": "Sudden limb weakness suggests stroke; a swollen calf "
                       "suggests DVT; together they suggest a right-to-left shunt.",
        "Response": "A patent foramen ovale.",
    }
    rec = normalise_medical_o1(raw, 0)
    assert rec is not None and rec.kind == "dialogue"
    assert rec.rationale is not None
    assert "patent foramen ovale" in rec.response


def test_chatdoctor_builds_question_from_patient_input_only():
    raw = {
        "instruction": "If you are a doctor, answer based on the description.",
        "input": "I woke up feeling the room spinning and felt nauseous.",
        "output": "This sounds like benign paroxysmal positional vertigo.",
    }
    rec = normalise_chatdoctor(raw, 0)
    assert rec is not None and rec.kind == "dialogue"
    assert "room spinning" in rec.question
    assert rec.response.startswith("This sounds like")


def test_chatdoctor_drops_empty_turns():
    assert normalise_chatdoctor(
        {"instruction": "x", "input": "", "output": "y"}, 0) is None
    assert normalise_chatdoctor(
        {"instruction": "x", "input": "y", "output": "  "}, 0) is None


def _medmcqa_row(i: int, *, usable: bool = True) -> dict:
    return {
        "id": f"r{i}",
        "question": f"Question number {i}?",
        "opa": "one", "opb": "two", "opc": "three", "opd": "four",
        "cop": i % 4,
        "choice_type": "single" if usable else "multi",
        "exp": "e" * 100,
        "subject_name": "Anatomy",
    }


def _fetch(rows):
    """Stand in for datasets.load_dataset so the sampling tests stay offline."""
    return lambda hf_id, config, split: rows


def test_load_returns_exactly_the_limit_when_enough_rows_survive():
    rows = [_medmcqa_row(i) for i in range(200)]
    got = load("medmcqa", limit=10, fetch=_fetch(rows))
    assert len(got) == 10
    assert all(isinstance(r, Record) for r in got)


def test_load_stops_early_instead_of_scanning_the_whole_source():
    rows = [_medmcqa_row(i) for i in range(500)]
    assert len(load("medmcqa", limit=5, fetch=_fetch(rows))) == 5


def test_load_raises_rather_than_silently_returning_short():
    # Only 8 of 20 survive the filters. Asking for 15 must fail loudly: a
    # training mix that comes up short reports a ratio it does not have.
    rows = [_medmcqa_row(i, usable=i < 8) for i in range(20)]
    with pytest.raises(SystemExit) as excinfo:
        load("medmcqa", limit=15, fetch=_fetch(rows))
    message = str(excinfo.value)
    assert "8" in message
    assert "--medmcqa" in message


def test_load_with_limit_zero_returns_every_surviving_row():
    rows = [_medmcqa_row(i, usable=i % 2 == 0) for i in range(20)]
    assert len(load("medmcqa", limit=0, fetch=_fetch(rows))) == 10


def test_load_is_deterministic_for_a_seed_and_varies_across_seeds():
    rows = [_medmcqa_row(i) for i in range(200)]
    first = [r.id for r in load("medmcqa", limit=10, fetch=_fetch(rows))]
    again = [r.id for r in load("medmcqa", limit=10, fetch=_fetch(rows))]
    other = [r.id for r in load("medmcqa", limit=10, seed=7, fetch=_fetch(rows))]
    assert first == again
    assert first != other


def test_load_threads_require_rationale_through_to_the_normaliser():
    rows = [dict(_medmcqa_row(i), exp="") for i in range(20)]
    with pytest.raises(SystemExit):
        load("medmcqa", limit=5, fetch=_fetch(rows))
    assert len(load("medmcqa", limit=5, require_rationale=False,
                    fetch=_fetch(rows))) == 5


def test_load_threads_require_single_choice_through_to_the_normaliser():
    rows = [dict(_medmcqa_row(i), choice_type="multi") for i in range(20)]
    with pytest.raises(SystemExit):
        load("medmcqa", limit=5, fetch=_fetch(rows))
    assert len(load("medmcqa", limit=5, require_single_choice=False,
                    fetch=_fetch(rows))) == 5
