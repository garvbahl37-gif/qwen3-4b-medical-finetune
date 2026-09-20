from __future__ import annotations

from training.sources import (
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


def test_chatdoctor_joins_instruction_and_input_into_the_question():
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
