from __future__ import annotations

import pytest

from training.prompts import build_messages, extract_letter, target_text
from training.records import Record

MCQ = Record(
    id="m1", source="medmcqa", kind="mcq",
    question="Which vitamin deficiency causes scurvy?",
    options={"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"},
    answer="B", rationale="Scurvy is caused by a lack of ascorbic acid.",
    response=None, subject="Medicine",
)
CHAT = Record(
    id="c1", source="chatdoctor", kind="dialogue",
    question="I have had a sore throat for three days. What should I do?",
    options=None, answer=None, rationale=None,
    response="A sore throat lasting three days is usually viral.", subject=None,
)


def test_mcq_prompt_lists_every_option_and_hides_the_answer():
    msgs = build_messages(MCQ, with_answer=False)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    for letter, text in MCQ.options.items():
        assert f"{letter}. {text}" in user
    assert "Scurvy is caused" not in user


def test_mcq_target_ends_with_a_parseable_answer_line():
    target = target_text(MCQ)
    assert target.rstrip().endswith("Answer: B")
    assert MCQ.rationale in target


def test_with_answer_appends_the_assistant_turn():
    msgs = build_messages(MCQ, with_answer=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == target_text(MCQ)


def test_dialogue_uses_the_chat_system_prompt_not_the_mcq_one():
    mcq_sys = build_messages(MCQ, with_answer=False)[0]["content"]
    chat_sys = build_messages(CHAT, with_answer=False)[0]["content"]
    assert mcq_sys != chat_sys
    assert target_text(CHAT) == CHAT.response


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C", "C"),
        ("**Answer:** D", "D"),
        ("answer - a", "A"),
        ("The reasoning is long.\n\nAnswer: B", "B"),
        ("Answer: B\nActually, Answer: D", "D"),
        ("(C)", "C"),
        ("C.", "C"),
        ("C) Vitamin D", "C"),
        ("```\nAnswer: A\n```", "A"),
        ("I cannot determine this.", None),
        ("", None),
        ("A 45-year-old man presents, most consistent with choice C.", "C"),
        ("A 23-year-old woman presents. The answer is B.", "B"),
        ("The correct option is D.", "D"),
        ("Reasoning about Vitamin C and Hepatitis A.\n\nAnswer: B", "B"),
    ],
)
def test_extract_letter_is_lenient_about_format(text, expected):
    assert extract_letter(text) == expected


def test_extract_letter_ignores_letters_inside_words():
    assert extract_letter("Diabetes And Cancer") is None


def test_extract_letter_prefers_an_explicit_answer_line_over_a_stray_letter():
    assert extract_letter("A patient presents.\nAnswer: D") == "D"


@pytest.mark.parametrize(
    "text",
    [
        "Vitamin D deficiency is the most likely cause given the presentation.",
        "This is likely due to Hepatitis B infection based on the serology.",
        "The patient has blood group A and is Rh negative.",
        "I am uncertain, but this could relate to A or B depending on labs.",
        "A 45-year-old man presents with acute chest pain radiating to the jaw.",
    ],
)
def test_extract_letter_refuses_to_invent_an_answer_from_clinical_prose(text):
    # A false positive is worse than None: None scores wrong for base and
    # tuned alike, while a guessed letter is indistinguishable from a real
    # answer in the aggregate accuracy the project reports.
    assert extract_letter(text) is None


def test_record_survives_a_dict_round_trip():
    assert Record.from_dict(MCQ.to_dict()) == MCQ
    assert MCQ.to_dict()["options"] == {"A": "Vitamin A", "B": "Vitamin C",
                                        "C": "Vitamin D", "D": "Vitamin K"}
