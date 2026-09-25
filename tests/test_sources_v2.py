from __future__ import annotations

from training.sources_v2 import (is_english, match_option, norm_medical_o1_direct,
                                 norm_medical_o1_reasoning, norm_medreason, norm_mmlu,
                                 norm_pubmedqa, norm_r1_distill, norm_reasonmed,
                                 norm_ultramedical, parse_option_lines,
                                 split_stem_and_options, stated_letter)

# Real rows, shortened (sampled 2026-09-25).
MEDREASON_MCQ = {
    "dataset_name": "medmcqa",
    "question": "Urogenital Diaphragm is made up of the following, except:",
    "options": "Answer Choices:\nA. Deep transverse Perineus\nB. Perinial membrane\n"
               "C. Colle's fascia\nD. Sphincter Urethrae",
    "answer": "Colle's fascia. Explanation: Colle's fascia does not contribute. "
              "It is superficial.",
    "reasoning": "Finding reasoning paths: ...\nConclusion: Colle's fascia.",
}
ULTRA = {
    "id": "u1", "type": "Exam", "answer": "D",
    "conversations": [
        {"from": "human", "value": "A pregnant woman has dysuria. Which is the best "
                                   "treatment?\n\nA. Ampicillin\nB. Ceftriaxone\n"
                                   "C. Doxycycline\nD. Nitrofurantoin"},
        {"from": "gpt", "value": "Nitrofurantoin is safe in pregnancy.\n\n"
                                 "So, the answer is D."},
    ],
}
REASONMED = {
    "instruction": "Please answer the following multiple-choice question:\n"
                   "The most likely poisoning is -?\nA. Mercury\nB. Lead\n"
                   "C. Arsenic\nD. Phosphorus",
    "input": "",
    "output": "Salivation and gum lines suggest a heavy metal. The pattern "
              "points to mercury exposure.",
}
PUBMED = {
    "pubid": 21645374, "question": "Do mitochondria play a role?",
    "final_decision": "yes", "long_answer": "Results depict mitochondrial dynamics.",
    "context": {"contexts": ["Background one.", "Methods two."]},
}


def test_parse_option_lines_needs_a_run_from_a():
    assert parse_option_lines(["Answer Choices:", "A. one", "B. two"]) == {"A": "one", "B": "two"}
    assert parse_option_lines(["A.: 12 mm", "B.: 24 mm"]) == {"A": "12 mm", "B": "24 mm"}
    assert parse_option_lines(["(A) x", "(B) y", "(C) z"]) == {"A": "x", "B": "y", "C": "z"}
    assert parse_option_lines(["B. two", "C. three"]) == {}
    assert parse_option_lines(["A. one", "B. two", "D. four"]) == {}
    assert parse_option_lines(["A. only"]) == {}


def test_split_stem_and_options_reads_trailing_option_lines():
    stem, opts = split_stem_and_options(
        "A 3-year-old girl has a rash.\nWhich drug?\n\nA. Ampicillin\nB. Ceftriaxone\n")
    assert stem == "A 3-year-old girl has a rash.\nWhich drug?"
    assert opts == {"A": "Ampicillin", "B": "Ceftriaxone"}
    assert split_stem_and_options("No options here.") == ("No options here.", {})


def test_match_option_by_prefix_decision_and_longest():
    opts = {"A": "Deep transverse Perineus", "B": "Perinial membrane",
            "C": "Colle's fascia", "D": "Sphincter Urethrae"}
    assert match_option("Colle's fascia. Explanation: it is superficial.", opts) == "C"
    assert match_option("Small cell carcinoma",
                        {"A": "Small cell carcinoma", "B": "Squamous"}) == "A"
    assert match_option("The final decision is: yes. Because...",
                        {"A": "Yes", "B": "No"}) == "A"
    assert match_option("Lead acetate poisoning",
                        {"A": "Lead", "B": "Lead acetate"}) == "B"
    assert match_option("Something else", opts) is None


def test_stated_letter_takes_the_last_and_ignores_words():
    assert stated_letter("maybe the answer is A... no, the answer is C.") == "C"
    assert stated_letter("So, the answer is A.") == "A"
    assert stated_letter("the answer is Aspirin") is None


def test_is_english():
    assert is_english("A patient with fever.")
    assert not is_english("患者发热三天")


def test_medreason_mcq_maps_answer_text_to_its_letter():
    rec = norm_medreason(MEDREASON_MCQ, 7)
    assert (rec.kind, rec.answer, rec.source, rec.subject) == ("mcq", "C", "medreason", "medmcqa")
    assert rec.rationale.startswith("Colle's fascia does not contribute")
    assert rec.reasoning.startswith("Finding reasoning paths")


def test_medreason_keeps_huatuo_as_free_text_and_drops_benchmark_subsets():
    free = norm_medreason({**MEDREASON_MCQ, "dataset_name": "huatuo", "options": "",
                           "answer": "Normal residual volume."}, 1)
    assert free.kind == "dialogue" and free.response == "Normal residual volume."
    assert free.reasoning
    for name in ("MMLU", "pubmedqa", "pubmedqa_artificial", "pubmedqa_unlabeled",
                 "LastHumanity"):
        assert norm_medreason({**MEDREASON_MCQ, "dataset_name": name}, 1) is None


def test_medreason_drops_rows_whose_answer_names_no_single_option():
    assert norm_medreason({**MEDREASON_MCQ, "answer": "None of these"}, 1) is None


def test_ultramedical_keeps_rows_whose_explanation_agrees_with_gold():
    rec = norm_ultramedical(ULTRA, 0)
    assert rec.answer == "D" and rec.options["D"] == "Nitrofurantoin"
    assert rec.reasoning.endswith("the answer is D.")
    assert rec.question == "A pregnant woman has dysuria. Which is the best treatment?"
    assert norm_ultramedical({**ULTRA, "answer": "B"}, 0) is None


def test_r1_distill_and_medical_o1_map_reasoning_and_reply():
    r = norm_r1_distill({"question": "Which drug reverses heparin?",
                         "reasoning (reasoning_content)": "think",
                         "response (content)": "reply"}, 3)
    assert (r.kind, r.reasoning, r.response, r.source) == ("dialogue", "think", "reply", "r1_distill")
    raw = {"Question": "Q?", "Complex_CoT": "cot", "Response": "resp"}
    o = norm_medical_o1_reasoning(raw, 4)
    assert (o.reasoning, o.response) == ("cot", "resp")
    d = norm_medical_o1_direct(raw, 4)
    assert d.reasoning is None and d.source == "medical_o1_direct" and d.id != o.id


def test_reasonmed_reads_the_option_its_conclusion_names():
    rec = norm_reasonmed(REASONMED, 0)
    assert rec.question == "The most likely poisoning is -?" and rec.answer == "A"
    assert norm_reasonmed({**REASONMED, "output": "Mercury or lead could both fit."}, 0) is None
    assert norm_reasonmed({**REASONMED, "output": "Long reasoning... the answer is B."}, 0).answer == "B"


def test_pubmedqa_includes_the_abstract_and_three_options():
    rec = norm_pubmedqa(PUBMED, 0)
    assert rec.question == ("Context: Background one. Methods two.\n\n"
                            "Question: Do mitochondria play a role?")
    assert rec.options == {"A": "yes", "B": "no", "C": "maybe"} and rec.answer == "A"
    assert rec.id == "pubmedqa-21645374"
    assert norm_pubmedqa(PUBMED, 0, source="pubmedqa_artificial").source == "pubmedqa_artificial"
    assert norm_pubmedqa({**PUBMED, "final_decision": "unsure"}, 0) is None


def test_mmlu_maps_choices_to_letters():
    rec = norm_mmlu({"question": "Q?", "subject": "anatomy",
                     "choices": ["a", "b", "c", "d"], "answer": 2}, 5)
    assert rec.answer == "C" and rec.options["C"] == "c"
    assert (rec.subject, rec.source, rec.id) == ("anatomy", "mmlu_medical", "mmlu-anatomy-5")
    assert norm_mmlu({"question": "Q?", "subject": "anatomy",
                      "choices": ["a", "b"], "answer": 0}, 5) is None


def test_r1_distill_keeps_medical_rows_only():
    row = {"question": "Calculate the net present value of an investment at 10%.",
           "reasoning (reasoning_content)": "think", "response (content)": "NPV is 1,372."}
    assert norm_r1_distill(row, 1) is None
    clinical = dict(row, question="A patient on warfarin has a raised INR. Next step?",
                    **{"response (content)": "Hold warfarin and give vitamin K."})
    assert norm_r1_distill(clinical, 2) is not None
