from __future__ import annotations

from training.dataset_audit import (answer_agreement, audit, base_think_text,
                                    degenerate_repetition, malformed_reasons,
                                    near_duplicate_pairs, render_report, render_samples,
                                    revised_answer,
                                    style_features, unterminated)
from training.records import Record


def mcq(**kw) -> Record:
    fields = dict(id="q1", source="ultramedical", kind="mcq", question="Which drug is given?",
                  options={"A": "Aspirin", "B": "Heparin", "C": "Warfarin"}, answer="B",
                  rationale=None, response="Answer: B", subject=None, reasoning=None)
    fields.update(kw)
    return Record(**fields)


def chat(**kw) -> Record:
    fields = dict(id="c1", source="medical_o1", kind="dialogue", question="Why is the sky blue?",
                  options=None, answer=None, rationale=None, response="Because.",
                  subject=None, reasoning="Let me think. Rayleigh scattering.")
    fields.update(kw)
    return Record(**fields)


def test_answer_agreement_reads_the_traces_stated_letter():
    assert answer_agreement(mcq(reasoning="... so the answer is B.")) == "agree"
    assert answer_agreement(mcq(reasoning="... so the answer is C.")) == "conflict"
    assert answer_agreement(mcq(reasoning="the answer is A. Wait, the answer is B.")) == "agree"
    assert revised_answer(mcq(reasoning="the answer is A. Wait, the answer is B."))
    assert not revised_answer(mcq(reasoning="the answer is B. Yes, the answer is B."))


def test_answer_agreement_falls_back_to_the_option_the_conclusion_names():
    assert answer_agreement(mcq(reasoning="Long thought. Heparin acts fastest here.")) == "agree_text"
    assert answer_agreement(mcq(reasoning="Aspirin or heparin, hard to say.")) == "unstated"
    assert answer_agreement(mcq(reasoning="Nothing named.")) == "unstated"
    # an option discussed before the conclusion does not count
    assert answer_agreement(mcq(reasoning="Warfarin is too slow.\n\nConclusion: give heparin.")) == "agree_text"
    assert answer_agreement(mcq(reasoning="Heparin is wrong.\n\nConclusion: warfarin.")) == "other_text"


def test_named_option_matches_stems_and_prefers_the_most_specific():
    from training.dataset_audit import named_option

    opts = {"A": "Placenta previa", "B": "Placenta abruption", "C": "Vasa previa"}
    assert named_option("Conclusion: the diagnosis is placental abruption.", opts) == "B"
    opts = {"A": "Stenosis", "B": "Mitral stenosis"}
    assert named_option("Conclusion: mitral valve stenosis.", opts) == "B"


def test_answer_agreement_is_na_without_reasoning_or_gold():
    assert answer_agreement(mcq()) == "n/a"
    assert answer_agreement(chat()) == "n/a"


def test_lowercase_answer_is_not_read_as_a_letter():
    # 'the answer is a combination' must not become A.
    assert answer_agreement(mcq(reasoning="the answer is a combination of heparin")) == "agree_text"


def test_degenerate_repetition_needs_a_long_line_three_times():
    line = "The patient has a fever and a cough today."
    assert degenerate_repetition("\n".join([line] * 3))
    assert not degenerate_repetition("\n".join([line] * 2))
    assert not degenerate_repetition("\n".join(["short"] * 10))


def test_unterminated():
    assert unterminated("and then the")
    assert not unterminated("It is heparin.")
    assert not unterminated("")


def test_style_features():
    s = style_features("Okay, let me see. Wait, actually that is wrong.")
    assert s["first_person_open"] and not s["markdown"] and s["self_checks_per_1k_words"] > 0
    assert style_features("### Finding reasoning paths:\n1. x")["kg_template"]
    assert style_features("## Step 1\nText")["markdown"]


def test_malformed_reasons():
    assert malformed_reasons(mcq(answer="D")) == ["answer not among options"]
    assert malformed_reasons(mcq(options={"A": "x", "B": "x"}, answer="A")) == ["duplicate option text"]
    assert malformed_reasons(chat(response="  ")) == ["empty response"]
    assert malformed_reasons(mcq()) == []


def test_near_duplicate_pairs_finds_reworded_copies_only():
    base = "a 45 year old man presents with crushing chest pain radiating to the left arm"
    texts = [base, base + " today", "a child with a rash on both legs after a sore throat"]
    pairs = near_duplicate_pairs(texts, n=3, threshold=0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_base_think_text_takes_the_thinking_part():
    assert base_think_text("<think>\nabc\n</think>\n\nAnswer: B") == "abc"
    assert base_think_text("<think>\nunfinished") == "unfinished"


def _result():
    recs = [mcq(id="u1", reasoning="Okay. So the answer is B."),
            mcq(id="u2", reasoning="So the answer is C."),
            mcq(id="m1", source="medqa", response="Answer: B"),
            mcq(id="m2", source="medqa", question="Which drug is given?"),
            chat(id="c1"),
            mcq(id="r1", source="medreason", reasoning="Finding reasoning paths: the answer is B."),
            mcq(id="x1", source="reasonmed", reasoning="<think>t</think> So the answer is B.")]
    return recs, audit(recs, token_len=lambda r: 100 if r.id != "u2" else 5000,
                       reasoning_len=lambda t: len(t.split()), max_seq=2496,
                       base_traces=["Okay, let me think. Wait, hmm."])


def test_audit_counts_per_source():
    _, res = _result()
    u, m = res["sources"]["ultramedical"], res["sources"]["medqa"]
    assert (u["rows"], u["reasoning_rows"], u["answer_verified"]) == (2, 2, 1)
    assert u["suspected_bad"] == 1 and u["truncated_at_max_seq"] == 1
    assert m["answer_verified"] == 2 and m["reasoning_rows"] == 0
    # every row shares one question text: all are exact duplicates
    assert u["exact_duplicate_questions"] == 2 and m["exact_duplicate_questions"] == 2
    assert "medreason:mcq" in res["sources"] and res["sources"]["medical_o1"]["answer_verified"] == 0
    assert res["qwen3_base_traces"]["traces"] == 1
    x = res["sources"]["reasonmed"]
    # no independent gold: agreement verifies nothing
    assert x["answer_verified"] == 0 and x["suspicious"]["think tags inside the trace"] == 1


def test_report_has_the_required_table_and_samples_render():
    recs, res = _result()
    md = render_report(res, {"sources": {}, "cap": 3072, "dropped_over_max_seq": 0})
    assert "| source | rows | reasoning_rows | answer_verified | suspected_bad | duplicates | truncated |" in md
    assert "| ultramedical | 2 | 2 | 1 | 1 |" in md
    samples = render_samples(recs, per_source=1)
    assert "## ultramedical" in samples and "**Gold:** B" in samples
