from __future__ import annotations

from training.decontam import Deduper, EvalIndex, ngrams

VIGNETTE = ("a 45 year old man presents with crushing chest pain radiating to the "
            "left arm and jaw for two hours")


def test_exact_match_after_normalisation():
    index = EvalIndex(["What is the most likely diagnosis?"])
    assert index.hits("what is the MOST likely diagnosis") == "exact"
    assert index.hits("A different question entirely") is None


def test_thirteen_word_overlap_is_caught_but_short_text_needs_an_exact_match():
    index = EvalIndex([VIGNETTE])
    assert index.hits("Case: " + VIGNETTE + " What next?") == "ngram"
    assert index.hits("crushing chest pain radiating to the left arm") is None


def test_ngrams_of_short_text_is_empty():
    assert ngrams("one two three", 13) == set()


def test_deduper_flags_the_second_occurrence():
    d = Deduper()
    assert d.first_time("Q one?") and not d.first_time("q ONE") and d.first_time("Q two?")


BOILER = "which of the following is the most likely cause of this patient's symptoms"


def test_a_shared_stock_phrase_is_not_contamination():
    index = EvalIndex(["A 30-year-old woman has a rash on her arms after hiking in the "
                       f"woods for a week. {BOILER}?"])
    assert index.hits(f"A 70-year-old man has chest pain and sweating after climbing "
                      f"stairs today. {BOILER}?") is None


def test_most_of_a_benchmark_question_inside_a_training_row_is_contamination():
    index = EvalIndex([VIGNETTE])
    near_copy = " ".join(VIGNETTE.split()[:16]) + " and nausea"
    assert index.hits(near_copy) == "ngram"
