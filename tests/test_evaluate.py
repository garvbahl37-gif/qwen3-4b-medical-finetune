from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training.evaluate import (
    batched,
    check_holdout_size,
    letter_token_ids,
    pick_from_logits,
    project_eval_seconds,
    require_aligned,
    score_constrained,
    score_generative,
    select_scorable_records,
)
from training.records import Record


def mk(rid, source, kind="mcq", answer="A"):
    return Record(id=rid, source=source, kind=kind, question="q",
                  options={"A": "a", "B": "b", "C": "c", "D": "d"},
                  answer=answer, rationale=None, response=None, subject=None)


# --- pick_from_logits (brief Step 1, verbatim) ------------------------------

def test_pick_from_logits_returns_the_highest_scoring_letter():
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert pick_from_logits([0.1, 0.2, 5.0, 0.3], ids) == "C"


def test_pick_from_logits_only_considers_the_four_letters():
    # index 7 is the global maximum but is not a letter token
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    logits = [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0]
    assert pick_from_logits(logits, ids) == "B"


def test_pick_from_logits_breaks_ties_deterministically():
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert pick_from_logits([1.0, 1.0, 1.0, 1.0], ids) == "A"


# --- letter_token_ids: the leading space must matter ------------------------

class _FakeTok:
    """Simulates a BPE tokenizer where ' A' and 'A' get different ids, and
    where a space-prefixed letter can encode to more than one token."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        mapping = {
            " A": [999, 100], " B": [999, 101], " C": [999, 102], " D": [999, 103],
            "A": [1], "B": [2], "C": [3], "D": [4],
        }
        return mapping[text]


def test_letter_token_ids_encodes_with_a_leading_space():
    ids = letter_token_ids(_FakeTok())
    # Must not match the bare-letter ids (1,2,3,4): that would mean the
    # leading space was dropped.
    assert ids == {"A": 100, "B": 101, "C": 102, "D": 103}


def test_letter_token_ids_takes_the_last_token_of_the_encoding():
    ids = letter_token_ids(_FakeTok())
    for letter, tid in ids.items():
        assert tid != 999, f"{letter} resolved to the shared prefix token, not the letter"


# --- select_scorable_records: deviation 4 ----------------------------------

def test_select_scorable_records_drops_dialogue_records():
    recs = [mk("1", "medqa", kind="mcq"), mk("2", "chatdoctor", kind="dialogue")]
    kept = select_scorable_records(recs)
    assert [r.id for r in kept] == ["1"]


def test_select_scorable_records_raises_on_an_mcq_with_no_answer_letter():
    recs = [mk("1", "medqa", answer=None)]
    with pytest.raises(SystemExit):
        select_scorable_records(recs)


def test_select_scorable_records_raises_on_an_mcq_with_an_invalid_answer():
    recs = [mk("1", "medqa", answer="E")]
    with pytest.raises(SystemExit):
        select_scorable_records(recs)


def test_select_scorable_records_error_message_does_not_crash_to_format():
    recs = [mk(str(i), "medqa", answer=None) for i in range(15)]
    try:
        select_scorable_records(recs)
    except SystemExit as exc:
        assert "15" in str(exc)
    else:
        pytest.fail("expected SystemExit")


# --- require_aligned: deviation 2 ------------------------------------------

def test_require_aligned_passes_when_all_three_lengths_match():
    recs = [mk("1", "medqa"), mk("2", "medqa")]
    require_aligned(recs, ["A", "B"], ["A", "A"], "constrained")


def test_require_aligned_raises_naming_the_three_lengths():
    recs = [mk("1", "medqa"), mk("2", "medqa")]
    with pytest.raises(SystemExit) as exc_info:
        require_aligned(recs, ["A"], ["A", "A"], "constrained")
    message = str(exc_info.value)
    assert "recs 2" in message
    assert "base_preds 1" in message
    assert "tuned_preds 2" in message


# --- check_holdout_size: deviation 1 ---------------------------------------

def test_check_holdout_size_passes_for_the_confirmed_counts():
    recs = [mk(str(i), "medqa") for i in range(1_273)] + \
           [mk(str(i), "medmcqa") for i in range(4_183)]
    check_holdout_size(recs)  # must not raise


def test_check_holdout_size_raises_when_medmcqa_is_short():
    # 2,858 is the known-wrong count produced by loading MedMCQA validation
    # with require_single_choice=True instead of False.
    recs = [mk(str(i), "medmcqa") for i in range(2_858)]
    with pytest.raises(SystemExit) as exc_info:
        check_holdout_size(recs)
    assert "2,858" in str(exc_info.value)
    assert "4,183" in str(exc_info.value)


def test_check_holdout_size_ignores_sources_it_does_not_know_about():
    recs = [mk(str(i), "synthetic_smoke_test") for i in range(3)]
    check_holdout_size(recs)  # must not raise


def test_batched_splits_into_full_groups_and_a_remainder():
    assert batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_a_batch_size_below_one():
    with pytest.raises(ValueError):
        batched([1], 0)


def test_pick_from_logits_refuses_non_finite_scores():
    # float16 overflow on a T4 turns logits into inf or nan, and argmax over
    # them silently returns the first letter for every question -- which
    # reads as a plausible accuracy. It must stop instead.
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    with pytest.raises(FloatingPointError):
        pick_from_logits([float("nan"), 0.0, 1.0, 2.0], ids)
    with pytest.raises(FloatingPointError):
        pick_from_logits([0.0, float("inf"), 1.0, 2.0], ids)


def test_scorers_refuse_a_right_padded_tokenizer():
    # Right padding puts a pad token at the last position of every shorter
    # sequence in a batch; the scorer would read it and score noise.
    class RightPadded:
        padding_side = "right"

    with pytest.raises(SystemExit, match="left"):
        score_constrained(None, RightPadded(), [])
    with pytest.raises(SystemExit, match="left"):
        score_generative(None, RightPadded(), [])


def test_project_eval_seconds_scales_measured_rates_to_both_models():
    timing = {"constrained_s_per_example": 0.5, "generative_s_per_example": 4.0}
    # 2 models x (1,000 x 0.5 + 100 x 4.0) = 2 x 900 = 1,800
    assert project_eval_seconds(timing, n_constrained=1000, n_generative=100) == 1800


def test_evaluation_never_imports_unsloth():
    for path in ("training/evaluate.py", "training/modeling.py"):
        tree = ast.parse(Path(path).read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                    for a in n.names}
        imported |= {n.module for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom) and n.module}
        assert not any(m.split(".")[0] == "unsloth" for m in imported), path
