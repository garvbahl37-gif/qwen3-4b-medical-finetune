from __future__ import annotations

from training.evalcore import (accuracy, by_subject, mcnemar_exact, paired_diff_ci,
                              paired_report)
from training.records import Record


def mk(rid, answer, subject):
    return Record(id=rid, source="medmcqa", kind="mcq", question="q",
                  options={"A": "a", "B": "b", "C": "c", "D": "d"},
                  answer=answer, rationale=None, response=None, subject=subject)


def test_accuracy_counts_an_unparseable_answer_as_wrong():
    assert accuracy(["A", None, "B"], ["A", "B", "B"]) == 2 / 3


def test_accuracy_of_an_empty_set_is_zero_not_a_crash():
    assert accuracy([], []) == 0.0


def test_mcnemar_counts_the_four_cells():
    base = [True, True, False, False]
    tuned = [True, False, True, False]
    r = mcnemar_exact(base, tuned)
    assert r["both_correct"] == 1
    assert r["regressions"] == 1   # base right, tuned wrong
    assert r["wins"] == 1          # base wrong, tuned right
    assert r["both_wrong"] == 1


def test_mcnemar_returns_p_one_when_wins_and_regressions_balance():
    r = mcnemar_exact([True, False], [False, True])
    assert r["p_value"] == 1.0


def test_mcnemar_is_significant_when_every_discordant_pair_is_a_win():
    base = [False] * 10
    tuned = [True] * 10
    r = mcnemar_exact(base, tuned)
    assert r["wins"] == 10 and r["regressions"] == 0
    assert r["p_value"] < 0.01


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    r = mcnemar_exact([True, True], [True, True])
    assert r["p_value"] == 1.0


def test_by_subject_breaks_accuracy_out_per_subject():
    recs = [mk("1", "A", "Anatomy"), mk("2", "B", "Anatomy"), mk("3", "C", "Pharmacology")]
    out = by_subject(recs, [True, False, True], [True, True, False])
    assert out["Anatomy"]["n"] == 2
    assert out["Anatomy"]["base"] == 0.5
    assert out["Anatomy"]["tuned"] == 1.0
    assert out["Pharmacology"]["n"] == 1


def test_by_subject_buckets_missing_subject_under_unknown_not_dropped():
    recs = [mk("1", "A", None), mk("2", "B", None), mk("3", "C", "Anatomy")]
    out = by_subject(recs, [True, False, True], [True, True, False])
    assert sum(v["n"] for v in out.values()) == 3
    assert "unknown" in out
    assert out["unknown"]["n"] == 2
    assert out["unknown"]["base"] == 0.5
    assert out["unknown"]["tuned"] == 1.0


def test_paired_report_counts_unparseable_predictions_as_wrong_not_excluded():
    recs = [mk("1", "A", "Anatomy"), mk("2", "B", "Anatomy")]
    rep = paired_report(recs, [None, "B"], ["A", "B"], mode="constrained")
    assert rep["n"] == 2
    assert rep["base_accuracy"] == 0.5
    assert rep["base_unparseable"] == 1
    assert rep["mcnemar"]["wins"] == 1


def test_paired_report_has_everything_the_writeup_quotes():
    recs = [mk("1", "A", "Anatomy"), mk("2", "B", "Anatomy")]
    rep = paired_report(recs, ["A", "A"], ["A", "B"], mode="constrained")
    assert rep["mode"] == "constrained"
    assert rep["n"] == 2
    assert rep["base_accuracy"] == 0.5
    assert rep["tuned_accuracy"] == 1.0
    assert rep["mcnemar"]["wins"] == 1
    assert "Anatomy" in rep["by_subject"]



def test_paired_diff_ci_matches_the_paired_wald_formula():
    base = [True] * 50 + [False] * 50
    tuned = [True] * 40 + [False] * 10 + [True] * 30 + [False] * 20
    ci = paired_diff_ci(base, tuned)
    assert ci["diff"] == 0.2
    half = 1.959964 * ((40 - 20 ** 2 / 100) / 100 ** 2) ** 0.5
    assert abs(ci["low"] - (0.2 - half)) < 1e-6 and abs(ci["high"] - (0.2 + half)) < 1e-6


def test_paired_diff_ci_of_nothing_is_zero():
    assert paired_diff_ci([], []) == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_paired_diff_ci_never_leaves_minus_one_to_one():
    ci = paired_diff_ci([False] * 4, [True, True, True, False])
    assert ci["diff"] == 0.75 and ci["high"] == 1.0 and -1.0 <= ci["low"] <= 1.0
