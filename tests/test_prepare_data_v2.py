from __future__ import annotations

from training.decontam import Deduper, EvalIndex
from training.prepare_data_v2 import MIX, MixEntry, choose_max_seq, collect, summarise_mix
from training.records import Record


def rec(i, *, kind="mcq", reasoning="r", q=None) -> Record:
    return Record(id=f"r{i}", source="s", kind=kind, question=q or f"question number {i}",
                  options={"A": "a", "B": "b"} if kind == "mcq" else None,
                  answer="A" if kind == "mcq" else None, rationale=None,
                  response=None if kind == "mcq" else "resp", subject=None,
                  reasoning=reasoning)


def test_collect_stops_at_target_and_counts_each_rejection():
    def normalise(raw, idx):
        i = raw["i"]
        if i == 0:
            return None
        if i == 1:
            return rec(1, kind="dialogue")
        if i == 2:
            return rec(2, q="blocked question")
        if i in (3, 4):
            return rec(i, q="same")
        return rec(i)

    kept, stats = collect(
        MixEntry("s", "mcq", 3), [(i, {"i": i}) for i in range(20)], normalise,
        index=EvalIndex(["blocked question"]), deduper=Deduper(),
        measure=lambda r: 999 if r.id == "r5" else 10, cap=100, max_candidates=100)
    assert [r.id for r, _ in kept] == ["r3", "r6", "r7"]
    assert stats == {"target": 3, "seen": 8, "unusable": 1, "other_kind": 1,
                     "contaminated_exact": 1, "contaminated_ngram": 0,
                     "duplicate": 1, "too_long": 1, "kept": 3}


def test_collect_gives_up_after_max_candidates():
    kept, stats = collect(MixEntry("s", None, 10), [(i, {}) for i in range(50)],
                          lambda raw, idx: None, index=EvalIndex([]), deduper=Deduper(),
                          measure=lambda r: 1, cap=10, max_candidates=5)
    assert kept == [] and stats["seen"] == 5 and stats["unusable"] == 5


def test_choose_max_seq_rounds_p99_up_to_64_and_caps():
    assert choose_max_seq([100] * 99 + [1000], cap=3072) == 128
    assert choose_max_seq(list(range(1, 5001)), cap=3072) == 3072


def test_summarise_mix_reports_fractions():
    s = summarise_mix([rec(1), rec(2, reasoning=None), rec(3, kind="dialogue")])
    assert s["reasoning_fraction"] == round(2 / 3, 4)
    assert s["mcq_fraction"] == round(2 / 3, 4)
    assert s["by_source"] == {"s": 3}


def test_the_mix_is_three_quarters_reasoning_and_eleven_thousand_rows():
    reasoning = {"medreason", "r1_distill", "ultramedical", "medical_o1", "reasonmed"}
    total = sum(e.target for e in MIX)
    assert total == 11_000
    assert sum(e.target for e in MIX if e.name in reasoning) == 8_250
