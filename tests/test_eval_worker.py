from __future__ import annotations

import pytest

from training.eval_worker import STAGES, fits_before, parse_stages, pick_letter, stage_order
from training.records import Record


def recs(n: int) -> list[Record]:
    return [Record(id=f"q{i}", source="s", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject=None) for i in range(n)]


def test_letter_stages_keep_every_question_in_file_order():
    assert [r.id for r in stage_order(recs(5), "letter", "medqa", 1)] == [f"q{i}" for i in range(5)]


def test_reasoning_order_is_a_seeded_shuffle_cut_to_size():
    a = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    b = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    assert [r.id for r in a] == [r.id for r in b] and len(a) == 1000
    assert [r.id for r in a] != [f"q{i}" for i in range(1000)]
    assert len(stage_order(recs(1273), "reasoning", "medqa", 7)) == 1273


def test_fits_before():
    assert fits_before(100.0, 50.0, 50.0) and not fits_before(100.0, 50.0, 51.0)


def test_parse_stages():
    assert parse_stages("letter:medqa,reasoning:pubmedqa") == [("letter", "medqa"),
                                                               ("reasoning", "pubmedqa")]
    assert parse_stages("") == list(STAGES)
    with pytest.raises(SystemExit):
        parse_stages("letter:nope")


def test_pick_letter_restricts_to_the_questions_options_and_breaks_ties_early():
    row = {10: 1.0, 11: 5.0, 12: 5.0, 13: 9.0}
    ids = {"A": 10, "B": 11, "C": 12, "D": 13}
    assert pick_letter(row, ids, ["A", "B", "C"]) == "B"
    assert pick_letter(row, ids, ["A", "B", "C", "D"]) == "D"
    with pytest.raises(FloatingPointError):
        pick_letter({10: float("nan"), 11: 0.0}, ids, ["A", "B"])


def test_split_on_oom_retries_halves_after_the_failure_is_released():
    import torch

    from training.eval_worker import _split_on_oom

    calls = []

    def fn(group):
        calls.append(len(group))
        if len(group) > 2:
            raise torch.cuda.OutOfMemoryError("too big")
        return [x * 10 for x in group]

    assert _split_on_oom(fn, [1, 2, 3, 4, 5, 6, 7, 8]) == [10, 20, 30, 40, 50, 60, 70, 80]
    assert calls == [8, 4, 2, 2, 4, 2, 2]


def test_chunks_splits_forced_prompts_into_small_groups():
    from training.eval_worker import FORCE_BATCH, chunks

    assert FORCE_BATCH == 2
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_sizes_override_the_default_reasoning_cut():
    a = stage_order(recs(1273), "reasoning", "medqa", 7, {"medqa": 200})
    full = stage_order(recs(1273), "reasoning", "medqa", 7)
    # the cut is a prefix of the same shuffle, so any two runs share questions
    assert [r.id for r in a] == [r.id for r in full[:200]]
    assert len(stage_order(recs(2000), "reasoning", "medmcqa", 7, {"medqa": 5})) == 1000
    assert len(stage_order(recs(2000), "reasoning", "medmcqa", 7, {"medmcqa": 0})) == 2000


def test_parse_counts():
    from training.eval_worker import parse_counts

    assert parse_counts("medqa=200, pubmedqa=0", minimum=0) == {"medqa": 200, "pubmedqa": 0}
    assert parse_counts("", minimum=1) == {}
    for bad in ("medqa=0", "nope=2", "medqa=x", "medqa"):
        with pytest.raises(SystemExit):
            parse_counts(bad, minimum=1)


def test_only_base_runs_without_an_adapter():
    from training.eval_worker import check_role

    assert check_role("base", "outputs/run3") is None
    assert check_role("run3_4bit", "outputs/run3") == "outputs/run3"
    with pytest.raises(SystemExit):
        check_role("tuned", None)
    with pytest.raises(SystemExit):
        check_role("Run 3", "x")


def test_think_length_counts_tokens_before_the_last_close():
    from training.eval_worker import think_length

    assert think_length([1, 2, 3, 99, 5], 99) == 3
    assert think_length([1, 99, 2, 99, 5], 99) == 3
    assert think_length([1, 2, 3], 99) == 3


def test_sample_zero_keeps_the_seeds_run_3_used():
    from training.eval_worker import SAMPLE_STRIDE, batch_seed

    assert batch_seed(1234, 32, 0) == 1234 * 1_000_003 + 32
    assert batch_seed(1234, 32, 1) - batch_seed(1234, 32, 0) == SAMPLE_STRIDE > 4183


def test_run_stage_scores_every_question_once_per_sample_sample_zero_first(tmp_path, monkeypatch):
    import json

    import training.eval_worker as w

    def fake(model, tok, g, *, seed, **kw):
        return [{"id": r.id, "letter": "A", "seed": seed} for r in g]

    monkeypatch.setattr(w, "reason_batch", fake)
    out = tmp_path / "reasoning__medqa.jsonl"
    status, written = w.run_stage(None, None, "reasoning", recs(5), out, batch_size=2,
                                  budget=8, seed=1, deadline=float("inf"), rates={},
                                  ids={}, think_id=0, stop_ids=set(), samples=2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert (status, written) == ("done", 10)
    assert [r["sample"] for r in rows] == [0] * 5 + [1] * 5
    assert [r["id"] for r in rows[:5]] == [r["id"] for r in rows[5:]]
    assert len({r["seed"] for r in rows}) == 6  # three batches per sample


def test_letter_stages_ignore_samples(tmp_path, monkeypatch):
    import training.eval_worker as w

    monkeypatch.setattr(w, "letters_for_prompts", lambda m, t, p, sets, ids: ["A"] * len(p))
    monkeypatch.setattr(w, "render_letter_prompt", lambda tok, r: "x")
    status, written = w.run_stage(None, None, "letter", recs(3), tmp_path / "l.jsonl",
                                  batch_size=2, budget=8, seed=1, deadline=float("inf"),
                                  rates={}, ids={}, think_id=0, stop_ids=set(), samples=2)
    assert (status, written) == ("done", 3)
