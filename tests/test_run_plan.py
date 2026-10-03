from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from training.eval_worker import STAGES
from training.experiments_config import ADAPTERS, E0
from training.run_plan import (MEASURED, SESSION_SECONDS, queue_seconds, stage_seconds,
                               worker_args)

ALL_STAGES = {f"{m}:{b}" for m, b in STAGES}
ADAPTER_DIRS = {"run3": "outputs/run3"}


def scheduled() -> Counter:
    return Counter((model, stage) for gpus in E0["sessions"].values()
                   for queue in gpus.values() for model, stages in queue for stage in stages)


def test_models_a_and_b_get_every_stage_once_and_c_its_control_stages():
    counts = scheduled()
    assert all(n == 1 for n in counts.values())
    for model in ("base", "run3"):
        assert {s for m, s in counts if m == model} == ALL_STAGES
    assert {s for m, s in counts if m == "run3_4bit"} == {
        "letter:medqa", "letter:medmcqa", "letter:pubmedqa", "letter:mmlu_medical",
        "reasoning:medqa"}


def test_every_queue_fits_a_session_even_at_the_pessimistic_estimate():
    for session, gpus in E0["sessions"].items():
        for gpu in gpus:
            assert queue_seconds(E0, session, gpu, step_factor=1.0) <= SESSION_SECONDS, (session, gpu)


def test_e0_follows_the_specification():
    assert E0["think_budget"] == 3072 and E0["samples"] == {"medqa": 2}
    assert {m["label"] for m in E0["models"].values()} == {"A", "B", "C"}
    assert E0["models"]["run3_4bit"]["base"] == "unsloth/Qwen3-4B-unsloth-bnb-4bit"


def test_worker_args_carry_the_shared_settings():
    args = worker_args(E0, "base", ("reasoning:medqa",), out_dir="o", eval_dir="d",
                       adapters=ADAPTER_DIRS, deadline=123.4)
    joined = " ".join(args)
    assert "--think-budget 3072" in joined and "--reasoning-batch-size 8" in joined
    assert "--sizes medqa=200,mmlu_medical=150,pubmedqa=150,medmcqa=150" in joined
    assert "--samples medqa=2" in joined and "--deadline 123" in joined
    assert "--adapter" not in args and "--load-in-4bit" not in args


def test_the_4bit_control_loads_the_training_base_and_answers_once():
    args = worker_args(E0, "run3_4bit", ("reasoning:medqa",), out_dir="o", eval_dir="d",
                       adapters=ADAPTER_DIRS)
    assert "--load-in-4bit" in args and "--samples" in args
    assert args[args.index("--samples") + 1] == "medqa=1"
    assert args[args.index("--adapter") + 1] == "outputs/run3"


def test_smoke_args_are_tiny_and_have_no_deadline():
    args = worker_args(E0, "run3", ("letter:medqa",), out_dir="o", eval_dir="d",
                       adapters=ADAPTER_DIRS, deadline=99.0, smoke=True)
    assert args[args.index("--limit") + 1] == "2" and "--deadline" not in args
    assert args[args.index("--think-budget") + 1] == "64"


def test_a_missing_adapter_stops():
    with pytest.raises(SystemExit):
        worker_args(E0, "run3", ("letter:medqa",), out_dir="o", eval_dir="d", adapters={})


def test_estimates_scale_with_budget_samples_and_quantization():
    base = stage_seconds(E0, "base", "reasoning:medqa", step_factor=1.0)
    # 200 questions x 2 samples, 2x the budget, half the batch
    assert base == pytest.approx(22.92 * 400 * 2 * 2)
    c = stage_seconds(E0, "run3_4bit", "reasoning:medqa", step_factor=1.0)
    assert c == pytest.approx(9.01 * 200 * 4 * 1.5)


@pytest.mark.skipif(not Path("results/run3/predictions/base/meta.json").exists(),
                    reason="run 3 predictions not present")
def test_measured_speeds_are_run_3s():
    for role in ("base", "tuned"):
        meta = json.loads(Path(f"results/run3/predictions/{role}/meta.json").read_text())
        assert meta["think_budget"] == 1536 and meta["batch_size"] == 16
        for key, stage in meta["stages"].items():
            assert MEASURED[role][key] == pytest.approx(stage["seconds"] / stage["scored"], abs=0.01)


@pytest.mark.skipif(not Path("results/run3/predictions/tuned/meta.json").exists(),
                    reason="run 3 predictions not present")
def test_the_adapter_checked_on_kaggle_is_the_one_run_3_evaluated():
    meta = json.loads(Path("results/run3/predictions/tuned/meta.json").read_text())
    assert meta["adapter_sha256"].startswith(ADAPTERS["run3"]["sha256_prefix"])
