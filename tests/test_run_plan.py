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


from training.experiments_config import E2  # noqa: E402


def test_e2_generation_shards_the_pool_with_a_longer_budget():
    entry = E2["sessions"]["gen"][0][0]
    args = worker_args(E2, entry[0], entry[1], out_dir="o", eval_dir="d", adapters={},
                       extra=entry[2], engine="hf")
    a = " ".join(args)
    assert "--shard 0/2" in a and "--think-budget 4096" in a and "--samples pool=2" in a
    assert "--reasoning-batch-size 6" in a and "--engine" not in args


def test_the_vllm_engine_uses_its_own_chunk_sizes():
    entry = E2["sessions"]["gen"][1][0]
    args = worker_args(E2, entry[0], entry[1], out_dir="o", eval_dir="d", adapters={},
                       extra=entry[2], engine="vllm")
    a = " ".join(args)
    assert "--engine vllm" in a and "--reasoning-batch-size 192" in a and "--batch-size 512" in a
    assert "--gpu-memory-utilization 0.9" in a and "--shard 1/2" in a


def test_e2_evaluates_both_models_on_dev_and_never_tunes_on_benchmarks():
    stages = {m: {s for sess, g in E2["sessions"].items() if sess != "gen"
                  for q in g.values() for e in q if e[0] == m for s in e[1]}
              for m in ("base", "e2")}
    dev = {"reasoning:dev_medqa", "reasoning:dev_medmcqa", "letter:dev_medqa", "letter:dev_medmcqa"}
    assert dev <= stages["base"] and dev <= stages["e2"]
    assert E2["training"]["lr"] < 1e-4 and E2["training"]["rank"] < 64


def test_train_and_trace_commands():
    from training.run_plan import traces_args, train_args

    t = " ".join(train_args(E2, data_dir="data/e2", out_dir="outputs/e2", max_seq=3200,
                            stop_after=7200))
    assert "-m training.train --format v2 --data data/e2 --out outputs/e2" in t
    assert "--rank 16" in t and "--lr 5e-05" in t and "--batch-size 1 --grad-accum 16" in t
    assert "--max-seq 3200" in t and "--stop-after-seconds 7200" in t and "--no-probe-abort" in t
    r = " ".join(traces_args(E2, gen_dirs=["g0", "g1"], pool="data/v3/eval_pool.jsonl",
                             out="data/e2"))
    assert "--gen g0 --gen g1" in r and "--max-think 3500" in r and "--direct-fraction 0.2" in r


def test_an_entry_that_wrote_rows_is_not_rerun(tmp_path):
    from training.run_plan import entry_wrote_rows

    assert not entry_wrote_rows(tmp_path, "base", ("reasoning:pool",))
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "reasoning__pool.jsonl").write_text('{"id": "q"}\n')
    assert entry_wrote_rows(tmp_path, "base", ("reasoning:pool",))


def test_enforce_eager_reaches_the_vllm_worker():
    entry = E2["sessions"]["gen"][0][0]
    args = worker_args(E2, entry[0], entry[1], out_dir="o", eval_dir="d", adapters={},
                       extra=entry[2], engine="vllm", enforce_eager=True)
    assert "--enforce-eager" in args
    hf = worker_args(E2, entry[0], entry[1], out_dir="o", eval_dir="d", adapters={},
                     extra=entry[2], engine="hf", enforce_eager=True)
    assert "--enforce-eager" not in hf


def test_a_missing_vllm_interpreter_falls_back_to_transformers(tmp_path, monkeypatch):
    import types

    import training.run_plan as rp

    calls = []

    def fake_run(cmd, env=None):
        calls.append(cmd[0])
        if cmd[0] == "/no/such/python":
            raise FileNotFoundError(cmd[0])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(rp.subprocess, "run", fake_run)
    code = rp.run_queue(E2, "gen", 0, out_dir=str(tmp_path), eval_dir="d", adapters={},
                        engine="vllm", vllm_python="/no/such/python")
    assert code == 0 and calls[0] == "/no/such/python" and calls[1] == rp.sys.executable
