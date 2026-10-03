"""Turn an experiment's sessions into worker commands, and run one GPU's queue.

    python -m training.run_plan show [--experiment E0] [--write experiments/E0/plan.json]
    python -m training.run_plan run --session s1 --gpu 0 --out-dir outputs/e0/s1 \\
        --eval-dir data/v2 --adapter run3=outputs/run3 --deadline 1767225600 [--smoke]

`run` is what a Kaggle notebook launches once per GPU: it runs that GPU's
queue in order, one worker process per entry, and carries on to the next
entry if one fails. The time estimates are arithmetic on run 3's measured
speeds, not measurements of E0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from training.experiments_config import EXPERIMENTS

# Seconds per question on one Tesla T4, fp16, batch 16, 1,536-token budget:
# run 3's evaluation (results/run3/predictions/{base,tuned}/meta.json).
# The base model never reached reasoning on PubMedQA or MedMCQA there; those
# take its MedQA speed, the slowest measured.
MEASURED = {
    "base": {"letter:medqa": 0.21, "letter:medmcqa": 0.08, "letter:pubmedqa": 0.29,
             "letter:mmlu_medical": 0.14, "reasoning:medqa": 22.92,
             "reasoning:mmlu_medical": 20.96, "reasoning:pubmedqa": 22.92,
             "reasoning:medmcqa": 22.92},
    "tuned": {"letter:medqa": 0.30, "letter:medmcqa": 0.11, "letter:pubmedqa": 0.41,
              "letter:mmlu_medical": 0.20, "reasoning:medqa": 9.01,
              "reasoning:mmlu_medical": 8.19, "reasoning:pubmedqa": 4.68,
              "reasoning:medmcqa": 6.92},
}
MEASURED_BUDGET, MEASURED_BATCH = 1536, 16
# Unmeasured: how much slower bitsandbytes 4-bit generation is than fp16 here.
FOUR_BIT_SLOWDOWN = 1.5
# One Kaggle session: 12 hours, the workers stop 40 minutes early, and setup
# (install, fetch, model download, smoke run) takes about half an hour.
SESSION_SECONDS = 12 * 3600 - 40 * 60 - 30 * 60
QUESTIONS = {"medqa": 1273, "medmcqa": 4183, "pubmedqa": 1000, "mmlu_medical": 1089}
SMOKE = ("--limit", "2", "--think-budget", "64", "--batch-size", "2",
         "--reasoning-batch-size", "2")


def model_samples(cfg: dict, model: str) -> dict[str, int]:
    return {**cfg["samples"], **cfg["models"][model].get("samples", {})}


def worker_args(cfg: dict, model: str, stages: tuple[str, ...], *, out_dir: str,
                eval_dir: str, adapters: dict[str, str], deadline: float = 0.0,
                smoke: bool = False) -> list[str]:
    """The eval_worker command line for one queue entry, without the interpreter."""
    spec = cfg["models"][model]
    args = ["-m", "training.eval_worker", "--role", model, "--base", spec["base"],
            "--eval-dir", eval_dir, "--out-dir", out_dir, "--device", "cuda",
            "--stages", ",".join(stages), "--seed", str(cfg["seed"]),
            "--sizes", ",".join(f"{b}={n}" for b, n in cfg["sizes"].items())]
    samples = model_samples(cfg, model)
    if samples:
        args += ["--samples", ",".join(f"{b}={n}" for b, n in samples.items())]
    if spec["adapter"]:
        if spec["adapter"] not in adapters:
            raise SystemExit(f"\nSTOP. {model} needs --adapter {spec['adapter']}=DIR.")
        args += ["--adapter", adapters[spec["adapter"]]]
    if spec["load_in_4bit"]:
        args.append("--load-in-4bit")
    if smoke:
        return args + list(SMOKE)
    args += ["--think-budget", str(cfg["think_budget"]),
             "--batch-size", str(cfg["batch_size"]),
             "--reasoning-batch-size", str(cfg["reasoning_batch_size"])]
    if deadline:
        args += ["--deadline", f"{deadline:.0f}"]
    return args


def stage_seconds(cfg: dict, model: str, stage: str, *, step_factor: float) -> float:
    """Estimated seconds for one stage of one model.

    Reasoning scales with the budget (most base-model batches run to it) and
    with the number of batches; `step_factor` is how long one decoding step
    of the smaller batch takes relative to a step of 16, which was not
    measured: 1.0 is the pessimistic end, ~0.6 the optimistic."""
    spec = cfg["models"][model]
    rates = MEASURED["base" if spec["adapter"] is None else "tuned"]
    mode, bench = stage.split(":")
    if mode == "letter":
        seconds = rates[stage] * QUESTIONS[bench]
    else:
        n = (cfg["sizes"].get(bench) or QUESTIONS[bench]) * model_samples(cfg, model).get(bench, 1)
        scale = (cfg["think_budget"] / MEASURED_BUDGET
                 * MEASURED_BATCH / cfg["reasoning_batch_size"] * step_factor)
        seconds = rates[stage] * n * scale
    return seconds * (FOUR_BIT_SLOWDOWN if spec["load_in_4bit"] else 1.0)


def queue_seconds(cfg: dict, session: str, gpu: int, *, step_factor: float) -> float:
    return sum(stage_seconds(cfg, model, stage, step_factor=step_factor)
               for model, stages in cfg["sessions"][session][gpu] for stage in stages)


def plan(cfg: dict) -> dict:
    """Every queue with its pessimistic and optimistic estimate."""
    out = {}
    for session, gpus in cfg["sessions"].items():
        for gpu, queue in gpus.items():
            out[f"{session}/gpu{gpu}"] = {
                "queue": [{"model": m, "stages": list(st)} for m, st in queue],
                "estimate_hours": {
                    "pessimistic": round(queue_seconds(cfg, session, gpu, step_factor=1.0) / 3600, 1),
                    "optimistic": round(queue_seconds(cfg, session, gpu, step_factor=0.6) / 3600, 1)},
                "session_hours": round(SESSION_SECONDS / 3600, 1)}
    return out


def run_queue(cfg: dict, session: str, gpu: int, **kw) -> int:
    """Run one GPU's queue in order. A failed entry is reported and the queue
    moves on: the next model's results are still worth having."""
    failed = []
    for model, stages in cfg["sessions"][session][gpu]:
        cmd = [sys.executable] + worker_args(cfg, model, stages, **kw)
        print("$", " ".join(cmd), flush=True)
        code = subprocess.run(cmd, env=os.environ.copy()).returncode
        print(f"{model}: exit {code}", flush=True)
        if code:
            failed.append(model)
    if failed:
        print("failed:", ", ".join(failed), flush=True)
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show")
    show.add_argument("--experiment", default="E0")
    show.add_argument("--write", type=Path, help="also save config and plan as JSON here")
    run = sub.add_parser("run")
    run.add_argument("--experiment", default="E0")
    run.add_argument("--session", required=True)
    run.add_argument("--gpu", type=int, required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--eval-dir", required=True)
    run.add_argument("--adapter", action="append", default=[], help="name=dir")
    run.add_argument("--deadline", type=float, default=0.0)
    run.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = EXPERIMENTS[args.experiment]
    if args.cmd == "show":
        result = plan(cfg)
        for key, item in result.items():
            est = item["estimate_hours"]
            print(f"{key}: {est['optimistic']}-{est['pessimistic']} h of "
                  f"{item['session_hours']} h")
            for entry in item["queue"]:
                print(f"    {entry['model']:<10} {', '.join(entry['stages'])}")
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(json.dumps({"config": cfg, "plan": result}, indent=2,
                                             default=list))
            print(f"wrote {args.write}")
        return
    adapters = dict(item.split("=", 1) for item in args.adapter)
    sys.exit(run_queue(cfg, args.session, args.gpu, out_dir=args.out_dir,
                       eval_dir=args.eval_dir, adapters=adapters,
                       deadline=args.deadline, smoke=args.smoke))


if __name__ == "__main__":
    main()
