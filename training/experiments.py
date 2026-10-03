"""Build experiments/results.csv from each experiment's record and raw predictions.

    python -m training.experiments                  # writes experiments/results.csv

Every score is recomputed here from the prediction files an experiment.json
points at, against the benchmark's gold letters. No score is typed in by hand:
an experiment whose predictions are missing gets empty score cells.

Within one experiment, each benchmark is scored on the questions every model
that ran it answered (first sample), so the models in a row group are
compared like for like. Thinking-mode scores use the first sample;
medqa_reasoning_any_correct says how often any sample was right.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.check_lengths import percentile
from training.eval_report import read_rows, read_samples, stage_files
from training.eval_worker import load_benchmark
from training.records import Record

COLUMNS = [
    # Phase 13's columns, in its order.
    "experiment_id", "git_commit", "dataset_version", "model", "quantization",
    "lora_rank", "lora_alpha", "learning_rate", "epochs", "batch_size",
    "gradient_accumulation", "max_seq_length", "warmup_ratio", "weight_decay",
    "train_loss", "validation_loss", "medqa_reasoning", "medqa_letter", "medmcqa",
    "pubmedqa", "mmlu_medical", "general_score", "safety_score",
    "median_thinking_length", "notes",
    # Added: what the scores above mean and rest on.
    "medqa_reasoning_any_correct", "medmcqa_letter", "pubmedqa_letter",
    "mmlu_medical_letter", "n_medqa_reasoning", "n_medmcqa", "n_pubmedqa",
    "n_mmlu_medical", "thinking_length_kind", "eval_protocol",
]
TRAINING_FIELDS = ("quantization", "lora_rank", "lora_alpha", "learning_rate", "epochs",
                   "batch_size", "gradient_accumulation", "max_seq_length",
                   "warmup_ratio", "weight_decay")
# Column -> stage. medmcqa, pubmedqa and mmlu_medical are thinking-mode
# accuracy, matching medqa_reasoning; their letter-choice scores have
# their own columns.
STAGE_COLUMNS = {"medqa_reasoning": "reasoning:medqa", "medqa_letter": "letter:medqa",
                 "medmcqa": "reasoning:medmcqa", "pubmedqa": "reasoning:pubmedqa",
                 "mmlu_medical": "reasoning:mmlu_medical",
                 "medmcqa_letter": "letter:medmcqa", "pubmedqa_letter": "letter:pubmedqa",
                 "mmlu_medical_letter": "letter:mmlu_medical"}
N_COLUMNS = {"n_medqa_reasoning": "reasoning:medqa", "n_medmcqa": "reasoning:medmcqa",
             "n_pubmedqa": "reasoning:pubmedqa", "n_mmlu_medical": "reasoning:mmlu_medical"}
# Run 1 predates the stage files: its evaluation JSON holds, per question,
# a letter-choice prediction ("constrained") scored without a /no_think switch.
V1_STAGES = {"eval_medqa.json": "letter:medqa", "eval_medmcqa.json": "letter:medmcqa"}


def load_v1(path: Path, role: str) -> dict[str, dict]:
    """Run 1's per-question letters for one role, as stage rows."""
    data = json.loads(path.read_text())
    return {p["id"]: {"id": p["id"], "letter": p["constrained"][role]}
            for p in data["predictions"]}


def stage_rows(exp: dict, root: Path) -> dict[str, dict[str, tuple[dict, dict]]]:
    """model -> stage -> (first-sample rows by id, all samples by id)."""
    ev = exp.get("evaluation") or {}
    out: dict[str, dict[str, tuple[dict, dict]]] = {}
    for model, spec in (ev.get("models") or {}).items():
        stages: dict[str, tuple[dict, dict]] = {}
        if ev.get("format") == "v1_eval_json":
            for name, stage in V1_STAGES.items():
                path = root / spec["dir"] / name
                if path.exists():
                    rows = load_v1(path, spec["role"])
                    stages[stage] = (rows, {k: [v] for k, v in rows.items()})
        else:
            files = stage_files([root / d for d in spec["dirs"]])
            for stage, path in files.items():
                stages[stage] = (read_rows(path), read_samples(path))
        out[model] = stages
    return out


def score(exp: dict, root: Path, bench_dir: Path) -> dict[str, dict]:
    """model -> column -> value, every score computed from predictions."""
    rows = stage_rows(exp, root)
    benches: dict[str, list[Record]] = {}
    out: dict[str, dict] = {m: {} for m in rows}
    for stage in sorted({s for st in rows.values() for s in st}):
        bench = stage.split(":")[1]
        if bench not in benches:
            benches[bench] = load_benchmark(bench_dir / f"eval_{bench}.jsonl", bench,
                                            check_size=False)
        have = [m for m in rows if stage in rows[m]]
        common = [r for r in benches[bench] if all(r.id in rows[m][stage][0] for m in have)]
        for m in have:
            firsts, samples = rows[m][stage]
            if not common:
                continue
            acc = sum(firsts[r.id]["letter"] == r.answer for r in common) / len(common)
            for col, st in STAGE_COLUMNS.items():
                if st == stage:
                    out[m][col] = round(acc, 4)
            for col, st in N_COLUMNS.items():
                if st == stage:
                    out[m][col] = len(common)
            if stage == "reasoning:medqa":
                picked = [firsts[r.id] for r in common]
                think = [p["think_tokens"] for p in picked if "think_tokens" in p]
                if think and len(think) == len(picked):
                    out[m]["median_thinking_length"] = percentile(think, 0.5)
                    out[m]["thinking_length_kind"] = "think_tokens"
                else:
                    out[m]["median_thinking_length"] = percentile(
                        [p.get("new_tokens", 0) for p in picked], 0.5)
                    out[m]["thinking_length_kind"] = "new_tokens (thinking + answer)"
                multi = [r for r in common if len(samples.get(r.id, [])) > 1]
                if multi and len(multi) == len(common):
                    out[m]["medqa_reasoning_any_correct"] = round(sum(
                        any(s["letter"] == r.answer for s in samples[r.id]) for r in multi)
                        / len(multi), 4)
    return out


def train_loss(exp: dict, root: Path) -> float | None:
    path = (exp.get("artifacts") or {}).get("train_stats")
    if path and (root / path).exists():
        return round(json.loads((root / path).read_text())["train_loss"], 4)
    return None


def rows_for(exp: dict, root: Path, bench_dir: Path) -> list[dict]:
    scores = score(exp, root, bench_dir) if (bench_dir / "eval_medqa.jsonl").exists() else {}
    training = exp.get("training") or {}
    ev = exp.get("evaluation") or {}
    out = []
    for model, spec in (ev.get("models") or {"-": {}}).items():
        row = {c: "" for c in COLUMNS}
        row.update({"experiment_id": exp["experiment_id"],
                    "git_commit": exp.get("git_commit", ""),
                    "dataset_version": exp.get("dataset_version", ""),
                    "model": spec.get("label", model),
                    "eval_protocol": ev.get("protocol", ""),
                    "notes": " ".join(filter(None, [exp.get("notes", ""), spec.get("notes", "")]))})
        if spec.get("trained", False):
            row.update({k: training.get(k, "") for k in TRAINING_FIELDS})
            loss = train_loss(exp, root)
            row["train_loss"] = "" if loss is None else loss
        else:
            row["quantization"] = spec.get("quantization", "")
        row.update(scores.get(model, {}))
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiments", type=Path, default=Path("experiments"))
    p.add_argument("--bench-dir", type=Path, default=Path("data/v2"))
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    if not (args.bench_dir / "eval_medqa.jsonl").exists():
        print(f"no benchmarks in {args.bench_dir}: score cells stay empty "
              "(run training.prepare_data_v2 to rebuild them)")
    rows = []
    for path in sorted(args.experiments.glob("*/experiment.json")):
        rows += rows_for(json.loads(path.read_text()), args.root, args.bench_dir)
    out = args.experiments / "results.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
