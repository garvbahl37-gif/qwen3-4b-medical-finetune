"""Build data/v3: questions for E2's self-generated traces, and a dev split.

    python -m training.prepare_pool_v3 --out data/v3

E2 trains Qwen3-4B on its own reasoning, kept only when it reaches the gold
answer, so it needs training questions with trustworthy gold letters:
MedQA-USMLE and MedMCQA training splits. From the same shuffled stream it
also sets aside a dev split, so that choosing between E2 variants never
looks at the benchmarks the final numbers are reported on.

Every question is checked against all four benchmarks (exact match, or at
least half of a benchmark question's 13-word runs) and deduplicated; pool
questions are also checked against the dev split. The four benchmark files
are copied in unchanged, so one Kaggle dataset carries everything E2 needs.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from training import sources
from training.decontam import Deduper, EvalIndex
from training.kaggle_paths import V3_DATA_FILES, data_fingerprint
from training.records import Record

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa", "mmlu_medical")
# MedQA is the target exam; MedMCQA adds breadth. The pool is larger than one
# session can generate for, and the worker's seeded order makes whatever it
# reaches a random sample.
DEV_SIZE = {"medqa": 300, "medmcqa": 300}
POOL_SIZE = {"medqa": 4000, "medmcqa": 2000}


def split_pool(candidates: dict[str, list[Record]], *, eval_questions: list[str],
               dev_size: dict[str, int], pool_size: dict[str, int]):
    """Dev first, then pool, from each source's (already shuffled) list."""
    bench = EvalIndex(eval_questions)
    seen = Deduper()
    dev: dict[str, list[Record]] = {}
    clean: dict[str, list[Record]] = {}
    stats = {}
    for name, recs in candidates.items():
        st = stats[name] = {"seen": len(recs), "contaminated": 0, "duplicate": 0,
                            "near_dev": 0, "dev": 0, "pool": 0}
        clean[name] = []
        for i, rec in enumerate(recs):
            if bench.hits(rec.question):
                st["contaminated"] += 1
            elif not seen.first_time(rec.question):
                st["duplicate"] += 1
            else:
                clean[name].append(Record(**{**rec.to_dict(),
                                             "id": f"{name}train-{rec.id.split('-')[-1] or i}"}))
        dev[name] = clean[name][:dev_size.get(name, 0)]
        st["dev"] = len(dev[name])
    dev_index = EvalIndex([r.question for recs in dev.values() for r in recs])
    pool: list[Record] = []
    for name in candidates:
        taken = 0
        for rec in clean[name][len(dev[name]):]:
            if taken >= pool_size.get(name, 0):
                break
            if dev_index.hits(rec.question):
                stats[name]["near_dev"] += 1
                continue
            pool.append(rec)
            taken += 1
        stats[name]["pool"] = taken
    return {"dev_medqa": dev.get("medqa", []), "dev_medmcqa": dev.get("medmcqa", []),
            "pool": pool}, stats


def load_candidates(seed: int) -> dict[str, list[Record]]:
    """Every usable training question, shuffled. MedMCQA keeps rows without an
    explanation (E2 needs only the gold letter) but drops multi-answer rows."""
    from datasets import load_dataset

    out = {}
    for name in ("medqa", "medmcqa"):
        hf_id, config, split = sources.SOURCES[name]
        ds = load_dataset(hf_id, config, split=split)
        order = list(range(len(ds)))
        random.Random(seed).shuffle(order)
        recs = []
        for idx in order:
            raw = ds[idx]
            rec = (sources.normalise_medqa(raw, idx) if name == "medqa" else
                   sources.normalise_medmcqa(raw, idx, require_rationale=False))
            if rec is not None:
                recs.append(Record(**{**rec.to_dict(), "id": f"{name}-{idx}"}))
        out[name] = recs
    return out


def write_jsonl(path: Path, recs: list[Record]) -> None:
    path.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--benchmarks", type=Path, default=Path("data/v2"))
    p.add_argument("--out", type=Path, default=Path("data/v3"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_questions = []
    for bench in BENCHMARKS:
        src = args.benchmarks / f"eval_{bench}.jsonl"
        shutil.copy(src, args.out / src.name)
        eval_questions += [json.loads(line)["question"] for line in src.read_text().splitlines()
                           if line]
    out, stats = split_pool(load_candidates(args.seed), eval_questions=eval_questions,
                            dev_size=DEV_SIZE, pool_size=POOL_SIZE)
    for name, recs in out.items():
        write_jsonl(args.out / f"eval_{name}.jsonl", recs)
    report = {"version": "v3", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "seed": args.seed, "sources": {n: sources.SOURCES[n] for n in ("medqa", "medmcqa")},
              "dev_size": DEV_SIZE, "pool_size": POOL_SIZE, "stats": stats,
              "sizes": {name: len(recs) for name, recs in out.items()},
              "benchmarks_from": str(args.benchmarks)}
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"sizes": report["sizes"], "stats": stats}, indent=2))
    print(f"wrote {args.out}, data fingerprint {data_fingerprint(args.out, V3_DATA_FILES)}")


if __name__ == "__main__":
    main()
