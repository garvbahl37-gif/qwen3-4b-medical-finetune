"""E3: corrective reasoning from a stronger Qwen3 teacher, on the questions the
base model gets wrong.

    python -m training.teacher_traces weak   --gen results/e2/attempt1/e2_generation \\
        --pool data/v3/eval_pool.jsonl --out data/e3
    python -m training.teacher_traces pilot  --root outputs/e3/pilot --weak data/e3/eval_weak.jsonl
    python -m training.teacher_traces traces --gen DIR [--gen DIR] --weak data/e3/eval_weak.jsonl \\
        --out data/e3t
    python -m training.teacher_traces merge  --sets data/e2 data/e3t --out data/e3

E2 trained on the base model's own correct reasoning and changed nothing: it
only showed the model questions it already answers. E3 targets the other
side. The "weak" questions are the training-pool questions on which the base
model's E2 sample was wrong or unfinished. A larger Qwen3 thinks through
them, and only answers that reach the gold letter are kept (rejection
sampling, as MMOPD/Qwen3-4B-OT3-medical did with a far larger teacher).

The teacher is chosen by a pilot on a slice of the weak set: the one that
rescues the most questions per GPU-hour, among those that rescue at least a
fifth. Qwen3 teachers keep Qwen3's own thinking style, unlike run 3's
GPT-4 / DeepSeek-R1 traces.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from training.eval_report import read_all
from training.format_v2 import render_training_text
from training.records import Record
from training.self_traces import build_e2, load_generation, reject_reason

MIN_RESCUE = 0.2


def weak_questions(gen: dict[str, list[dict]], pool: dict[str, Record]):
    """Questions none of whose samples was a finished, unforced, correct answer."""
    weak = [pool[qid] for qid in sorted(gen)
            if all(reject_reason(row, pool[qid].answer, 10**9) for row in gen[qid])]
    return weak, {"answered": len(gen), "weak": len(weak),
                  "by_source": dict(Counter(r.source for r in weak))}


def pilot_stats(roots: list[Path], weak: dict[str, Record]) -> dict[str, dict]:
    """Per teacher (a worker directory holding reasoning__weak.jsonl): how
    many weak questions it answered and rescued, and how long it took."""
    out: dict[str, dict] = {}
    for root in roots:
        for path in sorted(root.rglob("reasoning__weak.jsonl")):
            name = path.parent.name
            meta = json.loads((path.parent / "meta.json").read_text())
            seconds = meta["stages"]["reasoning:weak"]["seconds"]
            rows = [r for r in read_all(path) if r["id"] in weak]
            rescued = len({r["id"] for r in rows
                           if not reject_reason(r, weak[r["id"]].answer, 10**9)})
            answered = len({r["id"] for r in rows})
            out[name] = {"answered": answered, "rescued": rescued,
                         "rescue_rate": round(rescued / answered, 4) if answered else 0.0,
                         "seconds": seconds,
                         "rescued_per_hour": round(rescued / seconds * 3600, 1) if seconds else 0.0}
    return out


def choose_teacher(stats: dict[str, dict], min_rescue: float = MIN_RESCUE) -> str | None:
    """The most rescued questions per hour among teachers that rescue at
    least `min_rescue` of what they answered; a tie goes to the higher rate."""
    able = {n: s for n, s in stats.items() if s["answered"] and s["rescue_rate"] >= min_rescue}
    if not able:
        return None
    return max(able, key=lambda n: (able[n]["rescued_per_hour"], able[n]["rescue_rate"]))


def merge_sets(dirs: list[Path], out: Path, *, seed: int) -> dict:
    """Concatenate training sets (each a train.jsonl and its e2_report.json),
    shuffled; max_seq is the longest of theirs."""
    rows, counts, max_seq = [], {}, 0
    for d in dirs:
        lines = [line for line in (d / "train.jsonl").read_text().splitlines() if line]
        rows += lines
        counts[d.name] = len(lines)
        max_seq = max(max_seq, json.loads((d / "e2_report.json").read_text())["max_seq"])
    random.Random(seed).shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(line + "\n" for line in rows))
    report = {"rows": len(rows), "max_seq": max_seq, "from": counts, "seed": seed}
    (out / "e2_report.json").write_text(json.dumps(report, indent=2))
    return report


def _records(path: Path) -> dict[str, Record]:
    return {r.id: r for r in (Record.from_dict(json.loads(line))
                              for line in path.read_text().splitlines() if line)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weak")
    w.add_argument("--gen", type=Path, action="append", required=True)
    w.add_argument("--pool", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    pl = sub.add_parser("pilot")
    pl.add_argument("--root", type=Path, action="append", required=True)
    pl.add_argument("--weak", type=Path, required=True)
    pl.add_argument("--out", type=Path)
    t = sub.add_parser("traces")
    t.add_argument("--gen", type=Path, action="append", required=True)
    t.add_argument("--weak", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--tokenizer", default="unsloth/Qwen3-4B")
    t.add_argument("--max-think", type=int, default=3500)
    t.add_argument("--cap", type=int, default=4096)
    t.add_argument("--max-per-question", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--min-rows", type=int, default=100)
    m = sub.add_parser("merge")
    m.add_argument("--sets", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.cmd == "weak":
        pool = _records(args.pool)
        gen = {k: v for k, v in load_generation(args.gen).items() if k in pool}
        weak, report = weak_questions(gen, pool)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "eval_weak.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n"
                                                          for r in weak))
        (args.out / "weak_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    elif args.cmd == "pilot":
        stats = pilot_stats(args.root, _records(args.weak))
        chosen = choose_teacher(stats)
        result = {"stats": stats, "chosen": chosen, "min_rescue": MIN_RESCUE}
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))
        if chosen is None:
            raise SystemExit("\nSTOP. No teacher rescued enough weak questions.")
    elif args.cmd == "traces":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        weak = _records(args.weak)
        gen = {k: v for k, v in load_generation(args.gen, stage="weak").items() if k in weak}

        def measure(rec: Record) -> int:
            return len(tok(render_training_text(tok, rec), add_special_tokens=False)["input_ids"])

        rows, report = build_e2(gen, weak, measure=measure, max_think=args.max_think,
                                cap=args.cap, direct_fraction=0.0, seed=args.seed,
                                max_per_question=args.max_per_question, prefix="e3t",
                                source="e3_teacher")
        report.update({"max_think": args.max_think, "cap": args.cap, "seed": args.seed,
                       "max_per_question": args.max_per_question,
                       "max_seq": max(256, -(-report["max_rendered_tokens"] // 64) * 64)})
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "train.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n"
                                                      for r in rows))
        (args.out / "e2_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if report["reasoning_rows"] < args.min_rows:
            raise SystemExit(f"\nSTOP. Only {report['reasoning_rows']} verified teacher traces.")
    else:
        print(json.dumps(merge_sets(args.sets, args.out, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
