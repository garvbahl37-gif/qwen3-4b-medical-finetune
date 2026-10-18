"""Build E2's training set from base Qwen3-4B's own verified reasoning.

    python -m training.self_traces --gen outputs/e2_gen --pool data/v3/eval_pool.jsonl \\
        --out data/e2

The evaluation worker, run in reasoning mode on the training-question pool,
leaves every sampled completion in reasoning__pool.jsonl. A sample becomes a
training row only if all of these hold:
- the model closed its think block itself;
- it stated a letter without being forced;
- that letter is the gold answer;
- its thinking holds no nested think tags;
- its thinking fits the length limit.

From each question's kept samples, one is chosen at random. Choosing the
shortest would teach shorter thinking, which is run 3's failure. The thinking
goes in verbatim; nothing is written for the model. A share of answer-only
rows on the same questions keeps letter choice, the Qwen3 / Unsloth advice to
mix reasoning and non-reasoning rows.

Rows are Records in the run 2/3 format, so training.train --format v2 trains
on them unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from training.check_lengths import percentile
from training.eval_report import read_all
from training.format_v2 import render_training_text
from training.records import Record

_ANSWER_LINE = re.compile(r"^\s*Answer:\s*\(?[A-J]\)?\.?\s*", re.IGNORECASE)


def split_completion(text: str) -> tuple[str | None, str | None]:
    """(thinking, final answer) from a completion, or (None, None) if the
    think block never closed."""
    if "</think>" not in (text or ""):
        return None, None
    head, _, final = text.rpartition("</think>")
    thinking = head.split("<think>", 1)[-1] if head.lstrip().startswith("<think>") else head
    return thinking.strip(), final.replace("<|im_end|>", "").strip()


def reject_reason(row: dict, gold: str, max_think: int) -> str | None:
    if not row.get("closed") or row.get("forced"):
        return "unfinished_or_forced"
    if row.get("letter") != gold:
        return "wrong_letter"
    thinking, _ = split_completion(row.get("text", ""))
    if not thinking or "<think>" in thinking or "</think>" in thinking:
        return "nested_think"
    if row.get("think_tokens", 0) > max_think:
        return "thinking_too_long"
    return None


def trace_record(rec: Record, row: dict) -> Record:
    thinking, final = split_completion(row["text"])
    explanation = _ANSWER_LINE.sub("", final, count=1).strip()
    return Record(**{**rec.to_dict(), "id": f"e2-{rec.id}", "source": "e2_self",
                     "reasoning": thinking, "rationale": explanation or None})


def build_e2(rows_by_q: dict[str, list[dict]], pool: dict[str, Record], *, measure,
             max_think: int, cap: int, direct_fraction: float, seed: int):
    """The training rows and a report. `measure(record)` is the rendered
    training length in tokens, passed in so tests need no tokenizer."""
    rng = random.Random(seed)
    rejected: Counter[str] = Counter()
    histogram: Counter[str] = Counter()
    reasoning: list[Record] = []
    chosen_think: dict[str, int] = {}
    for qid in sorted(rows_by_q):
        rec = pool[qid]
        samples = sorted(rows_by_q[qid], key=lambda r: r.get("sample", 0))
        kept = []
        for row in samples:
            why = reject_reason(row, rec.answer, max_think)
            if why:
                rejected[why] += 1
            else:
                kept.append(row)
        histogram[f"{len(kept)}/{len(samples)}"] += 1
        if kept:
            pick = rng.choice(kept)
            reasoning.append(trace_record(rec, pick))
            chosen_think[f"e2-{qid}"] = pick.get("think_tokens", 0)
    lengths = {r.id: measure(r) for r in reasoning}
    fitting = [r for r in reasoning if lengths[r.id] <= cap]
    n_direct = round(len(fitting) * direct_fraction / (1 - direct_fraction)) if fitting else 0
    direct_ids = rng.sample(sorted(rows_by_q), min(n_direct, len(rows_by_q)))
    direct = [Record(**{**pool[qid].to_dict(), "id": f"e2d-{qid}", "source": "e2_direct",
                        "rationale": None, "reasoning": None}) for qid in direct_ids]
    direct = [r for r in direct if measure(r) <= cap]
    rows = fitting + direct
    rng.shuffle(rows)
    total = sum(len(v) for v in rows_by_q.values())
    think = [r.get("think_tokens", 0) for v in rows_by_q.values() for r in v]
    kept_think = [chosen_think[r.id] for r in fitting]
    report = {
        "questions": len(rows_by_q), "samples_generated": total,
        "samples_kept": total - sum(rejected.values()),
        "acceptance_rate": round((total - sum(rejected.values())) / total, 4) if total else 0,
        "rejected": dict(rejected), "pass_rate_histogram": dict(histogram),
        "questions_with_a_kept_trace": len(reasoning), "dropped_over_cap": len(reasoning) - len(fitting),
        "reasoning_rows": len(fitting), "direct_rows": len(direct), "rows": len(rows),
        "think_tokens_all_samples": ({"median": percentile(think, 0.5), "p90": percentile(think, 0.9)}
                                     if think else None),
        "think_tokens_kept": ({"median": percentile(kept_think, 0.5),
                               "p90": percentile(kept_think, 0.9)} if kept_think else None),
        "max_rendered_tokens": max((measure(r) for r in rows), default=0),
    }
    return rows, report


def load_generation(dirs: list[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in dirs:
        for path in sorted(d.rglob("reasoning__pool.jsonl")):
            for row in read_all(path):
                out.setdefault(row["id"], []).append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gen", type=Path, action="append", required=True,
                   help="a worker output directory to search for reasoning__pool.jsonl")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default="unsloth/Qwen3-4B")
    p.add_argument("--max-think", type=int, default=3500)
    p.add_argument("--cap", type=int, default=4096, help="longest rendered training row")
    p.add_argument("--direct-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-rows", type=int, default=200,
                   help="stop with an error below this many reasoning rows")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = {r.id: r for r in (Record.from_dict(json.loads(line))
                              for line in args.pool.read_text().splitlines() if line)}
    rows_by_q = {k: v for k, v in load_generation(args.gen).items() if k in pool}

    def measure(rec: Record) -> int:
        return len(tok(render_training_text(tok, rec), add_special_tokens=False)["input_ids"])

    rows, report = build_e2(rows_by_q, pool, measure=measure, max_think=args.max_think,
                            cap=args.cap, direct_fraction=args.direct_fraction, seed=args.seed)
    report.update({"max_think": args.max_think, "cap": args.cap,
                   "direct_fraction": args.direct_fraction, "seed": args.seed,
                   "max_seq": max(256, -(-report["max_rendered_tokens"] // 64) * 64)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    (args.out / "e2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["reasoning_rows"] < args.min_rows:
        raise SystemExit(f"\nSTOP. Only {report['reasoning_rows']} verified traces "
                         f"(need {args.min_rows}); not enough to train on.")


if __name__ == "__main__":
    main()
