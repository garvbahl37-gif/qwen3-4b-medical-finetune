from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.prompts import build_messages
from training.records import Record


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile. No numpy, so this runs anywhere."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(lengths: list[int]) -> dict:
    return {
        "n": len(lengths),
        "median": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure token lengths and optionally drop what will not fit.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    p.add_argument("--max-seq", type=int, default=0,
                   help="0 means report only and suggest a value")
    p.add_argument("--drop", action="store_true",
                   help="rewrite --data without examples that exceed --max-seq")
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy; imported only when needed

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]

    lengths = []
    for row in rows:
        msgs = build_messages(Record.from_dict(row), with_answer=True)
        # return_dict=False: newer transformers defaults tokenize=True to a
        # BatchEncoding (dict of input_ids/attention_mask), whose len() is the
        # key count, not the token count. Force the plain token-id list.
        ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
        lengths.append(len(ids))

    stats = summarise(lengths)
    print(f"tokens  n={stats['n']:,}  median={stats['median']}  "
          f"p90={stats['p90']}  p99={stats['p99']}  max={stats['max']}")

    if not args.max_seq:
        suggested = 64 * math.ceil(stats["p99"] / 64)
        print(f"\nsuggested --max-seq {suggested} (p99 rounded up to a multiple of 64)")
        print(f"it would drop {sum(1 for n in lengths if n > suggested):,} examples")
        return

    over = [i for i, n in enumerate(lengths) if n > args.max_seq]
    print(f"{len(over):,} of {len(rows):,} exceed max-seq {args.max_seq} "
          f"({len(over) / len(rows):.2%})")

    if not over:
        print("nothing to drop")
        return

    if not args.drop:
        # Truncating mid-answer teaches the model to emit unfinished responses,
        # which is a worse failure than losing a few long examples.
        raise SystemExit(
            f"\nSTOP. {len(over):,} examples would be truncated mid-answer.\n"
            f"FIX: re-run with --drop, or raise --max-seq to {stats['max']}.")

    keep = [row for i, row in enumerate(rows) if i not in set(over)]
    with args.data.open("w") as fh:
        for row in keep:
            fh.write(json.dumps(row) + "\n")
    print(f"dropped {len(over):,}; {len(keep):,} remain in {args.data}")


if __name__ == "__main__":
    main()
