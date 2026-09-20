from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from training.records import Record
from training.sources import load

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalise_question(text: str) -> str:
    """Collapse a question to a comparable key for overlap detection."""
    text = unicodedata.normalize("NFKD", text or "").lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def decontaminate(train: list[Record], holdouts: list[Record]) -> tuple[list[Record], int]:
    """Drop training rows whose question appears in a holdout, or twice in train.

    A benchmark whose answers are in the training set measures memorisation.
    Duplicates inside the training set are dropped for the same reason: they
    silently reweight whatever they duplicate.
    """
    blocked = {normalise_question(r.question) for r in holdouts}
    kept: list[Record] = []
    removed = 0
    for rec in train:
        key = normalise_question(rec.question)
        if key in blocked:
            removed += 1
            continue
        blocked.add(key)
        kept.append(rec)
    return kept, removed


def main() -> None:
    p = argparse.ArgumentParser(description="Build the medical training set.")
    p.add_argument("--medmcqa", type=int, default=14000)
    p.add_argument("--medqa", type=int, default=8000)
    p.add_argument("--medical-o1", type=int, default=6000)
    p.add_argument("--chatdoctor", type=int, default=12000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading held-out benchmarks first, so training can be cleaned against them.")
    holdouts = (
        load("medqa", limit=0, seed=args.seed, split="test")
        + load("medmcqa", limit=0, seed=args.seed, split="validation",
               require_rationale=False, require_single_choice=False)
    )
    print(f"  holdout pool: {len(holdouts):,} questions\n")

    print("Loading training sources.")
    train: list[Record] = []
    for name, want in (
        ("medmcqa", args.medmcqa), ("medqa", args.medqa),
        ("medical_o1", args.medical_o1), ("chatdoctor", args.chatdoctor),
    ):
        if want:
            train += load(name, limit=want, seed=args.seed)

    before = len(train)
    train, removed = decontaminate(train, holdouts)
    if before:
        print(f"\ndecontamination: dropped {removed:,} of {before:,} "
              f"({removed / before:.2%}) as holdout overlap or duplicate")
    else:
        print(f"\ndecontamination: dropped {removed:,} of {before:,} "
              "as holdout overlap or duplicate")

    random.Random(args.seed).shuffle(train)
    val, train = train[: args.val_size], train[args.val_size:]

    for name, rows in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        with path.open("w") as fh:
            for rec in rows:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        print(f"wrote {path}  {len(rows):,} rows")

    mix = {}
    for rec in train:
        mix[rec.source] = mix.get(rec.source, 0) + 1
    mcq = sum(1 for r in train if r.kind == "mcq")
    reasoning = sum(1 for r in train if r.source != "chatdoctor")

    report = {
        "train_size": len(train), "val_size": len(val),
        "holdout_pool": len(holdouts),
        "decontaminated_removed": removed, "decontaminated_from": before,
        "mix": mix,
        "mcq_fraction": round(mcq / len(train), 4) if train else 0.0,
        "reasoning_fraction": round(reasoning / len(train), 4) if train else 0.0,
        "seed": args.seed,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmix: {mix}")
    if train:
        print(f"multiple-choice {mcq / len(train):.0%}, "
              f"free-text {1 - mcq / len(train):.0%}")
        print(f"exam+reasoning {reasoning / len(train):.0%}, "
              f"patient dialogue {1 - reasoning / len(train):.0%}  "
              f"(spec target 70/30)")
    else:
        print("multiple-choice 0%, free-text 0%")
        print("exam+reasoning 0%, patient dialogue 0%  (spec target 70/30)")


if __name__ == "__main__":
    main()
