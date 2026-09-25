from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from training import sources
from training import sources_v2 as sv2
from training.check_lengths import percentile, summarise
from training.decontam import Deduper, EvalIndex
from training.format_v2 import v2_messages
from training.records import Record


@dataclass(frozen=True)
class MixEntry:
    name: str
    kind: str | None
    target: int

    @property
    def label(self) -> str:
        return f"{self.name}:{self.kind}" if self.kind else self.name


# 75% reasoning (8,250) and 25% direct (2,750): Unsloth's and Qwen's guidance
# for keeping Qwen3's thinking. 11,000 rows at ~925 tokens is about 4.5 hours
# at run 1's measured T4 throughput.
MIX: tuple[MixEntry, ...] = (
    MixEntry("medreason", "mcq", 2200),
    MixEntry("medreason", "dialogue", 1100),
    MixEntry("r1_distill", None, 1650),
    MixEntry("ultramedical", None, 1650),
    MixEntry("medical_o1", None, 1100),
    MixEntry("reasonmed", None, 550),
    MixEntry("medmcqa", None, 800),
    MixEntry("medqa", None, 550),
    MixEntry("pubmedqa_artificial", None, 750),
    MixEntry("medical_o1_direct", None, 650),
)

# name: (hf_id, config, split, streaming, reversed order). The big three are
# streamed through a shuffle buffer rather than downloaded whole. The direct
# medical-o1 slice reads the same shuffled order from the other end, so it
# never meets the reasoning slice's rows.
HF: dict[str, tuple[str, str | None, str, bool, bool]] = {
    "medreason": ("UCSC-VLAA/MedReason", None, "train", False, False),
    "r1_distill": ("FreedomIntelligence/Medical-R1-Distill-Data", "en", "train", False, False),
    "ultramedical": ("TsinghuaC3I/UltraMedical", None, "train", True, False),
    "medical_o1": ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", False, False),
    "medical_o1_direct": ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", False, True),
    "reasonmed": ("lingshu-medical-mllm/ReasonMed", None, "train", True, False),
    "pubmedqa_artificial": ("qiaojin/PubMedQA", "pqa_artificial", "train", True, False),
    "medmcqa": ("openlifescienceai/medmcqa", "default", "train", False, False),
    "medqa": ("GBaker/MedQA-USMLE-4-options", "default", "train", False, False),
}

NORMALISERS = {
    "medreason": sv2.norm_medreason,
    "r1_distill": sv2.norm_r1_distill,
    "ultramedical": sv2.norm_ultramedical,
    "medical_o1": sv2.norm_medical_o1_reasoning,
    "medical_o1_direct": sv2.norm_medical_o1_direct,
    "reasonmed": sv2.norm_reasonmed,
    "pubmedqa_artificial": partial(sv2.norm_pubmedqa, source="pubmedqa_artificial"),
    "medmcqa": sources.normalise_medmcqa,
    "medqa": sources.normalise_medqa,
}

STAT_KEYS = ("target", "seen", "unusable", "other_kind", "contaminated_exact",
             "contaminated_ngram", "duplicate", "too_long", "kept")


def collect(entry: MixEntry, rows, normalise, *, index: EvalIndex, deduper: Deduper,
            measure, cap: int, max_candidates: int):
    """Take rows in order until `entry.target` survive every filter.

    Filters run cheapest first, and every rejection is counted, so the data
    report shows exactly why a source came up short instead of hiding it.
    """
    stats = dict.fromkeys(STAT_KEYS, 0)
    stats["target"] = entry.target
    kept: list[tuple[Record, int]] = []
    for idx, raw in rows:
        if len(kept) >= entry.target or stats["seen"] >= max_candidates:
            break
        stats["seen"] += 1
        rec = normalise(raw, idx)
        if rec is None:
            stats["unusable"] += 1
            continue
        if entry.kind and rec.kind != entry.kind:
            stats["other_kind"] += 1
            continue
        hit = index.hits(rec.question)
        if hit:
            stats[f"contaminated_{hit}"] += 1
            continue
        if not deduper.first_time(rec.question):
            stats["duplicate"] += 1
            continue
        n_tokens = measure(rec)
        if n_tokens > cap:
            stats["too_long"] += 1
            continue
        kept.append((rec, n_tokens))
    stats["kept"] = len(kept)
    return kept, stats


def choose_max_seq(lengths: list[int], *, cap: int) -> int:
    return min(cap, 64 * math.ceil(percentile(lengths, 0.99) / 64))


def summarise_mix(recs: list[Record]) -> dict:
    n = len(recs) or 1
    by_source: dict[str, int] = {}
    for r in recs:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return {"by_source": dict(sorted(by_source.items())),
            "reasoning_fraction": round(sum(bool(r.reasoning) for r in recs) / n, 4),
            "mcq_fraction": round(sum(r.kind == "mcq" for r in recs) / n, 4)}


def iter_rows(name: str, *, seed: int):
    from datasets import load_dataset  # network; imported only when building

    hf_id, config, split, streaming, reverse = HF[name]
    if streaming:
        ds = load_dataset(hf_id, config, split=split, streaming=True)
        yield from enumerate(ds.shuffle(seed=seed, buffer_size=10_000))
        return
    ds = load_dataset(hf_id, config, split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    if reverse:
        order.reverse()
    for i in order:
        yield i, ds[i]


def load_eval_sets() -> dict[str, list[Record]]:
    from datasets import load_dataset

    sets = {
        "medqa": sources.load("medqa", limit=0, split="test"),
        "medmcqa": sources.load("medmcqa", limit=0, split="validation",
                                require_rationale=False, require_single_choice=False),
        "pubmedqa": [r for i, row in enumerate(
                         load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"))
                     if (r := sv2.norm_pubmedqa(row, i))],
        "mmlu_medical": [r for subject in sv2.MMLU_MEDICAL
                         for i, row in enumerate(load_dataset("cais/mmlu", subject, split="test"))
                         if (r := sv2.norm_mmlu(row, i))],
    }
    wrong = {k: len(v) for k, v in sets.items() if len(v) != sv2.EVAL_SIZES[k]}
    if wrong:
        raise SystemExit(f"\nSTOP. Benchmark sizes {wrong} differ from {sv2.EVAL_SIZES}.\n"
                         "FIX: a dataset on the Hub has changed; check its revision.")
    return sets


def write_jsonl(path: Path, recs: list[Record]) -> None:
    with path.open("w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec.to_dict()) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build run 2's training set and benchmark files.")
    p.add_argument("--out", type=Path, default=Path("data/v2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="unsloth/Qwen3-4B")
    p.add_argument("--cap", type=int, default=3072, help="longest example kept, in tokens")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every target (dry runs)")
    p.add_argument("--candidates-factor", type=int, default=8)
    p.add_argument("--show", type=int, default=0, help="print N rendered examples per source")
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy; imported only when building

    args.out.mkdir(parents=True, exist_ok=True)
    print("Benchmarks first, so training can be cleaned against them.")
    evals = load_eval_sets()
    for name, recs in evals.items():
        write_jsonl(args.out / f"eval_{name}.jsonl", recs)
        print(f"  eval_{name}.jsonl  {len(recs):,}")
    index = EvalIndex([r.question for recs in evals.values() for r in recs])

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def measure(rec: Record) -> int:
        return len(tok.apply_chat_template(
            v2_messages(rec, with_answer=True), tokenize=True, return_dict=False,
            enable_thinking=bool(rec.reasoning)))

    deduper = Deduper()
    kept_all: list[tuple[Record, int]] = []
    report_sources: dict[str, dict] = {}
    print("\nTraining sources.")
    for entry in MIX:
        target = max(1, round(entry.target * args.scale))
        scaled = MixEntry(entry.name, entry.kind, target)
        kept, stats = collect(
            scaled, iter_rows(entry.name, seed=args.seed), NORMALISERS[entry.name],
            index=index, deduper=deduper, measure=measure, cap=args.cap,
            max_candidates=target * args.candidates_factor)
        report_sources[scaled.label] = stats
        short = "" if stats["kept"] >= target else f"   SHORT by {target - stats['kept']:,}"
        print(f"  {scaled.label:<22} kept {stats['kept']:>6,} of {stats['seen']:>7,} seen"
              f" (contaminated {stats['contaminated_exact'] + stats['contaminated_ngram']:,},"
              f" duplicate {stats['duplicate']:,}, too long {stats['too_long']:,}){short}")
        for rec, n_tokens in kept[: args.show]:
            text = tok.apply_chat_template(v2_messages(rec, with_answer=True), tokenize=False,
                                           enable_thinking=bool(rec.reasoning))
            print(f"\n----- {scaled.label} {rec.id} ({n_tokens} tokens) -----\n"
                  f"{text[:1500]}\n...\n{text[-400:]}")
        kept_all += kept

    lengths = [n for _, n in kept_all]
    max_seq = choose_max_seq(lengths, cap=args.cap)
    final = [rec for rec, n in kept_all if n <= max_seq]
    random.Random(args.seed).shuffle(final)
    write_jsonl(args.out / "train.jsonl", final)

    report = {
        "version": "run2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "tokenizer": args.tokenizer, "cap": args.cap,
        "scale": args.scale, "max_seq": max_seq,
        "length": summarise([n for _, n in kept_all if n <= max_seq]),
        "train_size": len(final),
        "dropped_over_max_seq": len(kept_all) - len(final),
        **summarise_mix(final),
        "sources": report_sources,
        "eval_sizes": {name: len(recs) for name, recs in evals.items()},
    }
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ntrain.jsonl {len(final):,} rows | max_seq {max_seq} | "
          f"reasoning {report['reasoning_fraction']:.0%} | "
          f"multiple choice {report['mcq_fraction']:.0%}")


if __name__ == "__main__":
    main()
