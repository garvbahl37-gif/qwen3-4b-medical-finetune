from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from training.eval_worker import STAGES, load_benchmark, stage_order
from training.evalcore import paired_report
from training.records import Record


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {row["id"]: row for row in rows}


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pair(recs: list[Record], base_rows: dict, tuned_rows: dict):
    kept = [r for r in recs if r.id in base_rows and r.id in tuned_rows]
    return (kept, [base_rows[r.id]["letter"] for r in kept],
            [tuned_rows[r.id]["letter"] for r in kept])


def reasoning_counts(rows: dict, kept: list[Record]) -> dict:
    picked = [rows[r.id] for r in kept]
    tokens = [row.get("new_tokens", 0) for row in picked]
    return {"forced": sum(bool(row.get("forced")) for row in picked),
            "closed": sum(bool(row.get("closed")) for row in picked),
            "median_new_tokens": statistics.median(tokens) if tokens else 0}


def build_report(eval_dir: Path, bench_dir: Path) -> dict:
    """Pair the two workers stage by stage, on the questions both scored."""
    metas = {role: read_meta(eval_dir / role / "meta.json") for role in ("base", "tuned")}
    seed = metas["tuned"].get("seed", metas["base"].get("seed", 1234))
    limit = metas["tuned"].get("limit") or metas["base"].get("limit") or 0
    benches: dict[str, list[Record]] = {}
    stages: dict[str, dict] = {}
    pooled_in: dict[str, tuple[list, list, list]] = {"letter": ([], [], []),
                                                     "reasoning": ([], [], [])}
    for mode, bench in STAGES:
        base_rows = read_rows(eval_dir / "base" / f"{mode}__{bench}.jsonl")
        tuned_rows = read_rows(eval_dir / "tuned" / f"{mode}__{bench}.jsonl")
        if not base_rows and not tuned_rows:
            continue
        if bench not in benches:
            benches[bench] = load_benchmark(bench_dir / f"eval_{bench}.jsonl", bench,
                                            check_size=False)
        planned = stage_order(benches[bench], mode, bench, seed)
        if limit:
            planned = planned[:limit]
        kept, base_letters, tuned_letters = pair(planned, base_rows, tuned_rows)
        rep = paired_report(kept, base_letters, tuned_letters, mode=mode)
        rep.update({"benchmark": bench, "planned": len(planned), "scored": len(kept)})
        if mode == "reasoning":
            rep["base_counts"] = reasoning_counts(base_rows, kept)
            rep["tuned_counts"] = reasoning_counts(tuned_rows, kept)
        stages[f"{mode}:{bench}"] = rep
        for bucket, values in zip(pooled_in[mode], (kept, base_letters, tuned_letters)):
            bucket.extend(values)
    pooled = {}
    for mode, (kept, base_letters, tuned_letters) in pooled_in.items():
        if kept:
            rep = paired_report(kept, base_letters, tuned_letters, mode=mode)
            rep.pop("by_subject", None)
            pooled[mode] = rep
    return {"stages": stages, "pooled": pooled, "models": metas}


def format_table(report: dict) -> str:
    lines = [f"{'stage':<24}{'scored':>13}{'base':>8}{'tuned':>8}{'change':>8}"
             f"{'p':>10}  95% CI of the change"]
    rows = list(report["stages"].items())
    rows += [(f"pooled {mode}", rep) for mode, rep in report["pooled"].items()]
    for key, rep in rows:
        ci = rep["diff_ci"]
        scored = f"{rep.get('scored', rep['n']):,}/{rep.get('planned', rep['n']):,}"
        change = (rep["tuned_accuracy"] - rep["base_accuracy"]) * 100
        lines.append(f"{key:<24}{scored:>13}{rep['base_accuracy']:>8.1%}"
                     f"{rep['tuned_accuracy']:>8.1%}{change:>+8.1f}"
                     f"{rep['mcnemar']['p_value']:>10.4f}"
                     f"  [{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}]")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Pair base and tuned predictions into a report.")
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = build_report(args.eval_dir, args.bench_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(format_table(report))
    for role, meta in report["models"].items():
        print(f"{role}: {meta.get('status', 'missing')}", meta.get("error", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
