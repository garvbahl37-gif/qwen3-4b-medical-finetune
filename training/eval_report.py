from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from training.check_lengths import percentile
from training.eval_worker import STAGES, load_benchmark, stage_order
from training.evalcore import paired_report
from training.records import Record


def read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def read_rows(path: Path) -> dict[str, dict]:
    """Each question's first sample. Rows written before samples existed
    carry no 'sample' field and are sample 0."""
    return {row["id"]: row for row in read_all(path) if row.get("sample", 0) == 0}


def read_samples(path: Path) -> dict[str, list[dict]]:
    """Every sample of each question, in sample order."""
    out: dict[str, list[dict]] = {}
    for row in sorted(read_all(path), key=lambda r: r.get("sample", 0)):
        out.setdefault(row["id"], []).append(row)
    return out


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
    unpaired: list[str] = []
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
        if not kept:
            # One model reached this stage before the deadline and the other
            # did not: there is nothing to compare, and "0.0% vs 0.0%" would
            # read like a result.
            unpaired.append(f"{mode}:{bench}")
            continue
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
    return {"stages": stages, "pooled": pooled, "unpaired": unpaired, "models": metas}


def stage_files(dirs: list[Path]) -> dict[str, Path]:
    """'mode:bench' -> the one file holding it across a model's directories
    (a model can be scored over several sessions, one set of stages each)."""
    found: dict[str, Path] = {}
    for d in dirs:
        for mode, bench in STAGES:
            path = d / f"{mode}__{bench}.jsonl"
            if path.exists() and path.stat().st_size:
                key = f"{mode}:{bench}"
                if key in found:
                    raise SystemExit(f"\nSTOP. {key} is in both {found[key]} and {path}.")
                found[key] = path
    return found


def model_summary(recs: list[Record], firsts: dict, samples: dict, mode: str) -> dict:
    """One model's results on the questions it scored (first sample)."""
    gold = {r.id: r.answer for r in recs}
    scored = [qid for qid in gold if qid in firsts]
    correct = [firsts[q]["letter"] == gold[q] for q in scored]
    out = {"n": len(scored),
           "accuracy": sum(correct) / len(scored) if scored else None,
           "unparseable": sum(firsts[q]["letter"] is None for q in scored)}
    if mode == "reasoning":
        rows = [firsts[q] for q in scored]
        think = [r["think_tokens"] for r in rows if "think_tokens" in r]
        new = [r.get("new_tokens", 0) for r in rows]
        out.update({
            "forced": sum(bool(r.get("forced")) for r in rows),
            "closed": sum(bool(r.get("closed")) for r in rows),
            "median_new_tokens": percentile(new, 0.5) if new else None,
            # think_tokens exists from E0 on; earlier runs recorded only
            # new_tokens (thinking plus answer).
            "think_tokens": ({"median": percentile(think, 0.5), "mean": round(sum(think) / len(think), 1),
                              "p90": percentile(think, 0.9), "p95": percentile(think, 0.95)}
                             if think else None),
        })
        multi = [q for q in scored if len(samples.get(q, [])) > 1]
        if multi:
            k = min(len(samples[q]) for q in multi)
            full = [q for q in multi if len(samples[q]) >= k]
            out["any_correct"] = {
                "samples": k, "n": len(full),
                "accuracy": sum(any(s["letter"] == gold[q] for s in samples[q][:k])
                                for q in full) / len(full),
                "first_sample_accuracy": sum(samples[q][0]["letter"] == gold[q]
                                             for q in full) / len(full)}
    return out


def build_comparison(models: dict[str, list[Path]], bench_dir: Path, *,
                     baseline: str = "base") -> dict:
    """Any number of models, stage by stage.

    For each stage: every model's own result; the accuracy of each on the
    questions all of them scored (`common`); and each model paired with the
    baseline on the questions both scored, first sample only, with McNemar
    and a 95% interval for the change."""
    files = {name: stage_files(dirs) for name, dirs in models.items()}
    metas = {name: [read_meta(d / "meta.json") for d in dirs] for name, dirs in models.items()}
    benches: dict[str, list[Record]] = {}
    stages: dict[str, dict] = {}
    for mode, bench in STAGES:
        key = f"{mode}:{bench}"
        have = [name for name in models if key in files[name]]
        if not have:
            continue
        if bench not in benches:
            benches[bench] = load_benchmark(bench_dir / f"eval_{bench}.jsonl", bench,
                                            check_size=False)
        recs = benches[bench]
        firsts = {name: read_rows(files[name][key]) for name in have}
        samples = {name: read_samples(files[name][key]) for name in have}
        stage = {"models": {name: model_summary(recs, firsts[name], samples[name], mode)
                            for name in have}}
        common = [r for r in recs if all(r.id in firsts[name] for name in have)]
        stage["common"] = {"n": len(common), "accuracy": {
            name: (sum(firsts[name][r.id]["letter"] == r.answer for r in common) / len(common)
                   if common else None) for name in have}}
        if baseline in have:
            stage["vs_baseline"] = {}
            for name in have:
                if name == baseline:
                    continue
                kept, base_letters, other_letters = pair(recs, firsts[baseline], firsts[name])
                if kept:
                    rep = paired_report(kept, base_letters, other_letters, mode=mode)
                    rep.pop("by_subject", None)
                    stage["vs_baseline"][name] = rep
        stages[key] = stage
    return {"baseline": baseline, "stages": stages,
            "models": {name: {"dirs": [str(d) for d in models[name]], "metas": metas[name]}
                       for name in models}}


def export_rows(models: dict[str, list[Path]], bench_dir: Path):
    """One record per model, question and sample: the per-question table."""
    benches: dict[str, dict[str, Record]] = {}
    for name, dirs in models.items():
        for key, path in stage_files(dirs).items():
            mode, bench = key.split(":")
            if bench not in benches:
                benches[bench] = {r.id: r for r in load_benchmark(
                    bench_dir / f"eval_{bench}.jsonl", bench, check_size=False)}
            for row in read_all(path):
                rec = benches[bench][row["id"]]
                yield {"question_id": row["id"], "benchmark": bench, "mode": mode,
                       "model": name, "sample": row.get("sample", 0),
                       "gold_answer": rec.answer, "prediction": row["letter"],
                       "correct": row["letter"] == rec.answer,
                       "thinking_length": row.get("think_tokens"),
                       "new_tokens": row.get("new_tokens"),
                       "forced": row.get("forced"), "closed": row.get("closed"),
                       "subject": rec.subject}


def format_comparison(report: dict) -> str:
    base = report["baseline"]
    lines = [f"{'stage':<24}{'model':<14}{'n':>6}{'acc':>8}{'any':>8}{'think med':>11}"
             f"{'vs ' + base:>10}{'p':>9}  95% CI"]
    for key, stage in report["stages"].items():
        for name, m in stage["models"].items():
            anyc = m.get("any_correct")
            think = (m.get("think_tokens") or {}).get("median")
            cmp = stage.get("vs_baseline", {}).get(name)
            acc = f"{m['accuracy']:.1%}" if m["accuracy"] is not None else "-"
            any_acc = f"{anyc['accuracy']:.1%}" if anyc else "-"
            think_med = str(think) if think is not None else "-"
            line = f"{key:<24}{name:<14}{m['n']:>6,}{acc:>8}{any_acc:>8}{think_med:>11}"
            if cmp:
                ci = cmp["diff_ci"]
                change = (cmp["tuned_accuracy"] - cmp["base_accuracy"]) * 100
                line += (f"{change:>+10.1f}{cmp['mcnemar']['p_value']:>9.4f}"
                         f"  [{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}] on {cmp['n']:,}")
            lines.append(line)
    return "\n".join(lines)


def parse_models(items: list[str]) -> dict[str, list[Path]]:
    """['base=a,b', 'run3=c'] -> {'base': [a, b], 'run3': [c]}."""
    out = {}
    for item in items:
        name, _, dirs = item.partition("=")
        if not name or not dirs or name in out:
            raise SystemExit(f"\nSTOP. --model {item!r}: expected a new name=dir[,dir...].")
        out[name] = [Path(d) for d in dirs.split(",")]
    return out


def discover_models(roots: list[Path]) -> dict[str, list[Path]]:
    """Every worker directory under the roots (one holding meta.json),
    grouped by model name: <root>/.../gpu1/base/ belongs to 'base'."""
    out: dict[str, list[Path]] = {}
    for root in roots:
        for meta in sorted(root.rglob("meta.json")):
            out.setdefault(meta.parent.name, []).append(meta.parent)
    if not out:
        raise SystemExit(f"\nSTOP. No worker output (meta.json) under {[str(r) for r in roots]}.")
    return out


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
    p.add_argument("--eval-dir", type=Path, help="run 2/3 layout: base/ and tuned/ inside")
    p.add_argument("--model", action="append", default=[],
                   help="name=dir[,dir...]; repeat per model (E0 onwards)")
    p.add_argument("--root", action="append", default=[], type=Path,
                   help="find every model directory under here; repeatable")
    p.add_argument("--baseline", default="base")
    p.add_argument("--export", type=Path, help="write the per-question records here (JSONL)")
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if sum(map(bool, (args.eval_dir, args.model, args.root))) != 1:
        raise SystemExit("\nSTOP. Give --eval-dir, or --model per model, or --root.")
    if args.model or args.root:
        models = parse_models(args.model) if args.model else discover_models(args.root)
        report = build_comparison(models, args.bench_dir, baseline=args.baseline)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(format_comparison(report))
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            with args.export.open("w") as fh:
                for row in export_rows(models, args.bench_dir):
                    fh.write(json.dumps(row) + "\n")
            print(f"wrote {args.export}")
        print(f"\nwrote {args.out}")
        return
    report = build_report(args.eval_dir, args.bench_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(format_table(report))
    if report["unpaired"]:
        print("reached by one model only, not compared:", ", ".join(report["unpaired"]))
    for role, meta in report["models"].items():
        print(f"{role}: {meta.get('status', 'missing')}", meta.get("error", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
