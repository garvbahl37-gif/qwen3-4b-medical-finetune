from __future__ import annotations

from math import comb

from training.records import Record


def accuracy(preds: list[str | None], golds: list[str]) -> float:
    """An unparseable prediction is wrong, not excluded.

    Dropping them would flatter whichever model rambles more, which is
    usually the base model.
    """
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def mcnemar_exact(base: list[bool], tuned: list[bool]) -> dict:
    """Two-sided exact McNemar on paired outcomes. Standard library only."""
    both_correct = sum(b and t for b, t in zip(base, tuned))
    both_wrong = sum((not b) and (not t) for b, t in zip(base, tuned))
    regressions = sum(b and not t for b, t in zip(base, tuned))
    wins = sum((not b) and t for b, t in zip(base, tuned))

    n = wins + regressions
    if n == 0:
        p = 1.0
    else:
        k = min(wins, regressions)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)

    return {
        "wins": wins, "regressions": regressions,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "discordant": n, "p_value": round(p, 6),
    }


def by_subject(recs: list[Record], base: list[bool], tuned: list[bool]) -> dict:
    buckets: dict[str, dict] = {}
    for rec, b, t in zip(recs, base, tuned):
        key = rec.subject or "unknown"
        bucket = buckets.setdefault(key, {"n": 0, "base_n": 0, "tuned_n": 0})
        bucket["n"] += 1
        bucket["base_n"] += int(b)
        bucket["tuned_n"] += int(t)
    for bucket in buckets.values():
        bucket["base"] = bucket.pop("base_n") / bucket["n"]
        bucket["tuned"] = bucket.pop("tuned_n") / bucket["n"]
    return dict(sorted(buckets.items(), key=lambda kv: -kv[1]["n"]))


def paired_report(recs: list[Record], base_preds: list[str | None],
                  tuned_preds: list[str | None], *, mode: str) -> dict:
    golds = [r.answer or "" for r in recs]
    base_ok = [p == g for p, g in zip(base_preds, golds)]
    tuned_ok = [p == g for p, g in zip(tuned_preds, golds)]
    return {
        "mode": mode,
        "n": len(recs),
        "base_accuracy": accuracy(base_preds, golds),
        "tuned_accuracy": accuracy(tuned_preds, golds),
        "base_unparseable": sum(p is None for p in base_preds),
        "tuned_unparseable": sum(p is None for p in tuned_preds),
        "mcnemar": mcnemar_exact(base_ok, tuned_ok),
        "by_subject": by_subject(recs, base_ok, tuned_ok),
    }
