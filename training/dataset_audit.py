"""Audit run 2/3's training data source by source, before any cleaning.

    python -m training.dataset_audit --data data/v2 --out reports

Reads data/v2/train.jsonl (13,617 rows) and data_report.json, measures every
source independently, compares the style of its reasoning traces with
Qwen3-4B's own (taken from run 3's base-model predictions), and writes
reports/DATASET_AUDIT.md, reports/dataset_audit.json and
reports/dataset_audit_samples.md (rows drawn at random for a human to read).

It reports; it does not clean. Medical correctness of a trace cannot be
established by a script: the answer-agreement check only says whether the
trace's own conclusion matches the gold letter.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from training.check_lengths import percentile
from training.format_v2 import v2_messages
from training.prepare_data import normalise_question
from training.records import Record
from training.sources_v2 import stated_letter

# Who wrote each source's reasoning or answers, from its dataset card or paper
# (checked 2026-10-03). "verified" is what the publisher says, not what this
# audit can confirm.
GENERATORS = {
    "medreason": ("an LLM the card and paper (arXiv 2504.00993) do not name, writing "
                  "from knowledge-graph paths", "paths 'validated for consistency with "
                  "clinical logic'"),
    "r1_distill": ("DeepSeek-R1 (full)", "not stated: problems are verifiable, outputs "
                   "are not described as checked"),
    "ultramedical": ("GPT-4", "no; its `score` field is gpt-3.5-turbo's rating of the "
                     "question"),
    "medical_o1": ("GPT-4o", "yes, by a 'medical verifier' (card)"),
    "medical_o1_direct": ("GPT-4o (answer only)", "yes, by a 'medical verifier' (card)"),
    "reasonmed": ("three unnamed LLMs (card)", "multi-agent check of each trace (card)"),
    "medmcqa": ("human-written exam explanations", "gold letter from the exam"),
    "medqa": ("none: answer letter and option text only", "gold letter from the exam"),
    "pubmedqa_artificial": ("abstract authors' conclusion; the yes/no label is "
                            "generated heuristically by PubMedQA", "heuristic label"),
}

# Direct (no-reasoning) sources whose gold letter is an exam's answer key.
# PubMedQA's artificial labels are heuristic, so they do not count.
EXAM_GOLD = {"medqa", "medmcqa"}

_FIRST_PERSON = re.compile(r"^\s*(okay|ok|alright|so|hmm|let me|let's|i need|i'm|i am|"
                           r"first,? i|well|right)\b", re.IGNORECASE)
_SELF_CHECK = re.compile(r"\b(wait|hmm|let me (?:double[- ]check|think again|re-?check)|"
                         r"actually|on second thought)\b", re.IGNORECASE)
_MARKDOWN = re.compile(r"^\s*(#{1,6}\s|\*\*[^*]+\*\*\s*:?\s*$|\d+\.\s+\*\*)", re.MULTILINE)
_KG_STYLE = re.compile(r"finding reasoning paths|reasoning process:", re.IGNORECASE)
_AMBIGUOUS_OPTION = re.compile(r"\b(all of the above|none of the above|both|"
                               r"all of these|none of these)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_TERMINAL = ".?!)]\"'*`"


def length_stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": percentile(values, 0.5),
            "mean": round(statistics.fmean(values), 1),
            "p90": percentile(values, 0.9), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99), "max": max(values)}


_STATED = re.compile(r"[Aa]nswer is:?\s*\**\s*\(?([A-J])\b")
_CONCLUSION = re.compile(r"conclusion|final answer|in summary|therefore", re.IGNORECASE)
_STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "not", "all",
         "none", "above", "both", "these", "its", "into", "than", "which", "may", "can"}
# Phrases showing the writer was handed the answer before reasoning.
_ANSWER_GIVEN = re.compile(r"\b(provided|expected) answer\b|\banswer (provided|key)\b|"
                           r"\bgiven answer\b", re.IGNORECASE)
# Sources with no gold letter independent of the trace.
NO_GOLD = {"reasonmed"}
# Why agreement is not evidence for these sources.
CIRCULAR = {
    "reasonmed": "ReasonMed has no gold field: the build read the training letter from the "
                 "trace itself, so agreement is guaranteed and verifies nothing.",
    "ultramedical": "the build dropped rows whose stated letter differed from the dataset's "
                    "answer field (34 rows), so agreement is 100% by construction; it verifies "
                    "the trace against that field, which for synthetic questions is itself "
                    "model-written.",
}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOP}


def conclusion_of(text: str) -> str:
    """The trace's closing section: after its last 'conclusion'/'therefore'
    in the final 1,500 characters, else its last 400 characters."""
    tail = text[-1500:]
    found = list(_CONCLUSION.finditer(tail))
    return tail[found[-1].start():] if found else text[-400:]


def named_option(text: str, options: dict[str, str]) -> str | None:
    """The one option every content word of which (5-letter stems) appears
    in the trace's conclusion; the most specific wins a tie. None if no
    option or several equally specific ones are named."""
    words = _stems(conclusion_of(text))
    hits = {k: len(st) for k, v in options.items() if (st := _stems(v)) and st <= words}
    if not hits:
        return None
    best = max(hits.values())
    top = [k for k, n in hits.items() if n == best]
    return top[0] if len(top) == 1 else None


def answer_agreement(rec: Record) -> str:
    """Does the trace's own conclusion pick the gold option?

    'agree' / 'conflict': its last 'the answer is X' is / is not the gold
    letter. 'agree_text' / 'other_text': no letter stated; the option its
    conclusion names (`named_option`, a heuristic) is / is not the gold one.
    'unstated': neither. 'n/a': no gold letter or no reasoning."""
    if rec.kind != "mcq" or not rec.reasoning or not rec.options:
        return "n/a"
    letter = stated_letter(rec.reasoning)
    if letter in rec.options:
        return "agree" if letter == rec.answer else "conflict"
    named = named_option(rec.reasoning, rec.options)
    if named is None:
        return "unstated"
    return "agree_text" if named == rec.answer else "other_text"


def revised_answer(rec: Record) -> bool:
    """The trace states two different option letters as 'the answer'."""
    if not rec.reasoning or not rec.options:
        return False
    return len({x for x in _STATED.findall(rec.reasoning) if x in rec.options}) > 1


def degenerate_repetition(text: str, *, min_line: int = 30, times: int = 3) -> bool:
    """A substantial line repeated `times` or more: a looping generation."""
    lines = Counter(line.strip() for line in (text or "").splitlines()
                    if len(line.strip()) >= min_line)
    return any(count >= times for count in lines.values())


def unterminated(text: str) -> bool:
    """Ends mid-sentence: a sign the trace was cut off where it was produced."""
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] not in _TERMINAL


def style_features(text: str) -> dict:
    text = text or ""
    words = max(1, len(text.split()))
    return {"first_person_open": bool(_FIRST_PERSON.match(text)),
            "markdown": bool(_MARKDOWN.search(text)),
            "kg_template": bool(_KG_STYLE.search(text)),
            "self_checks_per_1k_words": round(1000 * len(_SELF_CHECK.findall(text)) / words, 2)}


def malformed_reasons(rec: Record) -> list[str]:
    reasons = []
    if not (rec.question or "").strip():
        reasons.append("empty question")
    if rec.kind == "mcq":
        if not rec.options:
            reasons.append("mcq without options")
        elif rec.answer not in rec.options:
            reasons.append("answer not among options")
        else:
            texts = [normalise_question(v) for v in rec.options.values()]
            if len(set(texts)) < len(texts):
                reasons.append("duplicate option text")
    elif not (rec.response or "").strip():
        reasons.append("empty response")
    if rec.reasoning is not None and not rec.reasoning.strip():
        reasons.append("empty reasoning")
    return reasons


def suspected_bad_row(rec: Record) -> bool:
    """A row a cleaning pass would look at first: its trace contradicts its
    gold letter, it is malformed, it loops, or it nests think tags."""
    trace = rec.reasoning or ""
    return (answer_agreement(rec) == "conflict" or bool(malformed_reasons(rec))
            or degenerate_repetition(trace) or "<think>" in trace or "</think>" in trace)


def ambiguous_options(rec: Record) -> bool:
    return bool(rec.options) and any(_AMBIGUOUS_OPTION.search(v) for v in rec.options.values())


def shingles(text: str, n: int) -> set[int]:
    words = _WORD.findall((text or "").lower())
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def near_duplicate_pairs(texts: list[str], *, n: int = 5, threshold: float = 0.8,
                         max_postings: int = 50) -> list[tuple[int, int, float]]:
    """Pairs of texts whose word n-gram sets have Jaccard >= threshold.

    An inverted index over n-grams finds candidates; n-grams shared by more
    than `max_postings` texts are boilerplate and are not used to propose
    pairs (they still count in the Jaccard)."""
    sets = [shingles(t, n) for t in texts]
    postings: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for g in s:
            postings[g].append(i)
    pairs = []
    for i, s in enumerate(sets):
        if not s:
            continue
        seen: Counter[int] = Counter()
        for g in s:
            ids = postings[g]
            if len(ids) > max_postings:
                continue
            for j in ids:
                if j > i:
                    seen[j] += 1
        for j in seen:
            jac = len(s & sets[j]) / len(s | sets[j])
            if jac >= threshold:
                pairs.append((i, j, round(jac, 3)))
    return pairs


def label_of(rec: Record) -> str:
    if rec.source == "medreason":
        return f"medreason:{rec.kind}"
    return rec.source


def audit(recs: list[Record], *, token_len, reasoning_len, max_seq: int,
          base_traces: list[str]) -> dict:
    """Every per-source measurement. `token_len(rec)` and `reasoning_len(text)`
    count tokens; they are passed in so tests need no tokenizer."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(recs):
        by_label[label_of(rec)].append(i)

    q_norm = [normalise_question(r.question) for r in recs]
    q_counts = Counter(q_norm)
    resp_counts = Counter(normalise_question(r.response) for r in recs if r.response)
    q_pairs = near_duplicate_pairs([r.question for r in recs], n=5, threshold=0.8)
    reasoning_idx = [i for i, r in enumerate(recs) if r.reasoning]
    t_pairs = near_duplicate_pairs([recs[i].reasoning[:3000] for i in reasoning_idx],
                                   n=8, threshold=0.6)
    near_q = defaultdict(int)
    for i, j, _ in q_pairs:
        near_q[i] += 1
        near_q[j] += 1
    near_t = defaultdict(int)
    cross_source_traces = 0
    for a, b, _ in t_pairs:
        i, j = reasoning_idx[a], reasoning_idx[b]
        near_t[i] += 1
        near_t[j] += 1
        if label_of(recs[i]) != label_of(recs[j]):
            cross_source_traces += 1

    lengths = [token_len(r) for r in recs]
    sources = {}
    for label, idx in sorted(by_label.items()):
        rows = [recs[i] for i in idx]
        reasoning_rows = [r for r in rows if r.reasoning]
        agreement = Counter(answer_agreement(r) for r in rows)
        traces = [r.reasoning for r in reasoning_rows]
        t_tokens = [reasoning_len(t) for t in traces]
        styles = [style_features(t) for t in traces]
        suspicious = Counter()
        for r in reasoning_rows:
            if "<think>" in r.reasoning or "</think>" in r.reasoning:
                suspicious["think tags inside the trace"] += 1
            if _ANSWER_GIVEN.search(r.reasoning):
                suspicious["refers to a provided answer"] += 1
            if degenerate_repetition(r.reasoning):
                suspicious["repeated lines (looping)"] += 1
            if unterminated(r.reasoning):
                suspicious["trace ends mid-sentence"] += 1
            if len(r.reasoning) < 200:
                suspicious["trace under 200 characters"] += 1
        for r in rows:
            text = (r.reasoning or "") + (r.response or "")
            if text and sum(not ch.isascii() for ch in text) / len(text) > 0.03:
                suspicious["over 3% non-ASCII"] += 1
            if re.search(r"\bas an ai\b", text, re.IGNORECASE):
                suspicious["'as an AI'"] += 1
        for r in reasoning_rows:
            if revised_answer(r):
                suspicious["states two different answers (self-correction?)"] += 1
        malformed = Counter(m for r in rows for m in malformed_reasons(r))
        if label in NO_GOLD:
            verified = 0
        elif reasoning_rows:
            verified = agreement["agree"] + agreement["agree_text"]
        elif label in EXAM_GOLD:
            verified = sum(1 for r in rows if r.kind == "mcq" and r.answer in (r.options or {}))
        else:
            verified = 0
        suspected_bad = sum(1 for r in rows if suspected_bad_row(r))
        sources[label] = {
            "rows": len(rows),
            "reasoning_rows": len(reasoning_rows),
            "mcq_rows": sum(1 for r in rows if r.kind == "mcq"),
            "generator": GENERATORS.get(label.split(":")[0], ("?", "?"))[0],
            "publisher_verification": GENERATORS.get(label.split(":")[0], ("?", "?"))[1],
            "answer_agreement": dict(agreement),
            "answer_verified": verified,
            "suspected_bad": suspected_bad,
            "exact_duplicate_questions": sum(1 for i in idx if q_counts[q_norm[i]] > 1),
            "duplicate_responses": sum(1 for r in rows if r.response
                                       and resp_counts[normalise_question(r.response)] > 1),
            "near_duplicate_questions": sum(1 for i in idx if near_q[i]),
            "near_duplicate_traces": sum(1 for i in idx if near_t[i]),
            "tokens": length_stats([lengths[i] for i in idx]),
            "reasoning_tokens": length_stats(t_tokens),
            "truncated_at_max_seq": sum(1 for i in idx if lengths[i] > max_seq),
            "ambiguous_options": sum(1 for r in rows if ambiguous_options(r)),
            "malformed": dict(malformed),
            "suspicious": dict(suspicious),
            "style": {
                "first_person_open": _share(styles, "first_person_open"),
                "markdown": _share(styles, "markdown"),
                "kg_template": _share(styles, "kg_template"),
                "self_checks_per_1k_words": round(statistics.fmean(
                    s["self_checks_per_1k_words"] for s in styles), 2) if styles else None,
            },
        }
    base_styles = [style_features(t) for t in base_traces]
    base = {"traces": len(base_traces),
            "reasoning_tokens": length_stats([reasoning_len(t) for t in base_traces]),
            "style": {k: _share(base_styles, k) for k in
                      ("first_person_open", "markdown", "kg_template")}
            | {"self_checks_per_1k_words": round(statistics.fmean(
                s["self_checks_per_1k_words"] for s in base_styles), 2) if base_styles else None}}
    return {"rows": len(recs), "max_seq": max_seq, "sources": sources,
            "near_duplicate_question_pairs": len(q_pairs),
            "near_duplicate_trace_pairs": len(t_pairs),
            "cross_source_trace_pairs": cross_source_traces,
            "qwen3_base_traces": base}


def _share(items: list[dict], key: str) -> float | None:
    return round(sum(bool(x[key]) for x in items) / len(items), 3) if items else None


def base_think_text(text: str) -> str:
    """The thinking part of a run 3 base-model completion."""
    text = text or ""
    head = text.split("</think>")[0]
    return head.replace("<think>", "").strip()


def render_report(result: dict, build_report: dict, notes: str = "") -> str:
    s = result["sources"]
    lines = [
        "# Dataset audit: run 2/3 training data (`data/v2/train.jsonl`)",
        "",
        "Generated by `python -m training.dataset_audit`. Measurements only; no row "
        "was changed. A script cannot judge whether a trace's medicine is right: "
        "*answer agreement* below only checks whether the trace's own conclusion "
        "names the gold option. Medical correctness needs the human review in "
        "`reports/dataset_audit_samples.md`.",
        "",
        f"{result['rows']:,} rows; training `max_seq` {result['max_seq']}.",
        "",
        "## Summary",
        "",
        "| source | rows | reasoning_rows | answer_verified | suspected_bad | duplicates | truncated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, v in s.items():
        dup = v["exact_duplicate_questions"] + v["near_duplicate_questions"]
        lines.append(f"| {label} | {v['rows']:,} | {v['reasoning_rows']:,} | "
                     f"{v['answer_verified']:,} | {v['suspected_bad']:,} | {dup:,} | "
                     f"{v['truncated_at_max_seq']:,} |")
    lines += [
        "",
        "*answer_verified*: reasoning rows whose trace concludes with the gold option, by "
        "its stated letter or, where it states none (every MedReason trace), by the option "
        "its conclusion names (a heuristic, see below); ReasonMed shows 0 because it has no "
        "gold independent of the trace; for MedQA and MedMCQA direct rows, every row whose target is the exam's answer "
        "key. Free-text rows and PubMedQA's heuristic labels have no gold answer to check, "
        "so they show 0. *suspected_bad*: letter conflicts, malformed rows, looping traces "
        "and traces containing think tags. *duplicates*: exact plus near "
        "(word 5-gram Jaccard >= 0.8) duplicate questions within the training set. "
        "*truncated*: rendered length above `max_seq`, cut at training time.",
        "",
        "## Who wrote the reasoning",
        "",
        "| source | generator | publisher's verification |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['generator']} | {v['publisher_verification']} |" for k, v in s.items()]
    base = result["qwen3_base_traces"]
    lines += [
        "",
        "None of the training reasoning was written by Qwen3.",
        "",
        "## Reasoning style against Qwen3-4B's own",
        "",
        f"Qwen3-4B's own traces: {base['traces']:,} thinking blocks from run 3's base model "
        "on MedQA (thinking mode, temperature 0.6, 1,536-token budget, so the longest are "
        "cut).",
        "",
        "| source | traces | median tokens | p90 | opens in first person | markdown | KG template | self-checks / 1k words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| **Qwen3-4B base** | {base['traces']:,} | {base['reasoning_tokens'].get('median', '-')} | "
        f"{base['reasoning_tokens'].get('p90', '-')} | {base['style']['first_person_open']:.0%} | "
        f"{base['style']['markdown']:.0%} | {base['style']['kg_template']:.0%} | "
        f"{base['style']['self_checks_per_1k_words']} |",
    ]
    for k, v in s.items():
        if not v["reasoning_rows"]:
            continue
        st = v["style"]
        lines.append(f"| {k} | {v['reasoning_rows']:,} | {v['reasoning_tokens']['median']} | "
                     f"{v['reasoning_tokens']['p90']} | {st['first_person_open']:.0%} | "
                     f"{st['markdown']:.0%} | {st['kg_template']:.0%} | "
                     f"{st['self_checks_per_1k_words']} |")
    lines += ["", "## Answer agreement of reasoning traces", "",
              "| source | agree (letter) | agree (text) | conflict (letter) | other option named (text) | unstated | states two answers |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in s.items():
        a = v["answer_agreement"]
        if v["reasoning_rows"] and v["mcq_rows"]:
            two = v["suspicious"].get("states two different answers (self-correction?)", 0)
            lines.append(f"| {k} | {a.get('agree', 0):,} | {a.get('agree_text', 0):,} | "
                         f"{a.get('conflict', 0):,} | {a.get('other_text', 0):,} | "
                         f"{a.get('unstated', 0):,} | {two:,} |")
    lines += ["", "*letter*: the trace's last 'the answer is X'. *text*: no letter stated; "
              "the one option all of whose content words appear in the trace's conclusion "
              "(a heuristic: paraphrases such as 'cerclage' for 'encirclage' are missed, and "
              "'not X' questions can match the wrong way). Only letter conflicts count as "
              "suspected bad. *states two answers*: 'the answer is X' for two different "
              "letters (judged on the last).", ""]
    lines += [f"- **{k}**: {why}" for k, why in CIRCULAR.items() if k in s]
    lines += ["", "## Lengths (rendered training tokens)", "",
              "| source | min | median | mean | p90 | p95 | p99 | max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in s.items():
        t = v["tokens"]
        lines.append(f"| {k} | {t['min']} | {t['median']} | {t['mean']} | {t['p90']} | "
                     f"{t['p95']} | {t['p99']} | {t['max']} |")
    lines += ["", "## Duplicates", "",
              f"- near-duplicate question pairs (Jaccard >= 0.8): "
              f"{result['near_duplicate_question_pairs']:,}",
              f"- near-duplicate trace pairs (word 8-gram Jaccard >= 0.6 on the first 3,000 "
              f"characters): {result['near_duplicate_trace_pairs']:,}, of which "
              f"{result['cross_source_trace_pairs']:,} cross sources",
              "", "| source | exact dup questions | near dup questions | duplicate responses | near dup traces |",
              "|---|---:|---:|---:|---:|"]
    for k, v in s.items():
        lines.append(f"| {k} | {v['exact_duplicate_questions']:,} | "
                     f"{v['near_duplicate_questions']:,} | {v['duplicate_responses']:,} | "
                     f"{v['near_duplicate_traces']:,} |")
    lines += ["", "## Malformed, suspicious and ambiguous rows", ""]
    for k, v in s.items():
        issues = {**{f"malformed: {m}": c for m, c in v["malformed"].items()},
                  **v["suspicious"]}
        if v["ambiguous_options"]:
            issues["options like 'all/none of the above'"] = v["ambiguous_options"]
        if issues:
            lines.append(f"- **{k}**: " + "; ".join(f"{m} {c:,}" for m, c in issues.items()))
    lines += ["", "## Filtering before these rows were kept", "",
              "From `data/v2/data_report.json` (the build that produced this file):", "",
              "| source | seen | unusable | contaminated | duplicate | too long | kept |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in build_report.get("sources", {}).items():
        lines.append(f"| {k} | {v['seen']:,} | {v['unusable']:,} | "
                     f"{v['contaminated_exact'] + v['contaminated_ngram']:,} | "
                     f"{v['duplicate']:,} | {v['too_long']:,} | {v['kept']:,} |")
    lines += ["", f"Rows over the {build_report.get('cap', '?'):,}-token cap were dropped "
              f"while sampling (*too long*), and {build_report.get('dropped_over_max_seq', 0):,} "
              f"more over `max_seq` ({build_report.get('max_seq')}) afterwards. Long traces were "
              "filtered out, not cut, so the kept reasoning is biased towards short traces. "
              "The *truncated* column re-renders every row in run 3's format (which adds the "
              "/think or /no_think switch) to check nothing now exceeds `max_seq`."]
    if notes.strip():
        lines += ["", "## Reviewer notes (hand-written, from `reports/dataset_audit_notes.md`)",
                  "", notes.strip()]
    return "\n".join(lines) + "\n"


def render_samples(recs: list[Record], *, per_source: int = 3, seed: int = 7) -> str:
    rng = random.Random(seed)
    by_label: dict[str, list[Record]] = defaultdict(list)
    for r in recs:
        by_label[label_of(r)].append(r)
    out = ["# Dataset audit: rows for human review", "",
           f"{per_source} rows per source, drawn at random (seed {seed}). Read each "
           "trace for medical correctness; the script cannot.", ""]
    for label, rows in sorted(by_label.items()):
        out.append(f"## {label}")
        for r in rng.sample(rows, min(per_source, len(rows))):
            out += ["", f"### {r.id} — answer agreement: {answer_agreement(r)}", "",
                    "**Question**", "", "> " + (r.question[:900]).replace("\n", "\n> ")]
            if r.options:
                out += [""] + [f"- {k}. {v}" for k, v in sorted(r.options.items())]
                out += ["", f"**Gold:** {r.answer}"]
            if r.reasoning:
                head, tail = r.reasoning[:1200], r.reasoning[-500:]
                out += ["", "**Reasoning (start)**", "", "```", head, "```"]
                if len(r.reasoning) > 1700:
                    out += ["", "**Reasoning (end)**", "", "```", tail, "```"]
            if r.response:
                out += ["", "**Response**", "", "> " + r.response[:900].replace("\n", "\n> ")]
            if r.rationale and r.kind == "mcq":
                out += ["", "**Explanation**", "", "> " + r.rationale[:500].replace("\n", "\n> ")]
        out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the run 2/3 training data per source.")
    p.add_argument("--data", type=Path, default=Path("data/v2"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--base-traces", type=Path,
                   default=Path("results/run3/predictions/base/reasoning__medqa.jsonl"))
    p.add_argument("--tokenizer", default="unsloth/Qwen3-4B")
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy; imported only when auditing

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    recs = [Record.from_dict(json.loads(line))
            for line in (args.data / "train.jsonl").read_text().splitlines() if line]
    build_report = json.loads((args.data / "data_report.json").read_text())
    base_traces = [base_think_text(json.loads(line)["text"])
                   for line in args.base_traces.read_text().splitlines() if line]

    def token_len(rec: Record) -> int:
        return len(tok.apply_chat_template(v2_messages(rec, with_answer=True), tokenize=True,
                                           return_dict=False, enable_thinking=bool(rec.reasoning)))

    def reasoning_len(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    result = audit(recs, token_len=token_len, reasoning_len=reasoning_len,
                   max_seq=build_report["max_seq"], base_traces=base_traces)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_audit.json").write_text(json.dumps(result, indent=2))
    notes_path = args.out / "dataset_audit_notes.md"
    notes = notes_path.read_text() if notes_path.exists() else ""
    (args.out / "DATASET_AUDIT.md").write_text(render_report(result, build_report, notes))
    (args.out / "dataset_audit_samples.md").write_text(render_samples(recs))
    print(f"wrote {args.out}/DATASET_AUDIT.md, dataset_audit.json, dataset_audit_samples.md")


if __name__ == "__main__":
    main()
