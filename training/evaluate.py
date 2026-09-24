from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

from training.evalcore import paired_report
from training.modeling import DEFAULT_BASE, load_model, release_memory
from training.prompts import LETTERS, extract_letter, render_inference_prompt
from training.records import Record

# Confirmed on a real Kaggle run. MedQA-USMLE test yields its full 1,273 rows
# under load()'s defaults. MedMCQA validation yields its full 4,183 only when
# loaded with require_rationale=False, require_single_choice=False -- the
# exact flags training/prepare_data.py used to decontaminate the training set
# against it. Loaded with the training defaults (require_single_choice=True)
# it drops the third of rows marked choice_type != "single" and comes out to
# roughly 2,858 instead. That file is built by a step outside this module (a
# notebook cell, not yet written), so this cannot force it to use the right
# flags -- but it can refuse to silently score whatever size shows up.
EXPECTED_HOLDOUT_SIZES = {"medqa": 1_273, "medmcqa": 4_183}


def letter_token_ids(tok) -> dict[str, int]:
    """Token id for each answer letter as it appears after 'Answer: '.

    The leading space matters: most BPE tokenizers give ' A' and 'A'
    different ids, and scoring the wrong one measures noise instead of the
    model's actual choice. The last token of the encoding is used rather
    than the first, in case a tokenizer splits ' <letter>' into more than
    one piece.
    """
    ids: dict[str, int] = {}
    for letter in LETTERS:
        encoded = tok.encode(f" {letter}", add_special_tokens=False)
        ids[letter] = encoded[-1]
    return ids


def pick_from_logits(logits_row, letter_ids: dict[str, int]) -> str:
    """Argmax restricted to the four answer letters. Ties go to the earliest.

    Refuses non-finite scores. float16 overflow on a T4 turns logits into inf
    or nan, and argmax over those silently returns the first letter for every
    question -- a plausible-looking accuracy that measures nothing.
    """
    scores = {letter: float(logits_row[letter_ids[letter]]) for letter in LETTERS}
    if not all(math.isfinite(s) for s in scores.values()):
        raise FloatingPointError(f"non-finite letter logits {scores}")
    best, best_score = None, None
    for letter in LETTERS:
        if best_score is None or scores[letter] > best_score:
            best, best_score = letter, scores[letter]
    return best


def check_holdout_size(recs: list[Record]) -> None:
    """Guard against the held-out set silently diverging from what training
    decontaminated against.

    If the flags used to build a `--test` file ever drift from
    prepare_data.py's (e.g. MedMCQA loaded with require_single_choice=True),
    the benchmark reported here quietly stops being the one training was
    cleaned against, and the accuracy numbers become meaningless without any
    error to say so. This counts rows by `source` and compares against the
    two confirmed sizes; unrecognised sources (ad hoc smoke-test fixtures)
    are left alone.
    """
    counts: dict[str, int] = {}
    for rec in recs:
        counts[rec.source] = counts.get(rec.source, 0) + 1
    mismatches = [
        f"{source} {counts[source]:,} (expected {expected:,})"
        for source, expected in EXPECTED_HOLDOUT_SIZES.items()
        if source in counts and counts[source] != expected
    ]
    if mismatches:
        raise SystemExit(
            "\nSTOP. Held-out set size does not match the benchmark training "
            "was decontaminated against: " + "; ".join(mismatches) + ".\n"
            "FIX: rebuild --test with training.sources.load(name, limit=0, "
            "split=..., require_rationale=False, require_single_choice=False) "
            "for MedMCQA validation; MedQA test takes load()'s defaults."
        )


def select_scorable_records(recs: list[Record]) -> list[Record]:
    """Keep only MCQ records, and require every one to carry a real answer.

    A free-text (dialogue) record has no letter to score against and would
    silently compare `None` to `""` in evalcore.accuracy, counting as wrong
    for both models without ever being flagged. An MCQ record with a missing
    or invalid answer is the same failure from the other side, so it must
    never reach paired_report either.
    """
    mcq = [r for r in recs if r.kind == "mcq"]
    bad = [r.id for r in mcq if r.answer not in LETTERS]
    if bad:
        shown = bad[:10]
        suffix = "..." if len(bad) > 10 else ""
        raise SystemExit(
            f"\nSTOP. {len(bad)} MCQ record(s) have no valid answer letter: "
            f"{shown}{suffix}.\n"
            "FIX: drop or correct these before scoring; left in, they would "
            "silently count as wrong for both models."
        )
    return mcq


def require_aligned(recs: list[Record], base_preds: list, tuned_preds: list,
                     mode: str) -> None:
    """evalcore's functions use zip(), which truncates silently on a length
    mismatch. Refuse to call them with predictions that do not line up with
    the records they were scored against.
    """
    if not (len(base_preds) == len(tuned_preds) == len(recs)):
        raise SystemExit(
            f"\nSTOP. {mode}: mismatched lengths going into paired_report -- "
            f"recs {len(recs):,}, base_preds {len(base_preds):,}, "
            f"tuned_preds {len(tuned_preds):,}.\n"
            "FIX: investigate the scoring loop; zip() would otherwise "
            "silently truncate to the shortest and under-report n."
        )


def batched(items: list, size: int) -> list[list]:
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _require_left_padding(tok) -> None:
    if getattr(tok, "padding_side", "left") != "left":
        raise SystemExit(
            "\nSTOP. The tokenizer pads on the right. Batched scoring reads the "
            "final position, which right padding fills with a pad token for "
            "every shorter sequence.\nFIX: load through training.modeling."
            "load_model, which sets padding_side='left'.")


def score_constrained(model, tok, recs: list[Record], *, batch_size: int = 16
                      ) -> list[str]:
    """One forward pass per batch; compare the four letter logits.

    Position ids are derived from the attention mask so a left-padded
    sequence sees the same positions it would see alone, and only the last
    position's logits are materialised -- the full vocabulary at every
    position would be 4GB per batch of 16 on a T4.
    """
    _require_left_padding(tok)
    import torch

    ids = letter_token_ids(tok)
    out: list[str] = []
    for group in batched(recs, batch_size):
        texts = [render_inference_prompt(tok, r, answer_prefix=True) for r in group]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        positions = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            last = model(**enc, position_ids=positions,
                         logits_to_keep=1).logits[:, -1, :]
        for row in last:
            try:
                out.append(pick_from_logits(row, ids))
            except FloatingPointError as exc:
                raise SystemExit(
                    f"\nSTOP. {exc}.\nHalf precision overflowed, so every "
                    "prediction from here on would be meaningless.\n"
                    "FIX: re-run with --device cpu, which runs in float32.")
    return out


def score_generative(model, tok, recs: list[Record], *, batch_size: int = 16,
                     max_new_tokens: int = 512) -> tuple[list[str | None], list[str]]:
    """Greedy decode, then extract the stated letter. Returns letters and texts.

    With left padding every prompt in a batch ends at the same column, so
    each completion is everything after that column.
    """
    _require_left_padding(tok)
    import torch

    letters: list[str | None] = []
    texts: list[str] = []
    for group in batched(recs, batch_size):
        prompts = [render_inference_prompt(tok, r) for r in group]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for row in generated:
            text = tok.decode(row[width:], skip_special_tokens=True)
            texts.append(text)
            letters.append(extract_letter(text))
    return letters, texts


def project_eval_seconds(timing: dict, *, n_constrained: int,
                         n_generative: int) -> int:
    """Scale a smoke pass's measured per-model rates to a full run of both models."""
    per_model = (n_constrained * timing["constrained_s_per_example"]
                 + n_generative * timing["generative_s_per_example"])
    return int(2 * per_model)


def main() -> None:
    p = argparse.ArgumentParser(description="Score base against tuned, paired.")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--base", default=DEFAULT_BASE, help="full-precision base weights")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gen-limit", type=int, default=300,
                   help="how many examples also get generative scoring")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = p.parse_args()

    recs = [Record.from_dict(json.loads(line))
            for line in args.test.read_text().splitlines() if line]
    check_holdout_size(recs)
    recs = select_scorable_records(recs)
    if args.limit:
        recs = recs[: args.limit]
    gen_recs = recs[: args.gen_limit]
    print(f"scoring {len(recs):,} constrained, {len(gen_recs):,} generative")

    started = time.time()
    model, tok, choice = load_model(args.base, args.adapter, device=args.device)
    load_s = time.time() - started
    print(f"loaded {args.base} + adapter on {choice.device} "
          f"({choice.dtype_name}) in {load_s:.0f}s", flush=True)

    preds: dict[str, dict] = {"constrained": {}, "generative": {}}
    completions: dict[str, list[str]] = {}
    spent = {"constrained": 0.0, "generative": 0.0}
    # One load, two models. The base is the same weights with the adapter
    # switched off, which guarantees base and tuned share identical base
    # weights and removes the second load that could run a T4 out of memory.
    for tag in ("base", "tuned"):
        ctx = model.disable_adapter() if tag == "base" else contextlib.nullcontext()
        with ctx:
            t = time.time()
            preds["constrained"][tag] = score_constrained(
                model, tok, recs, batch_size=args.batch_size)
            spent["constrained"] += time.time() - t
            t = time.time()
            letters, texts = score_generative(
                model, tok, gen_recs, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens)
            spent["generative"] += time.time() - t
        preds["generative"][tag] = letters
        completions[tag] = texts
        print(f"  {tag}: scored", flush=True)

    del model
    release_memory(choice.device)

    require_aligned(recs, preds["constrained"]["base"],
                    preds["constrained"]["tuned"], "constrained")
    require_aligned(gen_recs, preds["generative"]["base"],
                    preds["generative"]["tuned"], "generative")

    reports = {
        "constrained": paired_report(
            recs, preds["constrained"]["base"], preds["constrained"]["tuned"],
            mode="constrained"),
        "generative": paired_report(
            gen_recs, preds["generative"]["base"], preds["generative"]["tuned"],
            mode="generative"),
    }
    timing = {
        "load_s": round(load_s, 1),
        "constrained_s_per_example": round(
            spent["constrained"] / (2 * max(len(recs), 1)), 4),
        "generative_s_per_example": round(
            spent["generative"] / (2 * max(len(gen_recs), 1)), 4),
    }
    samples = [
        {"id": r.id, "answer": r.answer,
         "base": completions["base"][i], "tuned": completions["tuned"][i]}
        for i, r in enumerate(gen_recs[:12])
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "test_set": str(args.test), "base": args.base, "adapter": args.adapter,
        "device": choice.device, "timing": timing, "reports": reports,
        "samples": samples,
    }, indent=2))

    for mode, rep in reports.items():
        m = rep["mcnemar"]
        print(f"\n=== {mode}  n={rep['n']:,} ===")
        print(f"  base  {rep['base_accuracy']:.1%}   "
              f"tuned {rep['tuned_accuracy']:.1%}   "
              f"change {(rep['tuned_accuracy'] - rep['base_accuracy']) * 100:+.1f}")
        print(f"  wins {m['wins']}  regressions {m['regressions']}  "
              f"p = {m['p_value']:.4f}")
        print(f"  unparseable: base {rep['base_unparseable']}, "
              f"tuned {rep['tuned_unparseable']}")
    print(f"\ntiming {timing}\nwrote {args.out}")


if __name__ == "__main__":
    main()
