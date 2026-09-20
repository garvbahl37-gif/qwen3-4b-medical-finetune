from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from training.evalcore import paired_report
from training.prompts import LETTERS, build_messages, extract_letter
from training.records import Record

BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

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
    """Argmax restricted to the four answer letters. Ties go to the earliest."""
    best, best_score = None, None
    for letter in LETTERS:
        score = float(logits_row[letter_ids[letter]])
        if best_score is None or score > best_score:
            best, best_score = letter, score
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


def _load(adapter: str | None, max_seq: int):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter or BASE_MODEL,
        max_seq_length=max_seq, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def score_constrained(model, tok, recs: list[Record]) -> list[str]:
    import torch

    ids = letter_token_ids(tok)
    out: list[str] = []
    for rec in recs:
        msgs = build_messages(rec, with_answer=False)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True) + "Answer:"
        batch = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[0, -1]
        out.append(pick_from_logits(logits, ids))
    return out


def score_generative(model, tok, recs: list[Record], max_new_tokens: int = 320
                     ) -> list[str | None]:
    import torch

    out: list[str | None] = []
    for rec in recs:
        msgs = build_messages(rec, with_answer=False)
        batch = tok(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(gen[0][batch["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        out.append(extract_letter(completion))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Score base against tuned, paired.")
    p.add_argument("--adapter", required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seq", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gen-limit", type=int, default=300,
                   help="how many examples also get generative scoring")
    args = p.parse_args()

    recs = [Record.from_dict(json.loads(line))
            for line in args.test.read_text().splitlines() if line]
    check_holdout_size(recs)
    recs = select_scorable_records(recs)
    if args.limit:
        recs = recs[: args.limit]
    gen_recs = recs[: args.gen_limit]
    print(f"scoring {len(recs):,} constrained, {len(gen_recs):,} generative")

    preds: dict[str, dict] = {}
    for tag, adapter in (("base", None), ("tuned", args.adapter)):
        model, tok = _load(adapter, args.max_seq)
        print(f"  {tag}: constrained...", flush=True)
        preds.setdefault("constrained", {})[tag] = score_constrained(model, tok, recs)
        print(f"  {tag}: generative...", flush=True)
        preds.setdefault("generative", {})[tag] = score_generative(model, tok, gen_recs)

        # The model is loaded twice, sequentially, on a 15.6GB T4. `del` alone
        # leaves the CUDA allocator holding the freed blocks, and the second
        # from_pretrained can then OOM on a card that objectively has room --
        # which would waste the completed training run this is scoring.
        del model
        del tok
        gc.collect()
        import torch
        torch.cuda.empty_cache()

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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"test_set": str(args.test),
                                    "reports": reports}, indent=2))

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
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
