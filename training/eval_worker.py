from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from training.evaluate import resolve_eos_ids, sha256_of_file
from training.format_v2 import (final_letter, forced_suffix, option_letters,
                                render_letter_prompt, render_reasoning_prompt)
from training.modeling import DEFAULT_BASE, load_model
from training.records import Record
from training.sources_v2 import EVAL_SIZES

# Letter choice on everything first (fast, comparable with run 1), then
# reasoning in priority order: a deadline cuts from the end.
STAGES: tuple[tuple[str, str], ...] = (
    ("letter", "medqa"), ("letter", "medmcqa"), ("letter", "pubmedqa"),
    ("letter", "mmlu_medical"),
    ("reasoning", "medqa"), ("reasoning", "mmlu_medical"),
    ("reasoning", "pubmedqa"), ("reasoning", "medmcqa"),
)
REASONING_SIZES = {"medqa": 0, "mmlu_medical": 0, "pubmedqa": 500, "medmcqa": 1000}
# Qwen's recommended thinking-mode sampling. Greedy decoding in thinking mode
# degrades answers and loops, per both Qwen and Unsloth.
SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
ALL_LETTERS = tuple("ABCDEFGHIJ")
# Seconds per batch before the first one is measured.
FIRST_GUESS = {"letter": 15.0, "reasoning": 150.0}
# Forced-answer prompts scored per forward pass.
FORCE_BATCH = 2


def parse_stages(text: str) -> list[tuple[str, str]]:
    if not text:
        return list(STAGES)
    out = []
    for item in text.split(","):
        mode, _, bench = item.strip().partition(":")
        if (mode, bench) not in STAGES:
            raise SystemExit(f"\nSTOP. Unknown stage {item!r}; known: "
                             f"{[f'{m}:{b}' for m, b in STAGES]}")
        out.append((mode, bench))
    return out


def stage_order(recs: list[Record], mode: str, bench: str, seed: int) -> list[Record]:
    """The questions a stage scores, in scoring order. Reasoning stages use a
    fixed seeded shuffle, so a prefix cut by the deadline is a random sample,
    and both models see the same questions in the same order."""
    if mode == "letter":
        return list(recs)
    order = list(recs)
    random.Random(f"{seed}-{bench}").shuffle(order)
    size = REASONING_SIZES.get(bench, 0)
    return order[:size] if size else order


def fits_before(deadline: float, now: float, est_seconds: float) -> bool:
    return now + est_seconds <= deadline


def load_benchmark(path: Path, bench: str, *, check_size: bool = True) -> list[Record]:
    recs = [Record.from_dict(json.loads(line))
            for line in path.read_text().splitlines() if line]
    if check_size and len(recs) != EVAL_SIZES[bench]:
        raise SystemExit(f"\nSTOP. {path} holds {len(recs):,} questions, expected "
                         f"{EVAL_SIZES[bench]:,}.\nFIX: rebuild data/v2 and re-upload it.")
    bad = [r.id for r in recs if r.kind != "mcq" or r.answer not in (r.options or {})]
    if bad:
        raise SystemExit(f"\nSTOP. {len(bad)} unscorable questions in {path}: {bad[:5]}")
    return recs


def letter_token_ids(tok) -> dict[str, int]:
    return {L: tok.encode(f" {L}", add_special_tokens=False)[-1] for L in ALL_LETTERS}


def pick_letter(row, ids: dict[str, int], letters: list[str]) -> str:
    """Argmax over this question's own letters; ties go to the earliest.
    Non-finite scores mean fp16 overflowed and the result must not be used."""
    scores = {L: float(row[ids[L]]) for L in letters}
    if not all(math.isfinite(s) for s in scores.values()):
        raise FloatingPointError(f"non-finite letter logits {scores}")
    return max(letters, key=lambda L: (scores[L], -letters.index(L)))


def letters_for_prompts(model, tok, prompts, letter_sets, ids) -> list[str]:
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    positions = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        last = model(**enc, position_ids=positions, logits_to_keep=1,
                     use_cache=False).logits[:, -1, :]
    return [pick_letter(row, ids, letters) for row, letters in zip(last, letter_sets)]


def reason_batch(model, tok, recs, *, ids, think_id, stop_ids, budget, seed) -> list[dict]:
    """Think, then answer. A completion that states no letter -- the budget
    ran out mid-thought, or it never wrote 'Answer: X' -- is budget-forced:
    the think block is closed if needed, 'Answer:' is appended, and the
    letter is read from the logits. Both models get the same treatment."""
    import torch

    prompts = [render_reasoning_prompt(tok, r) for r in recs]
    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=budget,
                             pad_token_id=tok.pad_token_id, **SAMPLING)
    width = enc["input_ids"].shape[1]
    rows, to_force = [], []
    for i, (rec, seq) in enumerate(zip(recs, out)):
        new = seq[width:].tolist()
        while new and new[-1] in stop_ids:
            new.pop()
        closed = think_id in new
        letter = None
        if closed:
            after = new[len(new) - new[::-1].index(think_id):]
            letter = final_letter(tok.decode(after, skip_special_tokens=True),
                                  option_letters(rec))
        text = tok.decode(new, skip_special_tokens=False)
        rows.append({"id": rec.id, "letter": letter, "closed": closed,
                     "forced": letter is None, "new_tokens": len(new), "text": text})
        if letter is None:
            to_force.append(i)
    # Forced prompts carry the whole thinking trace, up to ~2,500 tokens, and
    # a no-cache forward pass over sixteen of them ran a T4 out of memory in
    # run 2. Two at a time is cheap next to the generation itself.
    for group in chunks(to_force, FORCE_BATCH):
        forced = _split_on_oom(
            lambda g: letters_for_prompts(
                model, tok,
                [prompts[i] + rows[i]["text"] + forced_suffix(rows[i]["closed"]) for i in g],
                [option_letters(recs[i]) for i in g], ids),
            group)
        for i, letter in zip(group, forced):
            rows[i]["letter"] = letter
    return rows


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _split_on_oom(fn, group):
    """A long evaluation can fragment GPU memory; halve the batch and retry
    rather than lose the stage.

    The retry happens outside the except block. Inside it, the traceback
    still holds the failed attempt's tensors, so the memory it should free is
    not free -- which is how run 2's base worker failed at every smaller size
    down to one."""
    import gc

    import torch

    try:
        return fn(group)
    except torch.cuda.OutOfMemoryError as exc:
        message = str(exc)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(group) == 1:
        raise torch.cuda.OutOfMemoryError(message)
    mid = len(group) // 2
    return _split_on_oom(fn, group[:mid]) + _split_on_oom(fn, group[mid:])


def run_stage(model, tok, mode, recs, out_path: Path, *, batch_size, budget, seed,
              deadline, rates, ids, think_id, stop_ids) -> tuple[str, int]:
    written = 0
    with out_path.open("w") as fh:
        for start in range(0, len(recs), batch_size):
            group = recs[start:start + batch_size]
            if not fits_before(deadline, time.time(), rates.get(mode, FIRST_GUESS[mode])):
                return "deadline", written
            began = time.time()
            if mode == "letter":
                def fn(g):
                    letters = letters_for_prompts(
                        model, tok, [render_letter_prompt(tok, r) for r in g],
                        [option_letters(r) for r in g], ids)
                    return [{"id": r.id, "letter": L} for r, L in zip(g, letters)]
            else:
                def fn(g, _start=start):
                    return reason_batch(model, tok, g, ids=ids, think_id=think_id,
                                        stop_ids=stop_ids, budget=budget,
                                        seed=seed * 1_000_003 + _start)
            rows = _split_on_oom(fn, group)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            written += len(rows)
            took = time.time() - began
            rates[mode] = took if mode not in rates else 0.8 * rates[mode] + 0.2 * took
    return "done", written


def main() -> None:
    p = argparse.ArgumentParser(description="Score one model, stage by stage, until a deadline.")
    p.add_argument("--role", choices=("base", "tuned"), required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0: none")
    p.add_argument("--think-budget", type=int, default=1536)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--stages", default="")
    p.add_argument("--limit", type=int, default=0, help="questions per stage (smoke runs)")
    p.add_argument("--no-size-check", action="store_true")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = p.parse_args()

    import peft
    import torch
    import transformers

    if args.role == "tuned" and not args.adapter:
        raise SystemExit("\nSTOP. --role tuned needs --adapter.")
    adapter = args.adapter if args.role == "tuned" else None
    stages = parse_stages(args.stages)
    deadline = args.deadline if args.deadline > 0 else float("inf")
    out_dir = args.out_dir / args.role
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"role": args.role, "base": args.base, "adapter": adapter,
            "seed": args.seed, "think_budget": args.think_budget,
            "batch_size": args.batch_size, "deadline": args.deadline,
            "limit": args.limit, "started": time.time(), "status": "running",
            "stages": {}}

    def save_meta() -> None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    save_meta()
    began = time.time()
    model, tok, choice = load_model(args.base, adapter, device=args.device)
    meta["load_seconds"] = round(time.time() - began, 1)
    meta["environment"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "device": choice.device, "dtype": choice.dtype_name,
        "device_name": (torch.cuda.get_device_name(0) if choice.device == "cuda"
                        else choice.device)}
    if adapter:
        meta["adapter_sha256"] = sha256_of_file(Path(adapter) / "adapter_model.safetensors")
    ids = letter_token_ids(tok)
    think_id = tok.convert_tokens_to_ids("</think>")
    stop_ids = resolve_eos_ids(getattr(model.generation_config, "eos_token_id", None),
                               tok.eos_token_id)
    if tok.pad_token_id is not None:
        stop_ids.add(tok.pad_token_id)
    save_meta()

    rates: dict[str, float] = {}
    benches: dict[str, list[Record]] = {}
    try:
        for mode, bench in stages:
            if bench not in benches:
                benches[bench] = load_benchmark(args.eval_dir / f"eval_{bench}.jsonl", bench,
                                                check_size=not args.no_size_check)
            recs = stage_order(benches[bench], mode, bench, args.seed)
            if args.limit:
                recs = recs[: args.limit]
            key = f"{mode}:{bench}"
            began = time.time()
            status, written = run_stage(
                model, tok, mode, recs, out_dir / f"{mode}__{bench}.jsonl",
                batch_size=args.batch_size, budget=args.think_budget, seed=args.seed,
                deadline=deadline, rates=rates, ids=ids, think_id=think_id,
                stop_ids=stop_ids)
            meta["stages"][key] = {"planned": len(recs), "scored": written,
                                   "status": status,
                                   "seconds": round(time.time() - began, 1)}
            save_meta()
            print(f"{args.role} {key}: {written:,}/{len(recs):,} {status} "
                  f"in {time.time() - began:.0f}s", flush=True)
            if status == "deadline":
                meta["status"] = "deadline"
                break
        else:
            meta["status"] = "finished"
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["finished"] = time.time()
        save_meta()


if __name__ == "__main__":
    main()
