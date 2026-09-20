from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

from training.budget import check, project
from training.prompts import build_messages
from training.records import Record

BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

# Qwen3's chat template renders the assistant turn after this marker. Loss is
# masked to everything after it, so the model is never scored on reproducing
# the clinical vignette it was given.
RESPONSE_MARKER = "<|im_start|>assistant\n"
INSTRUCTION_MARKER = "<|im_start|>user\n"


def pick_kwarg(func, *candidates: str) -> str:
    """Return the first name in `candidates` that `func`'s signature accepts.

    TRL renamed SFTTrainer's `tokenizer=` to `processing_class=`, and moved
    SFTConfig's `max_seq_length` too, at some point between versions this
    laptop has no TRL install to check. Kaggle installs whatever is current
    when the notebook runs, so this inspects the signature that is actually
    installed there instead of hardcoding a guess -- a wrong guess would
    only surface after the queue wait and the install, burning a session.

    Raises TypeError immediately, with the candidates and the real
    parameter names, if none match -- loud and in the first second, rather
    than a cryptic error from deep inside TRL after the model has loaded.

    There is deliberately no `**kwargs` catch-all fallback: this exists
    precisely to catch TRL renaming a field to a name not in `candidates`,
    and a catch-all would swallow that exact case silently -- the value
    would be accepted by the call and consumed by nothing, which is a
    silent wrong-training bug worse than the one this function prevents.
    """
    params = inspect.signature(func).parameters
    for name in candidates:
        if name in params:
            return name
    raise TypeError(
        f"none of {candidates!r} are accepted by {func!r}; "
        f"it takes {sorted(params)!r}"
    )


def build_dataset(path: Path, tok):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    texts = [
        tok.apply_chat_template(
            build_messages(Record.from_dict(row), with_answer=True), tokenize=False)
        for row in rows
    ]
    return Dataset.from_dict({"text": texts})


def verify_masking(trainer, tok) -> None:
    """Prove the completion-only mask did something before spending six hours.

    Unsloth matches the markers against tokenised subsequences. If that match
    fails it masks nothing, silently: training still runs, still finishes inside
    budget, and still produces an adapter -- just a worse one, with nothing
    anywhere to say why.
    """
    try:
        rows = [trainer.train_dataset[i] for i in range(2)]
        labels = trainer.data_collator(rows)["labels"][0]
        masked = int((labels == -100).sum())
        trained = int((labels != -100).sum())
    except Exception as exc:
        raise SystemExit(
            f"\nSTOP. Could not verify completion-only masking: "
            f"{type(exc).__name__}: {exc}\n"
            "FIX: investigate rather than skipping -- this check is the only "
            "evidence the mask works.")

    if masked == 0 or trained == 0:
        raise SystemExit(
            f"\nSTOP. Completion-only masking did nothing usable: "
            f"{masked} masked, {trained} trained.\n"
            "The markers did not match the tokenised sequence, so the model "
            "would train on the prompt as well as the response.\n"
            f"FIX: print tok.apply_chat_template(...) for one example and "
            f"correct RESPONSE_MARKER / INSTRUCTION_MARKER.")

    preview = tok.decode([t for t in labels.tolist() if t != -100][:40])
    print(f"masking verified: {masked:,} tokens masked, {trained:,} trained")
    print(f"  first trained tokens: {preview!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="QLoRA fine-tune on the medical mix.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seq", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--budget-seconds", type=int, default=21600)
    p.add_argument("--probe-steps", type=int, default=50)
    args = p.parse_args()

    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # TRL has renamed both of these between versions; pick whichever name the
    # installed SFTTrainer/SFTConfig actually accepts now, before spending
    # minutes on the model download and dataset tokenisation below. A third,
    # unrecognised rename must fail here, loudly, not after all that work.
    tokenizer_kwarg = pick_kwarg(SFTTrainer.__init__, "processing_class", "tokenizer")
    max_seq_kwarg = pick_kwarg(SFTConfig.__init__, "max_seq_length", "max_length")

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=args.max_seq,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # Evaluation now runs in its own, separate Kaggle notebook, and no
    # eval_strategy is set here, so a val split would be tokenised and never
    # consulted. Only the training set is built.
    train_ds = build_dataset(args.data / "train.jsonl", tok)
    print(f"train {len(train_ds):,}   max_seq {args.max_seq}")

    class Probe(TrainerCallback):
        """Abort at --probe-steps if the run will not fit the budget."""

        def __init__(self) -> None:
            self.started = time.time()
            self.fired = False

        def on_step_end(self, cfg, state, control, **kw):
            # >= rather than ==, with a one-shot guard: a run with fewer
            # total steps than --probe-steps would never hit the equality,
            # and would then train with no budget check at all while
            # appearing to be guarded.
            if self.fired or state.global_step < args.probe_steps:
                return
            self.fired = True
            proj = project(state.global_step, int(time.time() - self.started),
                           int(state.max_steps))
            print(f"\nprobe: {proj['seconds_per_step']:.2f}s/step, "
                  f"projected {proj['projected_seconds'] / 3600:.2f}h "
                  f"for {int(state.max_steps):,} steps", flush=True)
            message = check(proj, args.budget_seconds)
            if message:
                raise SystemExit(message)
            print("projection fits the budget; continuing\n", flush=True)

    # Build the config first, then force the EOS token onto the instance.
    # Signature inspection is not reliable here: Unsloth patches SFTConfig, and
    # inspecting a patched class describes the patch rather than the original.
    # An earlier fix resolved '<|im_end|>' correctly and still passed TRL the
    # unresolved placeholder '<EOS_TOKEN>', because the signature guard
    # silently evaluated False against the patched __init__.
    config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=42,
        output_dir=str(args.out),
        report_to="none",
        **{max_seq_kwarg: args.max_seq},
    )

    eos = tok.eos_token
    current = getattr(config, "eos_token", None)
    if eos and current != eos:
        try:
            config.eos_token = eos
        except Exception as exc:                      # frozen or property-backed
            raise SystemExit(
                f"\nSTOP. Could not set SFTConfig.eos_token: "
                f"{type(exc).__name__}: {exc}\n"
                f"Tokenizer EOS is {eos!r}; the config holds {current!r}.\n"
                "FIX: TRL rejects a config whose eos_token is not in the "
                "vocabulary, so this must be corrected before training.")

    print(f"SFTConfig.eos_token = {getattr(config, 'eos_token', '<absent>')!r} "
          f"(tokenizer EOS {eos!r}, id {tok.eos_token_id})")

    # A silent no-op is exactly what cost the earlier run: the assignment
    # above looked like it worked and TRL still saw the stale placeholder.
    # Verify the instance actually holds what was just assigned to it.
    if eos and getattr(config, "eos_token", None) != eos:
        raise SystemExit(
            f"\nSTOP. SFTConfig.eos_token is still "
            f"{getattr(config, 'eos_token', None)!r} after assignment, "
            f"not {eos!r}.\nFIX: TRL will reject this at trainer construction; "
            "the assignment is being overridden somewhere.")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        callbacks=[Probe()],
        **{tokenizer_kwarg: tok},
    )

    # Mask the prompt so gradient is spent only on the response.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_MARKER, response_part=RESPONSE_MARKER)

    verify_masking(trainer, tok)

    resume = any(args.out.glob("checkpoint-*")) if args.out.exists() else False
    if resume:
        print(f"resuming from a checkpoint in {args.out}")
    stats = trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "train_stats.json").write_text(json.dumps({
        "train_runtime_seconds": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
        "examples": len(train_ds),
        "max_seq": args.max_seq,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
    }, indent=2))
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
