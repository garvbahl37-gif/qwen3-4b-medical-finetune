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
    """
    params = inspect.signature(func).parameters
    for name in candidates:
        if name in params:
            return name
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates[0]
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

    train_ds = build_dataset(args.data / "train.jsonl", tok)
    val_ds = build_dataset(args.data / "val.jsonl", tok)
    print(f"train {len(train_ds):,}   val {len(val_ds):,}   max_seq {args.max_seq}")

    class Probe(TrainerCallback):
        """Abort at --probe-steps if the run will not fit the budget."""

        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, cfg, state, control, **kw):
            if state.global_step != args.probe_steps:
                return
            proj = project(state.global_step, int(time.time() - self.started),
                           int(state.max_steps))
            print(f"\nprobe: {proj['seconds_per_step']:.2f}s/step, "
                  f"projected {proj['projected_seconds'] / 3600:.2f}h "
                  f"for {int(state.max_steps):,} steps", flush=True)
            message = check(proj, args.budget_seconds)
            if message:
                raise SystemExit(message)
            print("projection fits the budget; continuing\n", flush=True)

    # TRL has renamed both of these between versions; pick whichever name
    # the installed SFTTrainer/SFTConfig actually accepts rather than
    # hardcoding one. See pick_kwarg's docstring.
    tokenizer_kwarg = pick_kwarg(SFTTrainer.__init__, "processing_class", "tokenizer")
    max_seq_kwarg = pick_kwarg(SFTConfig.__init__, "max_seq_length", "max_length")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(
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
        ),
        callbacks=[Probe()],
        **{tokenizer_kwarg: tok},
    )

    # Mask the prompt so gradient is spent only on the response.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_MARKER, response_part=RESPONSE_MARKER)

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
