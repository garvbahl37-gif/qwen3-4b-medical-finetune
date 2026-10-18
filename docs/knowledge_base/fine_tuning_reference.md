# Fine-tuning reference

## The trainer this project uses ([code] `training/train.py`)

- Unsloth `FastLanguageModel` on `unsloth/Qwen3-4B-unsloth-bnb-4bit`
  (QLoRA, NF4), LoRA on all seven projections, `lora_alpha = rank`,
  dropout 0, gradient checkpointing "unsloth", `random_state=42`.
- TRL `SFTTrainer` with `dataset_text_field="text"`: rows are rendered by
  `format_v2.render_training_text`.
- Optimizer and schedule: adamw_8bit, cosine schedule, warmup ratio 0.03,
  weight decay 0.01, fp16 on the T4, seed 42.
- `--stop-after-seconds` stops at a step boundary and saves the adapter. The
  cosine schedule is then left unfinished.
- Checkpoints every `--save-steps` steps. A rerun resumes from `checkpoint-*`
  if one is present.

## Loss masking

| Fact | Evidence |
|---|---|
| Loss covers only the assistant turn, via Unsloth `train_on_responses_only(instruction_part="<\|im_start\|>user\n", response_part="<\|im_start\|>assistant\n")` | [code] |
| Run 3's log: "masking verified: 442 tokens masked, 59 trained", with training starting at `<think>` | [measured] `results/run3/kaggle_run3_output.log` |
| `verify_masking` stops the run if nothing is masked or nothing is trained | [code] |
| TRL's own `assistant_only_loss=True` needs a chat template with `{% generation %}` markers, which Qwen3's lacks; it is not used here | [docs] TRL v0.24.0 sft_trainer.md |
| TRL prompt-completion datasets compute loss on the completion only by default | [docs] TRL v0.24.0 |

Loss masking is not a suspect in run 3's regression: it is measured to work.

## What the evidence says about SFT on a reasoning model

- **Run 3** **[measured]**: QLoRA, rank 64, lr 1e-4, one epoch, on 13,617
  rows of reasoning written by other models (GPT-4, GPT-4o, DeepSeek-R1,
  unnamed LLMs). Thinking-mode MedQA fell from 69.7% to 60.2%, and median new
  tokens from 1,036 to 403. Letter mode rose 1.5 points (not significant).
- **Unsloth** **[docs]**: "Use 75% reasoning and 25% non-reasoning in your
  dataset to make the model retain its reasoning capabilities." Runs 2 and 3
  followed this, and it did not prevent the loss. The mix ratio is not what
  went wrong; the source of the reasoning is the suspect
  ([open_questions](open_questions.md) H1).
- **"SFT Memorizes, RL Generalizes"** (arXiv 2501.17161) **[paper]**: SFT
  tends to memorise and generalises worse out of distribution than RL. The
  tasks are synthetic, so this supports our result only indirectly.
- **"LoRA Learns Less and Forgets Less"** (arXiv 2405.09673) **[paper]**: LoRA
  keeps out-of-domain performance better than full fine-tuning. It does not
  show that LoRA preserves an RL-trained reasoning style; run 3 suggests it
  does not.
- **Self-training on verified own traces** (STaR 2203.14465, RFT 2308.01825,
  ReST-EM 2312.06585) **[paper]**: models improve on their own filtered
  reasoning. RFT finds more distinct reasoning paths help more. None of these
  papers tests medicine or a hybrid-thinking model. This is the basis for E2.
- **Rule-based RL with multiple-choice rewards** (AlphaMed 2505.17952)
  **[paper]**: GRPO with a binary reward on MedQA and MedMCQA train lifted
  Llama-3.1-8B MedQA from 58.7 to 76.2. It used 8 A800 GPUs, full-parameter
  training, and a non-thinking backbone. Feasibility on a T4 is **[unknown]**.

## E2's settings and why ([code] `experiments_config.E2`)

| Setting | Value | Reason |
|---|---|---|
| Data | Base model's own completions, gold-verified, one random kept sample per question | Keeps the model's style and length. Random choice avoids a short-thinking bias. |
| Generation budget | 4,096 | A trace cut at the budget is rejected; a tight budget would select short thinking. m1 puts the medical optimum near 4K [paper]. |
| Rank, lr | 16, 5e-5 | Smaller update than run 3 (64, 1e-4). On-policy data needs only sharpening. **[hypothesis]** |
| Answer-only rows | 20% | Keeps letter-choice behaviour (the Unsloth mixing advice) |
| Batch | 1 × 16 accumulation | Rows up to 4,096 tokens on a T4 |

## Checklist before any training run

1. The data's rendered lengths are below `--max-seq` (`self_traces` drops
   longer rows; `prepare_data_v2` drops over-length rows).
2. `require_reasoning_rendered` and `verify_masking` pass. Both stop the run
   if not.
3. A dev-split comparison against the base is scheduled in the same session,
   or the next one.
4. The experiment's hypothesis and acceptance criteria are written down
   first ([open_questions](open_questions.md)).
