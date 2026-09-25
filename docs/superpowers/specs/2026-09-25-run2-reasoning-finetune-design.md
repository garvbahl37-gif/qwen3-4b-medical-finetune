# Run 2: reasoning fine-tune of Qwen3-4B — design

Date: 2026-09-25. Direction approved in chat: approach A (reasoning distillation
SFT) in one hands-off Kaggle session. Supersedes the training half of
`2026-09-20-medical-llm-finetune-design.md` for run 2; run 1's code stays as it
is, so its results remain reproducible.

## Why run 1 fell short

From `results/run1/`:

1. **It was taught not to reason.** Every example was rendered with an empty
   `<think>` block, and MedQA targets were `Answer: X` alone. Reasoning is where
   the base model is strongest: written answers 68.7% (base) vs 62.3% (tuned) on
   MedQA.
2. **Rationale-first answers were cut off.** 70 of the fine-tune's 300 MedMCQA
   written answers hit the 512-token cap before a letter.
3. **Noisy data.** 30% ChatDoctor. Training loss flattened by step 150 of 541.
4. **Against the model makers' guidance.** Unsloth and Qwen advise keeping about
   75% reasoning data to preserve Qwen3's thinking, and never decoding greedily
   in thinking mode.

Letter-choice accuracy did rise (+2.1 MedQA, n.s.; +1.6 MedMCQA, p = 0.018), so
the model did learn some medicine; the format undid it.

## Goal and success criteria

- **Primary:** in reasoning mode, the fine-tune beats base Qwen3-4B with McNemar
  p < 0.05 on at least two of the four benchmarks, and on their pooled questions.
- **Secondary:** letter-choice accuracy beats run 1's fine-tune on MedQA and
  MedMCQA.
- **Always:** the result is recorded as measured, including a loss.

## Constraints

- Kaggle T4 (compute 7.5: fp16 only, no FlashAttention 2), a 12-hour session,
  about 30 GPU-hours a week. One push; the session must finish without the
  laptop.
- Same base model, Qwen3-4B, so run 1 and run 2 compare.
- Datasets with an explicit Apache-2.0 or MIT licence only.
  `II-Medical-Reasoning-SFT` and `m23k` are excluded: no licence stated.
- At most two agents at a time during implementation.

## Architecture

```
this Mac                                   Kaggle (one session, T4 x2)
────────────────────────────────           ─────────────────────────────────────────
prepare_data_v2  ─► data/v2/               cell 1  hardware: two T4s
  9 sources          train.jsonl           cell 2  install (pinned) + verify
  filters            val.jsonl             cell 3  code + data: find, fingerprint
  decontamination    eval/*.jsonl          cell 4  train (GPU 0, time-guarded)
  lengths            data_report.json      cell 5  save adapter, loss curve
        │                                  cell 6  eval smoke + projection
        ▼                                  cell 7  eval stages (GPU 0 fine-tune,
push_kaggle.sh ─► medical-ft-code (code)           GPU 1 base, in parallel)
                  medical-ft-data (data)   cell 8  summary ─► Output panel
```

Data is prepared on this Mac and uploaded frozen, with a content fingerprint the
notebook checks. The notebook downloads only the two model checkpoints.

## Data

About 10,000 examples: 75% reasoning, 25% direct. The count comes from run 1's
measured throughput (about 700 padded tokens a second at batch 8) and an average
of about 1,100 tokens per example, for roughly 5 hours of training inside a
6-hour cap.

| Source (Hugging Face id) | Licence | Role | Target | Mapping |
|---|---|---|---:|---|
| `UCSC-VLAA/MedReason` | Apache-2.0 | reasoning | 2,500 | question + options → user; `reasoning` → think; gold letter → `Answer: X` + first explanation sentence of `answer` |
| `FreedomIntelligence/Medical-R1-Distill-Data` (`en`) | Apache-2.0 | reasoning | 1,500 | `question` → user; R1 `reasoning` → think; `response` → answer |
| `TsinghuaC3I/UltraMedical` (`Exam` and `Literature` types, all multiple choice) | MIT | reasoning | 1,500 | question → user; explanation, which ends "So, the answer is X.", → think; `answer` letter → `Answer: X` |
| `FreedomIntelligence/medical-o1-reasoning-SFT` (`en`) | Apache-2.0 | reasoning | 1,000 | `Complex_CoT` → think; `Response` → answer |
| `lingshu-medical-mllm/ReasonMed` | Apache-2.0 | reasoning | 500 | instruction → user; output → think + final answer line |
| `OpenMed/Medical-Reasoning-SFT-Trinity-Mini` | Apache-2.0 | reasoning | 500 | `reasoning_content` → think; `content` → answer |
| `openlifescienceai/medmcqa` (train) | Apache-2.0 | direct | 800 | `Answer: X`, then the explanation after it |
| `GBaker/MedQA-USMLE-4-options` (train) | MIT | direct | 500 | `Answer: X. <option text>` |
| `qiaojin/PubMedQA` (`pqa_artificial`) | MIT | direct | 600 | abstract + question → user, options A yes / B no / C maybe; `Answer: X` + `long_answer` |
| `FreedomIntelligence/medical-o1-reasoning-SFT` (`en`), other rows | Apache-2.0 | direct | 600 | `Response` alone, without the reasoning: concise direct answers for chat (replaces ChatDoctor) |

A row is multiple choice when its gold answer is a single option letter. A
2026-09-25 sample of 3,000 UltraMedical rows held only `Exam` (66%) and
`Literature` (34%) types, all multiple choice, so no open-ended slice comes from
it. ChatDoctor is dropped: templated replies, and no exam signal. The medical-o1
reasoning and direct slices use disjoint rows.

Filters, in order:

1. **Parseable.** Multiple-choice rows need 2 to 5 options and a gold answer that
   maps to exactly one. English only, by an ASCII-share check (MedReason's
   `huatuo` rows are Chinese-origin).
2. **Answer agrees.** Where a trace states a final letter, it must match gold,
   or the row is dropped.
3. **Decontaminated** against every evaluation question below: exact match after
   normalisation (lowercase, alphanumerics, collapsed spaces), plus any shared
   13-word sequence for questions of 13 words or more. MedReason's `MMLU` rows
   are dropped whole, since they may come from MMLU's test split. Counts go into
   the data report per source.
4. **Deduplicated** across sources on the normalised question.
5. **Length.** Tokenised with Qwen3's template; `max_seq` is the measured p99
   rounded up to 64, capped at 3,072; longer rows are dropped, never truncated.

Sampling is seeded and stratified by source. 2% of the mix is held out as a
validation file for loss only.

## Format

The system prompts stay as run 1's. A system line asks for the answer first
after thinking.

- **Reasoning rows**, rendered with `enable_thinking=True`:
  `<think>\n{reasoning}\n</think>\n\nAnswer: C\n\n{one or two sentences}`,
  or prose for open-ended questions.
- **Direct rows**, rendered with `enable_thinking=False`:
  `<think>\n\n</think>\n\nAnswer: C\n\n{explanation}`.

The answer letter always comes first after the think block, so no cap can
remove it. Verified on Qwen3's real template: reasoning goes in via the
`reasoning_content` field, and direct rows keep the empty block run 1 trained on.

## Training

| Setting | Value | Why |
|---|---|---|
| Base | `unsloth/Qwen3-4B-unsloth-bnb-4bit`, QLoRA | same as run 1 |
| Versions | `unsloth==2026.9.7` and the stack it installed in run 1 | the one Unsloth version proven on this image |
| LoRA | r 64, alpha 64, dropout 0, all 7 projections | alpha = r and dropout 0 per Unsloth; more capacity for reasoning |
| Optimiser | adamw_8bit, lr 1e-4, cosine, 3% warmup, weight decay 0.01 | a lower lr than run 1's 2e-4, to avoid overwriting Qwen3's reasoning |
| Batch | 2 per step x 8 accumulation = 16 | longer sequences |
| Batching | length-grouped, no packing | without FlashAttention on a T4, packing lets examples attend to each other |
| Loss | completion only (the assistant turn, think block included) | as run 1 |
| Epochs | 1 | |
| GPU | GPU 0 only | proven single-GPU Unsloth path |

**Time guard.** A callback logs projected run time at step 20. If elapsed
training time reaches 6 hours it stops cleanly and saves the adapter. The
dataset is sized so the guard should not fire; if it does, the adapter still
exists and the report says so. Checkpoints are saved every 200 steps.

## Evaluation, in the same session

**Benchmarks.** MedQA-USMLE test (1,273), MedMCQA validation (4,183), PubMedQA
labeled (1,000; options yes / no / maybe; the abstract is given), and MMLU
medical: anatomy, clinical knowledge, college biology, college medicine, medical
genetics, professional medicine (1,089 test).

**Mode 1: letter choice.** Thinking off, as run 1: the letter's logit after
`Answer:`, restricted to the question's own options. All questions, both models.

**Mode 2: reasoning.** Thinking on, Qwen's recommended sampling (temperature
0.6, top-p 0.95, top-k 20), fixed seeds, a thinking budget of 1,536 new tokens.

- If the model closes its think block and writes `Answer: X`, that letter
  counts.
- **Budget forcing.** If the budget runs out first, the notebook appends
  `\n</think>\n\nAnswer:` and reads the letter's logit. No question is lost to
  the cap, and both models get the same budget.
- The report counts natural and forced answers per model.
- Sizes: MedQA 1,273, MMLU-medical 1,089, PubMedQA 500 (a fixed seeded
  sample), MedMCQA 1,000 (a fixed seeded sample).

**Execution.** The base model on GPU 1 and the fine-tune on GPU 0 run as
parallel processes over the same questions in the same order. Plain transformers
+ peft, full-precision fp16 base. Run 1's one-layer LoRA check runs in cell 2;
if an old torchao trips it, the cell uninstalls torchao and checks again, and
stops only if peft still cannot wrap a layer. torchao is not removed
unconditionally, because Unsloth shares this environment. No vLLM: its current engine needs compute 8.0+, and
LoRA has open compile bugs on T4s.

**Stages and time gate.** A 16-question smoke run measures speed first. Stages
run in this order: letter choice on everything, then reasoning on MedQA,
MMLU-medical, PubMedQA and MedMCQA. Before each stage the projection is checked
against the time left in the 12-hour session, keeping a 30-minute margin; a stage
that will not fit is skipped and listed. Each stage writes its JSON to
`/kaggle/working` when it finishes.

**Reported per benchmark and mode.** Accuracy for both models, the change, exact
McNemar p, a 95% interval for the paired difference, per-subject accuracy, and
the natural and forced counts. A pooled figure covers all questions scored in
each mode. Every question's prediction from both models is kept.

## Testing before the push

- Unit tests (TDD) for every pure function: the source mappers, filters,
  decontamination, length filtering, formatting, answer extraction, the
  budget-forcing text operations, statistics, and the time gate.
- **Real data prep** on this Mac: every source fetched, the report read by eye,
  and ten examples per source printed through the real template.
- **Local smoke run** of both evaluation modes on `Qwen/Qwen3-0.6B` with a random
  adapter, in the pinned transformers 5.5.0 / peft 0.19.1 environment, with
  torchao 0.10.0 installed and then removed (run 1's failure, reproduced).
- Unsloth training on CUDA can only run on Kaggle. Mitigation: the pinned
  version that already trained run 1, the step-20 probe, and the time guard.
- One independent review of the whole change before the push.

## Risks

| Risk | Mitigation |
|---|---|
| The pinned Unsloth no longer installs | The verify cell stops before any GPU work, naming the version it found |
| Thinking outputs are longer than budgeted | Budget forcing; the time gate skips late stages instead of overrunning |
| A source mapping is wrong | Printed samples reviewed before the push; per-source counts in the report |
| The base model beats the fine-tune in reasoning mode | Reported as measured; GRPO is the planned follow-up |
| Weekly GPU quota runs short | The run needs about 10–11 hours; the remaining quota shows on the owner's Kaggle settings page, since the API does not report it |

## Out of scope

GRPO (the follow-up session), multi-GPU training, vLLM, the frontend (Plan 3,
paused on branch `plan3-task1`), and a change of base model.
