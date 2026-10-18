# Evaluation reference

## Benchmarks ([code] `sources_v2.EVAL_SIZES`, `prepare_data_v2.load_eval_sets`)

| Benchmark | Split | Questions | Options |
|---|---|---:|---|
| MedQA (USMLE, `GBaker/MedQA-USMLE-4-options`) | test | 1,273 | A–D |
| MedMCQA (`openlifescienceai/medmcqa`) | validation (test labels are hidden) | 4,183 | A–D |
| PubMedQA (`qiaojin/PubMedQA` pqa_labeled) | all 1,000 labelled | 1,000 | yes/no/maybe |
| MMLU medical (`cais/mmlu`: anatomy, clinical knowledge, college biology, college medicine, medical genetics, professional medicine) | test | 1,089 | A–D |
| **dev_medqa**, **dev_medmcqa** (`data/v3`) | 300 + 300 from the *training* splits, decontaminated | 600 | A–D |

**The dev split is for choosing between models**. The four benchmarks are
for reporting. MedQA test and MedMCQA validation have been looked at across
runs 1–3, but no hyperparameter was tuned on them. From E2 on, every
selection decision reads only dev.

## Scoring ([code] `eval_worker.py`)

- **Letter mode.** The prompt has thinking off (`/no_think`,
  `enable_thinking=False`) and ends with "Answer:". The prediction is the
  argmax of the next-token logits over the question's own letters. It is
  deterministic, and non-finite logits stop the run.
- **Reasoning mode.**
  - The prompt has thinking on (`/think`), with sampling at T 0.6, top-p 0.95,
    top-k 20 and a token budget.
  - The letter is read from the text after the last `</think>` ("Answer: X",
    else the extractor).
  - No letter means "budget forcing": close the think block if it is open,
    append "Answer:", and take the argmax letter. `forced` and `closed` are
    recorded per row.
- **Per row**: `letter`, `closed`, `forced`, `new_tokens`, `think_tokens`
  (tokens before the last `</think>`, from E0 on), `seed`, `sample`, `text`.
- **Order.** Reasoning stages run a seeded shuffle (seed 1234), so a deadline
  cut leaves a random sample and every model sees the same questions. With
  several samples, sample 0 covers every question before sample 1 starts.
- **Engines.** transformers (`modeling.load_model`) or vLLM
  (`vllm_engine.py`). Both use the same prompts (token ids, no special tokens
  added), the same row builder (`row_from_tokens`) and the same letter rule.
  Whether vLLM's fp16 numerics on a T4 reproduce transformers' letter choices
  is **[unknown]** until measured. Never compare models scored by different
  engines without that check (`meta.json` records the engine).

## Statistics ([code] `evalcore.py`, `eval_report.py`)

- Pairwise comparisons use only the questions both models answered (first
  sample): exact McNemar p and a 95% Wald interval on the paired difference.
- `common`: each model's accuracy on the questions all models answered.
- `any_correct`, `majority_accuracy` (self-consistency) and
  `first_sample_accuracy` on the same questions, for multi-sample stages.
- `experiments.py` re-scores every number in `experiments/results.csv` from
  raw predictions; nothing is typed in.

## Known evaluation problems

| Problem | Status |
|---|---|
| Runs 2–3 used a 1,536-token budget; the base's p90 is at the cap, so it was under-measured | [measured]. E0 and E2 use 3,072. |
| The "median thinking 1,036 vs 403" figure is new tokens (thinking + answer) | [measured]. `think_tokens` from E0 on. |
| Run 3's base never reached PubMedQA or MedMCQA reasoning | [measured]. Those comparisons are missing. |
| Run 1's letter scores use a different prompt format | [code]. Compare within run 1 only. |
| A reasoning score depends on sampling; one sample has noise | [hypothesis]. E0 adds 2 MedQA samples and majority vote. |
| Benchmark accuracy is not clinical competence | Always state it. Safety and general-capability evaluations are not built yet. |
