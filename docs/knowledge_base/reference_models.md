# Reference medical fine-tunes (studied 2026-10-19)

Read from each model card (Hugging Face API and raw README), and from
shallow clones of each repository under `third_party/refs/` (git-ignored).
**None of these numbers was reproduced here.** Every metric is as reported
by its author.

| | MMOPD/Qwen3-4B-OT3-medical | XformAI-india/Qwen3-4B-medicaldataset | vireshk/qwen3-8b-medmcqa-qlora | HK2184/medqa-qwen3-lora | ceresnam/qwen3-finetune | roymacias/medical-qa-qlora-finetune | TheRedCan/medical-qlora-finetune |
|---|---|---|---|---|---|---|---|
| **Base, size** | MMOPD/Qwen3-4B-OT3-2ep (Qwen3-4B-**Base** + 2 epochs on OpenThoughts3-1.2M), 4B. Not Qwen/Qwen3-4B. | Qwen/Qwen3-4B, 4B | Qwen/Qwen3-8B @ b968826, 8B | Qwen/Qwen3-1.7B | Qwen3-4B, 8B, 14B | google/gemma-3-4b-it | Qwen2.5-3B instruct and base, Qwen2.5-0.5B (MCQA attempts) |
| **Method** | full SFT, bf16, ZeRO-2 | SFT (kind not stated; full weights uploaded) | QLoRA, NF4 | LoRA, bf16 | QLoRA, NF4 | QLoRA, NF4 | QLoRA, NF4 |
| **Data, n** | 40,853 traces on MedQA-USMLE **train** (20,360 4-option, 20,493 5-option; 15,427 prompt variants) | FreedomIntelligence/medical-o1-reasoning-SFT, n not stated | 30,000 MedMCQA train rows | MedMCQA train: first 500 (card) / 2,000 (code) | first 2,000 rows of medical-o1-reasoning-SFT | 50,000 stratified MedMCQA train | about 8,000 MedMCQA train |
| **How examples were made** | Teacher Qwen3.6-35B-A3B (card: "MedQA 94.6%"), 8 samples per question at T 1.0; thinking capped at 16k characters | dataset's GPT-4o chain of thought | human `exp` explanation, answer letter first | dataset letter and explanation | dataset GPT-4o chain of thought | letter + human explanation | `exp` as chain of thought |
| **Wrong teacher answers filtered?** | **yes**: only traces with the gold letter, at most 3 per question, shortest first | per the dataset card (medical verifier); not by the author | n/a (human explanations); answer-key boilerplate stripped | no | no | n/a | n/a |
| **Seq len, LR, epochs, LoRA** | 8,192 packed; 1e-5, 5% warmup; 4 epochs; no LoRA | not stated; 2e-5; 3 epochs; batch 8 × accumulation 4 | 1,024; 2e-4 cosine; 1 epoch (1,875 steps); r 16, α 32, dropout 0.05, all 7 projections | **card and code disagree**: 128 vs 256, 1e-4 vs 2e-4, 1 vs 2 epochs, r 4 vs 8, α 16, q and v only | not set; 2e-4; 1 epoch; r 64, α 16, all 7 projections; **loss on the prompt too** (plain LM collator) | 1,024; 2e-4 cosine; 1 epoch; r 16, α 32, all 7; assistant-only loss | 1,024; 2e-4; 1 epoch; r 16, α 32, all 7 |
| **Hardware** | not stated for this model; its init was trained on A100-80GB with ZeRO-2 | not stated | 1 × A40, 5 h 04 m | AMD MI300X, about 5 minutes | not stated | 1 × L4 (24 GB) | Kaggle or Colab T4 |
| **Evaluation and reported metrics** | MedQA 69.8 → **80.7** (split not stated; T 1.0, top-p 1.0, long budget); MedXpertQA 13.7 → 22.3; PubMedQA 75.2 → 77.2. **General loss:** AIME24 66.3 → 52.9, LiveCodeBench 51.7 → 39.9, IFEval 51.0 → 41.2 | **none reported** | frozen 1,000 items from MedMCQA validation, letter by log-prob, **thinking off**: 56.8 → 62.9 (+6.1, McNemar p < 0.0001); 3 seeds 62.9 / 63.9 / 61.7; retrieval 67.0 beats it; free text **below** base | eval script scores the first 100 MedMCQA validation questions; **no number published** | none: the "test" set is the same `train[0:2000]` slice (a speed benchmark) | MedMCQA held-out (n 1,573): 46.15 → **44.95**; MedQA (OOD): 53.34 → **49.80**; MedGemma 56.64 / 66.22 | MCQA "neutral-to-slightly-negative, never significant" over 4 configs (numbers not published); then pivoted to extraction |
| **Leakage prevention** | "8-gram decontaminated against the evaluation sets" (stated; traces unpublished, so not checkable) | none stated | content-hash gate (0 leaked rows); contamination probed (~9% of test stems recalled) | none | none (train = test) | MD5 of the normalised question, plus a cross-dataset check | sentence-overlap check (extraction task) |
| **Code and gaps** | **no code, no traces** published; teacher outputs, prompts and filtering code unavailable | no code; sequence length, LoRA settings and evaluation all missing | github.com/vireshkoli/Fine-Tune-vs-RAG: configs, CI, report (no LICENSE file) | github.com/HK2184/MedQA-Medical-AI-on-AMD-ROCm (no LICENSE file); its card contradicts its code | notebooks; MIT | complete pipeline, configs and predictions committed; MIT | src, tests and Kaggle scripts; no LICENSE file |
| **Licence** | Apache-2.0 | MIT (on an Apache-2.0 base) | Apache-2.0 adapter; MedQA deliberately not used ("licence unknown") | MIT | MIT | MIT | none |
| **Transferable to us** | **the recipe**: a stronger *same-family* thinking teacher, gold-verified rejection sampling, several traces per question. Its compute (35B teacher, full fine-tune, 4 epochs, 8k context) does not transfer. | nothing measured | content-hash leakage gate; paired McNemar; reporting seed variance; letter-mode gains do not mean thinking-mode gains | nothing | throughput datapoint only | negative evidence: explanation SFT hurt OOD MedQA | negative evidence matching our runs 2–3; similar Kaggle tooling |

## What the references say together

1. **Teacher-written reasoning gave the only large thinking-mode gain** among
   these references, and only in MMOPD. Its conditions differ from ours: a
   teacher far stronger than the student, every answer gold-verified, the
   same thinking format as the student, and a lot of compute. **[paper-like
   claim on a model card; not reproduced]**
2. **SFT on human or dataset explanations for multiple choice is neutral to
   harmful** in roymacias (−1.2 in-distribution, −3.5 OOD) and TheRedCan, and
   was harmful in our runs 2–3. Where it helps (vireshk, +6.1), it helps
   letter-mode scoring with thinking off, and free text gets worse.
   **[reported + our measured runs]**
3. **Our E2** (own verified traces) tied the base. It saw only questions the
   base already gets right. **[measured]**

## The three techniques chosen, as E3

- **A. Teacher reasoning with verified answers.** A stronger Qwen3 (8B-AWQ,
  14B-AWQ or 30B-A3B-GPTQ, chosen by a measured pilot) thinks through the
  questions. Only traces that close their thinking on the gold letter are
  kept, up to 2 of 3 samples per question. Same-family traces keep Qwen3's
  style, which run 3's GPT-4 / R1 traces did not.
- **B. Target the weaknesses.** The questions are the 1,418 pool questions
  the base got wrong in E2's generation (952 MedQA, 466 MedMCQA), not
  questions it already answers.
- **C. Leakage-safe data and paired dev evaluation.** The pool is
  decontaminated against all four benchmarks (exact match or ≥ 50% of
  13-grams) and kept apart from the 600-question dev split. Base, E2 and E3
  are compared on the same dev questions with the same decoding, using
  paired McNemar tests. The benchmarks are not used for selection.

Also adopted from vireshk: re-score the base in the same session, as a check
on run-to-run noise.

**Not adopted:**
- shortest-first trace selection (MMOPD): it would teach shorter thinking,
  run 3's failure;
- full fine-tuning and multi-epoch training: a T4 cannot afford them;
- MedQA-licence caution (vireshk): this project already uses MedQA.
