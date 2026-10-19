# Open questions, ranked by expected information per GPU-hour

Each hypothesis states its success and failure criteria **before** its
result is seen. "Dev" means `dev_medqa` and `dev_medmcqa` (`data/v3`), which
are used for selection. The benchmarks are reported, never tuned on.

| # | Hypothesis | Evidence so far | Confidence | Smallest decisive experiment | Success | Failure | Status |
|---|---|---|---|---|---|---|---|
| H1 | Training on **other models'** traces is what degraded reasoning; the model's **own** verified traces will not | run 3 [measured]; STaR / RFT / ReST-EM [paper] | medium | **E2** (one session) | E2 ≥ base on dev_medqa thinking-mode accuracy (paired), median `think_tokens` within ±20% of the base | E2 < base on dev by a significant margin, or thinking shrinks > 20% | **answered 2026-10-19: tie.** Dev MedQA 70.0 → 69.0 (p 0.76), dev MedMCQA 69.0 → 67.7 (p 0.64), MedQA test 70.9 → 71.0. Thinking −21% (dev MedQA), −12% (MedQA), −7% (dev MedMCQA). Own traces avoid run 3's 9.5-point loss, but add nothing. |
| H2 | The 1,536-token budget under-measured the base model | base p90 at the cap [measured]; m1's optimum near 4K [paper] | high | base at 3,072 on full MedQA test (E2's base evaluation runs this) | accuracy at 3,072 > 69.7% | ≤ 69.7% | **answered: 70.9% vs 69.7%, +1.2 points, p 0.24** (paired with run 3's answers). The direction is as predicted, but not significant. |
| H3 | Self-consistency (majority of k samples) raises thinking-mode accuracy with no training | general SC literature; nothing medical verified | medium | base, k = 5, on dev_medqa (needs vLLM speed) | maj@5 − first-sample ≥ +2 points | < +1 point | **next.** vLLM works on the T4 (8.1 s per MedQA question per GPU at 3,072) |
| H4 | Run 3's update was too large (rank 64, lr 1e-4) | none direct | low–medium | E1: same v2 data, rank 16, lr 2e-5 | E1 thinking ≥ run 3 + 3 points on dev | ≤ run 3 | deprioritised (decisions D2) |
| H5 | GRPO with a binary MCQ reward beats E2 | AlphaMed [paper], on 8B, 8 × A800, non-thinking backbone; an Unsloth T4 GRPO notebook exists [docs] | low on a T4 | ~200-step QLoRA GRPO pilot, short completions, MedQA train | dev gain over base and E2 | no gain, or length collapse | only after E2 (the earlier plan's rule: no RL before E2 evidence) |
| H6 | vLLM fp16 on a T4 reproduces transformers' letter choices | none | unknown | score dev letter mode with both engines in one session | ≥ 98% agreement, accuracy within 0.5 points | below either | **answered: base letter choice by vLLM (E2) vs transformers (run 3) agrees on 99.6–100% of questions, accuracy within 0.08 points.** Engines are interchangeable for comparisons. |
| H8 | Corrective traces from a stronger Qwen3, on the base's weak questions, raise dev accuracy where self-traces (E2) could not | MMOPD +10.9 MedQA with 40k verified teacher traces [reported, not reproduced]; E2 tie [measured] | low–medium at our scale (about 1,000 traces, LoRA, 1 epoch) | **E3** (E3a teacher traces, E3b train and score) | E3 > base on pooled dev reasoning (600 paired), McNemar p < 0.05, E3 ≥ E2, thinking within 20% | no significant gain: the base stays champion | **running** (E3a launched 2026-10-19 20:36) |
| H7 | Data defects in v2 (nested think, contradictions) cause part of run 3's loss | audit [measured]: about 1–2% of rows | low | v2 minus flagged rows, otherwise like run 3 | thinking improves ≥ 2 points | no change | low priority; E2 replaces v2 |

## After E2 (2026-10-19): the "tie" branch

E2 trained only on questions the base already answered correctly: the
deadline left one sample per question. It reinforced what the model already
does and taught it nothing new. The next experiments target questions the
base gets *sometimes* right:
1. **H3, self-consistency** (no training). Measure majority vote over 5
   samples on dev, then MedQA. With vLLM, MedQA × 5 is about 14 GPU-hours,
   one session on two GPUs.
2. **H5, a small GRPO pilot** with the binary gold-letter reward, on pool
   questions where the base passes some but not all of 4 samples. That is
   where an on-policy signal exists. Use Unsloth's T4 GRPO recipe.

## What each E2 outcome would mean (written before the result)

- **E2 beats the base on dev.** Confirm it on the benchmarks (MedQA reasoning
  is already in-session). Then try a second round (ReST-EM style), or more
  samples per question for hard questions.
- **E2 ties the base.** Self-distillation keeps reasoning but adds no
  knowledge. Next, test H3 (test-time voting) and H5 (RL).
- **E2 loses to the base.** Even on-policy SFT harms this model at this
  update size. Try a smaller update or stop SFT. Report that the base is the
  best model (the earlier plan's case D).
