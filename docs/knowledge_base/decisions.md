# Decisions

| # | Date | Decision | Evidence | Cost if wrong |
|---|---|---|---|---|
| D1 | 2026-10-03 | Freeze run 3 as the baseline fine-tune (tags, `experiments/run3`) | run 3 is the best-documented fine-tune | none |
| D2 | 2026-10-19 | Run **E2 before E1** | Run 3's loss tracks *whose* reasoning was trained on: 67% of its traces are written unlike Qwen3, and thinking shrank by 60%. Self-training evidence [paper]. E1 would test only update size. | one session if E2 is uninformative; E1 is still built and ready |
| D3 | 2026-10-19 | Select models on a **dev split** from the training sets; report the benchmarks | the benchmarks have been looked at in runs 1–3; selection on them would overfit | 600 fewer pool questions |
| D4 | 2026-10-19 | Generate with **vLLM in an isolated virtual environment**, falling back to transformers per queue entry | transformers took 22.9 s per question [measured]; vLLM supports compute capability 7.5 [docs]; Unsloth's T4 GRPO notebook uses it [docs] | fallback is slower, so fewer traces |
| D5 | 2026-10-19 | E2 generation budget **4,096**, evaluation 3,072 | a trace cut at the budget is rejected, so a short budget biases selection to short thinking; m1's optimum is near 4K [paper] | longer generation per question |
| D6 | 2026-10-19 | Choose **one random kept sample** per question, not the shortest | the shortest would repeat run 3's shortening | none |
| D7 | 2026-10-19 | E2 at **rank 16, lr 5e-5**, one epoch, batch 1 × 16 | on-policy data needs a smaller update than run 3's (64, 1e-4) **[hypothesis]**; dev decides | too small to change anything, which E2-vs-base on dev would show |
| D8 | 2026-10-19 | Keep **20% answer-only rows** in E2 | Unsloth's 75/25 guidance [docs]; letter mode is also reported | slight dilution of thinking rows |
| D9 | 2026-10-19 | No GRPO until E2 reports | the earlier plan's rule; AlphaMed's compute is far above a T4 [paper] | a delay, if RL would have been the winner |
| D10 | 2026-10-19 | The HF token pasted in chat is **not used or stored** | public models and datasets need no token | none; rotate it |
| D11 | 2026-10-19 | **The base model stays the best model**; the E2 adapter is kept (`training/outputs/e2`) but not adopted | E2 failed its pre-registered success criterion on dev (−1.0 and −1.3 points, not significant) and is level on the MedQA test (+0.2) | none: the selection rule says to keep the base until a candidate wins on dev |
| D12 | 2026-10-19 | Next: self-consistency (H3), then a GRPO pilot on partially solved questions (H5). Not more E2-style SFT. | E2 trained only on already-solved questions and added nothing; vLLM makes multi-sample work affordable on a T4 [measured] | one session each |
