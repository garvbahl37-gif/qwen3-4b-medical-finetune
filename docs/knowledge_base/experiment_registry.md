# Experiment registry

The machine-readable registry is `experiments/<id>/experiment.json`.
`experiments/results.csv` is generated from it and from raw predictions
(`python -m training.experiments`). This page is the readable index; its
numbers come from that CSV.

| ID | What changed | Status | Thinking MedQA (base → model) | Letter MedQA (base → model) | Takeaway |
|---|---|---|---|---|---|
| run1 | First QLoRA SFT: rank 32, lr 2e-4, v1 data, letter-choice evaluation only | done 2026-09-24 | not measured | 56.6 → 58.8 (v1 prompt) | small letter gain; written answers worse |
| run2 | v2 data with reasoning traces, no mode switch | done 2026-09-26 | 68.2 → 57.8 (192 questions) | 57.4 → 58.1 | the model learned to skip thinking |
| run3 | `/think` and `/no_think` added | done 2026-10-01; frozen (tags `run3-train`, `run3-eval`) | **69.7 → 60.2** (1,273 questions) | 57.0 → 58.4 | thinks on every question, but shorter and worse |
| E0 | baseline at a 3,072 budget: base, run 3 fp16, run 3 4-bit; 2 MedQA samples | configured, not run | — | — | superseded in part by E2's in-session base evaluation |
| E1 | gentle SFT: lr 2e-5, rank 16, same v2 data | not built | — | — | lower priority than E2 (decisions D2) |
| E2 | SFT on the base model's own gold-verified reasoning (v3 pool; 3,182 traces + 796 answer-only rows), rank 16, lr 5e-5 | **done 2026-10-19** (generation d3ed0cd; training and evaluation e2t v2, ce5206d) | **70.9 → 71.0** (1,273 questions, budget 3,072, vLLM; p 0.94). Dev MedQA 70.0 → 69.0 (p 0.76) | 57.0 → 58.4 (p 0.14) | **tie**: repairs run 3's damage but adds no accuracy; thinking −7% to −21%. The base stays the best model. |

| E3 | E2's set + a stronger Qwen3's gold-verified traces on the base's 1,418 weak pool questions; trained as E2 | **E3a launched 2026-10-19 20:36** (code 98f2ebd): teacher pilot and generation | pending | dev only | pre-registered criteria in open_questions H8 |

Model selection rule: a candidate must beat the base on the dev split, with
no meaningful loss on any benchmark and no collapse in thinking length,
before it replaces the base. Until then the base model is the best model.
