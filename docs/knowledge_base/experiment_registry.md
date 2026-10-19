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
| E2 | SFT on the base model's own gold-verified reasoning (v3 pool) | attempt 1 (code d3ed0cd): generation done, training failed on an offline tokenizer lookup. **Attempt 2 launched 2026-10-19** from the saved traces (`gb1105/qwen3-4b-medical-e2-train-and-evaluate`, code d60735e) | pending | pending | 4,608 completions, 69% verified, 3,182 traces; selection on the dev split |

Model selection rule: a candidate must beat the base on the dev split, with
no meaningful loss on any benchmark and no collapse in thinking length,
before it replaces the base. Until then the base model is the best model.
