# TODO

Status of the medical fine-tune. Plan:
`docs/superpowers/plans/2026-09-20-training-and-evaluation.md`
Spec: `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`

## Plan 1 — training and evaluation

- [x] **Task 1** shared prompt template and `Record` — 39 tests
      Took 3 review rounds. The answer extractor invented answers from clinical
      prose ("A 45-year-old man ... choice C" scored as A), then over-corrected
      and missed real ones. Settled against a 34-case adversarial corpus.
- [x] **Task 2** four dataset normalisers and `load()` — 17 tests
      Live keep rates: MedMCQA 49.0%, MedQA 100%, medical-o1 100%, ChatDoctor 100%.
      `load()` now raises instead of silently returning fewer rows than asked for.
- [x] **Task 3** decontamination and the prep CLI — 5 tests
      Every question hashed; any training row colliding with a benchmark is dropped
      and the count reported.
- [ ] **Task 3 fix (R14)** holdout must use the full MedMCQA validation set (4,183,
      not 2,858) or the reported number is not comparable to published MedMCQA
- [ ] **Task 4** measure token lengths, derive `max_seq`, drop outliers
- [ ] **Task 5** scoring: accuracy, exact McNemar, per-subject breakdown
- [ ] **Task 6** evaluation CLI: constrained logits + generative decoding
- [ ] **Task 7** QLoRA training with the step-50 budget probe and checkpointing
- [ ] **Task 8** Kaggle dataset upload and notebook push

## Plan 2 — serving and safety (not started)

- [ ] FastAPI server, GPU and GGUF-CPU backends
- [ ] `screen_message()` red-flag detector, validated against a benign control set
- [ ] ngrok deployment

## Plan 3 — frontend (not started)

- [ ] Next.js chat with streaming and multi-turn history
- [ ] Benchmark tab: base vs fine-tuned, both scoring modes, per-subject
- [ ] Method tab
- [ ] `MOCK_BACKEND=1` offline path

## Then

- [ ] Push the notebook to Kaggle, run on GPU T4 x2 with Internet on
- [ ] Record the result honestly, including if fine-tuning does not beat base

## Housekeeping

- [ ] **Rotate the Kaggle token** `KGAT_...` at kaggle.com/settings — it was pasted
      into a chat session
