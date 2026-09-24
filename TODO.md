# TODO

Qwen3-4B fine-tuned for medical question answering on a free Kaggle T4.

Spec: `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`
Plan 1: `docs/superpowers/plans/2026-09-20-training-and-evaluation.md`

## Where it stands

**Training is done.** One clean run on 2026-09-21 produced the adapter.
**Nothing has been evaluated yet**, so there is no claim yet that fine-tuning
helped. The next GPU job is the base-vs-tuned comparison.

## Plan 1 — training (complete)

- [x] **Task 1** shared prompt template and `Record`
      Three review rounds. The answer extractor first invented answers from
      clinical prose ("A 45-year-old man ... choice C" scored as A), then
      over-corrected and missed real ones. Settled on a 34-case adversarial corpus.
- [x] **Task 2** four dataset normalisers and `load()`
      Live keep rates: MedMCQA 50.0%, MedQA / medical-o1 / ChatDoctor 100%.
      `load()` raises instead of silently returning fewer rows than asked for.
- [x] **Task 3** decontamination and the prep CLI
      Holdout uses the full 4,183 MedMCQA validation questions, not a filtered
      2,858, so the score stays comparable to published MedMCQA figures.
- [x] **Task 4** sequence length measured, not guessed
      Caught `len()` counting dict keys instead of tokens: every example would
      have measured 2 tokens and training would have run at `max_seq=64`.
- [x] **Task 5** scoring: accuracy, exact McNemar, per-subject breakdown
- [x] **Task 6** evaluation CLI: constrained logits and generative decoding
      Letter token ids verified on the real tokenizer: `" A"`=362, `"A"`=32.
- [x] **Task 7** QLoRA trainer with the step-50 budget probe and checkpointing
- [x] **Task 8** Kaggle dataset upload and training notebook

### The training run

| | |
|---|---|
| examples trained | 17,285 (18,000 requested; 90 decontaminated, rest over-length) |
| mix | 69.9% exam and reasoning, 30.1% patient dialogue |
| held out, never trained on | 5,456 (1,273 MedQA test + 4,183 MedMCQA validation) |
| `max_seq` | 960, from a measured p99 of 931 |
| runtime | 4.81 h on one T4, 541 steps, 1.00 examples/sec |
| mean training loss | 1.796 (3.03 at the first logged step) |
| adapter | LoRA rank 32 on all seven projections, 264 MB |

Scaled down from 40,000 because the budget probe measured the T4 at 0.9
examples/sec: 40,000 would have taken 12 hours against Kaggle's 9-hour cap.

### Seven Kaggle attempts before that one

Each died in the first minutes, so the GPU quota barely moved. Every cause is
now guarded in code:

1. `datasets create` is asynchronous; the kernel started before the data existed
2. the dataset mounted at `/kaggle/input/datasets/...`, not where assumed
3. a guard's own error message raised while reporting a missing directory
4. the kernel id did not match the slug Kaggle derives from the title
5. `trl` was imported before `unsloth`, so the EOS token was set on the wrong class
6. the budget probe aborted a 12-hour run at step 50, as designed

## Plan 2 — evaluate and serve (next)

- [ ] Run `evaluate.py` end to end **locally** on a tiny Qwen3 before any GPU time,
      so the Kaggle run cannot fail on untested code
- [ ] Upload the adapter as a Kaggle dataset
- [ ] Evaluation notebook: an 8-question smoke pass, then the full 5,456, in one
      session
- [ ] Record the result honestly, including if fine-tuning does not beat base
- [ ] Merge the adapter into full-precision weights for serving
- [ ] FastAPI server with `screen_message()`: emergency red flags shown above the
      answer, validated against a benign control set so it does not fire on
      everything

## Plan 3 — frontend

- [ ] Next.js chat: streaming, multi-turn, safety banner, red-flag callout
- [ ] Benchmark tab: base vs fine-tuned, both scoring modes, per-subject table
- [ ] Method tab: how the data was filtered, decontaminated and scored
- [ ] `MOCK_BACKEND=1` so the UI builds and tests without the model

## Possibly later

- [ ] A second training session continuing from this adapter on a fresh,
      disjoint slice. Kaggle's 9-hour cap is per session, so more data means
      more sessions, not a longer one. Worth doing only if evaluation says the
      first 17k helped.

## Housekeeping

- [ ] **Force-push `main` to GitHub.** PR #1 merged the branch from before the
      author email was corrected, so GitHub still credits 12 commits to
      `jiteshbhalla1-web`. Local `main` has the corrected history, and every file
      on the remote is also present locally, so nothing is lost:
      `git push --force-with-lease origin main`
- [ ] **Rotate the Kaggle token** `KGAT_...` at kaggle.com/settings: it was pasted
      into a chat session
