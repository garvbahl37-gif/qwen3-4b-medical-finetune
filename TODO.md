# TODO

Qwen3-4B fine-tuned for medical question answering on a free Kaggle T4.

Spec: `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`
Plan 1: `docs/superpowers/plans/2026-09-20-training-and-evaluation.md`

## Where it stands

**Training is done.** One clean run on 2026-09-21 produced the adapter, now
kept at `training/outputs/run1` (264 MB, outside git). **Nothing has been
evaluated yet**, so there is no claim yet that fine-tuning helped.

Training loss levelled off at about step 150, roughly 4,800 examples in, and
barely moved over the remaining 390 steps. Only evaluation can say whether the
second half helped, but it is a reason not to assume more of the same data is
the lever.

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

The notebook exactly as it ran on Kaggle, and its output, are kept in
`results/run1/` (`kaggle_training_notebook.ipynb`, `kaggle_training_output.txt`),
next to the loss curve, the data report and the training stats.

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

## Plan 2 — evaluation (built, launching next)

`docs/superpowers/plans/2026-09-24-evaluation.md`. Drops Unsloth from evaluation so
the scoring path can run on this Mac before it runs on Kaggle.

- [x] **Task 1** load models with plain transformers + peft; render prompts with
      `enable_thinking=False`. Writing the plan found the scoring prompt left Qwen3
      in thinking mode, while every training example put the answer after an empty
      think block: the fine-tune would have been scored off-format.
- [x] **Task 2** batched scoring from one load (base = adapter switched off), left
      padding, position ids from the mask, a hard stop on non-finite logits.
      The reviewer traced all of it into the installed transformers and peft source.
- [x] **Task 3** end-to-end smoke run on Qwen3-0.6B locally: batch 1 and batch 4
      give identical answers with up to 18 tokens of real padding, and the disabled
      adapter scores exactly like a freshly loaded base
- [x] **Task 4** evaluation notebook with an in-session smoke pass and a 7.5 h
      budget gate. Review caught that Kaggle was still serving the old code
      (`modeling.py` missing entirely) and the upload script could not tell: it now
      waits for Kaggle's file listing to match, and each notebook checks a content
      fingerprint of the code it was built for
- [x] **The real adapter on the real base, locally**: 504 of 504 weights load,
      252 of 252 LoRA matrices non-zero, and it changes the model's answers
- [ ] **Final-review fixes before the push**: pin the GPU to a T4 (without it the
      new kernel would likely get a P100 and stop at cell 1); make the upload wait
      immune to a CLI version warning; save every per-question prediction so this
      one GPU session can answer why, not just what
- [ ] **The one Kaggle start**, then record the result honestly, including if
      fine-tuning does not beat base

**Early signal, not a result:** on 16 MedQA test questions run locally, the base
model scored 12/16 and the fine-tune 9/16 (3 regressions, 0 wins, p = 0.25). Too few
questions to conclude anything, and exactly what the full 5,456 will settle.

### What the full evaluation must state, whatever it finds

- The adapter was trained on the 4-bit base and is scored on the full-precision
  base. If that biases anything, it biases against the fine-tune.
- Answer-choice scoring asks for the letter immediately. For MedQA that is exactly
  how it was trained; for MedMCQA training put a rationale first, so it may
  understate the fine-tune there.
- MedMCQA answers lean towards A (32%). A model can gain on MedMCQA by guessing A
  more often, which is not medical knowledge; per-question predictions will show
  whether that happened. MedQA's answers are balanced, so it is the cleaner test.

## Plan 3 — serving and frontend

`docs/superpowers/plans/2026-09-24-serving-and-frontend.md`

- [ ] **Task 1** `screen_message()`: seven emergencies in lay language, notes on
      dosing, stopping medication and diagnosis. Pasted exam vignettes are read as
      exam context, so the red panel is kept for people describing themselves.
      Its patterns pass all 42 of their own tests, verified before the plan was
      committed.
- [ ] **Task 2** FastAPI server: every stream opens with the screen, before the
      first token; mock backend for building without the model
- [ ] **Task 3** web app shell and design system: operating-theatre green with red
      reserved for emergencies; Atkinson Hyperlegible for the interface, STIX Two
      for the model's answers
- [ ] **Task 4** Ask view: streamed answers under their safety screen
- [ ] **Task 5** Results view, laid out as a laboratory report; the loss chart
      follows the dataviz checks (its line colour was re-picked after the
      validator failed the page green on chroma)
- [ ] **Task 6** Method view, Playwright end-to-end tests, README
- [ ] Screenshot every page at 1280px and 360px and fix what collides or overflows

## Possibly later

- [ ] A second training session continuing from this adapter on a fresh,
      disjoint slice. Kaggle's 9-hour cap is per session, so more data means
      more sessions, not a longer one. Worth doing only if evaluation says the
      first 17k helped.

## Housekeeping

- [x] **Force-push `main` to GitHub** (2026-09-25). PR #1 had merged the branch
      from before the author email was corrected, crediting 12 commits to
      `jiteshbhalla1-web`. After the push, all 43 commits are authored by
      garvbahl37-gif, and GitHub's contributor list shows only that account.
- [ ] **Rotate the Kaggle token** `KGAT_...` at kaggle.com/settings: it was pasted
      into a chat session
