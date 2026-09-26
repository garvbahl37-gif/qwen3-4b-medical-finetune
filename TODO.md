# TODO

Qwen3-4B fine-tuned for medical question answering on a free Kaggle T4.

Spec: `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`
Plan 1: `docs/superpowers/plans/2026-09-20-training-and-evaluation.md`

## Where it stands

**Training is done.** One clean run on 2026-09-21 produced the adapter, now
kept at `training/outputs/run1` (264 MB, outside git). **Evaluation is done** (version 2 on Kaggle, 2026-09-24, 1.2 hours on a T4):

| Benchmark | Scoring | n | Base | Fine-tune | Change | Wins / regressions | p |
|---|---|---:|---:|---:|---:|---:|---:|
| MedQA-USMLE test | answer choice | 1,273 | 56.6% | 58.8% | +2.1 | 112 / 85 | 0.064 |
| MedMCQA validation | answer choice | 4,183 | 53.8% | 55.4% | +1.6 | 399 / 334 | 0.018 |
| MedQA-USMLE test | written answer | 300 | 68.7% | 62.3% | −6.3 | 19 / 38 | 0.016 |
| MedMCQA validation | written answer | 300 | 57.7% | 42.7% | −15.0 | 22 / 67 | 0.000002 |

- **Answer-choice scoring**, the headline, shows a small gain on both
  benchmarks. It is significant on MedMCQA (p = 0.018) and not on MedQA
  (p = 0.064). The MedMCQA gain does not come from guessing A: the fine-tune
  picked A 1,176 times against the base model's 1,499, with 1,348 correct As.
- **Written answers** are worse, and mostly for reasons of format, not
  knowledge. On MedMCQA, 70 of the fine-tune's 300 answers ran past the
  512-token cap before stating a letter (the base: 0). On the 230 questions
  where neither model hit the cap, they score alike: 56.5% and 55.7%. On MedQA
  the fine-tune answers at once, as it was trained to (median 9 characters),
  while the base model reasons first (median 884 characters), and reasoning
  first wins here.

Training loss levelled off at about step 150, roughly 4,800 examples in, and
barely moved over the remaining 390 steps. Only evaluation can say whether the
second half helped, but it is a reason not to assume more of the same data is
the lever.

## Run 2 — reasoning fine-tune (complete; primary goal not met)

Spec `docs/superpowers/specs/2026-09-25-run2-reasoning-finetune-design.md`, plan
`docs/superpowers/plans/2026-09-25-run2-reasoning-finetune.md`.

- [x] Research: Unsloth and Qwen guidance (keep 75% reasoning data; never decode
      greedily in thinking mode), the huggingface/skills and unsloth-buddy
      fine-tuning skills, and six new licensed datasets
- [x] Data built locally and frozen as a Kaggle dataset: 13,617 examples, 75%
      reasoning in Qwen3's think block, the answer letter always first. Sources:
      MedReason, R1-Distill (medical rows only), UltraMedical, medical-o1,
      ReasonMed, MedMCQA, MedQA, PubMedQA; ChatDoctor dropped. Decontaminated
      against all four benchmarks and deduplicated across sources
      (`results/run2/data_report.json`)
- [x] Code: 208 tests; the evaluation rehearsed end to end on Qwen3-0.6B in both
      the local and the pinned Kaggle stack, and the notebook's own cells
      rehearsed against a mock Kaggle layout
- [x] **Launched 2026-09-25 13:27 UTC**: [the run 2 kernel](https://www.kaggle.com/code/gb1105/qwen3-4b-medical-fine-tune-run-2) (private), T4 x2,
      one 12-hour session, no laptop needed. It trains (time guard at 5.75 h),
      saves the adapter, runs an evaluation smoke test, then scores base and
      fine-tune in parallel on MedQA, MedMCQA, PubMedQA and MMLU-medical, by
      letter choice and by reasoning, until 40 minutes before the limit. The
      notebook as pushed is `results/run2/kaggle_run2_notebook.ipynb`
- [x] **Completed 2026-09-26.** Training ran its full schedule: 852 steps, 5.68 h,
      mean loss 0.977, no early stop. Evaluation ran until the deadline. Results,
      every prediction and the Kaggle log are in `results/run2/`; the adapter is in
      `training/outputs/run2` (515 MB, outside git)

| Benchmark (letter choice, all questions) | n | Base | Fine-tune | Change | p |
|---|---:|---:|---:|---:|---:|
| MedQA-USMLE test | 1,273 | 57.4% | 58.1% | +0.6 | 0.61 |
| MedMCQA validation | 4,183 | 55.0% | 55.3% | +0.3 | 0.66 |
| PubMedQA labeled | 1,000 | 72.0% | 75.1% | +3.1 | 0.008 |
| MMLU medical | 1,089 | 73.3% | 74.1% | +0.8 | 0.45 |
| All four, pooled | 7,545 | 60.3% | 61.1% | +0.8 | 0.067 |

Reasoning mode could be compared on only 192 MedQA questions: base 68.2%,
fine-tune 57.8% (p = 0.004). Two things went wrong there:

- **The fine-tune mostly stopped thinking.** It wrote a real reasoning trace on
  302 of 1,273 MedQA questions (median 15 new tokens) and otherwise answered at
  once, so its reasoning-mode score (56.8%) equals its letter-choice score. The
  cause is the data format: the 25% of direct-answer rows carried their empty
  think block inside the trained answer, under the same system prompt as the
  reasoning rows, so the model learned to choose "no thinking" itself for
  exam-style questions. Qwen3's intended signal is `/no_think` in the prompt.
- **The base model's evaluation worker ran out of GPU memory** after 192
  questions: it thinks at length (median 1,025 tokens), and the budget-forcing
  pass re-read 16 prompts of up to ~2,500 tokens at once on a 15 GB T4. The
  fine-tune's worker finished; its reasoning-mode scores without a base to
  compare against are MedQA 56.8% (1,273), MMLU-medical 71.2% (1,089),
  PubMedQA 74.6% (500) and MedMCQA 51.6% (304).

**Verdict:** the one significant gain is PubMedQA letter choice (+3.1 points).
Letter choice elsewhere is flat, and no better than run 1's fine-tune. The
reasoning goal was not met, for a cause that is now identified and fixable.

- [ ] Run 3 (not started; needs the owner's go and ~11 GPU-hours): mark direct
      rows with `/no_think` in the user turn so thinking is switched by the
      prompt, not guessed from the question's style; make the budget-forcing pass
      score forced prompts in small batches and free the failed attempt's memory
      before retrying; run the base model's reasoning at batch 8
- [ ] The independent final review never finished (stopped with the laptop); rerun
      it on the run 3 changes instead

## Documentation

- [x] `README.md` and three guides, written 2026-09-25 from the code and
      `results/run1/`: `docs/training.md` (data, prompt format, QLoRA, the failed
      attempts), `docs/evaluation.md` (method, report JSON, biases) and
      `docs/kaggle.md` (reproducing both runs on your own account)
- [x] Add the evaluation result to the README

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
| `max_seq` | 960, from a measured p99 of 929 |
| runtime | 4.81 h on one T4, 541 steps, 1.00 examples/sec |
| mean training loss | 1.796 (2.912 at the first logged step) |
| adapter | LoRA rank 32 on all seven projections, 264 MB |

The notebook exactly as it ran on Kaggle, and its output, are kept in
`results/run1/` (`kaggle_training_notebook.ipynb`, `kaggle_training_output.txt`),
next to the loss curve, the data report and the training stats.

Scaled down from 40,000 because the budget probe measured the T4 at 0.9
examples/sec: 40,000 would have taken 12 hours against Kaggle's 9-hour cap.

### What stopped the attempts before it

Run 1 is version 7 of the training notebook. The attempts before it stopped
early, most within minutes, so the GPU quota barely moved. Every cause is now
guarded in code:

1. `datasets create` is asynchronous; the kernel started before the data existed
2. the dataset mounted at `/kaggle/input/datasets/...`, not where assumed
3. a guard's own error message raised while reporting a missing directory
4. the kernel id did not match the slug Kaggle derives from the title
5. `trl` was imported before `unsloth`, so the EOS token was set on the wrong class
6. the budget probe aborted a 12-hour run at step 50, as designed

## Plan 2 — evaluation (complete)

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
- [x] **Final-review fixes before the push**: pin the GPU to a T4 (without it the
      new kernel would likely get a P100 and stop at cell 1); make the upload wait
      immune to a CLI version warning; save every per-question prediction so this
      one GPU session can answer why, not just what. All nine landed and passed a
      final review, and the pinned transformers 5.5.0 / peft 0.19.1 pair passed the
      local smoke run
- [x] **Kaggle start, version 1** (2026-09-24 19:45 UTC): stopped after about
      four minutes, at its first model load. Every gate before that passed on
      Kaggle: a Tesla T4, the pinned transformers 5.5.0 / peft 0.19.1, the code
      fingerprint, and the 1,273 + 4,183 held-out questions. The image ships
      torchao 0.10.0, and peft 0.19.1 raises on any torchao older than 0.16.0
      while it wraps each layer. Neither local environment had torchao, so no
      local run could see it. Reproduced locally by installing torchao 0.10.0,
      then fixed: the install cell removes torchao, which evaluation never uses,
      and the verify cell wraps one tiny layer with LoRA, so a broken optional
      package stops a run in seconds instead of after the model download
- [x] **Kaggle start, version 2** (2026-09-24 19:55 UTC): running,
      [the evaluation kernel](https://www.kaggle.com/code/gb1105/qwen3-4b-medical-fine-tune-evaluation) (private). Same reviewed evaluation code
      (fingerprint `487877cf9130fcab`) with the notebook fix; the notebook as
      pushed is kept at `results/run1/kaggle_eval_notebook.ipynb`
- [x] Collected `eval_medqa.json`, `eval_medmcqa.json` and the run's log
      (`kaggle_eval_output.log`) into `results/run1/`; the result is recorded
      at the top of this file and in the README

The early local signal (16 MedQA questions, base 12/16, fine-tune 9/16) was
generative scoring, and the full run agrees with it: the fine-tune is worse at
written answers on MedQA.

### What the full evaluation must state, whatever it finds

- The adapter was trained on the 4-bit base and is scored on the full-precision
  base. If that biases anything, it biases against the fine-tune.
- Answer-choice scoring asks for the letter immediately. For MedQA that is exactly
  how it was trained; for MedMCQA training put a rationale first, so it may
  understate the fine-tune there.
- MedMCQA answers lean towards A (32%). A model can gain on MedMCQA by guessing A
  more often, which is not medical knowledge; per-question predictions will show
  whether that happened. MedQA's answers are balanced, so it is the cleaner test.
- Generation stops at 512 new tokens. A completion cut off before its final
  answer loses that answer, so the cap costs whichever model writes longer
  answers; the report's hit-cap counts show how often each model hit it.
- The fine-tune was trained to answer MedQA at once. If the base reasons before
  it answers, generative MedQA compares reasoning first with answering directly,
  which can favour the base.

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
- [ ] **Task 6** Method view, Playwright end-to-end tests, and the chat frontend's
      run commands added to the README
- [ ] Screenshot every page at 1280px and 360px and fix what collides or overflows

## Possibly later

- [ ] Before more of the same data, fix the format the fine-tune learned:
      train MedQA with reasoning before the answer line, as the base model
      answers when it does best, and keep MedMCQA rationales short enough to
      finish inside 512 tokens. Evaluation says the first 17k helped only a
      little, so a second session on more of the same data is not the lever.
- [ ] Add the result to the Results page of the chat frontend (Plan 3, Task 5)

## Housekeeping

- [x] **Force-push `main` to GitHub** (2026-09-25). PR #1 had merged the branch
      from before the author email was corrected, crediting 12 commits to
      `jiteshbhalla1-web`. After the push, all 43 commits are authored by
      garvbahl37-gif, and GitHub's contributor list shows only that account.
- [ ] Correct the comment in `training/train.py` claiming the probe's `>=` check
      protects runs shorter than `--probe-steps`: such runs never reach it. After
      the evaluation run, since any change to `training/*.py` changes the code
      fingerprint the notebooks check
- [ ] **Rotate the Kaggle token** `KGAT_...` at kaggle.com/settings: it was pasted
      into a chat session
