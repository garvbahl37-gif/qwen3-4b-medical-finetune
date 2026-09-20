# Fine-tuning Qwen3-4B for medical question answering

Design document. Written 2026-09-20, approved before implementation.

## Goal

A 4-billion-parameter Qwen3 model fine-tuned on medical exam reasoning and
patient-facing dialogue, trained on a free Kaggle T4, measured against the base
model on two held-out benchmarks it never saw, and served behind a chatbot.

The project succeeds if it can state, honestly, whether fine-tuning helped —
including if the answer is no. A negative result reported clearly beats a
positive one that cannot be trusted.

## Why this shape

The sibling text2sql project earned its credibility from execution accuracy: an
objective, unambiguous metric that could not be argued with. Medical text has no
such executor, so the metric here is multiple-choice accuracy on standardised
exam questions. It is the closest available analogue — gradeable without
judgement, and drawn from benchmarks that exist independently of this project.

The conversational half of the training mix exists so the artifact is a chatbot
rather than an exam-taker. It is deliberately not the thing being measured,
because measuring it honestly would require an LLM judge, and a soft number
dressed up as a hard one is worse than no number.

## Non-goals

- Not a diagnostic tool. Not a medical device. Framed as educational throughout.
- No claim that it is safe for clinical use, and no benchmark that implies so.
- No LLM-as-judge score for chat quality.

## 1. Data

### Sources

All four verified reachable and ungated on 2026-09-20.

| source | pool | take | contributes |
|---|---|---|---|
| `openlifescienceai/medmcqa` | 182,822 | 14,000 | answer + written explanation |
| `GBaker/MedQA-USMLE-4-options` (train) | 10,178 | 8,000 | long clinical vignettes |
| `FreedomIntelligence/medical-o1-reasoning-SFT` (en) | 19,704 | 6,000 | explicit chain-of-thought |
| `lavita/ChatDoctor-HealthCareMagic-100k` | 112,165 | 12,000 | patient-facing dialogue |

Target 40,000 examples: 70% exam/reasoning, 30% conversational. Every count is a
CLI flag, not a constant.

### Measured quality problems

Sampling 600 MedMCQA training rows across six offsets:

- **12.5%** have an empty `exp` field.
- **12.5%** more have an `exp` under 80 characters — too short to teach reasoning.
- **33.8%** carry `choice_type != "single"` while still exposing a single `cop`
  index. This is the known-noisy portion of the dataset.

So the usable-with-explanation pool is roughly 137,000, not 182,822. Prep drops
the rest and reports what it dropped, the way the sibling project reports that
21% of its raw SQL does not execute.

### Decontamination

The benchmark is worthless if its answers are in the training set. Every
question — train and eval — is normalised (lowercased, punctuation and
whitespace collapsed) and hashed. Any training row whose hash collides with an
eval row is deleted. The collision count is printed and recorded in the run
report. A run that cannot report this number is not publishable.

### Prompt format

One chat template for all four sources, so the model never has to infer which
dataset a prompt came from:

- **MCQ**: system prompt, then the question with lettered options, then a
  response that gives reasoning followed by `Answer: X`.
- **Dialogue**: system prompt, patient description, doctor-style response.

The trailing `Answer: X` is what makes generative scoring extractable.

## 2. Training

Qwen3-4B, Unsloth, 4-bit QLoRA, rank 32, alpha 32, one T4 pinned via
`CUDA_VISIBLE_DEVICES=0` before torch is imported. A 4B model in 4-bit is ~3.3GB
against 15.6GB of card; splitting across both T4s buys nothing and pays PCIe on
every step.

### Three decisions that are easy to get wrong

**Completion-only loss.** Training on the full sequence spends capacity learning
to generate clinical vignettes, which nothing downstream needs. Loss is masked
to the response span only.

**Sequence length is measured, not assumed.** MedQA vignettes plus chain-of-
thought responses are substantially longer than text2sql's 832-token maximum.
Prep emits a length histogram; `max_seq` is set from the measured p99, and
outliers are dropped rather than truncated. Truncating mid-answer teaches the
model to stop early.

**The throughput probe stops the run.** At step 50 the notebook extrapolates
total training time from observed it/s. Kaggle allows 9 hours per GPU session
and 30 GPU-hours per week; the budget is 30 min prep, 6 h training, 1.5 h
evaluation, 1 h spare. If the projection exceeds the 6-hour training allowance
it raises `SystemExit` with the specific fix (halve batch, double grad accum).
The sibling notebook asks the operator to do this arithmetic manually;
automating it is the difference between losing a minute and losing six hours.
This matters more here because the run goes straight to full scale with no
calibration pass.

**Checkpointing.** Adapter state is saved to `/kaggle/working` every 200 steps
and the trainer resumes from the latest checkpoint if one exists. A session that
dies at hour five loses minutes, not the run.

Executed as *Save Version → Save & Run All*, headless, not an interactive
session that idles out.

## 3. Evaluation

### Held-out sets, never trained on

- **MedQA-USMLE test** — 1,273 questions. A published benchmark.
- **MedMCQA validation** — 4,183 questions, same filters as training.

### Two scoring modes, because they measure different things

**Constrained.** One forward pass; compare logits over the ` A`/` B`/` C`/` D`
continuations and take the argmax. Deterministic, no decoding, cheap enough to
run both full sets. This is the headline number.

**Generative.** Greedy decode, then a deliberately lenient extractor for the
letter — tolerant of markdown, preamble and restated options, because penalising
formatting rather than correctness would distort the comparison. Run on a
300-example subsample of each set.

Both are reported. Constrained accuracy alone can hide a model that knows the
answer and cannot state it, which is precisely the failure that would break the
chatbot.

### Method

Paired base-vs-tuned on identical prompts with identical decoding. Significance
by McNemar's exact test on the paired outcomes. Reported per run:

- accuracy for base and tuned on each set, both modes
- wins / both-correct / regressions / both-wrong
- per-subject breakdown from MedMCQA `subject_name` — where dilution shows
- the decontamination collision count

Scoring lives in one module shared by every entry point, so two runs cannot
report subtly different numbers.

### Chat quality

No invented metric. A hand-written qualitative set of patient questions, read
and reported as prose, plus the safety-guard pass rate from section 4. Labelled
as qualitative.

## 4. Serving and safety

FastAPI, mirroring `serving/app.py` in the sibling project. Backend selected by
environment variable: transformers on GPU, or a q8_0 GGUF on CPU. Deployed
locally behind the already-reserved ngrok domain.

### `screen_message()`

The medical counterpart to the sibling project's `lint_sql()`: the serving layer
**detects rather than repairs**. It flags, before the model answers:

- Emergencies — stroke FAST signs, cardiac chest pain, anaphylaxis, suicidal
  ideation, severe bleeding, infant fever.
- Out-of-scope asks — definitive diagnosis, specific dosing, "should I stop this
  medication".

On an emergency hit the UI shows emergency guidance **above** the model's answer
rather than suppressing it; hiding the response teaches users to work around the
filter. Out-of-scope hits append the disclaimer.

Validated against a control set of benign medical questions and required to fire
on none of them. A guard that fires on everything is equivalent to no guard.

## 5. Frontend

Next.js 16 / React 19, matching the sibling `web/`. Three tabs:

- **Chat** — streaming, multi-turn, persistent safety banner, red-flag callout.
- **Benchmark** — base vs fine-tuned, both scoring modes, per-subject table.
- **Method** — how the data was filtered, decontaminated and scored.

Requests go through `app/api/chat/route.ts`; the browser never addresses the
model host, so no token reaches the client. `MOCK_BACKEND=1` allows building the
UI offline while the GPU run is in flight.

## 6. Kaggle mechanics

Training code uploads as a **private Kaggle Dataset** via the API; the notebook
attaches it rather than cloning GitHub, keeping this project independent of the
text2sql repo. Notebook pushed with `kaggle kernels push`.

Sidebar requirements: accelerator **GPU T4 x2**, Internet **On**. The notebook
verifies compute capability and stops with instructions on a P100, whose
capability 6.0 has no kernels in modern PyTorch builds and which cannot be
changed through the API.

## 7. Layout

```
training/   data preparation, training, evaluation, the Kaggle notebook
serving/    inference server and the safety screen
web/        Next.js chatbot: chat, benchmark, method
tests/      end-to-end check against a deployed URL
docs/       this spec, and results
```

## 8. Risks

| risk | mitigation |
|---|---|
| Full run fails late, burning weekly quota | throughput probe at step 50, checkpoint every 200 steps, resume on restart |
| Fine-tuning does not beat base | reported as a negative result; the sibling project's four runs never beat run 1 |
| Benchmark contamination | hash-based decontamination with a reported count |
| Model states dangerous advice confidently | serving-layer screen, control-set validated, disclaimers in UI |
| Sequence length underestimated, answers truncated | histogram first, p99-derived `max_seq`, drop rather than truncate |

## 9. Explicitly deferred

- GGUF conversion and CPU serving — after a run is worth publishing.
- Publishing weights to the Hugging Face Hub.
- Any second training run.
