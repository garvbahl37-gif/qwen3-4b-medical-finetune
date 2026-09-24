# Training

This page describes how run 1, the only training run so far, was built and
what it measured. The code is in `training/`. The run's own record is in
[`results/run1/`](../results/run1/): the stats, the data report, the loss
curve, and the exact notebook that trained the adapter with its output.

## The steps

Run 1 ran these commands inside the Kaggle notebook, from `/kaggle/working/ft`.
They are copied from [the notebook that ran](../results/run1/kaggle_training_notebook.ipynb)
and [its output](../results/run1/kaggle_training_output.txt):

```bash
python -m training.prepare_data --medmcqa 6300 --medqa 3700 --medical-o1 2600 --chatdoctor 5400 --val-size 500 --out data
python -m training.check_lengths --data data/train.jsonl
python -m training.check_lengths --data data/train.jsonl --max-seq 960 --drop
python -m training.check_lengths --data data/val.jsonl --max-seq 960 --drop
python -m training.train --data data --out outputs/run1 --max-seq 960 --batch-size 8 --grad-accum 4 --rank 32 --epochs 1 --save-steps 200 --budget-seconds 21600
```

The notebook reads 960 from the first `check_lengths` call, which prints
`suggested --max-seq 960`, and passes it to the calls after it.

`scripts/build_notebook.py` generates the notebook. `training/kaggle_medical.ipynb`
is its current build, and it has changed since run 1: its code-fetch cell, for
example, now checks that the attached code is the code the notebook was built
for. `results/run1/kaggle_training_notebook.ipynb` is the record of what ran.

## Data sources and filtering

Each source is normalised into one `Record` (`training/records.py`) with the
fields `id`, `source`, `kind` (`mcq` or `dialogue`), `question`, `options`,
`answer`, `rationale`, `response` and `subject`.

| Key | Hugging Face id, config, split | Kind | What a row becomes |
|---|---|---|---|
| `medmcqa` | `openlifescienceai/medmcqa`, `default`, `train` | mcq | question, options A to D, answer from `cop`, rationale from `exp`, subject from `subject_name` |
| `medqa` | `GBaker/MedQA-USMLE-4-options`, `default`, `train` | mcq | question, options A to D, answer from `answer_idx`, no rationale, subject from `meta_info` |
| `medical_o1` | `FreedomIntelligence/medical-o1-reasoning-SFT`, `en`, `train` | dialogue | question from `Question`; response is `Complex_CoT`, a blank line, then `Response` |
| `chatdoctor` | `lavita/ChatDoctor-HealthCareMagic-100k`, `default`, `train` | dialogue | question from `input`, response from `output` |

Each dataset has its own terms of use; read its card on Hugging Face
([MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa),
[MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options),
[medical-o1](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT),
[ChatDoctor](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k))
before reusing it.

The filters in `training/sources.py`:

- MedMCQA drops a row when its `choice_type` is not `single`, when any of the
  four options is blank, when `cop` is not an integer from 0 to 3, when `exp`
  is shorter than 80 characters, or when the question is blank.
- MedQA drops a row unless its options are exactly A to D and all non-empty,
  its `answer_idx` is one of A to D, and its question is non-empty.
- medical-o1 and ChatDoctor drop a row whose question or response is empty.

The MedMCQA filters come from a sample of 600 MedMCQA training rows taken for
the [design spec](superpowers/specs/2026-09-20-medical-llm-finetune-design.md):
12.5% had an empty `exp`, a further 12.5% had one under 80 characters, and
33.8% were marked `choice_type` other than `single` while still giving a single
answer index. The comment in `sources.py` explains the 80-character floor:
below that length the field says "correct" rather than why, and teaches the
model nothing.

`sources.load()` shuffles a split's row indices with the seed (42) and walks
them, keeping rows that pass the filters, until it has the number requested.
If the filters cannot supply that many, it stops with an error that names the
flag to lower. It does not return fewer rows, because a short source would make
the reported mix false. A limit of 0 keeps every row that passes, which is how
the benchmarks are loaded.

What run 1 asked for and got:

| Source | Requested | Rows inspected | Kept | Keep rate |
|---|---|---|---|---|
| `medmcqa` | 6,300 | 12,527 | 6,300 | 50.3% |
| `medqa` | 3,700 | 3,700 | 3,700 | 100.0% |
| `medical_o1` | 2,600 | 2,600 | 2,600 | 100.0% |
| `chatdoctor` | 5,400 | 5,400 | 5,400 | 100.0% |

The requests hold the spec's 70/30 split: 12,600 exam and reasoning examples
against 5,400 patient dialogues. The `prepare_data.py` defaults (14,000, 8,000,
6,000 and 12,000) are the spec's original 40,000-example plan, which the
budget probe ruled out (see [the step-50 budget probe](#the-step-50-budget-probe)).

## Decontamination and the held-out sets

`prepare_data.py` loads both benchmarks before any training data:

| Benchmark | Dataset and split | Loaded with | Questions |
|---|---|---|---|
| MedQA-USMLE test | `GBaker/MedQA-USMLE-4-options`, `test` | the `load()` defaults | 1,273 |
| MedMCQA validation | `openlifescienceai/medmcqa`, `validation` | `require_rationale=False`, `require_single_choice=False` | 4,183 |

Together that is 5,456 questions, and run 1 kept every row of both. MedMCQA
validation is loaded with both training filters off. With them on, the
benchmark would lose the rows without an explanation and the rows marked
multi-choice, and the score would stop being comparable with published MedMCQA
figures. The noisy rows cost the base model and the fine-tune alike.

Every question, from training and from the benchmarks, is reduced to a key:
Unicode NFKD, lowercase, punctuation replaced by spaces, whitespace collapsed.
A training row is dropped when its key matches a benchmark question or a
training row kept before it. The step therefore also removes duplicates within
the training set, which would otherwise reweight whatever they duplicate.

Run 1 dropped 90 of 18,000 rows (0.50%). The script counts benchmark matches
and in-set duplicates together, so the output does not say how many of the 90
were benchmark questions.

Two limits follow from the method. Matching is exact after normalisation, so a
paraphrased benchmark question would pass. And it compares question text only,
so two items that share a stem but differ in their options count as one, and
the second is dropped.

After decontamination the rows are shuffled (seed 42). The first 500 go to
`data/val.jsonl` and the rest to `data/train.jsonl`, and the counts go to
`data/report.json`, kept as [`results/run1/data_report.json`](../results/run1/data_report.json):

| Source | Rows in `train.jsonl` |
|---|---|
| `medmcqa` | 6,043 |
| `medqa` | 3,601 |
| `medical_o1` | 2,520 |
| `chatdoctor` | 5,246 |
| total | 17,410 |

The report counts the mix two ways. By kind, 55.4% of rows are multiple choice
and 44.6% free text (`mcq_fraction` 0.5539). By source, 69.9% are exam and
reasoning and 30.1% patient dialogue (`reasoning_fraction` 0.6987), against the
spec's 70/30 target; medical-o1 is free text but counts as reasoning. Both
figures describe the 17,410 rows before the length drop below.

`train.py` does not read `val.jsonl`: no evaluation strategy is set, because
evaluation runs as a separate notebook. The same 5,456 questions are the
evaluation's test sets, and [`evaluate.py` stops](evaluation.md) on a test file
whose MedQA rows are not 1,273 or whose MedMCQA rows are not 4,183.

## Prompt format

`training/prompts.py` builds every prompt, for training and for evaluation, so
the two cannot drift apart. Multiple-choice rows get `SYSTEM_MCQ`:

> You are a medical education assistant. Answer the multiple-choice question by reasoning briefly from the clinical findings, then stating your choice on a final line in the exact form 'Answer: X'.

Dialogue rows get `SYSTEM_CHAT`:

> You are a medical education assistant. Explain clearly and carefully in plain language. You are not a substitute for a clinician: do not give a definitive diagnosis, and advise the person to seek in-person care when their description warrants it.

A multiple-choice question becomes this user turn:

```text
{question}

A. {option A}
B. {option B}
C. {option C}
D. {option D}
```

and this assistant target:

```text
{rationale}

Answer: C
```

A row with no rationale, which is every MedQA row, gets the `Answer:` line
alone. A dialogue row uses `SYSTEM_CHAT`, the patient's text or the question
as the user turn, and the response as the target.

`train.py` renders each example with the tokenizer's chat template. Qwen3's
template writes the assistant turn after an empty think block,
`<think>\n\n</think>\n\n`, and run 1's log shows an example's first trained
tokens starting with it. The evaluation reproduces that prefix at inference;
see [prompt parity](evaluation.md#prompt-parity-with-training).

The final `Answer: X` line is what both evaluation modes read. Constrained
scoring appends `Answer:` to the prompt and compares the next-token logits of
the four letters, and generative scoring extracts the letter the model states.

## Sequence length

`check_lengths.py` renders each example with its answer, tokenizes it with the
tokenizer of `unsloth/Qwen3-4B-unsloth-bnb-4bit` (the `--model` default), and
counts tokens. It reports nearest-rank percentiles and suggests `max_seq` as
the p99 rounded up to a multiple of 64. With `--max-seq` and `--drop` it
rewrites the file without the longer examples. With `--max-seq` alone it stops
if any example is longer, because truncating mid-answer teaches the model to
emit unfinished responses.

| File | Examples | Median | p90 | p99 | Max | Over 960, dropped |
|---|---|---|---|---|---|---|
| `train.jsonl` | 17,410 | 270 | 621 | 929 | 3,403 | 125 (0.72%) |
| `val.jsonl` | 500 | 276 | 615 | 881 | 1,357 | 2 (0.40%) |

A p99 of 929 rounds up to 960, and 17,285 training examples remained.

Token counts come from `apply_chat_template(..., tokenize=True, return_dict=False)`.
Newer transformers releases return a dict-like `BatchEncoding` for
`tokenize=True`, and its `len()` is the number of keys. [`TODO.md`](../TODO.md)
records that this was caught during development: every example would have
measured 2 tokens, and training would have run at a `max_seq` of 64.

## QLoRA setup

| Setting | Value |
|---|---|
| Base weights | `unsloth/Qwen3-4B-unsloth-bnb-4bit`, loaded in 4-bit |
| LoRA rank, alpha, dropout | 32, 32, 0.0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | Unsloth's (`"unsloth"`) |
| Batch | 8 per device, with 4 gradient-accumulation steps: 32 examples per optimizer step |
| Epochs | 1 |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.03 |
| Optimizer | `adamw_8bit`, weight decay 0.01 |
| Precision | fp16 when Unsloth reports no bf16 support, which it did on the T4 (`Bfloat16 = FALSE`) |
| `max_seq` | 960 |
| Seed | 42 |
| Logging and saving | loss every 10 steps; a checkpoint every 200 steps, the last 2 kept |

Loss is computed on the response only. Unsloth's `train_on_responses_only`
masks everything outside the assistant turns, found by the markers
`<|im_start|>user\n` and `<|im_start|>assistant\n`, so the model is not trained
to reproduce the vignette it was given. If the markers fail to match, the mask
silently does nothing, so `verify_masking` collates two rows before training
starts and stops the run if no token is masked or none is left to train on. Run 1 printed `masking verified: 90 tokens masked, 351 trained`.

TRL checks `SFTConfig.eos_token` against the tokenizer's vocabulary. `train.py`
imports Unsloth before transformers and TRL, since Unsloth patches both at
import time. It passes the tokenizer's EOS token in the `SFTConfig`
constructor, assigns it on the instance as well, stops if the assignment did
not hold, and prints the config class it built. Run 1 printed
`SFTConfig.eos_token = '<|im_end|>' (tokenizer EOS '<|im_end|>', id 151645)`.

TRL has renamed two arguments between releases. `pick_kwarg` inspects the
installed signatures and passes the tokenizer as `processing_class` or
`tokenizer`, and the length as `max_seq_length` or `max_length`. If neither
name is accepted it raises `TypeError` before the model downloads.

The notebook sets `CUDA_VISIBLE_DEVICES=0` before importing torch. Its comment
gives the reason: a 4B model in 4-bit is about 3.3 GB against 15.6 GB of card,
so splitting it across two T4s buys nothing and pays PCIe traffic on every
forward and backward pass.

Run 1's output records these library versions: Unsloth 2026.9.7, transformers
5.5.0, TRL 0.24.0, peft 0.19.1 and torch 2.10.0+cu128. The training notebook
installs its packages unpinned (`unsloth` and `unsloth_zoo`, then `trl`,
`peft`, `accelerate` and `bitsandbytes` with `--no-deps`, then `datasets`), so
a new run may get newer releases.

## The step-50 budget probe

The probe is a trainer callback. At the first optimizer step at or past
`--probe-steps` (default 50), it divides the elapsed time by the steps done,
multiplies by the run's total steps, and prints the projection. If the
projection exceeds `--budget-seconds` (21,600 s, or 6 hours), it stops the run
and names the fix: halve `--batch-size` and double `--grad-accum`, which keeps
the effective batch, or cut the training set with the `prepare_data.py` flags.
It fires once, on the first step at or past the threshold.

The 6-hour budget comes from the spec, which records Kaggle's limits as 9 hours
per GPU session and 30 GPU-hours per week. It planned one session as 30
minutes of preparation, 6 hours of training, 1.5 hours of evaluation and an
hour spare. Evaluation later moved to its own notebook and session.

The probe stopped one attempt before run 1. That attempt requested the spec's
40,000 examples, and its probe measured 35.5 s per step at an effective batch
of 32, or 0.9 examples per second. At that rate 40,000 examples take about 12
hours, against a 9-hour session cap, so the probe stopped the run at step 50.
Run 1 requested 18,000 examples instead, held the 70/30 mix, and kept
`--budget-seconds` at 21,600.

Run 1's probe printed `probe: 33.16s/step, projected 4.98h for 541 steps`. The
run took 17,332 s (4.81 hours).

## Checkpointing

With `--save-steps 200`, the trainer writes a checkpoint into `--out` every 200
steps and keeps the latest two. `train.py` resumes when `--out` already holds a
`checkpoint-*` directory. In the notebook, `--out` is `outputs/run1` under
`/kaggle/working/ft`, and nothing in the repo copies checkpoints into a new
Kaggle session. Run 1 finished in one session and did not resume.

At the end, `train.py` saves the adapter and tokenizer to `--out` and writes
`train_stats.json`. The notebook copies `data/report.json` to
`/kaggle/working/data_report.json` and zips `outputs/run1` into
`/kaggle/working/run1-adapter.zip` for download.

## What run 1 measured

| Measure | Value | Where it is recorded |
|---|---|---|
| Examples trained | 17,285 | `train_stats.json` |
| Optimizer steps | 541 | the output's probe line |
| Runtime | 17,332 s (4.81 h) | `train_stats.json` |
| Throughput | 0.997 examples/s, 0.031 steps/s | the output's final trainer line |
| Mean training loss | 1.796 | `train_stats.json` |
| Logged loss at step 10 | 2.912 | `loss_curve.json` |
| Logged loss at step 150 | 1.754 | `loss_curve.json` |
| Logged loss from step 150 to step 540 | between 1.666 and 1.827 | `loss_curve.json` |
| Logged loss at step 540 | 1.752 | `loss_curve.json` |
| Adapter size | 264 MB | [`TODO.md`](../TODO.md) |

[`loss_curve.json`](../results/run1/loss_curve.json) holds the 54 losses the
trainer logged, one every 10 steps, as epoch and loss pairs.
[`train_stats.json`](../results/run1/train_stats.json) also records `max_seq`,
`batch_size`, `grad_accum` and `rank`.

TODO.md notes that training loss levelled off at about step 150, roughly
4,800 examples in, and barely moved over the remaining 390 steps. It adds that
only evaluation can say whether the second half helped, but that the flat
curve is a reason not to assume more of the same data is the lever. Training
loss is not a benchmark score, and no evaluation has run yet.

## Kaggle attempts that failed before run 1

Run 1 was version 7 of the training kernel. [`TODO.md`](../TODO.md) lists what
stopped the attempts before it, and each cause left a guard in the code:

| What stopped the attempt | Guard now in the code |
|---|---|
| `kaggle datasets create` is asynchronous, and the kernel started before the dataset existed. | `scripts/kaggle_dataset.sh` waits until Kaggle reports the dataset ready and its file listing matches the upload. |
| The dataset mounted under `/kaggle/input/datasets/...`, not at `/kaggle/input/medical-ft-code`. Two runs died on that path. | `find_code_dir` in `training/kaggle_paths.py` searches `/kaggle/input` for the directory that holds every required module. `tests/test_kaggle_paths.py` covers it. |
| A guard's own error message raised an exception while it reported a missing directory. | The guards check that `/kaggle/input` exists before listing anything under it, and the diagnostic prints in `train.py` sit inside `try`/`except` so they cannot crash the run they explain. |
| The kernel id did not match the slug Kaggle derives from the title, so the first push landed under another name and the second was rejected with a 409. | Each kernel id equals the slug of its title, and `tests/test_kernel_metadata.py` checks both kernels. |
| `trl` was imported before `unsloth`, so the EOS token was set on a different `SFTConfig` class from the one TRL validated. | `train.py` imports Unsloth first, sets `eos_token` in the constructor and on the instance, verifies it, and prints the class it built. |
| The step-50 probe stopped the 40,000-example run, projected at about 12 hours, as designed. | The training set was resized to 18,000 requested examples, from measured throughput. |

The [evaluation plan](superpowers/plans/2026-09-24-evaluation.md) counts three
failed training runs caused by Unsloth's patching of TRL, which is one reason
the evaluation does not use Unsloth. [`docs/kaggle.md`](kaggle.md#failure-catalogue)
lists these together with the guards added for the evaluation run.

## Command-line reference

`python -m training.prepare_data`:

| Flag | Default | Run 1 |
|---|---|---|
| `--medmcqa` | 14000 | 6300 |
| `--medqa` | 8000 | 3700 |
| `--medical-o1` | 6000 | 2600 |
| `--chatdoctor` | 12000 | 5400 |
| `--val-size` | 500 | 500 |
| `--seed` | 42 | 42 |
| `--out` | `data` | `data` |

`python -m training.check_lengths`:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | the JSONL file to measure |
| `--model` | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | whose tokenizer counts the tokens |
| `--max-seq` | 0 | 0 reports and suggests a value; any other value checks against it |
| `--drop` | off | rewrite `--data` without the examples over `--max-seq` |

`python -m training.train`:

| Flag | Default | Run 1 |
|---|---|---|
| `--data` | `data` | `data` |
| `--out` | required | `outputs/run1` |
| `--max-seq` | required | 960 |
| `--batch-size` | 8 | 8 |
| `--grad-accum` | 4 | 4 |
| `--rank` | 32 (alpha is set equal to the rank) | 32 |
| `--epochs` | 1.0 | 1 |
| `--lr` | 2e-4 | 2e-4 |
| `--save-steps` | 200 | 200 |
| `--budget-seconds` | 21600 | 21600 |
| `--probe-steps` | 50 | 50 |
