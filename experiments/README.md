# Experiments

One directory per experiment, each with an `experiment.json` that records:
- the git commit it ran from;
- the data version;
- the training configuration;
- where its artifacts and raw predictions are.

`results.csv` is generated from those records and the predictions they point at:

```bash
.venv/bin/python -m training.experiments      # rewrites experiments/results.csv
```

No score in `results.csv` is typed in by hand. Every accuracy is recomputed
from the prediction files against the benchmark's gold letters. An experiment
whose predictions have not been downloaded gets empty score cells.

| directory | what it is | status |
|---|---|---|
| `run1/` | First fine-tune: rank 32, lr 2e-4, v1 data, letter-choice evaluation only | done |
| `run2/` | v2 data with reasoning traces, no /think switch | done |
| `run3/` | v2 data with /think and /no_think. Frozen as the baseline fine-tune (tags `run3-train`, `run3-eval`) | done |
| `E0/` | Baseline evaluation: base (A), run 3 fp16 (B), run 3 on its 4-bit training base (C) | planned, not run |

## How to read `results.csv`

The first 25 columns follow Phase 13 of the plan, in its order. The rest
explain those columns:

- **Thinking-mode columns.** `medqa_reasoning`, `medmcqa`, `pubmedqa` and
  `mmlu_medical` are thinking-mode accuracy on the first sample. The
  letter-choice scores are in `medqa_letter` and `*_letter`.
- **Like-for-like scoring.** Within one experiment, each benchmark is scored
  on the questions every model that ran it answered. `n_*` gives the count.
  Run 3's base reached only 480 MMLU-medical questions, so both run 3 rows
  are scored on those 480.
- **Thinking length.** `median_thinking_length` is real thinking tokens from
  E0 on (`thinking_length_kind = think_tokens`). Earlier runs recorded only
  `new_tokens`, which is thinking plus the answer. The 1,036 vs 403 figure
  quoted for run 3 is that measure.
- **Not yet measured.** `general_score`, `safety_score` and `validation_loss`
  are empty: no run has measured them, and no run had a validation split.
- **Run 1 is not comparable.** Run 1's letter scores use its own prompt
  format, so compare them within run 1 only.

## E0: how to run it

E0's settings live in `training/experiments_config.py`, and nowhere else.
Show the plan and its time estimates with:

```bash
.venv/bin/python -m training.run_plan show
```

The estimates are arithmetic on run 3's measured T4 speeds, not measurements
of E0 (see the note on estimates below).

E0 runs over two Kaggle sessions, each about 12 hours on T4 x2. Run both
sessions from the same commit, and record that commit in
`E0/experiment.json`.

```bash
# 1. The run 3 adapter must already be on Kaggle as gb1105/medical-ft-adapter-run3
#    (it is; scripts/push_adapter.sh training/outputs/run3 re-uploads it).
# 2. Upload the code and push session 1. The script rebuilds e0s1's notebook first.
bash scripts/push_kaggle.sh e0s1
# 3. Session 2 can run at the same time if your quota allows, or after session 1.
bash scripts/push_kaggle.sh e0s2
# 4. When each finishes, download its output.
~/.local/bin/kaggle kernels output gb1105/qwen3-4b-medical-e0-baseline-session-1 -p results/e0/s1
~/.local/bin/kaggle kernels output gb1105/qwen3-4b-medical-e0-baseline-session-2 -p results/e0/s2
# 5. Join both sessions into E0's report and per-question table, then refresh the CSV.
.venv/bin/python -m training.eval_report \
    --root results/e0/s1/e0_s1_predictions --root results/e0/s2/e0_s2_predictions \
    --bench-dir data/v2 --out results/e0/e0_report.json \
    --export results/e0/e0_per_question.jsonl
.venv/bin/python -m training.experiments
```

Each session notebook does the following:
1. Checks the code, data and adapter fingerprints.
2. Downloads the base models once.
3. Smoke-runs every GPU queue on two questions.
4. Runs the queues until 40 minutes before Kaggle's limit.

The outputs are:
- `e0_<session>_predictions/gpu<N>/<model>/`: every answer, one JSONL per
  stage, plus `meta.json` with the settings, environment, adapter sha256 and
  per-stage timing;
- `e0_<session>_report.json`;
- `e0_<session>_per_question.jsonl`;
- the queue logs.

`e0_per_question.jsonl` has one record per model, question and sample, with
the fields Phase 2 asks for: `question_id`, `gold_answer`, `prediction`,
`thinking_length`, `correct`, `model` and `mode`. It also has `benchmark`,
`sample`, `forced`, `closed` and `subject`.

In `e0_report.json`, each stage gives:
- every model's accuracy on its own questions;
- the accuracy of all models on the questions they all answered (`common`);
- each model paired with the base (McNemar and a 95% interval for the change);
- any-correct over MedQA's two samples, next to first-sample accuracy on the
  same questions;
- thinking-length median, mean, p90 and p95.

### Estimates and the deadline

The specification wanted the full suite at a 3,072-token budget. Run 3's base
model took 22.9 s per MedQA question at 1,536 tokens, batch 16. At 3,072
tokens the batch has to drop to 8 to fit a T4. That makes reasoning cost 2.4
to 4 times as much per question: the budget doubles, and the per-step speedup
from the smaller batch is unmeasured. The whole suite would come to well
over 100 T4-hours, so reasoning runs on seeded subsets:
- MedQA 200 questions, with 2 samples for A and B;
- MMLU-medical, PubMedQA and MedMCQA 150 each;
- letter choice on all 7,545 questions.

`run_plan show` puts every GPU queue at 4.4 to 10.2 hours of a 10.8-hour
session. If the pessimistic end proves optimistic, the deadline stops the
queue and keeps whatever was scored. Reasoning runs in a seeded order, so a
cut leaves a random sample. On MedQA, sample 0 covers every question before
sample 1 starts.
