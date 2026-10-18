# Qwen3-4B medical fine-tune

The goal is to make Qwen3-4B more accurate on medical multiple-choice QA
(MedQA, MedMCQA, PubMedQA, MMLU-medical) on a free Kaggle T4, without losing
its own reasoning.

**Current state (2026-10-19).**
- **The bar.** The base model is the best model so far: MedQA 69.7% in
  thinking mode at a 1,536-token budget.
- **Run 3.** Fine-tuning on other models' traces lowered that to 60.2% and
  shortened thinking from 1,036 to 403 tokens.
- **E2 is running on Kaggle.** It trains on the base model's own
  gold-verified reasoning.

## Before any major change, read

- `docs/knowledge_base/README.md`: the index, with a table of which document
  to read for which task.
- `docs/knowledge_base/known_failures.md`: what already broke, and the guard
  for each.
- `docs/knowledge_base/open_questions.md`: the ranked hypotheses and their
  pre-registered success and failure criteria.

## Entry points

- Data:
  - `python -m training.prepare_data_v2` builds runs 2–3's data.
  - `python -m training.prepare_pool_v3` builds E2's pool and dev split.
- Training: `python -m training.train --format v2 ...` (CUDA and Unsloth;
  runs on Kaggle).
- Evaluation:
  - `python -m training.eval_worker` scores one model.
  - `python -m training.eval_report --root DIR` writes comparisons and
    per-question records.
- Experiments:
  - `training/experiments_config.py` holds every setting.
  - `python -m training.run_plan show|run` runs the queues.
  - `python -m training.experiments` rebuilds `experiments/results.csv`.
- Kaggle:
  - `python scripts/build_notebook.py NAME` builds one notebook.
  - `bash scripts/push_kaggle.sh NAME` pushes it.
  - `bash scripts/push_data.sh DIR SLUG` uploads data.

## Tests

`.venv/bin/python -m pytest` (`pytest.ini` adds `-q`; pass `-o addopts=""`
for a summary). Every change needs a failing test first. Rehearse new
Kaggle wiring locally on `Qwen/Qwen3-0.6B` with `--device mps` before
pushing.

## Rules

- **Compare against the base.** Every candidate is compared with the base
  model, and with the best existing adapter, under identical generation
  settings and the same engine. Select on the **dev** split; report the
  benchmarks; never tune on them.
- **Write the hypothesis first.** Record it and its acceptance criteria in
  `open_questions.md` before the run.
- **Record every result** in `experiments/<id>/experiment.json`, regenerated
  into `results.csv`. No hand-typed scores.
- **Never overwrite** `results/run*` or a past run's notebook.
- **Never write** a Kaggle or Hugging Face token anywhere.
- **Commits** are credited to garvbahl37-gif only, with no co-author or
  "generated" trailers, and pushed straight to `main`.
- **Benchmark accuracy is not clinical competence.** Say so wherever
  results are presented.
