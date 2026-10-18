# Project architecture

All of this is **[code]** unless tagged otherwise.

## From raw data to a score

```
HF datasets ──▶ sources.py / sources_v2.py  normalise to Record (records.py)
            ──▶ decontam.py                 drop benchmark copies (exact, or ≥50% of 13-gram runs) and duplicates
            ──▶ prepare_data_v2.py           run 2/3 training mix   → data/v2/train.jsonl
                prepare_pool_v3.py           E2 question pool + dev → data/v3/eval_pool.jsonl, eval_dev_*.jsonl
            ──▶ format_v2.py                 Record → chat messages: system prompt, question + options,
                                             "/think" or "/no_think", assistant {reasoning_content, "Answer: X\n\n…"}
            ──▶ train.py                     Unsloth QLoRA on unsloth/Qwen3-4B-unsloth-bnb-4bit, TRL SFTTrainer,
                                             loss masked to the assistant turn (train_on_responses_only)
            ──▶ adapter (training/outputs/<run>, git-ignored; uploaded to Kaggle for evaluation)
            ──▶ eval_worker.py               one model per process, stage by stage, until a deadline:
                                             letter: logits after "Answer:" restricted to the options
                                             reasoning: sample with Qwen's settings; if no letter, budget-force
                                             engine: transformers (modeling.py) or vLLM (vllm_engine.py)
            ──▶ eval_report.py               pair models per stage (McNemar, 95% CI), any-correct, majority vote,
                                             per-question export
            ──▶ experiments.py               experiments/results.csv, recomputed from predictions
```

## Entry points

| Command | What it does |
|---|---|
| `python -m training.prepare_data_v2 --out data/v2` | the run 2/3 training data |
| `python -m training.prepare_pool_v3 --out data/v3` | E2's question pool, dev split, and benchmark copies |
| `python -m training.dataset_audit` | `reports/DATASET_AUDIT.md` |
| `python -m training.train --format v2 ...` | training (CUDA plus Unsloth only; runs on Kaggle) |
| `python -m training.eval_worker --role NAME ...` | scores one model |
| `python -m training.run_plan show / run` | an experiment's GPU queues, from `experiments_config.py` |
| `python -m training.self_traces ...` | E2's training set from generated completions |
| `python -m training.eval_report --root DIR ...` | comparison report and per-question records |
| `python -m training.experiments` | rebuilds `experiments/results.csv` |
| `python scripts/build_notebook.py NAME` | builds one Kaggle notebook (e0s1, e0s2, e2, …) |
| `bash scripts/push_kaggle.sh NAME` | uploads `training/*.py` as `medical-ft-code` and pushes that kernel |
| `bash scripts/push_data.sh data/v3 medical-ft-data-v3` | uploads a data directory |

## How a Kaggle run is wired

- **Code** goes up flat as the `medical-ft-code` dataset. Each notebook
  carries the code fingerprint it was built for (`kaggle_paths.code_fingerprint`)
  and stops if the attached code differs. The data fingerprint and adapter
  sha256 are checked the same way.
- **Notebooks are generated** by `scripts/build_notebook.py` from shared cells:
  the hardware check and session clock, a pinned install (unsloth 2026.9.7,
  transformers 5.5.0, trl 0.24.0, peft 0.19.1), a verify step, and the fetch.
- **Evaluation runs in subprocesses**, one per GPU (`CUDA_VISIBLE_DEVICES`).
  Each worker stops 40 minutes before Kaggle's 12-hour limit and keeps what it
  scored.
- **vLLM (E2 on)** is installed into a separate virtual environment
  (`/tmp/vllm-env`) so the training stack stays pinned. `run_plan` runs the
  worker under that interpreter. An entry that fails before writing anything
  is retried with transformers.

## Where things live

| Path | Contents |
|---|---|
| `results/run1..3/` | each run's notebook as pushed, Kaggle log, training stats, loss curve, every prediction |
| `experiments/<id>/experiment.json` | commit, data fingerprint, config, artifact paths |
| `reports/` | the dataset audit |
| `data/` (git-ignored) | built data, rebuilt by the prepare scripts |
| `third_party/trl` (git-ignored) | a reference clone of huggingface/trl, for its docs and its `trl-training` skill |
