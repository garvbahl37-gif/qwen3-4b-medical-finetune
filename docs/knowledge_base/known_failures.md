# Known failures and their fixes

| # | Failure | Evidence | Fix / guard |
|---|---|---|---|
| 1 | Run 1 trained a rationale-first answer format; 70 of 300 MedMCQA written answers never reached a letter within 512 tokens | [measured] results/run1 | v2 puts "Answer: X" first (`format_v2.answer_text`) |
| 2 | Run 2's model learned to skip thinking (real thinking on 302 of 1,273 MedQA questions) | [measured] results/run2 | `/think` or `/no_think` in every prompt (run 3) |
| 3 | Run 2's base worker ran out of memory in the forcing pass and could not recover | [measured] | forced answers 2 at a time; out-of-memory retry outside `except`; expandable segments |
| 4 | Run 3's first evaluation lost the session to two workers downloading the model at once | [measured] | download once, then `HF_HUB_OFFLINE=1` |
| 5 | **SFT on other models' traces degraded native reasoning**: MedQA 69.7 → 60.2, thinking 1,036 → 403 tokens | [measured] run 3 | E2 trains on the model's own verified traces |
| 6 | 135 v2 rows nest `<think>…</think>` inside the thinking, as real special tokens | [measured] audit | not fixed in v2; E2 rejects such samples (`self_traces.reject_reason`) |
| 7 | MedReason traces were written knowing the answer and sometimes contradict it | [measured] audit | v2 unchanged; not used by E2 |
| 8 | Some notebook code assumed `python` was on PATH | [measured] earlier runs | `argv()` maps it to `sys.executable` |
| 9 | Dataset mounts moved under `/kaggle/input/datasets/...` | [measured] earlier runs | `find_code_dir` / `find_data_dir` search by content |
| 10 | Kaggle's file listing pages at 20, so the code upload check never matched once the code had 26 files | [measured] 2026-10-19 | `--page-size 200` in `kaggle_dataset.sh` |
| 11 | `--sizes` / `--samples` rejected `pool` and dev names | [measured] local rehearsal 2026-10-19 | names come from `ALL_STAGES` (with a test) |
| 12 | Stopping a phase's `run_plan` process left its worker holding the GPU | [code] review | phases start in their own process group; `stop()` kills the group |
| 13 | A missing vLLM interpreter raised instead of falling back | [code] test | `run_queue` catches `OSError` and retries with transformers |
| 14 | Rebuilding all notebooks rewrote the fingerprints of runs that had already happened | [measured] 2026-10-03 | `build_notebook.py NAME` rebuilds one; `push_kaggle.sh` passes its kernel |
| 15 | Thinking length was reported as new tokens (thinking + answer) | [code] | `think_tokens` recorded from E0 on |
| 16 | E2 attempt 1 died before training step 1: Unsloth's tokenizer loader looked up a repository the download cell had not fetched, and `HF_HUB_OFFLINE=1` blocked it | [measured] `results/e2/attempt1/e2_train_failed.log` | training runs with the network allowed; only the vLLM workers stay offline. The generation was kept and reused (e2t). |
| 17 | e2t version 1 died 10 minutes in: the train-and-evaluate cell opened a log in `outputs/e2/` before anything had created it | [measured] `results/e2/attempt2/kaggle_e2t_v1_output.log` | `phase()` creates its folder. `tests/test_notebook_cells.py` now executes the notebook's orchestration cells against stand-ins for the GPU processes, the check the module tests could not make. |

Approaches that have been tried and did not beat the base in thinking mode:
the run 2 and run 3 data mixes (external traces, 75/25), at rank 64 and
lr 1e-4.
