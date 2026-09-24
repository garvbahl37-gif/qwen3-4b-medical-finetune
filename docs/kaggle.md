# Reproducing the runs on Kaggle

This runbook reproduces both runs on your own Kaggle account: the training run
that produces the LoRA adapter, and the evaluation that scores it against the
base model. Both run as private Kaggle notebooks on a T4 GPU. The code reaches
Kaggle as a private Kaggle Dataset, and the scripts in `scripts/` do the
uploading and pushing.

## What you need

- A Kaggle account that can use GPUs.
- The Kaggle command-line tool. The scripts look for it at
  `$HOME/.local/bin/kaggle`; set `KAGGLE_BIN` to use another path. This
  project used version 2.2.4.
- `python3` on your `PATH`, which the scripts use to read JSON files.
- A Python to run `scripts/build_notebook.py`. `push_kaggle.sh` uses
  `.venv/bin/python` unless `PYTHON` is set. The builder needs only the
  standard library and `training/kaggle_paths.py`, so `PYTHON=python3` works
  without a virtualenv.
- For the evaluation, an adapter: `adapter_config.json` and
  `adapter_model.safetensors` in one directory. The run-1 adapter is not
  published, so in practice this means running the training first.

## Credentials

The Kaggle command-line tool reads your API credentials from
`~/.kaggle/kaggle.json`. `scripts/kaggle_dataset.sh` also reads that file's
`username` field, and names the datasets it creates after it. Nothing in the
repository stores a credential, and `.gitignore` lists `kaggle.json`.
**Never commit `kaggle.json` or paste its contents anywhere.**

## Change the ids to your account

The kernel metadata names the owner `gb1105`. Replace it with your Kaggle
username in these places:

| File | Field | In the repository |
|---|---|---|
| `training/kernel-metadata.json` | `id` | `gb1105/qwen3-4b-medical-fine-tune-training` |
| `training/kernel-metadata.json` | `dataset_sources` | `gb1105/medical-ft-code` |
| `evaluation/kernel-metadata.json` | `id` | `gb1105/qwen3-4b-medical-fine-tune-evaluation` |
| `evaluation/kernel-metadata.json` | `dataset_sources` | `gb1105/medical-ft-code`, `gb1105/medical-ft-adapter` |
| `tests/test_kernel_metadata.py` | the expected `dataset_sources` | `gb1105/medical-ft-code`, `gb1105/medical-ft-adapter` |

For example, the training kernel's two fields become:

```json
"id": "YOUR_USERNAME/qwen3-4b-medical-fine-tune-training",
"dataset_sources": ["YOUR_USERNAME/medical-ft-code"]
```

Change only the owner. Kaggle derives a kernel's slug from its title and
ignores an id whose slug does not match; the first push then lands under
another name and the second fails with a 409. `tests/test_kernel_metadata.py`
checks that each id's slug equals its title lowercased, with every run of other
characters replaced by `-` and the ends trimmed. If you change a title, change
the slug to match.

The scripts need no edits. The dataset slugs `medical-ft-code` and
`medical-ft-adapter` are fixed in them, and the owner comes from
`kaggle.json`.

## The upload scripts

### `scripts/kaggle_dataset.sh`

```bash
bash scripts/kaggle_dataset.sh <slug> "<title>" <dir>
```

It creates or versions a private Kaggle Dataset from a directory and blocks
until Kaggle serves the new files. Both push scripts call it; you rarely need
to run it yourself.

1. It writes `dataset-metadata.json` into `<dir>`, with the title and the id
   `<username>/<slug>`.
2. It records the name and byte size of every other file in `<dir>`.
3. If `kaggle datasets status` finds the dataset, it uploads a new version
   with `datasets version`; otherwise it runs `datasets create`. Both use
   `--dir-mode skip`, which uploads files and skips subdirectories.
4. It polls `datasets status` every 5 seconds, up to 120 times. On `ready`, it
   also fetches `datasets files --csv` and compares each file's name and size
   with the upload, because after a new version the status can still read
   `ready` for the previous one. It exits 0 only when both agree, exits 1 as
   soon as the status contains `error` or `failed`, and exits 1 after 10
   minutes.

Every `kaggle` call passes `-W`, which turns off the CLI's version warning, and
the file listing is read only after its `name,size,creationDate` header. Each
of these keeps an outdated CLI's warning, which it prints to standard output,
from being read as a file; without them the listing would never match.

### `scripts/push_adapter.sh`

```bash
bash scripts/push_adapter.sh                 # uploads training/outputs/run1
bash scripts/push_adapter.sh path/to/adapter
```

It checks that `adapter_config.json` and `adapter_model.safetensors` exist and
are not empty, stages only those two files, and uploads them as the dataset
`medical-ft-adapter`, titled "Medical Fine-tune Adapter". The tokenizer saved
next to them is the base model's own, and the evaluation loads it from the
base.

### `scripts/push_kaggle.sh`

```bash
bash scripts/push_kaggle.sh              # the training notebook
bash scripts/push_kaggle.sh evaluation   # the evaluation notebook
```

1. It rebuilds both notebooks with `scripts/build_notebook.py`, so the code
   fingerprint built into each one matches the code about to go up. A push
   therefore rewrites `training/kaggle_medical.ipynb` and
   `evaluation/kaggle_eval.ipynb`.
2. It stages `training/*.py` and `training/requirements.txt` flat, with no
   subdirectories. A zipped directory might or might not be extracted on
   Kaggle, and the notebook rebuilds the `training` package from the flat
   files itself.
3. It uploads them as the dataset `medical-ft-code`, titled "Medical Fine-tune
   Code", through `kaggle_dataset.sh`, which waits until Kaggle serves them.
4. It pushes the notebook in the given directory with
   `kaggle -W kernels push -p <dir>` and prints the command's output. The push
   command prints errors such as "Kernel push error..." or "...not valid
   dataset sources..." and still exits 0, so the script reads the output. It
   exits 1 if the output contains "error" or "not valid", or does not contain
   "successfully pushed".
5. It prints the kernel's URL, `https://www.kaggle.com/code/<id>`.

## Run the training

After changing the ids:

```bash
bash scripts/push_kaggle.sh
```

The push uploads the notebook as a new version of the kernel, and that
version runs; run 1 was version 7 of the training kernel. If you start a run
from the Kaggle editor instead, the notebook's first cell says to use Save
Version and then Save & Run All, rather than an interactive session that idles
out.

What run 1 printed along the way, for comparison
([full output](../results/run1/kaggle_training_output.txt)):

| Stage | Run 1 |
|---|---|
| hardware check | `Tesla T4, 15360 MiB`, one visible GPU, capability 7.5 |
| data preparation | 17,410 training rows written after decontamination |
| length check | `suggested --max-seq 960`; 125 examples dropped |
| budget probe at step 50 | `33.16s/step, projected 4.98h for 541 steps` |
| training | 17,332 s |

When it finishes, download `run1-adapter.zip` from the notebook's Output
panel and extract it so that `adapter_config.json` and
`adapter_model.safetensors` sit directly in `training/outputs/run1`, the
directory `push_adapter.sh` reads by default. `training/outputs/` is ignored by
git.

The training notebook installs its packages unpinned. Run 1 got Unsloth
2026.9.7, transformers 5.5.0, TRL 0.24.0, peft 0.19.1 and torch 2.10.0+cu128,
and a new run may get newer releases. `train.py` handles two TRL changes
already (renamed arguments, and the EOS token check) and stops with a message
if a later release breaks either; see the [failure catalogue](#failure-catalogue).

## Run the evaluation

Push the adapter first, because the evaluation kernel attaches it:

```bash
bash scripts/push_adapter.sh
bash scripts/push_kaggle.sh evaluation
```

The notebook builds the two held-out sets, runs a smoke pass on 32 questions
per benchmark, and starts the full run only if the projection fits 27,000 s.
The evaluation plan expects the smoke pass to report within about 15 minutes.
[`docs/evaluation.md`](evaluation.md#the-kaggle-evaluation-notebook) lists
every cell and the condition that stops each one.

## The pinned T4

Both `kernel-metadata.json` files set `"machine_shape": "NvidiaTeslaT4"`, and
Kaggle CLI 2.2.4 sends that field with the push. Without it, a newly created
kernel is likely to be given a P100, whose compute capability of 6.0 has no
kernels in modern PyTorch builds.

Cell 1 of both notebooks is a hardware check. It stops the session on any GPU
below compute capability 7, and its message says to set the accelerator to
GPU T4 x2 in the sidebar if a P100 arrives despite the pin. That accelerator
offers two T4s; both notebooks set `CUDA_VISIBLE_DEVICES=0` before importing
torch and use one.

The T4 has no native bfloat16. Training ran in fp16 (Unsloth reported
`Bfloat16 = FALSE`), and the evaluation loads the model in float16.

Run 1 predates the pin, and its kernel was given a Tesla T4.

## Watching a run and collecting its output

- `push_kaggle.sh` ends with the kernel's URL. Open it to follow the log.
- To confirm that the push landed, list your kernels:

  ```bash
  ~/.local/bin/kaggle kernels list --mine
  ```

- The guards end the session with a message that starts with `STOP.` and
  usually has a `FIX:` line naming what to change. A failed command in a
  notebook step ends it with `Step failed (exit N)` and the command.

Both notebooks copy their results to `/kaggle/working`, which Kaggle shows in
the notebook's Output panel:

| Notebook | Files in the Output panel |
|---|---|
| training | `run1-adapter.zip` (everything `train.py` wrote to `outputs/run1`) and `data_report.json` |
| evaluation | `eval_medqa.json` and `eval_medmcqa.json` |

Run 1's copies are in [`results/run1/`](../results/run1/), with the exact
notebook that ran and its output.

## Session limits

| Limit | Value | How the code handles it |
|---|---|---|
| GPU session length | 9 hours | both notebooks budget below it |
| GPU quota | 30 GPU-hours per week | a run that will not fit stops early instead of spending the quota |
| Training budget | 21,600 s (6 h), `--budget-seconds` | the step-50 probe stops a run projected to exceed it |
| Evaluation budget | 27,000 s (7.5 h), `SESSION_BUDGET` in the notebook | the smoke pass stops the session if the projected full run exceeds it |

The Kaggle limits are as the [design spec](superpowers/specs/2026-09-20-medical-llm-finetune-design.md)
recorded them on 2026-09-20; check Kaggle's current limits before you plan a
run. The 9-hour cap is per session, so more training data means more
sessions, not a longer one. Nothing in the repository carries checkpoints from
one session to the next; `TODO.md` lists a second session continuing from this
adapter as possible later work.

## Failure catalogue

Each row is a failure this project met or guarded against, with the guard now
in the code. Six of them stopped training attempts before run 1: the
asynchronous upload, the kernel slug, the mount path, the guard's own error
message, the EOS token and the step-50 probe.

### Uploads and pushes

| Symptom | Cause | Guard |
|---|---|---|
| The kernel starts with no dataset attached. | `datasets create` and `datasets version` are asynchronous, and the kernel was pushed before the dataset was ready. | `kaggle_dataset.sh` waits for `ready` and a matching file listing before `push_kaggle.sh` pushes. |
| The first push lands under a different kernel name, and the second fails with a 409. | The id's slug did not match the slug Kaggle derives from the title. | Each id equals its slugified title, and `tests/test_kernel_metadata.py` checks it. |
| The kernel runs code older than the repository. | A new version of `medical-ft-code` was still processing, or an older one was attached, while the status read `ready`. | `kaggle_dataset.sh` compares the served file listing with the upload, and cell 3 of each notebook checks the code fingerprint. `TODO.md` records a review catching Kaggle still serving code without `modeling.py`. |
| The upload waits 10 minutes and times out, although the upload worked. | An outdated CLI prints a version warning to standard output, which the listing check would read as a file. | `-W` on every `kaggle` call, and the listing is read only after its header. |
| A failed dataset upload waits the full 10 minutes. | Kaggle reports that failure as `failed`, not `error`. | The wait exits 1 on either. |
| `push_kaggle.sh` prints the kernel URL although the push was rejected. | `kaggle kernels push` prints its error and exits 0. | The script reads the push output and exits 1 on error text or without "successfully pushed". |

### Inside the notebooks

| Symptom | Cause | Guard |
|---|---|---|
| The code cell cannot find `/kaggle/input/medical-ft-code`. | Kaggle mounted the dataset under `/kaggle/input/datasets/...`. Two runs died on that path. | `find_code_dir` and `find_adapter_dir` search `/kaggle/input` by content. |
| A traceback comes from inside a guard's own error message. | The message listed a directory that did not exist. | The guards check that the path exists first, and `train.py`'s diagnostics sit inside `try`/`except`. |
| Cell 1 stops with "Compute capability 6.0 ... has no kernels in modern PyTorch builds". | A P100 was assigned. | `machine_shape` pins `NvidiaTeslaT4`; if it happens anyway, set GPU T4 x2 in the sidebar. |
| Cell 2 of the training notebook prints `INSTALL FAILED`. | Importing Unsloth, TRL, peft or transformers failed after the install cell. | The cell prints a forced reinstall of `unsloth` and `unsloth_zoo` to run, then restart the session and skip the install cell. |
| Cell 2 of the evaluation notebook stops on library versions. | The pinned transformers 5.5.0 or peft 0.19.1 did not take effect. | Run > Restart session, then Run All again. |
| TRL rejects the config's `eos_token`. | Unsloth patches TRL at import time. With `trl` imported first, the `eos_token` fix landed on a different `SFTConfig` class from the one TRL validated. | `train.py` imports Unsloth first, sets `eos_token` in the constructor and on the instance, and verifies it. |
| `train.py` raises `TypeError` naming candidate argument names. | TRL renamed an argument to a name `pick_kwarg` does not know. | The error lists the names the installed TRL accepts; add the new one to the `pick_kwarg` call in `train.py`. |
| Training stops before step 1: completion-only masking did nothing usable. | The chat-template markers did not match the tokenized text. | The message says to print one rendered example and correct `RESPONSE_MARKER` or `INSTRUCTION_MARKER`. |
| Training stops at step 50: "Projected training time is ...". | The run will not fit the 6-hour budget. This is the probe working, and it stopped the 40,000-example attempt. | Halve `--batch-size` and double `--grad-accum`, or request fewer examples with the `prepare_data.py` flags. |
| The evaluation stops at cell 4 on a held-out set size. | The dataset on the Hugging Face Hub has changed since 1,273 and 4,183 were confirmed. | Check the dataset's revision before scoring anything. |
| The smoke pass stops on unparseable answers. | More than half of either model's generative answers could not be read: truncated output, or fp16 garbage from the base. | Read the completions in `outputs/smoke_<name>.json`. If they are cut off, raise `--max-new-tokens`; if they are garbage, half precision is failing on this GPU and the result must not be reported. |
| The smoke pass stops on the projected runtime. | The full run would exceed 27,000 s. | Lower `GEN_LIMIT`; generation dominates the runtime. |
| `evaluate.py` stops on non-finite letter logits. | float16 overflowed on the GPU. | There is no T4 fallback, and the result must not be reported. `--device cpu` is a local diagnostic only. |
