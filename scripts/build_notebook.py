from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Qwen3-4B medical fine-tune (training)

Trains a QLoRA adapter on ~40,000 examples from four medical sources. This
notebook only trains. Evaluation against two held-out benchmarks it never
saw -- MedQA-USMLE test (1,273) and MedMCQA validation (4,183) -- runs as a
**separate** Kaggle notebook once the evaluation code lands, fed the adapter
this notebook produces.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Then Save Version ->
Save & Run All, rather than an interactive session that idles out.

The run goes straight to full scale with no calibration pass, so the probe at
step 50 aborts rather than letting a bad configuration burn six hours.
"""),
    ("code", '''# --- 1. Hardware check (stops here if the GPU is unusable) -----------------
import os, subprocess

# Two T4s are offered, but a 4B model in 4-bit is ~3.3GB against 15.6GB of card.
# Splitting it buys nothing and pays PCIe on every forward and backward. Pin to
# one GPU BEFORE torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
major, minor = torch.cuda.get_device_capability()
print(f"GPU         : {name}")
print(f"visible GPUs: {torch.cuda.device_count()} (pinned to one on purpose)")
print(f"capability  : {major}.{minor}")
print(f"torch       : {torch.__version__}   bf16: {torch.cuda.is_bf16_supported()}")

if major < 7:
    raise SystemExit(
        f"\\nSTOP. Compute capability {major}.{minor} ({name.split(',')[0]}) has no "
        "kernels in modern PyTorch builds.\\nFIX: sidebar -> Accelerator -> "
        "'GPU T4 x2', then Run All. The GPU type cannot be set through the API.")
print("\\nGPU supported." if torch.cuda.is_bf16_supported()
      else "\\nGPU supported. Turing has no bf16; fp16 is selected automatically.")'''),
    ("code", """%%capture
!pip install -q --upgrade pip
!pip install -q unsloth unsloth_zoo
!pip install -q --no-deps trl peft accelerate bitsandbytes
!pip install -q datasets"""),
    ("code", '''# --- 2. Verify the install before spending GPU time on it ------------------
try:
    from unsloth import FastLanguageModel
    import trl, peft, transformers
    print(f"ok | transformers {transformers.__version__} | "
          f"trl {trl.__version__} | peft {peft.__version__}")
except Exception as e:
    print("INSTALL FAILED:", type(e).__name__, e)
    print("\\nFallback, then Run > Restart session and skip the install cell:")
    print("  !pip install -q --upgrade --force-reinstall --no-cache-dir "
          "unsloth unsloth_zoo")'''),
    ("code", '''# --- 3. Get the code from the attached dataset -----------------------------
import os, shutil, subprocess, sys
from pathlib import Path

SRC  = Path("/kaggle/input/medical-ft-code")
WORK = Path("/kaggle/working/ft")
WORK.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC / "training", WORK / "training", dirs_exist_ok=True)
(WORK / "training" / "__init__.py").touch()
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("cwd:", os.getcwd(), "|", len(list((WORK / "training").glob("*.py"))), "modules")

# A failing `!python x.py` returns non-zero but does not raise in Jupyter, so the
# notebook would sail past a dead step and fail later somewhere confusing.
def step(cmd: str):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise SystemExit(f"\\nStep failed (exit {p.returncode}):\\n  {cmd}")
    print("\\nok\\n", flush=True)'''),
    ("markdown", """## 4. Build the training set

14,000 MedMCQA + 8,000 MedQA + 6,000 medical-o1 + 12,000 ChatDoctor, which is
70% exam/reasoning and 30% conversational. Both benchmarks load first so every
training row can be decontaminated against them."""),
    ("code", '''step("python -m training.prepare_data --medmcqa 14000 --medqa 8000 "
     "--medical-o1 6000 --chatdoctor 12000 --val-size 500 --out data")

import json
print(json.dumps(json.load(open("data/report.json")), indent=2))'''),
    ("markdown", """## 5. Measure sequence length

Vignettes plus chain-of-thought run far longer than a guessed default. `max_seq`
comes from the measured p99, and the outliers above it are dropped rather than
truncated mid-answer."""),
    ("code", '''import re, subprocess

out = subprocess.run("python -m training.check_lengths --data data/train.jsonl",
                     shell=True, capture_output=True, text=True).stdout
print(out)
MAX_SEQ = int(re.search(r"suggested --max-seq (\\d+)", out).group(1))
print("MAX_SEQ =", MAX_SEQ)

step(f"python -m training.check_lengths --data data/train.jsonl "
     f"--max-seq {MAX_SEQ} --drop")
step(f"python -m training.check_lengths --data data/val.jsonl "
     f"--max-seq {MAX_SEQ} --drop")'''),
    ("markdown", """## 6. Train

The probe at step 50 extrapolates and aborts if the run will not fit six hours.
If it stops here, halve `--batch-size` and double `--grad-accum` as instructed
and re-run: the effective batch is unchanged."""),
    ("code", '''step(f"python -m training.train --data data --out outputs/run1 "
     f"--max-seq {MAX_SEQ} --batch-size 8 --grad-accum 4 --rank 32 "
     f"--epochs 1 --save-steps 200 --budget-seconds 21600")'''),
    ("code", '''# --- 7. Training stats, and save everything to the Output panel ------------
import json, shutil
from pathlib import Path

stats = json.loads(Path("outputs/run1/train_stats.json").read_text())
print("\\n########## Training stats ##########")
for key, value in stats.items():
    print(f"  {key}: {value}")

out = Path("/kaggle/working")
shutil.copy("data/report.json", out / "data_report.json")
shutil.make_archive(str(out / "run1-adapter"), "zip", "outputs/run1")
print("\\nSaved to /kaggle/working -- download run1-adapter.zip from the "
      "Output panel.")'''),
    ("markdown", """## Done

`run1-adapter.zip` and `data_report.json` are in the **Output** panel.
Download `run1-adapter.zip` and feed it to the evaluation notebook, which
runs as a separate Kaggle session once `training/evalcore.py` and
`training/evaluate.py` exist."""),
]

nb = {
    "cells": [
        {"cell_type": kind, "metadata": {},
         **({"source": src.splitlines(keepends=True)} if kind == "markdown"
            else {"source": src.splitlines(keepends=True),
                  "execution_count": None, "outputs": []})}
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("training/kaggle_medical.ipynb")
path.write_text(json.dumps(nb, indent=1))
print(f"wrote {path}, {len(CELLS)} cells")
