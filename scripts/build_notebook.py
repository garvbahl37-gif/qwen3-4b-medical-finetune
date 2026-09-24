from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.kaggle_paths import code_fingerprint

# The notebook's code-fetch cell cannot `import training.kaggle_paths`: the
# whole point of find_code_dir is to locate the uploaded code before it has
# been copied anywhere importable. So its real implementation lives in
# training/kaggle_paths.py (where tests/test_kaggle_paths.py can import and
# test it directly), and this reads that file's source and inlines the part
# below its marker comment into the cell verbatim, rather than hand-copying
# it into both places and letting them drift apart.
_KAGGLE_PATHS = Path("training/kaggle_paths.py").read_text()
_MARKER = "# --- inline below ---\n"
if _MARKER not in _KAGGLE_PATHS:
    raise SystemExit(f"marker {_MARKER!r} not found in training/kaggle_paths.py")
FIND_CODE_DIR_SRC = _KAGGLE_PATHS.split(_MARKER, 1)[1].rstrip("\n")

# The exact code this build produces for upload. If Kaggle ever mounts an
# older version -- a new upload still processing, or an old one attached by
# hand -- the modules carry the right names and the wrong code, and nothing
# else would notice. Both fetch cells check this before doing anything else.
CODE_FINGERPRINT = code_fingerprint(Path("training"))

CHECK_FINGERPRINT_SRC = ('''EXPECTED_FINGERPRINT = "''' + CODE_FINGERPRINT + '''"
if code_fingerprint(SRC) != EXPECTED_FINGERPRINT:
    raise SystemExit(
        f"\\nSTOP. The attached code ({code_fingerprint(SRC)}) is not the code this "
        f"notebook was built for ({EXPECTED_FINGERPRINT}).\\n"
        "Kaggle may still be processing a new version of medical-ft-code, or an "
        "older version is attached.\\n"
        "FIX: wait a minute and re-run; if it persists, run scripts/push_kaggle.sh again.")''')

CELL_GET_CODE = ('''# --- 3. Get the code from the attached dataset -----------------------------
import os, shutil, subprocess, sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/ft")
PKG   = WORK / "training"
PKG.mkdir(parents=True, exist_ok=True)

''' + FIND_CODE_DIR_SRC + '''

SRC = find_code_dir(INPUT)
''' + CHECK_FINGERPRINT_SRC + '''
print("found the code at:", SRC)

for src_file in sorted(SRC.glob("*.py")):
    shutil.copy(src_file, PKG / src_file.name)
(PKG / "__init__.py").touch()

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print("cwd:", os.getcwd(), "|", len(list(PKG.glob("*.py"))), "modules:",
      ", ".join(sorted(p.stem for p in PKG.glob("*.py"))))

# A failing `!python x.py` returns non-zero but does not raise in Jupyter, so the
# notebook would sail past a dead step and fail later somewhere confusing.
def step(cmd: str):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise SystemExit(f"\\nStep failed (exit {p.returncode}):\\n  {cmd}")
    print("\\nok\\n", flush=True)''')

CELL_GET_CODE_EVAL = ('''# --- 3. Get the code and the adapter from the attached datasets ------------
import os, shlex, shutil, subprocess, sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/ft")
PKG   = WORK / "training"
PKG.mkdir(parents=True, exist_ok=True)

''' + FIND_CODE_DIR_SRC + '''

SRC = find_code_dir(INPUT, EVAL_REQUIRED)
''' + CHECK_FINGERPRINT_SRC + '''
ADAPTER = find_adapter_dir(INPUT)
# step() runs through a shell, so the path is quoted: a mount path with a space
# in it would otherwise split into two arguments.
ADAPTER_ARG = shlex.quote(str(ADAPTER))
print("code:   ", SRC)
print("adapter:", ADAPTER)

for src_file in sorted(SRC.glob("*.py")):
    shutil.copy(src_file, PKG / src_file.name)
(PKG / "__init__.py").touch()

os.chdir(WORK)
sys.path.insert(0, str(WORK))
Path("data").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# A failing `!python x.py` returns non-zero but does not raise in Jupyter, so the
# notebook would sail past a dead step and fail later somewhere confusing.
def step(cmd: str):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise SystemExit(f"\\nStep failed (exit {p.returncode}):\\n  {cmd}")
    print("\\nok\\n", flush=True)''')

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Qwen3-4B medical fine-tune (training)

Trains a QLoRA adapter on ~17,700 examples from four medical sources -- sized
to measured T4 throughput, not the original 40,000-example estimate; see
"Build the training set" below. This notebook only trains. Evaluation
against two held-out benchmarks it never
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
        "kernels in modern PyTorch builds.\\nFIX: kernel-metadata.json pins "
        "machine_shape to a T4 already; if a P100 still arrived, set sidebar -> "
        "Accelerator -> 'GPU T4 x2' by hand, then Run All again.")
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
    ("code", CELL_GET_CODE),
    ("markdown", """## 4. Build the training set

6,300 MedMCQA + 3,700 MedQA + 2,600 medical-o1 + 5,400 ChatDoctor -- 18,000
requested, holding the spec's 70% exam/reasoning and 30% patient-dialogue
split (12,600 / 5,400). This size is set by measured throughput, not chosen
in advance: the step-50 probe on run 1 measured 35.5s/step at effective
batch 32 on a single T4, i.e. 0.9 examples/sec, and 40,000 examples at that
rate is a 12-hour run against a 9-hour Kaggle session cap. 18,000 lands
around 17,700 after decontamination and the length drop, which is 552 steps
-- about 5.4 hours, inside the 6-hour training budget with real margin.
Both benchmarks load first so every training row can be decontaminated
against them."""),
    ("code", '''step("python -m training.prepare_data --medmcqa 6300 --medqa 3700 "
     "--medical-o1 2600 --chatdoctor 5400 --val-size 500 --out data")

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

`run1-adapter.zip` and `data_report.json` are in the **Output** panel. The
adapter was trained on ~17,700 examples (not the spec's original 40,000 --
see "Build the training set" above for why). Download `run1-adapter.zip`
and feed it to the evaluation notebook, which runs as a separate Kaggle
session once `training/evalcore.py` and `training/evaluate.py` exist."""),
]

EVAL_CELLS: list[tuple[str, str]] = [
    ("markdown", """# Qwen3-4B medical fine-tune (evaluation)

Scores the fine-tuned adapter against the base model on two benchmarks it never
trained on: MedQA-USMLE test (1,273) and MedMCQA validation (4,183). Same
prompts, same greedy decoding, both models from one load -- the base is the same
weights with the adapter switched off.

It runs in two stages in one session. A smoke pass scores 32 questions per
benchmark through the exact code the full run uses, measures its speed, and
stops if the full run would not fit the session. Only then does the full run
start.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach `medical-ft-code`
and `medical-ft-adapter`."""),
    CELLS[1],
    ("code", """%%capture
# Pin the exact pair that already trained the adapter on this image: peft
# 0.19.1 wrote the adapter's config, and the image's default transformers has
# never been confirmed -- an older default could ignore dtype= and load 16GB
# of fp32 weights onto a 15GB T4. --no-deps on peft keeps its own dependency
# resolution from replacing the transformers version just pinned.
!pip install -q "transformers==5.5.0"
!pip install -q --no-deps "peft==0.19.1"
!pip install -q datasets"""),
    ("code", '''# --- 2. Verify the install before spending GPU time on it ------------------
import torch, transformers, peft
print(f"ok | torch {torch.__version__} | transformers {transformers.__version__} "
      f"| peft {peft.__version__}")

if transformers.__version__ != "5.5.0" or peft.__version__ != "0.19.1":
    raise SystemExit(
        f"\\nSTOP. transformers {transformers.__version__} / peft {peft.__version__} "
        "-- expected the pinned transformers==5.5.0 / peft==0.19.1 this adapter "
        "was trained with.\\nFIX: the pip install above did not take effect on "
        "this kernel image. Run > Restart session, then Run All again.")'''),
    ("code", CELL_GET_CODE_EVAL),
    ("markdown", """## 4. Build the held-out sets

Exactly the sets training was decontaminated against. MedMCQA validation is
loaded with both filters off: training drops rows without an explanation and
rows marked multi-choice, but the benchmark keeps all 4,183, or the score would
not be comparable to any published MedMCQA figure."""),
    ("code", '''import json
from training.evaluate import EXPECTED_HOLDOUT_SIZES as EXPECTED
from training.sources import load

HOLDOUTS = {
    "medqa": ("test", {}),
    "medmcqa": ("validation", {"require_rationale": False,
                               "require_single_choice": False}),
}

for name, (split, flags) in HOLDOUTS.items():
    recs = load(name, limit=0, split=split, **flags)
    if len(recs) != EXPECTED[name]:
        raise SystemExit(
            f"\\nSTOP. {name} {split} gave {len(recs):,} rows, expected "
            f"{EXPECTED[name]:,}.\\n"
            "FIX: the dataset on the Hugging Face Hub has changed since these "
            "sizes were confirmed. Check its revision before scoring anything.")
    with open(f"data/holdout_{name}.jsonl", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec.to_dict()) + "\\n")
    print(f"data/holdout_{name}.jsonl  {len(recs):,}")'''),
    ("markdown", """## 5. Smoke pass

32 questions per benchmark through the exact code the full run uses. It proves
the path works on this GPU, measures real throughput, and stops here if the full
run would not fit the session. It also stops if either model's answers mostly
cannot be parsed -- for the fine-tune that means truncated output, and for the
base it can also mean fp16 garbage, either of which would produce a comparison
that flatters the fine-tune rather than a real result."""),
    ("code", '''from training.evaluate import project_eval_seconds

SESSION_BUDGET = 27_000      # 7.5h of Kaggle's 9h GPU session; the rest is margin
GEN_LIMIT = 300

projected = 0
for name in ("medqa", "medmcqa"):
    step(f"python -m training.evaluate --adapter {ADAPTER_ARG} "
         f"--test data/holdout_{name}.jsonl --limit 32 --gen-limit 16 "
         f"--out outputs/smoke_{name}.json")
    smoke = json.load(open(f"outputs/smoke_{name}.json"))
    projected += project_eval_seconds(smoke["timing"],
                                      n_constrained=EXPECTED[name],
                                      n_generative=GEN_LIMIT)
    gen = smoke["reports"]["generative"]
    cap = smoke["generation"]["hit_cap"]
    print(f"{name}: unparseable tuned {gen['tuned_unparseable']}/{gen['n']}, "
          f"base {gen['base_unparseable']}/{gen['n']}  |  "
          f"hit cap tuned {cap['tuned']}/{gen['n']}, base {cap['base']}/{gen['n']}")
    if (gen["tuned_unparseable"] > gen["n"] // 2
            or gen["base_unparseable"] > gen["n"] // 2):
        raise SystemExit(
            f"\\nSTOP. Unparseable answers exceed half of {gen['n']}: tuned "
            f"{gen['tuned_unparseable']}, base {gen['base_unparseable']}.\\n"
            "A base this broken (truncation, fp16 garbage) would flatter the "
            "fine-tune, so neither direction is a real result.\\n"
            "FIX: raise --max-new-tokens in the full run.")

print(f"\\nprojected full run: {projected / 3600:.2f}h "
      f"against a {SESSION_BUDGET / 3600:.1f}h budget")
if projected > SESSION_BUDGET:
    raise SystemExit(
        f"\\nSTOP. The full evaluation projects to {projected / 3600:.1f}h.\\n"
        "FIX: lower GEN_LIMIT; generation dominates the runtime.")
print("fits; starting the full run")'''),
    ("markdown", """## 6. Full evaluation

All 1,273 and all 4,183 questions by constrained scoring, and the first 300 of
each by greedy generation, paired against the base model."""),
    ("code", '''for name in ("medqa", "medmcqa"):
    step(f"python -m training.evaluate --adapter {ADAPTER_ARG} "
         f"--test data/holdout_{name}.jsonl --gen-limit {GEN_LIMIT} "
         f"--out outputs/eval_{name}.json")'''),
    ("code", '''# --- 7. Results, and save everything to the Output panel -------------------
out = Path("/kaggle/working")
for label, name in (("MedQA-USMLE test", "medqa"),
                    ("MedMCQA validation", "medmcqa")):
    data = json.load(open(f"outputs/eval_{name}.json"))
    print(f"\\n########## {label} ##########")
    for mode, rep in data["reports"].items():
        m = rep["mcnemar"]
        delta = (rep["tuned_accuracy"] - rep["base_accuracy"]) * 100
        print(f"  {mode:<12} n={rep['n']:>5,}  base {rep['base_accuracy']:6.1%}  "
              f"tuned {rep['tuned_accuracy']:6.1%}  ({delta:+.1f})  "
              f"wins {m['wins']} / regressions {m['regressions']}  "
              f"p={m['p_value']:.4f}")
        if m["p_value"] >= 0.05:
            print("               not significant at p<0.05: "
                  "indistinguishable from base")
    shutil.copy(f"outputs/eval_{name}.json", out / f"eval_{name}.json")
print("\\nSaved eval_medqa.json and eval_medmcqa.json to the Output panel.")'''),
    ("markdown", """## Done

`eval_medqa.json` and `eval_medmcqa.json` are in the **Output** panel: accuracy
for both models in both modes, McNemar significance, a per-subject breakdown,
timing, and twelve raw completions per model for reading by eye."""),
]


def write_notebook(cells: list[tuple[str, str]], path: Path) -> None:
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             **({"source": src.splitlines(keepends=True)} if kind == "markdown"
                else {"source": src.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})}
            for kind, src in cells
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path}, {len(cells)} cells")


write_notebook(CELLS, Path("training/kaggle_medical.ipynb"))
write_notebook(EVAL_CELLS, Path("evaluation/kaggle_eval.ipynb"))
