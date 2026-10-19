from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.kaggle_paths import code_fingerprint, data_fingerprint

# `python scripts/build_notebook.py [name ...]` rebuilds only the named
# notebooks (a kernel directory name: training, evaluation, run3, eval3, e0s1,
# e0s2); no names rebuilds them all. Rebuilding bakes in the current code
# fingerprint, so a notebook that already ran is left alone unless asked for.
NOTEBOOKS = ("training", "evaluation", "run3", "eval3", "e0s1", "e0s2", "e2", "e2t")
WANT = set(sys.argv[1:])
if WANT - set(NOTEBOOKS):
    raise SystemExit(f"unknown notebook {sorted(WANT - set(NOTEBOOKS))}; known: {NOTEBOOKS}")


def wanted(name: str) -> bool:
    return not WANT or name in WANT

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
!pip install -q datasets
# The image ships torchao 0.10.0. peft 0.19.1 checks torchao while it wraps
# every layer and raises on anything older than 0.16.0 instead of skipping it,
# which stopped the first run at its first model load. Evaluation never uses
# torchao, so remove it rather than upgrade it against the image's torch.
!pip uninstall -y -q torchao"""),
    ("code", '''# --- 2. Verify the install before spending GPU time on it ------------------
import torch, transformers, peft
print(f"ok | torch {torch.__version__} | transformers {transformers.__version__} "
      f"| peft {peft.__version__}")

if transformers.__version__ != "5.5.0" or peft.__version__ != "0.19.1":
    raise SystemExit(
        f"\\nSTOP. transformers {transformers.__version__} / peft {peft.__version__} "
        "-- expected the pinned transformers==5.5.0 / peft==0.19.1 this adapter "
        "was trained with.\\nFIX: the pip install above did not take effect on "
        "this kernel image. Run > Restart session, then Run All again.")

# peft checks every optional quantization package on the image while it wraps
# each layer, and some of those checks raise on an old version instead of
# skipping it. Wrap one tiny layer the same way, so a broken optional package
# stops the run here in seconds, not after an 8GB model download.
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class _OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.q_proj(x)

try:
    get_peft_model(_OneLayer(), LoraConfig(r=2, target_modules=["q_proj"]))
except ImportError as e:
    raise SystemExit(
        f"\\nSTOP. peft cannot wrap a layer on this image: {e}\\n"
        "FIX: add a pip uninstall of the package it names to the install cell, "
        "then Run > Restart session and Run All again.")
print("ok | peft can wrap a layer on this image")'''),
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
            f"FIX: read the completions in outputs/smoke_{name}.json. If "
            "they are cut off, raise --max-new-tokens. If they are garbage, "
            "half precision is failing on this GPU and the result must not "
            "be reported.")

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
    gen = data["reports"]["generative"]
    cap = data["generation"]["hit_cap"]
    print(f"  generative answers with no letter: base {gen['base_unparseable']}, "
          f"tuned {gen['tuned_unparseable']} of {gen['n']}  |  "
          f"hit the {data['generation']['max_new_tokens']}-token cap: "
          f"base {cap['base']}, tuned {cap['tuned']}")
    shutil.copy(f"outputs/eval_{name}.json", out / f"eval_{name}.json")
print("\\nSaved eval_medqa.json and eval_medmcqa.json to the Output panel.")'''),
    ("markdown", """## Done

`eval_medqa.json` and `eval_medmcqa.json` are in the **Output** panel: accuracy
for both models in both modes, McNemar significance, a per-subject breakdown,
timing, every question's prediction from both models, hit-cap counts, and
twelve raw completions per model for reading by eye."""),
]


RUN2_DATA = Path("data/v2")

RUN2_INTRO = """# Qwen3-4B medical fine-tune, run 2

One session, no laptop needed. It trains a reasoning fine-tune of Qwen3-4B on
about 11,000 examples from eight medical sources, then scores it against the
base model on four benchmarks it never saw -- MedQA, MedMCQA, PubMedQA and
MMLU-medical -- by letter choice and by reasoning.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach `medical-ft-code`
and `medical-ft-data`.

Training stops by itself in time for evaluation. Evaluation stops 40 minutes
before Kaggle's 12-hour limit and reports whatever both models have scored."""

RUN2_HARDWARE = '''# --- 1. Hardware check, and the session clock ------------------------------
import subprocess, time
from pathlib import Path

# Kaggle ends a GPU session at 12 hours. Every later step measures itself
# against this clock; evaluation stops scoring 40 minutes before the end.
SESSION_START = time.time()
DEADLINE = SESSION_START + 12 * 3600 - 40 * 60

import torch

gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip().splitlines()
N_GPUS = torch.cuda.device_count()
major, minor = torch.cuda.get_device_capability(0)
for i, gpu in enumerate(gpus):
    print(f"GPU {i}      : {gpu}")
print(f"capability : {major}.{minor}")
print(f"torch      : {torch.__version__}")

if major < 7:
    raise SystemExit(
        f"\\nSTOP. Compute capability {major}.{minor} has no kernels in modern "
        "PyTorch builds.\\nFIX: kernel-metadata.json pins a T4; if a P100 still "
        "arrived, set sidebar -> Accelerator -> 'GPU T4 x2' and Run All again.")
plan = ("the fine-tune on GPU 0 and the base model on GPU 1, in parallel"
        if N_GPUS > 1 else "both models on GPU 0, one after the other")
print(f"\\n{N_GPUS} GPU(s). Training uses GPU 0; evaluation runs {plan}.")'''

RUN2_INSTALL = '''%%capture
# The exact stack that trained run 1 on this image: Unsloth 2026.9.7, with the
# unsloth_zoo current that day, printed "Transformers: 5.5.0" there, alongside
# trl 0.24.0 and peft 0.19.1. Run 1 got it by installing whatever was newest;
# pinning it is how run 2 gets the same thing again.
!pip install -q "unsloth==2026.9.7" "unsloth_zoo==2026.9.6"
!pip install -q --no-deps "transformers==5.5.0" "trl==0.24.0" "peft==0.19.1"'''

RUN2_VERIFY = '''# --- 2. Verify the install before spending GPU time on it ------------------
import importlib.metadata as md
import sys

WANT = {"unsloth": "2026.9.7", "transformers": "5.5.0", "trl": "0.24.0",
        "peft": "0.19.1"}
got = {name: md.version(name) for name in WANT}
print("ok |", " | ".join(f"{k} {v}" for k, v in got.items()),
      f"| torch {md.version('torch')}")
wrong = {k: v for k, v in got.items() if v != WANT[k]}
if wrong:
    raise SystemExit(
        f"\\nSTOP. Installed {wrong}, expected {WANT}.\\nFIX: the install cell did "
        "not take effect. Run > Restart session, then Run All again.")

# peft checks every optional quantization package on the image while it wraps
# each layer, and some checks raise on an old version instead of skipping it:
# run 1's first evaluation died on this image's torchao 0.10.0. Wrap one tiny
# layer in a fresh process, the way training and evaluation will.
LORA_CHECK = """
import torch.nn as nn
from peft import LoraConfig, get_peft_model
class OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
    def forward(self, x):
        return self.q_proj(x)
get_peft_model(OneLayer(), LoraConfig(r=2, target_modules=["q_proj"]))
"""
check = subprocess.run([sys.executable, "-c", LORA_CHECK], capture_output=True, text=True)
if check.returncode != 0 and "torchao" in check.stderr:
    print("peft rejects this image's torchao; removing it (nothing here uses it)")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    check = subprocess.run([sys.executable, "-c", LORA_CHECK], capture_output=True, text=True)
if check.returncode != 0:
    raise SystemExit(f"\\nSTOP. peft cannot wrap a layer on this image:\\n{check.stderr[-2000:]}")
print("ok | peft can wrap a layer on this image")

# transformers 5 replaced group_by_length with train_sampling_strategy. Check
# the class training will really use: Unsloth's patched SFTConfig.
FIELD_CHECK = """
import unsloth, dataclasses
from trl import SFTConfig
print("train_sampling_strategy" in {f.name for f in dataclasses.fields(SFTConfig)})
"""
check = subprocess.run([sys.executable, "-c", FIELD_CHECK], capture_output=True, text=True)
if check.stdout.strip().splitlines()[-1:] != ["True"]:
    raise SystemExit("\\nSTOP. SFTConfig has no train_sampling_strategy here:\\n"
                     f"{check.stdout[-1000:]}{check.stderr[-1500:]}")
print("ok | SFTConfig accepts train_sampling_strategy")'''

RUN2_TRAIN = '''# --- 4. Train on GPU 0 -------------------------------------------------------
# Leave the evaluation about 4.25 hours, and never train for more than 5.75.
STOP_AFTER = int(min(5.75 * 3600, DEADLINE - time.time() - 4.25 * 3600))
if STOP_AFTER < 2 * 3600:
    raise SystemExit(f"\\nSTOP. Only {STOP_AFTER / 3600:.1f}h are left for training.\\n"
                     "FIX: setup was unusually slow; Run All again.")
print(f"training stops by itself after {STOP_AFTER / 3600:.2f}h at the latest")

step(f"python -m training.train --format v2 --data data/v2 --out outputs/run2 "
     f"--max-seq {MAX_SEQ} --batch-size 2 --grad-accum 8 --rank 64 --lr 1e-4 "
     f"--epochs 1 --save-steps 200 --group-by-length --probe-steps 20 "
     f"--no-probe-abort --stop-after-seconds {STOP_AFTER}",
     env={"CUDA_VISIBLE_DEVICES": "0"})'''

RUN2_SAVE = '''# --- 5. Save the adapter and the training record to the Output panel -------
import json, shutil

OUT = Path("/kaggle/working")
shutil.copytree("outputs/run2", OUT / "run2-adapter", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("checkpoint-*"))
for name in ("train_stats.json", "loss_curve.json"):
    shutil.copy(Path("outputs/run2") / name, OUT / name)
shutil.copy(DATA / "data_report.json", OUT / "data_report.json")
print("########## Training ##########")
for key, value in json.loads((OUT / "train_stats.json").read_text()).items():
    print(f"  {key}: {value}")'''

RUN2_EVAL_SMOKE = '''# --- 6. Evaluation smoke run: every stage, four questions, both models -----
def worker(role: str, gpu: int, out_dir: str, extra: str):
    cmd = (f"python -m training.eval_worker --role {role} --eval-dir data/v2 "
           f"--out-dir {out_dir} --device cuda "
           + ("--adapter outputs/run2 " if role == "tuned" else "") + extra)
    log = open(f"{out_dir}-{role}.log", "w")
    proc = subprocess.Popen(argv(cmd), stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                 # hours of variable-length batches fragment memory
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return proc, log

def run_pair(out_dir: str, extra_tuned: str, extra_base: str, poll: int) -> dict:
    """Both models: in parallel on two GPUs, one after the other on one."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if N_GPUS > 1:
        procs = {"tuned": worker("tuned", 0, out_dir, extra_tuned),
                 "base": worker("base", 1, out_dir, extra_base)}
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(poll)
            written = {role: sum(1 for f in Path(out_dir, role).glob("*.jsonl")
                                 for _ in open(f)) for role in procs}
            print(time.strftime("%H:%M"), "answers written:", written, flush=True)
        codes = {role: p.returncode for role, (p, _) in procs.items()}
    else:
        codes = {}
        for role, extra in (("tuned", extra_tuned), ("base", extra_base)):
            proc, _ = worker(role, 0, out_dir, extra)
            codes[role] = proc.wait()
    for role in ("tuned", "base"):
        print(f"--- {role} log tail ---")
        print("".join(open(f"{out_dir}-{role}.log").readlines()[-15:]))
    return codes

SMOKE = "--limit 4 --think-budget 128 --batch-size 4"
codes = run_pair("outputs/eval_smoke", SMOKE, SMOKE, poll=15)
if any(codes.values()):
    raise SystemExit(f"\\nSTOP. The evaluation smoke run failed: {codes}. The adapter "
                     "is already in the Output panel; the log tails above say why.")
step("python -m training.eval_report --eval-dir outputs/eval_smoke --bench-dir data/v2 "
     "--out outputs/eval_smoke/report.json")'''

RUN2_EVAL_FULL = '''# --- 7. Full evaluation, until the deadline ---------------------------------
SETTINGS = "--think-budget 1536 --batch-size 16 --seed 1234"
if N_GPUS > 1:
    codes = run_pair("outputs/eval", f"--deadline {DEADLINE:.0f} {SETTINGS}",
                     f"--deadline {DEADLINE:.0f} {SETTINGS}", poll=600)
else:
    half = time.time() + (DEADLINE - time.time()) / 2
    codes = run_pair("outputs/eval", f"--deadline {half:.0f} {SETTINGS}",
                     f"--deadline {DEADLINE:.0f} {SETTINGS}", poll=600)
print("worker exit codes:", codes)
step("python -m training.eval_report --eval-dir outputs/eval --bench-dir data/v2 "
     "--out /kaggle/working/run2_eval.json")
shutil.copytree("outputs/eval", "/kaggle/working/run2_eval_predictions",
                dirs_exist_ok=True)
print("Saved run2_eval.json and every prediction to the Output panel.")'''

RUN2_DONE = """## Done

In the **Output** panel: `run2-adapter/`, `train_stats.json`, `loss_curve.json`,
`data_report.json`, `run2_eval.json` (every stage and the pooled results, with
McNemar p and a 95% interval for each change) and `run2_eval_predictions/`
(every question's answer from both models, with the reasoning text)."""


def run2_fetch_cell(data_fp: str) -> str:
    return ('''# --- 3. Get the code and the data from the attached datasets --------------
import json, os, shlex, shutil, subprocess, sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/ft")
PKG   = WORK / "training"
DATA  = WORK / "data" / "v2"
PKG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

''' + FIND_CODE_DIR_SRC + '''

SRC = find_code_dir(INPUT, RUN2_REQUIRED)
''' + CHECK_FINGERPRINT_SRC + '''
DATA_SRC = find_data_dir(INPUT)
EXPECTED_DATA_FINGERPRINT = "''' + data_fp + '''"
if data_fingerprint(DATA_SRC) != EXPECTED_DATA_FINGERPRINT:
    raise SystemExit(
        f"\\nSTOP. The attached data ({data_fingerprint(DATA_SRC)}) is not the data "
        f"this notebook was built for ({EXPECTED_DATA_FINGERPRINT}).\\n"
        "FIX: wait a minute and re-run; if it persists, run scripts/push_data.sh again.")
print("code:", SRC)
print("data:", DATA_SRC)

for src_file in sorted(SRC.glob("*.py")):
    shutil.copy(src_file, PKG / src_file.name)
(PKG / "__init__.py").touch()
for name in DATA_FILES:
    shutil.copy(DATA_SRC / name, DATA / name)

os.chdir(WORK)
sys.path.insert(0, str(WORK))
Path("outputs").mkdir(exist_ok=True)

REPORT = json.loads((DATA / "data_report.json").read_text())
MAX_SEQ = REPORT["max_seq"]
print(f"train {REPORT['train_size']:,} examples | max_seq {MAX_SEQ} | reasoning "
      f"{REPORT['reasoning_fraction']:.0%} | multiple choice {REPORT['mcq_fraction']:.0%}")
for label, stats in REPORT["sources"].items():
    print(f"  {label:<24} kept {stats['kept']:>6,} of target {stats['target']:>6,}")

# No shell: every command here is built from constants, so it splits into a
# plain list. "python" becomes this kernel's own interpreter -- the one the
# install cell installed Unsloth into -- rather than whatever PATH finds.
def argv(cmd: str) -> list[str]:
    args = shlex.split(cmd)
    return [sys.executable] + args[1:] if args[0] == "python" else args

# A failing command returns non-zero without raising in Jupyter.
def step(cmd: str, env: dict | None = None):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(argv(cmd), env={**os.environ, **(env or {})})
    if p.returncode != 0:
        raise SystemExit(f"\\nStep failed (exit {p.returncode}):\\n  {cmd}")
    print("\\nok\\n", flush=True)''')


def run2_cells(data_fp: str) -> list[tuple[str, str]]:
    return [
        ("markdown", RUN2_INTRO),
        ("code", RUN2_HARDWARE),
        ("code", RUN2_INSTALL),
        ("code", RUN2_VERIFY),
        ("code", run2_fetch_cell(data_fp)),
        ("markdown", "## 4. Train\n\nThe thinking format on GPU 0, with a time guard "
                     "that saves the adapter instead of overrunning the session."),
        ("code", RUN2_TRAIN),
        ("code", RUN2_SAVE),
        ("markdown", "## 6. Evaluate\n\nA smoke run of every stage first, then the full "
                     "run: letter choice on all 7,545 questions, then reasoning, until "
                     "the deadline."),
        ("code", RUN2_EVAL_SMOKE),
        ("code", RUN2_EVAL_FULL),
        ("markdown", RUN2_DONE),
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


if wanted("training"):
    write_notebook(CELLS, Path("training/kaggle_medical.ipynb"))
if wanted("evaluation"):
    write_notebook(EVAL_CELLS, Path("evaluation/kaggle_eval.ipynb"))
def run3_cells(data_fp: str) -> list[tuple[str, str]]:
    """Run 3 is run 2's notebook with its own names. run2/kaggle_run2.ipynb
    stays as it ran; run 3's change is in the code it uploads (every prompt
    states /think or /no_think, and the budget-forcing pass no longer runs a
    T4 out of memory)."""
    return [(kind, src.replace("run2", "run3").replace("run 2", "run 3"))
            for kind, src in run2_cells(data_fp)]


if not wanted("run3"):
    pass
elif (RUN2_DATA / "data_report.json").exists():
    write_notebook(run3_cells(data_fingerprint(RUN2_DATA)), Path("run3/kaggle_run3.ipynb"))
else:
    print("skipped run3/kaggle_run3.ipynb: no data/v2 (run training.prepare_data_v2)")


# --- Run 3 evaluation only ----------------------------------------------
# Run 3 trained to completion and then lost its evaluation to a download race:
# both workers fetched the 8GB base model at once and one failed to load its
# tokenizer. This notebook scores that adapter without retraining, downloads
# the base model once before either worker starts, and gives the evaluation
# the whole session.
RUN3_ADAPTER = Path("training/outputs/run3")

EVAL3_INTRO = """# Qwen3-4B medical fine-tune, run 3 evaluation

Scores run 3's adapter -- trained to completion in the run 3 session -- against
the base model on MedQA, MedMCQA, PubMedQA and MMLU-medical, by letter choice
and by reasoning, with the whole 12-hour session for evaluation.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach `medical-ft-code`,
`medical-ft-data` and `medical-ft-adapter-run3`."""


def eval3_adapter_cell(adapter_sha: str) -> str:
    return '''# --- 3b. Run 3's adapter, checked against the one trained -----------------
import hashlib

ADAPTER_SRC = find_adapter_dir(INPUT)
digest = hashlib.sha256()
with open(ADAPTER_SRC / "adapter_model.safetensors", "rb") as fh:
    for block in iter(lambda: fh.read(1 << 20), b""):
        digest.update(block)
if not digest.hexdigest().startswith("''' + adapter_sha + '''"):
    raise SystemExit(f"\\nSTOP. The attached adapter ({digest.hexdigest()[:16]}) is not "
                     "run 3's (''' + adapter_sha + ''').\\nFIX: attach medical-ft-adapter-run3.")
shutil.copytree(ADAPTER_SRC, "outputs/run3", dirs_exist_ok=True)
print("adapter:", ADAPTER_SRC, "-> outputs/run3")'''


EVAL3_DOWNLOAD = '''# --- 4. Download the base model once, before the two workers start ---------
# Run 3's two workers fetched the same 8GB model at the same moment and one of
# them failed to load the tokenizer. One download, retried, then the workers
# read the local copy only.
from huggingface_hub import snapshot_download

for attempt in range(1, 5):
    try:
        print("base model at", snapshot_download("unsloth/Qwen3-4B"))
        break
    except Exception as exc:
        print(f"download attempt {attempt} failed: {type(exc).__name__}: {exc}")
        time.sleep(60)
else:
    raise SystemExit("\\nSTOP. unsloth/Qwen3-4B could not be downloaded from the Hub.")
os.environ["HF_HUB_OFFLINE"] = "1"'''


def eval3_cells(data_fp: str, adapter_sha: str) -> list[tuple[str, str]]:
    cells = dict(enumerate(run3_cells(data_fp)))
    return [
        ("markdown", EVAL3_INTRO),
        cells[1], cells[2], cells[3], cells[4],
        ("code", eval3_adapter_cell(adapter_sha)),
        ("code", EVAL3_DOWNLOAD),
        cells[8], cells[9], cells[10], cells[11],
    ]


if not wanted("eval3"):
    pass
elif (RUN2_DATA / "data_report.json").exists() and (RUN3_ADAPTER / "adapter_model.safetensors").exists():
    import hashlib

    _digest = hashlib.sha256()
    with open(RUN3_ADAPTER / "adapter_model.safetensors", "rb") as _fh:
        for _block in iter(lambda: _fh.read(1 << 20), b""):
            _digest.update(_block)
    write_notebook(eval3_cells(data_fingerprint(RUN2_DATA), _digest.hexdigest()[:16]),
                   Path("eval3/kaggle_eval3.ipynb"))
else:
    print("skipped eval3/kaggle_eval3.ipynb: needs data/v2 and training/outputs/run3")


# --- E0: the baseline evaluation, two sessions ----------------------------
# Each session runs training/run_plan.py once per GPU. The queues, budget,
# subsets and sample counts all come from training/experiments_config.py; the
# notebook holds no evaluation settings of its own.
from training.experiments_config import ADAPTERS, E0

E0_PLAN_LINE = 'print(f"\\n{N_GPUS} GPU(s). Training uses GPU 0; evaluation runs {plan}.")'
E0_PLAN_NEW = 'print(f"\\n{N_GPUS} GPU(s). Each GPU runs its own queue of E0 workers.")'


def e0_models(session: str) -> list[str]:
    return [m for queue in E0["sessions"][session].values() for m, _ in queue]


def e0_intro(session: str) -> str:
    other = "s2" if session == "s1" else "s1"
    queues = []
    for gpu, queue in E0["sessions"][session].items():
        for model, stages in queue:
            label = E0["models"][model]["label"]
            queues.append(f"- GPU {gpu}: **{model}** ({label}): " + ", ".join(stages))
    sizes = ", ".join(f"{b} {n}" for b, n in E0["sizes"].items())
    return (f"# Qwen3-4B medical fine-tune, E0 baseline, session {session[1]}\n\n"
            "E0 scores base Qwen3-4B (A), run 3's adapter on the fp16 base (B) and "
            "run 3's adapter on its 4-bit training base (C) with the same generation "
            f"settings: a {E0['think_budget']:,}-token thinking budget, temperature 0.6, "
            f"top-p 0.95, top-k 20, seed {E0['seed']}. Reasoning runs on fixed seeded "
            f"subsets ({sizes} questions); MedQA gets {E0['samples']['medqa']} samples "
            f"per question. Session {other[1]} runs the rest.\n\nThis session:\n\n"
            + "\n".join(queues) +
            "\n\n**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach "
            "`medical-ft-code`, `medical-ft-data` and `medical-ft-adapter-run3`. "
            "Workers stop 40 minutes before Kaggle's 12-hour limit and keep what they scored.")


def e0_download_cell(session: str) -> str:
    repos = sorted({E0["models"][m]["base"] for m in e0_models(session)})
    four_bit = any(E0["models"][m]["load_in_4bit"] for m in e0_models(session))
    check = ('\nimport bitsandbytes\nprint("bitsandbytes", bitsandbytes.__version__)'
             if four_bit else "")
    return ('''# --- 4. Download each base model once, before any worker starts ------------
# Two workers fetching the same 8GB model at once lost run 3's first
# evaluation to a race. One download each, retried; then offline only.
from huggingface_hub import snapshot_download

for repo in ''' + repr(repos) + ''':
    for attempt in range(1, 5):
        try:
            print(repo, "at", snapshot_download(repo))
            break
        except Exception as exc:
            print(f"{repo}: download attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(60)
    else:
        raise SystemExit(f"\\nSTOP. {repo} could not be downloaded from the Hub.")
os.environ["HF_HUB_OFFLINE"] = "1"''' + check)


def e0_run_cell(session: str) -> str:
    return ('''# --- 5. A smoke run of every queue, then the session's real run ------------
SESSION = "''' + session + '''"
GPUS = ''' + repr(sorted(E0["sessions"][session])) + '''

def driver(gpu: int, out: str, extra: str):
    cmd = (f"python -m training.run_plan run --session {SESSION} --gpu {gpu} "
           f"--out-dir {out}/gpu{gpu} --eval-dir data/v2 --adapter run3=outputs/run3 {extra}")
    log = open(f"{out}-gpu{gpu}.log", "w")
    return subprocess.Popen(argv(cmd), stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ,
                                 "CUDA_VISIBLE_DEVICES": str(gpu if N_GPUS > 1 else 0),
                                 # hours of variable-length batches fragment memory
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

def run_all(out: str, extra: str, poll: int) -> dict:
    """Each GPU's queue in parallel; on a single GPU, one after the other."""
    Path(out).mkdir(parents=True, exist_ok=True)
    if N_GPUS > 1:
        procs = {g: driver(g, out, extra) for g in GPUS}
        while any(p.poll() is None for p in procs.values()):
            time.sleep(poll)
            written = sum(1 for f in Path(out).rglob("*.jsonl") for _ in open(f))
            print(time.strftime("%H:%M"), "answers written:", written, flush=True)
        codes = {g: p.returncode for g, p in procs.items()}
    else:
        codes = {g: driver(g, out, extra).wait() for g in GPUS}
    for g in GPUS:
        print(f"--- GPU {g} log tail ---")
        print("".join(open(f"{out}-gpu{g}.log").readlines()[-20:]))
    return codes

codes = run_all("outputs/e0_smoke", "--smoke", poll=15)
if any(codes.values()):
    raise SystemExit(f"\\nSTOP. The smoke run failed: {codes}. The log tails above say why.")
step("python -m training.eval_report --root outputs/e0_smoke --bench-dir data/v2 "
     "--out outputs/e0_smoke/report.json")

codes = run_all(f"outputs/e0/{SESSION}", f"--deadline {DEADLINE:.0f}", poll=600)
print("queue exit codes:", codes)
OUT = Path("/kaggle/working")
PRED = OUT / f"e0_{SESSION}_predictions"
shutil.copytree(f"outputs/e0/{SESSION}", PRED, dirs_exist_ok=True)
for log in Path("outputs/e0").glob(f"{SESSION}-gpu*.log"):
    shutil.copy(log, OUT / f"e0_{log.name}")
REPORT = OUT / f"e0_{SESSION}_report.json"
PER_Q = OUT / f"e0_{SESSION}_per_question.jsonl"
step(f"python -m training.eval_report --root {PRED} --bench-dir data/v2 "
     f"--out {REPORT} --export {PER_Q}")
# The working copy holds the code, the data and a second copy of the adapter.
# None of it is a result, and all of it would download with the output.
os.chdir(OUT)
shutil.rmtree(WORK)
print("Saved the predictions, the queue logs, the session report and the "
      "per-question records to the Output panel.")''')


def e0_cells(session: str, data_fp: str, adapter_sha: str) -> list[tuple[str, str]]:
    cells = dict(enumerate(run3_cells(data_fp)))
    hardware = cells[1][1]
    if E0_PLAN_LINE not in hardware:
        raise SystemExit("the hardware cell's plan line changed; update E0_PLAN_LINE")
    return [
        ("markdown", e0_intro(session)),
        ("code", hardware.replace(E0_PLAN_LINE, E0_PLAN_NEW)),
        cells[2], cells[3], cells[4],
        ("code", eval3_adapter_cell(adapter_sha)),
        ("code", e0_download_cell(session)),
        ("code", e0_run_cell(session)),
        ("markdown", f"## Done\n\nIn the **Output** panel: `e0_{session}_predictions/` "
                     "(every answer, per GPU and model, beside each worker's `meta.json`), "
                     f"`e0_{session}_report.json`, `e0_{session}_per_question.jsonl` and "
                     "the queue logs. `experiments/README.md` says how to join both "
                     "sessions into E0's report."),
    ]


for _session in ("s1", "s2"):
    if not wanted(f"e0{_session}"):
        continue
    if not (RUN2_DATA / "data_report.json").exists():
        print(f"skipped e0{_session}: no data/v2 (run training.prepare_data_v2)")
        continue
    write_notebook(e0_cells(_session, data_fingerprint(RUN2_DATA),
                            ADAPTERS["run3"]["sha256_prefix"]),
                   Path(f"e0{_session}/kaggle_e0{_session}.ipynb"))


# --- E2: self-distillation, one session -----------------------------------
# Generation, training and evaluation in one 12-hour session. Every setting
# comes from training/experiments_config.py (E2); the cells only sequence the
# phases and hold the clock.
from training.experiments_config import E2
from training.kaggle_paths import E2_DATA_FILES, V3_DATA_FILES

V3_DATA = Path("data/v3")

E2_INTRO = """# Qwen3-4B medical fine-tune, E2: self-distillation

Run 3 trained Qwen3-4B on reasoning written by other models. Its MedQA accuracy
in thinking mode fell from 69.7% to 60.2%, and its median thinking from 1,036
tokens to 403. E2 trains on the base model's **own** reasoning instead, kept
only when it reaches the gold answer. Nothing is written for the model.

1. **Generate** (both GPUs): base Qwen3-4B thinks through MedQA and MedMCQA
   training questions (`data/v3` pool, decontaminated against every benchmark),
   2 samples each, a 4,096-token budget.
2. **Select**: a sample is kept only if it closed its own think block and stated
   the gold letter. One kept sample per question, chosen at random. 20% of rows
   are answer-only.
3. **Train** (GPU 0): LoRA rank 16, lr 5e-5, one epoch, run 3's tested trainer.
   Meanwhile GPU 1 scores the base model on the dev split and on MedQA.
4. **Evaluate** (both GPUs): the E2 adapter on the dev split, the four
   benchmarks by letter choice, and MedQA by reasoning.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach `medical-ft-code` and
`medical-ft-data-v3`. Generation uses vLLM if it installs and passes a smoke
test, and transformers otherwise."""

E2_PLAN_NEW = 'print(f"\\n{N_GPUS} GPU(s). E2 generates on all of them, trains on GPU 0.")'


def e2_traces_fetch(traces_fp: str) -> str:
    """The saved E2 training set (a finished generation's train.jsonl)."""
    return ('''
TRACES_SRC = find_data_dir(INPUT, E2_DATA_FILES)
if data_fingerprint(TRACES_SRC, E2_DATA_FILES) != "''' + traces_fp + '''":
    raise SystemExit(
        f"\\nSTOP. The attached E2 training set ({data_fingerprint(TRACES_SRC, E2_DATA_FILES)}) "
        "is not the one this notebook was built for (''' + traces_fp + ''').\\n"
        "FIX: run scripts/push_data.sh data/e2 medical-ft-data-e2 again.")
Path("data/e2").mkdir(parents=True, exist_ok=True)
for name in E2_DATA_FILES:
    shutil.copy(TRACES_SRC / name, Path("data/e2") / name)
print("E2 training set:", TRACES_SRC)''')


def e2_fetch_cell(data_fp: str, traces_fp: str | None = None) -> str:
    return ('''# --- 3. Get the code and the data from the attached datasets --------------
import json, os, shlex, shutil, subprocess, sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/ft")
PKG   = WORK / "training"
DATA  = WORK / "data" / "v3"
PKG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

''' + FIND_CODE_DIR_SRC + '''

SRC = find_code_dir(INPUT, RUN2_REQUIRED | {"run_plan", "self_traces", "vllm_engine"})
''' + CHECK_FINGERPRINT_SRC + '''
DATA_SRC = find_data_dir(INPUT, V3_DATA_FILES)
EXPECTED_DATA_FINGERPRINT = "''' + data_fp + '''"
if data_fingerprint(DATA_SRC, V3_DATA_FILES) != EXPECTED_DATA_FINGERPRINT:
    raise SystemExit(
        f"\\nSTOP. The attached data ({data_fingerprint(DATA_SRC, V3_DATA_FILES)}) is not "
        f"the data this notebook was built for ({EXPECTED_DATA_FINGERPRINT}).\\n"
        "FIX: wait a minute and re-run; if it persists, run "
        "scripts/push_data.sh data/v3 medical-ft-data-v3 again.")
print("code:", SRC)
print("data:", DATA_SRC)

for src_file in sorted(SRC.glob("*.py")):
    shutil.copy(src_file, PKG / src_file.name)
(PKG / "__init__.py").touch()
for name in V3_DATA_FILES:
    shutil.copy(DATA_SRC / name, DATA / name)

os.chdir(WORK)
sys.path.insert(0, str(WORK))
Path("outputs").mkdir(exist_ok=True)
print(json.dumps(json.loads((DATA / "data_report.json").read_text())["sizes"]))
''' + (e2_traces_fetch(traces_fp) if traces_fp else "") + '''

def argv(cmd: str) -> list[str]:
    args = shlex.split(cmd)
    return [sys.executable] + args[1:] if args[0] == "python" else args

def step(cmd: str, env: dict | None = None):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(argv(cmd), env={**os.environ, **(env or {})})
    if p.returncode != 0:
        raise SystemExit(f"\\nStep failed (exit {p.returncode}):\\n  {cmd}")
    print("\\nok\\n", flush=True)''')


E2_DOWNLOAD = '''# --- 4. Download both base models once, then work offline -----------------
# The fp16 model generates and evaluates; Unsloth's 4-bit copy is the one
# training.train fine-tunes (run 1-3's base).
from huggingface_hub import snapshot_download

for repo in ("unsloth/Qwen3-4B", "unsloth/Qwen3-4B-unsloth-bnb-4bit"):
    for attempt in range(1, 5):
        try:
            print(repo, "at", snapshot_download(repo))
            break
        except Exception as exc:
            print(f"{repo}: download attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(60)
    else:
        raise SystemExit(f"\\nSTOP. {repo} could not be downloaded from the Hub.")
os.environ["HF_HUB_OFFLINE"] = "1"'''

E2_VLLM = '''# --- 5. vLLM in its own environment, then a smoke test --------------------
# vLLM pins its own torch and transformers, so it lives in a separate virtual
# environment and the training stack above stays exactly as run 3 had it. If
# it fails to install or to answer four questions, everything falls back to
# transformers generation, which run 3 used: slower, so fewer traces.
VENV = Path("/tmp/vllm-env")
VLLM_PY = str(VENV / "bin" / "python")
ENGINE, VLLM_EAGER = "hf", ""
SMOKE_V = (f"{VLLM_PY} -m training.eval_worker --engine vllm --role base --device cuda "
           "--eval-dir data/v3 --out-dir outputs/vllm_smoke --stages "
           "letter:dev_medqa,reasoning:dev_medqa --limit 4 --think-budget 256 "
           "--batch-size 4 --reasoning-batch-size 4")

def run_logged(cmd: list[str], timeout: int, env: dict | None = None) -> int:
    try:
        p = subprocess.run(cmd, env={**os.environ, **(env or {})}, timeout=timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"timed out after {timeout}s: {' '.join(cmd[:4])} ...")
        return -1
    print((p.stdout + p.stderr)[-3000:])
    return p.returncode

began = time.time()
ok = run_logged([sys.executable, "-m", "pip", "install", "-q", "uv"], 300) == 0
ok = ok and run_logged([sys.executable, "-m", "uv", "venv", "--python", sys.executable,
                        str(VENV)], 300) == 0
# --torch-backend=auto picks the torch build that matches this machine's
# CUDA driver, as vLLM's installation guide does.
ok = ok and run_logged([sys.executable, "-m", "uv", "pip", "install", "--python", VLLM_PY,
                        "vllm", "--torch-backend=auto"], 1500) == 0
print(f"vLLM install {'ok' if ok else 'FAILED'} in {time.time() - began:.0f}s")
if ok:
    for extra in ("", " --enforce-eager"):
        code = run_logged(shlex.split(SMOKE_V + extra), 1200, env={"CUDA_VISIBLE_DEVICES": "0"})
        rows = sum(1 for f in Path("outputs/vllm_smoke/base").glob("*.jsonl") for _ in open(f)) \\
            if Path("outputs/vllm_smoke/base").exists() else 0
        if code == 0 and rows >= 8:
            ENGINE = "vllm"
            VLLM_EAGER = extra
            break
        print(f"vLLM smoke{extra or ''} failed (exit {code}, {rows} rows)")
if ENGINE == "vllm":
    meta = json.loads(Path("outputs/vllm_smoke/base/meta.json").read_text())
    print("vLLM", meta["environment"].get("vllm"), "smoke passed:", meta["stages"])
print("ENGINE =", ENGINE)'''


def e2_phase_cell(smoke: tuple[str, ...] = ("gen", "base_eval")) -> str:
    return '''# --- 6. Phase runner: one run_plan queue per GPU --------------------------
from training.experiments_config import E2
from training.run_plan import traces_args, train_args

GPUS = [0, 1] if N_GPUS > 1 else [0]
GEN_END = min(SESSION_START + 5.0 * 3600, DEADLINE - 6.0 * 3600)

def phase(name: str, gpu: int, out: str, extra: str = "", device: int | None = None):
    """GPU `gpu`'s queue of session `name`, run on `device` (default: the same GPU)."""
    cmd = (f"python -m training.run_plan run --experiment E2 --session {name} --gpu {gpu} "
           f"--out-dir {out}/gpu{gpu} --eval-dir data/v3 --engine {ENGINE} "
           f"--vllm-python {VLLM_PY}{VLLM_EAGER} {extra}")
    # The log sits beside the output folder; create the folder's parent here,
    # since a phase can be the first thing to write under it.
    Path(out).mkdir(parents=True, exist_ok=True)
    log = open(f"{out}-gpu{gpu}.log", "w")
    # Its own process group, so stopping it also stops the worker it started.
    return subprocess.Popen(argv(cmd), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env={**os.environ,
                                 "CUDA_VISIBLE_DEVICES": str(gpu if device is None else device),
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

def stop(proc) -> None:
    import signal
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

def wait(procs: dict, out: str, poll: int) -> dict:
    while any(p.poll() is None for p in procs.values()):
        time.sleep(poll)
        written = sum(1 for f in Path(out).rglob("*.jsonl") for _ in open(f))
        print(time.strftime("%H:%M"), "answers written:", written, flush=True)
    for g in procs:
        print(f"--- {out} GPU {g} log tail ---")
        print("".join(open(f"{out}-gpu{g}.log").readlines()[-12:]))
    return {g: p.returncode for g, p in procs.items()}

def run_phase(name: str, out: str, extra: str = "", poll: int = 600, gpus=None) -> dict:
    Path(out).mkdir(parents=True, exist_ok=True)
    gpus = [g for g in (gpus or GPUS) if g in E2["sessions"][name]]
    return wait({g: phase(name, g, out, extra) for g in gpus}, out, poll)

# Every queue that can run before training, on two questions first: wiring
# errors cost minutes here instead of hours later. (The E2 queues need the
# adapter training produces; run_plan retries any vLLM entry that fails
# before writing an answer with transformers.)
codes = {}
for session in ''' + repr(smoke) + ''':
    codes.update({f"{session}{g}": c for g, c in
                  run_phase(session, f"outputs/smoke/{session}", "--smoke", poll=20).items()})
if any(codes.values()):
    raise SystemExit(f"\\nSTOP. A smoke run failed: {codes}. The log tails above say why.")
print("smoke runs passed:", codes)'''


E2_GENERATE = '''# --- 7. Generate: base Qwen3-4B thinks through the training pool ----------
codes = run_phase("gen", "outputs/e2/gen", f"--deadline {GEN_END:.0f}", poll=900)
print("generation exit codes:", codes)
OUT = Path("/kaggle/working")
shutil.copytree("outputs/e2/gen", OUT / "e2_generation", dirs_exist_ok=True)
for log in Path("outputs/e2").glob("gen-gpu*.log"):
    shutil.copy(log, OUT / f"e2_{log.name}")
print("generation saved to the Output panel")'''

E2_SELECT = '''# --- 8. Keep only verified traces, one per question ------------------------
step("python " + " ".join(traces_args(E2, gen_dirs=["outputs/e2/gen"],
                                      pool="data/v3/eval_pool.jsonl", out="data/e2")))
TRACES = json.loads(Path("data/e2/e2_report.json").read_text())
shutil.copy("data/e2/e2_report.json", OUT / "e2_traces_report.json")
MAX_SEQ = TRACES["max_seq"]
print(f"{TRACES['rows']:,} training rows ({TRACES['reasoning_rows']:,} traces, "
      f"{TRACES['direct_rows']:,} answer-only); acceptance {TRACES['acceptance_rate']:.0%}; "
      f"max_seq {MAX_SEQ}")'''

E2_TRAIN_EVAL = '''# --- 9. Train on GPU 0 while GPU 1 scores the base; then score E2 ---------
OUT = Path("/kaggle/working")
TRACES = json.loads(Path("data/e2/e2_report.json").read_text())
MAX_SEQ = TRACES["max_seq"]
# Leave the new adapter at least 3 hours of evaluation.
STOP_AFTER = int(min(E2["training"]["max_hours"] * 3600, DEADLINE - time.time() - 3.0 * 3600))
if STOP_AFTER < 1800:
    raise SystemExit(f"\\nSTOP. Only {STOP_AFTER / 60:.0f} minutes left for training.")
print(f"{TRACES['rows']:,} rows, max_seq {MAX_SEQ}; training stops by itself after "
      f"{STOP_AFTER / 3600:.2f}h at the latest")
train_cmd = "python " + " ".join(train_args(E2, data_dir="data/e2", out_dir="outputs/e2_adapter",
                                            max_seq=MAX_SEQ, stop_after=STOP_AFTER))
print("$", train_cmd)
# Unsloth's loader looks up a tokenizer repository that the download cell does
# not fetch; with HF_HUB_OFFLINE=1 the first E2 session died on exactly that.
# Training runs alone on GPU 0, so it may use the network. The vLLM workers,
# which read the cached models, stay offline.
online = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
train_log = open("outputs/e2-train.log", "w")
trainer = subprocess.Popen(argv(train_cmd), stdout=train_log, stderr=subprocess.STDOUT,
                           env={**online, "CUDA_VISIBLE_DEVICES": "0"})
# GPU 1 scores the base model meanwhile: dev, letters, then MedQA reasoning,
# stopping an hour before the end so E2's letter stages still fit after it.
base_procs = ({1: phase("base_eval", 1, "outputs/e2/base_eval",
                        f"--deadline {DEADLINE - 3600:.0f}")} if N_GPUS > 1 else {})
while trainer.poll() is None:
    time.sleep(300)
    tail = open("outputs/e2-train.log").readlines()[-2:]
    print(time.strftime("%H:%M"), "".join(tail).strip()[-300:], flush=True)
print("".join(open("outputs/e2-train.log").readlines()[-25:]))
shutil.copy("outputs/e2-train.log", OUT / "e2_train.log")
if trainer.returncode != 0:
    for proc in base_procs.values():
        stop(proc)
    if Path("outputs/e2/base_eval").exists():
        shutil.copytree("outputs/e2/base_eval", OUT / "e2_base_eval_predictions", dirs_exist_ok=True)
    raise SystemExit(f"\\nSTOP. Training failed (exit {trainer.returncode}); log above.")
shutil.copytree("outputs/e2_adapter", OUT / "e2-adapter", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("checkpoint-*"))
print("adapter saved:", sorted(p.name for p in (OUT / "e2-adapter").iterdir()))

# GPU 0 scores E2 at once (dev, then MedQA reasoning). GPU 1 first finishes the
# base model, whose MedQA reasoning is the other half of that comparison, then
# scores E2 by letter choice.
E2_ARGS = f"--deadline {DEADLINE:.0f} --adapter e2=outputs/e2_adapter"
e2_procs = {0: phase("e2_eval", 0, "outputs/e2/e2_eval", E2_ARGS)}
if base_procs:
    print("base evaluation exit codes:", wait(base_procs, "outputs/e2/base_eval", 300))
    e2_procs[1] = phase("e2_eval", 1, "outputs/e2/e2_eval", E2_ARGS)
else:
    wait(e2_procs, "outputs/e2/e2_eval", 300)
    e2_procs = {1: phase("e2_eval", 1, "outputs/e2/e2_eval", E2_ARGS, device=0)}
print("e2 evaluation exit codes:", wait(e2_procs, "outputs/e2/e2_eval", 600))

for name in ("base_eval", "e2_eval"):
    if Path(f"outputs/e2/{name}").exists():
        shutil.copytree(f"outputs/e2/{name}", OUT / f"e2_{name}_predictions", dirs_exist_ok=True)
for log in Path("outputs/e2").glob("*_eval-gpu*.log"):
    shutil.copy(log, OUT / f"e2_{log.name}")
roots = " ".join(f"--root {OUT / f'e2_{n}_predictions'}" for n in ("base_eval", "e2_eval")
                 if (OUT / f"e2_{n}_predictions").exists())
step(f"python -m training.eval_report {roots} --bench-dir data/v3 "
     f"--out {OUT / 'e2_report.json'} --export {OUT / 'e2_per_question.jsonl'}")
os.chdir(OUT)
shutil.rmtree(WORK)
print("Done. Everything is in the Output panel.")'''

E2_DONE = """## Done

In the **Output** panel:
- `e2-adapter/`: the adapter, `train_stats.json` and `loss_curve.json`.
- `e2_generation/`: every sampled completion on the training pool, per GPU.
- `e2_traces_report.json`: generated, kept and rejected counts, the acceptance
  rate and the thinking-length distribution.
- `e2_base_eval_predictions/` and `e2_eval_predictions/`: every answer, with
  each worker's settings.
- `e2_report.json`: base against E2, stage by stage, with McNemar tests.
- `e2_per_question.jsonl`: one record per model, question and sample.
- The logs.

Model selection reads the **dev** stages. The benchmark stages are reported,
not tuned on."""


def e2_cells(data_fp: str) -> list[tuple[str, str]]:
    cells = dict(enumerate(run3_cells(data_fingerprint(RUN2_DATA))))
    hardware = cells[1][1]
    if E0_PLAN_LINE not in hardware:
        raise SystemExit("the hardware cell's plan line changed; update E0_PLAN_LINE")
    return [
        ("markdown", E2_INTRO),
        ("code", hardware.replace(E0_PLAN_LINE, E2_PLAN_NEW)),
        cells[2], cells[3],
        ("code", e2_fetch_cell(data_fp)),
        ("code", E2_DOWNLOAD),
        ("code", E2_VLLM),
        ("code", e2_phase_cell()),
        ("code", E2_GENERATE),
        ("code", E2_SELECT),
        ("code", E2_TRAIN_EVAL),
        ("markdown", E2_DONE),
    ]


if wanted("e2"):
    if (V3_DATA / "data_report.json").exists() and (RUN2_DATA / "data_report.json").exists():
        write_notebook(e2_cells(data_fingerprint(V3_DATA, V3_DATA_FILES)),
                       Path("e2/kaggle_e2.ipynb"))
    else:
        print("skipped e2: needs data/v2 and data/v3 (run training.prepare_pool_v3)")


# --- E2, train and evaluate from a saved generation ------------------------
# The first E2 session generated and selected its training set (4,608
# completions; 3,182 verified traces) and then lost training to a tokenizer
# lookup that offline mode blocked. That training set is uploaded as
# medical-ft-data-e2, so this session skips the five hours of generation and
# spends them on training and on scoring both models.
E2_DATA = Path("data/e2")

E2T_INTRO = """# Qwen3-4B medical fine-tune, E2: train and evaluate

E2 trains Qwen3-4B on its **own** reasoning, kept only when it reached the gold
answer. The first session generated that training set: 4,608 completions on
held-out MedQA and MedMCQA training questions, 69% verified, giving 3,182
traces and 796 answer-only rows (`medical-ft-data-e2`). That session then
failed at training, before any update. This session:

1. **Trains** on GPU 0: LoRA rank 16, lr 5e-5, one epoch, run 3's trainer.
   Meanwhile GPU 1 scores the base model on the dev split, the four
   benchmarks by letter choice, and MedQA by reasoning.
2. **Evaluates** the E2 adapter on the same stages.
3. Writes the base-vs-E2 comparison.

**Sidebar: Accelerator `GPU T4 x2`, Internet `On`.** Attach `medical-ft-code`,
`medical-ft-data-v3` and `medical-ft-data-e2`."""

E2T_DONE = """## Done

In the **Output** panel:
- `e2-adapter/`: the adapter, `train_stats.json` and `loss_curve.json`.
- `e2_base_eval_predictions/` and `e2_eval_predictions/`: every answer, with
  each worker's settings.
- `e2_report.json`: base against E2, stage by stage, with McNemar tests.
- `e2_per_question.jsonl`: one record per model, question and sample.
- `e2_train.log` and the queue logs.

Model selection reads the **dev** stages; the benchmark stages are reported."""


def e2t_cells(data_fp: str, traces_fp: str) -> list[tuple[str, str]]:
    cells = dict(enumerate(run3_cells(data_fingerprint(RUN2_DATA))))
    hardware = cells[1][1]
    if E0_PLAN_LINE not in hardware:
        raise SystemExit("the hardware cell's plan line changed; update E0_PLAN_LINE")
    return [
        ("markdown", E2T_INTRO),
        ("code", hardware.replace(E0_PLAN_LINE, 'print(f"\\n{N_GPUS} GPU(s). E2 trains on GPU 0 '
                                                'and evaluates on all of them.")')),
        cells[2], cells[3],
        ("code", e2_fetch_cell(data_fp, traces_fp)),
        ("code", E2_DOWNLOAD),
        ("code", E2_VLLM),
        ("code", e2_phase_cell(smoke=("base_eval",))),
        ("code", E2_TRAIN_EVAL),
        ("markdown", E2T_DONE),
    ]


if wanted("e2t"):
    if all((d / "data_report.json").exists() for d in (V3_DATA, RUN2_DATA)) and \
            (E2_DATA / "e2_report.json").exists():
        write_notebook(e2t_cells(data_fingerprint(V3_DATA, V3_DATA_FILES),
                                 data_fingerprint(E2_DATA, E2_DATA_FILES)),
                       Path("e2t/kaggle_e2t.ipynb"))
    else:
        print("skipped e2t: needs data/v2, data/v3 and data/e2")
