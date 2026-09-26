from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.kaggle_paths import code_fingerprint, data_fingerprint

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


write_notebook(CELLS, Path("training/kaggle_medical.ipynb"))
write_notebook(EVAL_CELLS, Path("evaluation/kaggle_eval.ipynb"))
def run3_cells(data_fp: str) -> list[tuple[str, str]]:
    """Run 3 is run 2's notebook with its own names. run2/kaggle_run2.ipynb
    stays as it ran; run 3's change is in the code it uploads (every prompt
    states /think or /no_think, and the budget-forcing pass no longer runs a
    T4 out of memory)."""
    return [(kind, src.replace("run2", "run3").replace("run 2", "run 3"))
            for kind, src in run2_cells(data_fp)]


if (RUN2_DATA / "data_report.json").exists():
    write_notebook(run3_cells(data_fingerprint(RUN2_DATA)), Path("run3/kaggle_run3.ipynb"))
else:
    print("skipped run3/kaggle_run3.ipynb: no data/v2 (run training.prepare_data_v2)")
