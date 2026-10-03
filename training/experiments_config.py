"""What each experiment runs. E0 is the baseline every later one is judged by.

    python -m training.run_plan show          # sessions, queues, time estimates

E0 scores three models with identical generation settings:
  A  base       Qwen3-4B, fp16, no adapter
  B  run3       run 3's adapter on the fp16 base (how it is served)
  C  run3_4bit  run 3's adapter on the 4-bit base it was trained on (control)

The specification asked for a 3,072-token reasoning budget on all four
benchmarks with two samples per MedQA question. At run 3's measured T4 speed
(22.9 s per base-model MedQA question at 1,536 tokens, batch 16) the whole
suite would take well over 100 T4-hours, so reasoning runs on fixed seeded
subsets. A subset is a prefix of the same shuffle run 2 and run 3 used, so
every model answers the same questions, and a deadline cut leaves a random
sample. Letter choice is cheap and runs on all 7,545 questions.
"""
from __future__ import annotations

from training.eval_worker import SAMPLING
from training.modeling import DEFAULT_BASE, TRAINING_BASE_4BIT

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa", "mmlu_medical")
LETTER = tuple(f"letter:{b}" for b in BENCHMARKS)
REASONING_REST = ("reasoning:mmlu_medical", "reasoning:pubmedqa", "reasoning:medmcqa")

# Run 3's adapter, as uploaded to Kaggle and checked by sha256 there.
ADAPTERS = {
    "run3": {"kaggle_dataset": "gb1105/medical-ft-adapter-run3",
             "local_dir": "training/outputs/run3",
             "sha256_prefix": "8240aa382a18d8b9"},
}

E0 = {
    "experiment_id": "E0",
    "purpose": "Baseline: base Qwen3-4B against run 3's adapter (fp16 and on its 4-bit "
               "training base), letter choice and thinking mode, same settings for all.",
    "seed": 1234,
    "think_budget": 3072,
    "batch_size": 16,
    # A batch of 16 at 3,072 new tokens needs ~8.7 GB of KV cache beside the
    # 8 GB of fp16 weights, more than a T4's 15 GB; 8 needs ~4.3 GB.
    "reasoning_batch_size": 8,
    "sizes": {"medqa": 200, "mmlu_medical": 150, "pubmedqa": 150, "medmcqa": 150},
    "samples": {"medqa": 2},
    "sampling": SAMPLING,
    "models": {
        "base": {"label": "A", "base": DEFAULT_BASE, "adapter": None,
                 "load_in_4bit": False},
        "run3": {"label": "B", "base": DEFAULT_BASE, "adapter": "run3",
                 "load_in_4bit": False},
        # The control answers each MedQA question once: it asks whether the
        # base the adapter runs on changes its behaviour, not how often a
        # second sample rescues it.
        "run3_4bit": {"label": "C", "base": TRAINING_BASE_4BIT, "adapter": "run3",
                      "load_in_4bit": True, "samples": {"medqa": 1}},
    },
    # session -> GPU -> queue of (model, stages), run in order until the
    # deadline, one worker per entry. Each GPU writes to its own directory
    # (<out>/<session>/gpu<N>/<model>/), so a model may be split across GPUs
    # and sessions; the report joins its stages back together. The base
    # model's two MedQA samples, the most expensive and most important stage,
    # get a GPU to themselves.
    "sessions": {
        "s1": {0: (("base", ("reasoning:medqa",)),),
               1: (("base", LETTER),
                   ("run3", LETTER + ("reasoning:medqa",) + REASONING_REST))},
        "s2": {0: (("base", ("reasoning:mmlu_medical", "reasoning:pubmedqa")),),
               1: (("run3_4bit", LETTER + ("reasoning:medqa",)),
                   ("base", ("reasoning:medmcqa",)))},
    },
    "kaggle": {"s1": "gb1105/qwen3-4b-medical-e0-baseline-session-1",
               "s2": "gb1105/qwen3-4b-medical-e0-baseline-session-2"},
}

EXPERIMENTS = {"E0": E0}
