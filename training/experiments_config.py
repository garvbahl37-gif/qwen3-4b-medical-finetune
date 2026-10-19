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


# E2: Qwen3-4B trained on its own gold-verified reasoning, in one Kaggle session.
# Generation, then training on GPU 0 while GPU 1 scores the base model, then
# both GPUs score the new adapter. Selection reads the dev split (data/v3),
# never the benchmarks; the benchmark stages are reported, not tuned on.
DEV = ("reasoning:dev_medqa", "reasoning:dev_medmcqa", "letter:dev_medqa", "letter:dev_medmcqa")
E2 = {
    "experiment_id": "E2",
    "purpose": "Self-distillation: fine-tune on base Qwen3-4B's own reasoning, kept only "
               "when it reaches the gold letter, and compare with the base on the dev split.",
    "seed": 1234,
    # Evaluation uses E0's budget. Generation gets more (see gen entries): the
    # m1 paper puts the medical optimum near 4K tokens, and a trace cut at the
    # budget is rejected, so a tight budget would select for short thinking.
    "think_budget": 3072,
    "batch_size": 16,
    "reasoning_batch_size": 8,
    "vllm": {"batch_size": 512, "reasoning_batch_size": 192, "gpu_memory_utilization": 0.90},
    "sizes": {"medqa": 0, "pool": 0, "dev_medqa": 0, "dev_medmcqa": 0},
    "samples": {"pool": 2},
    "sampling": SAMPLING,
    "models": {
        "base": {"label": "A", "base": DEFAULT_BASE, "adapter": None, "load_in_4bit": False},
        "e2": {"label": "E2", "base": DEFAULT_BASE, "adapter": "e2", "load_in_4bit": False},
    },
    "sessions": {
        "gen": {0: (("base", ("reasoning:pool",),
                     {"shard": "0/2", "think_budget": 4096, "hf_reasoning_batch_size": 6}),),
                1: (("base", ("reasoning:pool",),
                     {"shard": "1/2", "think_budget": 4096, "hf_reasoning_batch_size": 6}),)},
        # Letter choice is minutes with vLLM, so both models get it on every
        # benchmark from the same engine; MedQA reasoning is the long stage.
        "base_eval": {1: (("base", DEV + LETTER + ("reasoning:medqa",)),)},
        "e2_eval": {0: (("e2", DEV + ("reasoning:medqa",)),),
                    1: (("e2", LETTER),)},
    },
    "traces": {"max_think": 3500, "cap": 4096, "direct_fraction": 0.2, "seed": 42,
               "min_rows": 200},
    # Conservative next to run 3 (rank 64, lr 1e-4): the data is the model's
    # own, so the update only has to sharpen what it already does.
    "training": {"rank": 16, "lr": 5e-5, "epochs": 1, "batch_size": 1, "grad_accum": 16,
                 "save_steps": 100, "probe_steps": 20, "max_hours": 4.0},
    "kaggle": {"e2": "gb1105/qwen3-4b-medical-e2-self-distillation"},
}

EXPERIMENTS = {"E0": E0, "E2": E2}


# E3: corrective traces from a stronger Qwen3 teacher on the questions the base
# gets wrong ("weak": data/e3/eval_weak.jsonl, 1,418 pool questions on which the
# base's E2 sample was wrong or unfinished). E3's training set is E2's plus the
# teacher's verified traces, trained exactly as E2 was, so the corrective
# examples are the one change. Session 1 pilots three teachers on 120 weak
# questions and lets the one with the most rescued questions per hour
# generate; session 2 trains and scores on the dev split only.
TEACHERS = {"t8b": "Qwen/Qwen3-8B-AWQ", "t14b": "Qwen/Qwen3-14B-AWQ",
            "t30b": "Qwen/Qwen3-30B-A3B-GPTQ-Int4"}
WEAK = ("reasoning:weak",)
PILOT = {"think_budget": 4096, "sizes": {"weak": 120}, "samples": {"weak": 1},
         "no_fallback": True}
TEACH = {"think_budget": 4096, "no_fallback": True}
E3 = {
    "experiment_id": "E3",
    "purpose": "Teacher-generated, gold-verified reasoning on the base model's weak "
               "questions, added to E2's training set; selection on the dev split.",
    "seed": 1234,
    "think_budget": E2["think_budget"],
    "batch_size": E2["batch_size"],
    "reasoning_batch_size": E2["reasoning_batch_size"],
    "vllm": E2["vllm"],
    "sizes": {"weak": 0, "dev_medqa": 0, "dev_medmcqa": 0},
    # Three teacher samples per weak question; up to two verified ones are kept.
    "samples": {"weak": 3},
    "sampling": SAMPLING,
    "models": {
        "base": {"label": "A", "base": DEFAULT_BASE, "adapter": None, "load_in_4bit": False},
        "e3": {"label": "E3", "base": DEFAULT_BASE, "adapter": "e3", "load_in_4bit": False},
        **{name: {"label": name, "base": repo, "adapter": None, "load_in_4bit": False,
                  "plain": True, "gpu_memory_utilization": 0.95,
                  **({"tensor_parallel": 2} if name == "t30b" else {})}
           for name, repo in TEACHERS.items()},
    },
    "sessions": {
        "pilot": {0: (("t14b", WEAK, {**PILOT, "shard": ""}),),
                  1: (("t8b", WEAK, {**PILOT, "shard": ""}),)},
        "pilot_tp": {0: (("t30b", WEAK, PILOT),)},
        **{f"teach_{t}": ({0: ((t, WEAK, TEACH),)} if t == "t30b" else
                          {0: ((t, WEAK, {**TEACH, "shard": "0/2"}),),
                           1: ((t, WEAK, {**TEACH, "shard": "1/2"}),)})
           for t in TEACHERS},
        "base_dev": {1: (("base", DEV),)},
        "e3_dev": {0: (("e3", ("reasoning:dev_medqa", "letter:dev_medqa")),),
                   1: (("e3", ("reasoning:dev_medmcqa", "letter:dev_medmcqa")),)},
    },
    "traces": {"max_think": 3500, "cap": 4096, "max_per_question": 2, "seed": 42,
               "min_rows": 100},
    "training": E2["training"],
    "kaggle": {"e3a": "gb1105/qwen3-4b-medical-e3-teacher-traces",
               "e3b": "gb1105/qwen3-4b-medical-e3-train-and-evaluate"},
}

EXPERIMENTS = {"E0": E0, "E2": E2, "E3": E3}
