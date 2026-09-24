# Medical Fine-tune: Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the trained adapter against the base model on the two held-out
benchmarks, in one Kaggle session that cannot fail on untested code.

**Architecture:** Evaluation drops Unsloth and loads through plain
`transformers` + `peft`, so the identical code path runs on this Mac first. The
base and tuned models come from one load: the base is the same weights with the
adapter switched off. Scoring is batched with left padding and renders prompts
with `enable_thinking=False`, which is the only way the prompt matches the
trained format. A local smoke run on Qwen3-0.6B proves the whole path; the
Kaggle notebook then runs a 32-question smoke pass in-session, projects the full
runtime, and only then starts the full 5,456.

**Tech Stack:** Python 3.12 (local) / 3.11 (Kaggle), `transformers` 5.x, `peft`,
`torch`, `pytest`, Kaggle CLI 2.2.4.

**Spec:** `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`
(section 3, Evaluation). Continues Plan 1,
`docs/superpowers/plans/2026-09-20-training-and-evaluation.md`, whose training
run completed on 2026-09-21.

## Global Constraints

- Every module starts with `from __future__ import annotations`.
- No `scipy`, no numpy.
- **No `torch`, `transformers`, `peft` or `unsloth` import at module scope** in
  any file under test. The pytest suite runs with no GPU and no network.
- **Evaluation never imports Unsloth.** Its patching of TRL caused three failed
  training runs, and it cannot run on the Mac where this code is smoke-tested.
- Seed is `42` everywhere a seed is taken.
- Answer letters are the four capitals `A B C D`.
- Held-out sets are exactly the ones training was decontaminated against:
  **MedQA-USMLE test 1,273** (`load()` defaults) and **MedMCQA validation 4,183**
  (`require_rationale=False, require_single_choice=False`). Total 5,456.
- Every inference prompt is rendered with `enable_thinking=False`. The trained
  assistant turn begins `<think>\n\n</think>\n\n`; Qwen3's template only emits
  that prefix when thinking is disabled.
- Evaluation base weights: `unsloth/Qwen3-4B` (full precision). `float16` on
  CUDA, `bfloat16` on MPS, `float32` on CPU.
- Kaggle GPU sessions stop at 9 hours. The evaluation budget is **27,000 s**
  (7.5 h), leaving 1.5 h of margin.
- Kaggle user `gb1105`. Datasets `medical-ft-code` and `medical-ft-adapter`.
  A kernel's id slug must equal the slug Kaggle derives from its title, or the
  second push fails with a 409.
- Commits name **garvbahl37-gif as sole contributor**. The repo's git config
  already sets the author; do not override it. No `Co-Authored-By` trailer and
  no "Generated with Claude Code" line on any commit.

## Review Focus

1. **Prompt drift.** A scoring prompt that does not end in the trained
   `<think>\n\n</think>\n\n` prefix scores the fine-tune off-format and sends the
   base model into a thinking block that eats its token budget. Pinned by
   Task 1 (the flag is passed) and Task 3 (a real Qwen3 tokenizer renders the
   exact prefix).
2. **Half-precision overflow on the T4.** Non-finite logits make argmax return
   `"A"` for every question, which reads as a plausible accuracy. Must stop, not
   score. Pinned by Task 2.
3. **Batching silently changing answers.** Right padding reads a pad token at the
   last position; wrong position ids shift every prediction. Batch size must not
   change a single prediction. Pinned by Task 2 (right padding refused) and
   Task 3 (batch 1 and batch 4 agree on a real model).
4. **"Base" that is not the base.** If disabling the adapter left any of it
   active, every reported gain would be understated by an unknown amount. Pinned
   by Task 3 (adapter-disabled scores equal a freshly loaded plain base).
5. **A full run that cannot finish in the session.** Generation dominates the
   runtime; a guess could run past 9 hours and lose everything. Pinned by
   Task 2 (the projection arithmetic) and Task 4 (the in-session smoke pass stops
   before the full run if the projection does not fit).

---

### Task 1: Load models without Unsloth, and render prompts in the trained format

**Files:**
- Create: `training/modeling.py`
- Modify: `training/prompts.py` (add `render_chat`, `render_inference_prompt`)
- Create: `tests/test_modeling.py`
- Modify: `tests/test_prompts.py` (append three tests)

**Interfaces:**
- Consumes: `prompts.build_messages(rec, *, with_answer)`, `records.Record`.
- Produces:
  - `modeling.DEFAULT_BASE: str = "unsloth/Qwen3-4B"`
  - `modeling.DeviceChoice` — frozen dataclass, fields `device: str`,
    `dtype_name: str`
  - `modeling.choose_device(requested: str | None, *, cuda: bool, mps: bool) -> DeviceChoice`
  - `modeling.load_model(base: str, adapter: str | None, *, device: str | None = None) -> tuple[model, tokenizer, DeviceChoice]`
    — tokenizer pads on the **left**; with an adapter the model is a
    `PeftModel`, so it has `.disable_adapter()`
  - `modeling.release_memory(device: str) -> None`
  - `prompts.render_chat(tok, messages: list[dict]) -> str`
  - `prompts.render_inference_prompt(tok, rec: Record, *, answer_prefix: bool = False) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_modeling.py`:

```python
from __future__ import annotations

import pytest

from training.modeling import DEFAULT_BASE, choose_device


def test_cuda_is_preferred_and_runs_in_float16():
    # The T4 has no native bfloat16; Unsloth reported "Bfloat16 = FALSE" on it.
    choice = choose_device(None, cuda=True, mps=True)
    assert (choice.device, choice.dtype_name) == ("cuda", "float16")


def test_apple_silicon_uses_mps_in_bfloat16():
    choice = choose_device(None, cuda=False, mps=True)
    assert (choice.device, choice.dtype_name) == ("mps", "bfloat16")


def test_cpu_falls_back_to_float32():
    choice = choose_device(None, cuda=False, mps=False)
    assert (choice.device, choice.dtype_name) == ("cpu", "float32")


def test_an_explicit_device_wins_when_it_is_available():
    assert choose_device("cpu", cuda=True, mps=True).device == "cpu"


def test_requesting_an_unavailable_device_stops_loudly():
    with pytest.raises(SystemExit, match="cuda"):
        choose_device("cuda", cuda=False, mps=True)


def test_an_unknown_device_name_stops_loudly():
    with pytest.raises(SystemExit, match="tpu"):
        choose_device("tpu", cuda=True, mps=True)


def test_the_default_base_is_full_precision_qwen3_4b():
    # The adapter was trained on Unsloth's 4-bit copy; LoRA deltas apply to
    # the unquantised weights unchanged, and those are what gets served.
    assert DEFAULT_BASE == "unsloth/Qwen3-4B"
    assert "bnb" not in DEFAULT_BASE
```

Append to `tests/test_prompts.py`, and add `render_chat, render_inference_prompt`
to its existing `from training.prompts import (...)` line:

```python
class _RecordingTokenizer:
    """Stands in for a real tokenizer; records what the template was asked."""

    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return "<rendered>"


def test_render_chat_disables_thinking_to_match_the_trained_format():
    tok = _RecordingTokenizer()
    render_chat(tok, [{"role": "user", "content": "hi"}])
    assert tok.kwargs["enable_thinking"] is False
    assert tok.kwargs["add_generation_prompt"] is True
    assert tok.kwargs["tokenize"] is False


def test_render_inference_prompt_sends_the_question_without_its_answer():
    tok = _RecordingTokenizer()
    render_inference_prompt(tok, MCQ)
    assert [m["role"] for m in tok.messages] == ["system", "user"]


def test_answer_prefix_appends_the_cue_so_the_next_token_is_the_letter():
    tok = _RecordingTokenizer()
    assert render_inference_prompt(tok, MCQ, answer_prefix=True) == "<rendered>Answer:"
    assert render_inference_prompt(tok, MCQ) == "<rendered>"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_modeling.py tests/test_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.modeling'` and
`ImportError: cannot import name 'render_chat'`.

- [ ] **Step 3: Write `training/modeling.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

# The adapter was trained on Unsloth's 4-bit copy of Qwen3-4B. Evaluation and
# serving load the full-precision weights instead: a LoRA adapter's deltas
# apply to the unquantised base unchanged, and scoring the full-precision model
# is scoring what actually gets served.
DEFAULT_BASE = "unsloth/Qwen3-4B"

_DTYPES = {"cuda": "float16", "mps": "bfloat16", "cpu": "float32"}


@dataclass(frozen=True)
class DeviceChoice:
    device: str
    dtype_name: str


def choose_device(requested: str | None, *, cuda: bool, mps: bool) -> DeviceChoice:
    """Pick where to run and at what precision. Pure, so it is testable.

    CUDA runs float16: the T4 has no native bfloat16. MPS runs bfloat16, which
    Apple silicon supports and which is Qwen3's native precision. CPU runs
    float32, because half precision on CPU is slow where it works at all.
    """
    if requested is not None and requested not in _DTYPES:
        raise SystemExit(
            f"\nSTOP. Unknown device {requested!r}; expected one of "
            f"{sorted(_DTYPES)}.")
    available = {"cuda": cuda, "mps": mps, "cpu": True}
    if requested is not None and not available[requested]:
        raise SystemExit(
            f"\nSTOP. --device {requested} was requested but is not available "
            "on this machine.")
    device = requested or ("cuda" if cuda else "mps" if mps else "cpu")
    return DeviceChoice(device, _DTYPES[device])


def load_model(base: str, adapter: str | None, *, device: str | None = None):
    """Load the base weights, wrap them with a LoRA adapter if given.

    Returns (model, tokenizer, DeviceChoice). With an adapter the model is a
    PeftModel, whose disable_adapter() context yields the untouched base --
    which is how evaluation scores both models from a single load.

    The tokenizer pads on the LEFT. Batched scoring reads the logits at the
    final position; with right padding that position is a pad token for every
    sequence shorter than the longest in the batch, and scores noise.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    choice = choose_device(device, cuda=torch.cuda.is_available(),
                           mps=torch.backends.mps.is_available())

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=getattr(torch, choice.dtype_name))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(choice.device)
    model.eval()
    return model, tok, choice


def release_memory(device: str) -> None:
    """Return freed memory to the device after the caller drops its model.

    `del model` alone leaves the CUDA allocator holding the freed blocks, so a
    following load can fail on a card that objectively has room.
    """
    import gc

    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
```

- [ ] **Step 4: Add the two renderers to `training/prompts.py`**

Insert directly after `build_messages`:

```python
def render_chat(tok, messages: list[dict]) -> str:
    """Render a conversation for the model to continue, in the trained format.

    Every training example rendered the assistant turn after an empty think
    block, '<think>\\n\\n</think>\\n\\n'. Qwen3's chat template only emits that
    prefix at inference when told enable_thinking=False. At its default the
    model starts in thinking mode instead: the fine-tune is then scored on a
    format it never saw, and the base spends its token budget thinking.
    """
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def render_inference_prompt(tok, rec: Record, *, answer_prefix: bool = False) -> str:
    """The prompt a benchmark question is scored with.

    answer_prefix=True appends 'Answer:' for constrained scoring, so the very
    next token is the choice itself -- ' A' to ' D', exactly what follows
    'Answer:' in every training target.
    """
    text = render_chat(tok, build_messages(rec, with_answer=False))
    return text + "Answer:" if answer_prefix else text
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_modeling.py tests/test_prompts.py -q`
Expected: PASS — 7 in `test_modeling.py`, 42 in `test_prompts.py`.

- [ ] **Step 6: Confirm the format against the real Qwen3 tokenizer**

The tokenizer is already cached locally from Plan 1. This is the check that
matters: the fake tokenizer only proves the flag is passed.

```bash
.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
from training.prompts import render_inference_prompt
from training.records import Record
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-unsloth-bnb-4bit")
rec = Record(id="x", source="medqa", kind="mcq", question="Which drug?",
             options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer="B",
             rationale=None, response=None, subject=None)
out = render_inference_prompt(tok, rec, answer_prefix=True)
want = "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
assert out.endswith(want), repr(out[-80:])
print("prompt ends with the trained prefix:", repr(out[-len(want):]))
PY
```

Expected: `prompt ends with the trained prefix: '<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:'`

- [ ] **Step 7: Run the full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — every test passes.

```bash
git add training/modeling.py training/prompts.py tests/test_modeling.py tests/test_prompts.py
git commit -m "Load models without Unsloth, and render prompts as they were trained

Evaluation now loads through plain transformers and peft, so the code
path that scores the model on Kaggle can run on a Mac first. Seven
Kaggle runs failed on code that had never executed anywhere, and
Unsloth cannot run locally.

The renderer passes enable_thinking=False. Every training example put
the answer after an empty think block, and Qwen3's template only emits
that block when thinking is disabled. At its default, the fine-tune was
about to be scored on a format it never saw, and the base model would
have spent its token budget thinking."
```

---

### Task 2: Batched, guarded scoring from a single load

**Files:**
- Modify: `training/evaluate.py`
- Modify: `tests/test_evaluate.py` (append six tests)

**Interfaces:**
- Consumes: `modeling.load_model`, `modeling.release_memory`,
  `modeling.DEFAULT_BASE`, `prompts.render_inference_prompt`,
  `prompts.extract_letter`, `prompts.LETTERS`, `evalcore.paired_report`.
- Produces:
  - `evaluate.batched(items: list, size: int) -> list[list]`
  - `evaluate.pick_from_logits(logits_row, letter_ids) -> str` — now raises
    `FloatingPointError` on non-finite scores
  - `evaluate.score_constrained(model, tok, recs, *, batch_size: int = 16) -> list[str]`
  - `evaluate.score_generative(model, tok, recs, *, batch_size: int = 16, max_new_tokens: int = 512) -> tuple[list[str | None], list[str]]`
  - `evaluate.project_eval_seconds(timing: dict, *, n_constrained: int, n_generative: int) -> int`
  - CLI: `python -m training.evaluate --adapter DIR --test FILE --out FILE
    [--base NAME] [--limit N] [--gen-limit N] [--batch-size N]
    [--max-new-tokens N] [--device cuda|mps|cpu]`
  - Report JSON keys: `test_set`, `base`, `adapter`, `device`, `timing`
    (`load_s`, `constrained_s_per_example`, `generative_s_per_example`, each
    per model), `reports` (`constrained`, `generative`), `samples` (up to 12
    `{id, answer, base, tuned}` raw completions)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py`, adding `batched, project_eval_seconds,
score_constrained, score_generative` to its existing import from
`training.evaluate`, plus `import ast` and `from pathlib import Path` at the top:

```python
def test_batched_splits_into_full_groups_and_a_remainder():
    assert batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_a_batch_size_below_one():
    with pytest.raises(ValueError):
        batched([1], 0)


def test_pick_from_logits_refuses_non_finite_scores():
    # float16 overflow on a T4 turns logits into inf or nan, and argmax over
    # them silently returns the first letter for every question -- which
    # reads as a plausible accuracy. It must stop instead.
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    with pytest.raises(FloatingPointError):
        pick_from_logits([float("nan"), 0.0, 1.0, 2.0], ids)
    with pytest.raises(FloatingPointError):
        pick_from_logits([0.0, float("inf"), 1.0, 2.0], ids)


def test_scorers_refuse_a_right_padded_tokenizer():
    # Right padding puts a pad token at the last position of every shorter
    # sequence in a batch; the scorer would read it and score noise.
    class RightPadded:
        padding_side = "right"

    with pytest.raises(SystemExit, match="left"):
        score_constrained(None, RightPadded(), [])
    with pytest.raises(SystemExit, match="left"):
        score_generative(None, RightPadded(), [])


def test_project_eval_seconds_scales_measured_rates_to_both_models():
    timing = {"constrained_s_per_example": 0.5, "generative_s_per_example": 4.0}
    # 2 models x (1,000 x 0.5 + 100 x 4.0) = 2 x 900 = 1,800
    assert project_eval_seconds(timing, n_constrained=1000, n_generative=100) == 1800


def test_evaluation_never_imports_unsloth():
    for path in ("training/evaluate.py", "training/modeling.py"):
        tree = ast.parse(Path(path).read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                    for a in n.names}
        imported |= {n.module for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom) and n.module}
        assert not any(m.split(".")[0] == "unsloth" for m in imported), path
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -q`
Expected: FAIL — `ImportError: cannot import name 'batched'`.

- [ ] **Step 3: Rewrite the loading and scoring half of `training/evaluate.py`**

Replace the imports at the top of the file with:

```python
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

from training.evalcore import paired_report
from training.modeling import DEFAULT_BASE, load_model, release_memory
from training.prompts import LETTERS, extract_letter, render_inference_prompt
from training.records import Record
```

Delete `BASE_MODEL` and `_load`. Keep `EXPECTED_HOLDOUT_SIZES`,
`letter_token_ids`, `check_holdout_size`, `select_scorable_records` and
`require_aligned` exactly as they are.

Replace `pick_from_logits` with:

```python
def pick_from_logits(logits_row, letter_ids: dict[str, int]) -> str:
    """Argmax restricted to the four answer letters. Ties go to the earliest.

    Refuses non-finite scores. float16 overflow on a T4 turns logits into inf
    or nan, and argmax over those silently returns the first letter for every
    question -- a plausible-looking accuracy that measures nothing.
    """
    scores = {letter: float(logits_row[letter_ids[letter]]) for letter in LETTERS}
    if not all(math.isfinite(s) for s in scores.values()):
        raise FloatingPointError(f"non-finite letter logits {scores}")
    best, best_score = None, None
    for letter in LETTERS:
        if best_score is None or scores[letter] > best_score:
            best, best_score = letter, scores[letter]
    return best
```

Replace `score_constrained` and `score_generative`, and add `batched`,
`_require_left_padding` and `project_eval_seconds`, all above `main`:

```python
def batched(items: list, size: int) -> list[list]:
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _require_left_padding(tok) -> None:
    if getattr(tok, "padding_side", "left") != "left":
        raise SystemExit(
            "\nSTOP. The tokenizer pads on the right. Batched scoring reads the "
            "final position, which right padding fills with a pad token for "
            "every shorter sequence.\nFIX: load through training.modeling."
            "load_model, which sets padding_side='left'.")


def score_constrained(model, tok, recs: list[Record], *, batch_size: int = 16
                      ) -> list[str]:
    """One forward pass per batch; compare the four letter logits.

    Position ids are derived from the attention mask so a left-padded
    sequence sees the same positions it would see alone, and only the last
    position's logits are materialised -- the full vocabulary at every
    position would be 4GB per batch of 16 on a T4.
    """
    _require_left_padding(tok)
    import torch

    ids = letter_token_ids(tok)
    out: list[str] = []
    for group in batched(recs, batch_size):
        texts = [render_inference_prompt(tok, r, answer_prefix=True) for r in group]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        positions = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            last = model(**enc, position_ids=positions,
                         logits_to_keep=1).logits[:, -1, :]
        for row in last:
            try:
                out.append(pick_from_logits(row, ids))
            except FloatingPointError as exc:
                raise SystemExit(
                    f"\nSTOP. {exc}.\nHalf precision overflowed, so every "
                    "prediction from here on would be meaningless.\n"
                    "FIX: re-run with --device cpu, which runs in float32.")
    return out


def score_generative(model, tok, recs: list[Record], *, batch_size: int = 16,
                     max_new_tokens: int = 512) -> tuple[list[str | None], list[str]]:
    """Greedy decode, then extract the stated letter. Returns letters and texts.

    With left padding every prompt in a batch ends at the same column, so
    each completion is everything after that column.
    """
    _require_left_padding(tok)
    import torch

    letters: list[str | None] = []
    texts: list[str] = []
    for group in batched(recs, batch_size):
        prompts = [render_inference_prompt(tok, r) for r in group]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for row in generated:
            text = tok.decode(row[width:], skip_special_tokens=True)
            texts.append(text)
            letters.append(extract_letter(text))
    return letters, texts


def project_eval_seconds(timing: dict, *, n_constrained: int,
                         n_generative: int) -> int:
    """Scale a smoke pass's measured per-model rates to a full run of both models."""
    per_model = (n_constrained * timing["constrained_s_per_example"]
                 + n_generative * timing["generative_s_per_example"])
    return int(2 * per_model)
```

- [ ] **Step 4: Replace `main` with the single-load version**

```python
def main() -> None:
    p = argparse.ArgumentParser(description="Score base against tuned, paired.")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--base", default=DEFAULT_BASE, help="full-precision base weights")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gen-limit", type=int, default=300,
                   help="how many examples also get generative scoring")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = p.parse_args()

    recs = [Record.from_dict(json.loads(line))
            for line in args.test.read_text().splitlines() if line]
    check_holdout_size(recs)
    recs = select_scorable_records(recs)
    if args.limit:
        recs = recs[: args.limit]
    gen_recs = recs[: args.gen_limit]
    print(f"scoring {len(recs):,} constrained, {len(gen_recs):,} generative")

    started = time.time()
    model, tok, choice = load_model(args.base, args.adapter, device=args.device)
    load_s = time.time() - started
    print(f"loaded {args.base} + adapter on {choice.device} "
          f"({choice.dtype_name}) in {load_s:.0f}s", flush=True)

    preds: dict[str, dict] = {"constrained": {}, "generative": {}}
    completions: dict[str, list[str]] = {}
    spent = {"constrained": 0.0, "generative": 0.0}
    # One load, two models. The base is the same weights with the adapter
    # switched off, which guarantees base and tuned share identical base
    # weights and removes the second load that could run a T4 out of memory.
    for tag in ("base", "tuned"):
        ctx = model.disable_adapter() if tag == "base" else contextlib.nullcontext()
        with ctx:
            t = time.time()
            preds["constrained"][tag] = score_constrained(
                model, tok, recs, batch_size=args.batch_size)
            spent["constrained"] += time.time() - t
            t = time.time()
            letters, texts = score_generative(
                model, tok, gen_recs, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens)
            spent["generative"] += time.time() - t
        preds["generative"][tag] = letters
        completions[tag] = texts
        print(f"  {tag}: scored", flush=True)

    del model
    release_memory(choice.device)

    require_aligned(recs, preds["constrained"]["base"],
                    preds["constrained"]["tuned"], "constrained")
    require_aligned(gen_recs, preds["generative"]["base"],
                    preds["generative"]["tuned"], "generative")

    reports = {
        "constrained": paired_report(
            recs, preds["constrained"]["base"], preds["constrained"]["tuned"],
            mode="constrained"),
        "generative": paired_report(
            gen_recs, preds["generative"]["base"], preds["generative"]["tuned"],
            mode="generative"),
    }
    timing = {
        "load_s": round(load_s, 1),
        "constrained_s_per_example": round(
            spent["constrained"] / (2 * max(len(recs), 1)), 4),
        "generative_s_per_example": round(
            spent["generative"] / (2 * max(len(gen_recs), 1)), 4),
    }
    samples = [
        {"id": r.id, "answer": r.answer,
         "base": completions["base"][i], "tuned": completions["tuned"][i]}
        for i, r in enumerate(gen_recs[:12])
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "test_set": str(args.test), "base": args.base, "adapter": args.adapter,
        "device": choice.device, "timing": timing, "reports": reports,
        "samples": samples,
    }, indent=2))

    for mode, rep in reports.items():
        m = rep["mcnemar"]
        print(f"\n=== {mode}  n={rep['n']:,} ===")
        print(f"  base  {rep['base_accuracy']:.1%}   "
              f"tuned {rep['tuned_accuracy']:.1%}   "
              f"change {(rep['tuned_accuracy'] - rep['base_accuracy']) * 100:+.1f}")
        print(f"  wins {m['wins']}  regressions {m['regressions']}  "
              f"p = {m['p_value']:.4f}")
        print(f"  unparseable: base {rep['base_unparseable']}, "
              f"tuned {rep['tuned_unparseable']}")
    print(f"\ntiming {timing}\nwrote {args.out}")
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -q`
Expected: PASS — the 14 existing tests plus the 6 new ones, 20 in all.

- [ ] **Step 6: Run the full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — every test passes.

```bash
git add training/evaluate.py tests/test_evaluate.py
git commit -m "Score both models from one load, in batches, and refuse overflow

The base is now the same weights with the adapter switched off, which
guarantees base and tuned share identical base weights and removes the
second load that could have run a T4 out of memory.

Scoring is batched with left padding and position ids taken from the
attention mask, so batch size cannot change an answer. Only the last
position's logits are computed; the full vocabulary at every position
would be 4GB per batch.

Non-finite logits now stop the run. float16 overflow on a T4 would
otherwise make argmax pick A for every question, which reads as a
plausible accuracy."
```

---

### Task 3: Prove the whole path locally on a tiny real Qwen3

**Files:**
- Create: `scripts/smoke_eval.py`
- Create: `requirements-smoke.txt`

**Interfaces:**
- Consumes: `modeling.load_model`, `modeling.release_memory`,
  `evaluate.score_constrained`, `evaluate.main` (via the CLI),
  `prompts.render_inference_prompt`, `records.Record`.
- Produces: a passing local run, recorded verbatim in the task report. No new
  importable API.

This is the task that makes the Kaggle run safe. It exercises the exact code the
notebook runs — the loader, both scorers, the adapter switch, the CLI, the report
— on `Qwen/Qwen3-0.6B`, which shares Qwen3-4B's tokenizer and chat template, with
a small random LoRA adapter so "tuned" genuinely differs from "base".

- [ ] **Step 1: Install the smoke dependencies**

```bash
printf 'torch>=2.6\npeft>=0.17\naccelerate>=1.0\n' > requirements-smoke.txt
.venv/bin/pip install -q -r requirements-smoke.txt
.venv/bin/python -c "import torch, peft; print('torch', torch.__version__, '| peft', peft.__version__)"
```

Expected: a version line for each. These stay out of `requirements-dev.txt` so
the fast suite keeps needing neither.

- [ ] **Step 2: Write `scripts/smoke_eval.py`**

```python
"""End-to-end smoke test of the evaluation path, on a tiny real Qwen3.

Runs the exact code the Kaggle evaluation runs -- modeling.load_model, the
batched scorers, the adapter switch, evaluate.py's CLI and report -- against
Qwen3-0.6B, which shares Qwen3-4B's tokenizer and chat template, with a small
random LoRA adapter. Seven Kaggle runs failed on code that had never executed
anywhere; this makes sure the evaluation cannot fail that way.

    .venv/bin/python scripts/smoke_eval.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TINY = "Qwen/Qwen3-0.6B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAINED_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

QUESTIONS = [
    ("Deficiency of which vitamin causes scurvy?",
     ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"], "B"),
    ("Which organ produces insulin?",
     ["Liver", "Kidney", "Pancreas", "Spleen"], "C"),
    ("The commonest cause of community-acquired pneumonia is:",
     ["Streptococcus pneumoniae", "Klebsiella", "Pseudomonas", "Legionella"], "A"),
    ("First-line treatment of anaphylaxis is:",
     ["Oral antihistamine", "Intramuscular adrenaline", "IV hydrocortisone",
      "Nebulised salbutamol"], "B"),
    ("The normal adult resting heart rate is:",
     ["20-40", "40-60", "60-100", "100-140"], "C"),
    ("Peaked T waves on an ECG suggest:",
     ["Hypokalaemia", "Hyperkalaemia", "Hyponatraemia", "Hypocalcaemia"], "B"),
    ("Koplik spots are seen in:",
     ["Measles", "Mumps", "Rubella", "Chickenpox"], "A"),
    ("The antidote for paracetamol overdose is:",
     ["Naloxone", "Flumazenil", "N-acetylcysteine", "Atropine"], "C"),
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.evaluate import score_constrained
    from training.modeling import load_model, release_memory
    from training.prompts import render_inference_prompt
    from training.records import Record

    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="medsmoke-"))
    recs = [
        Record(id=f"smoke-{i}", source="fixture", kind="mcq", question=q,
               options=dict(zip("ABCD", opts)), answer=a, rationale=None,
               response=None, subject="Smoke")
        for i, (q, opts, a) in enumerate(QUESTIONS)
    ]
    test_file = work / "fixtures.jsonl"
    test_file.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))

    # 1. The real template renders exactly the trained prefix plus the cue.
    tok = AutoTokenizer.from_pretrained(TINY)
    rendered = render_inference_prompt(tok, recs[0], answer_prefix=True)
    assert rendered.endswith(TRAINED_PREFIX + "Answer:"), repr(rendered[-80:])
    print("ok  the prompt ends with the trained prefix and 'Answer:'")

    # 2. A small random LoRA adapter, so 'tuned' genuinely differs from 'base'.
    torch.manual_seed(42)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=TARGETS, lora_dropout=0.0))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    adapter = work / "adapter"
    peft_model.save_pretrained(adapter)
    del peft_model, base
    print(f"ok  a random LoRA adapter saved to {adapter}")

    # 3. Batch size must not change a single prediction.
    model, tok, choice = load_model(TINY, str(adapter), device="cpu")
    one = score_constrained(model, tok, recs, batch_size=1)
    four = score_constrained(model, tok, recs, batch_size=4)
    assert one == four, f"batching changed predictions: {one} vs {four}"
    print(f"ok  batch 1 and batch 4 agree on all {len(one)}: {''.join(one)}")

    # 4. disable_adapter() must score the true base, nothing in between.
    with model.disable_adapter():
        via_disable = score_constrained(model, tok, recs, batch_size=4)
    del model
    release_memory(choice.device)
    plain, plain_tok, _ = load_model(TINY, None, device="cpu")
    via_plain = score_constrained(plain, plain_tok, recs, batch_size=4)
    assert via_disable == via_plain, (
        f"disabled adapter differs from a plain base: {via_disable} vs {via_plain}")
    print("ok  the disabled adapter scores identically to a freshly loaded base")
    del plain
    release_memory("cpu")

    # 5. The CLI end to end, exactly as the notebook calls it.
    out = work / "report.json"
    subprocess.run(
        [sys.executable, "-m", "training.evaluate", "--base", TINY,
         "--adapter", str(adapter), "--test", str(test_file), "--out", str(out),
         "--gen-limit", "4", "--max-new-tokens", "48", "--batch-size", "4",
         "--device", "cpu"],
        check=True, cwd=Path(__file__).resolve().parents[1])
    report = json.loads(out.read_text())
    assert report["reports"]["constrained"]["n"] == 8
    assert report["reports"]["generative"]["n"] == 4
    assert {"load_s", "constrained_s_per_example",
            "generative_s_per_example"} <= set(report["timing"])
    assert len(report["samples"]) == 4
    print("ok  evaluate.py end to end: both modes, timing and samples reported")

    print(f"\nSMOKE PASSED in {time.time() - started:.0f}s  ({work})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python scripts/smoke_eval.py`
Expected: five `ok` lines, then `SMOKE PASSED`. The first run downloads
Qwen3-0.6B, about 1.5 GB.

Any failure here is a real defect in Tasks 1-2 and must be fixed in those files,
not worked around in this script. In particular: if step 3 finds batch 1 and
batch 4 disagree, the padding or position-id handling in `score_constrained` is
wrong; if step 4 finds the disabled adapter differs from a plain base, the base
score in `main` would be wrong.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_eval.py requirements-smoke.txt
git commit -m "Prove the evaluation path end to end before it touches a GPU

Runs the exact code the Kaggle notebook runs against Qwen3-0.6B, which
shares Qwen3-4B's tokenizer and chat template, with a random LoRA
adapter. It checks what no unit test can: that the real template
renders the trained prefix, that batch size changes no prediction,
that the disabled adapter scores the true base, and that the CLI
writes a complete report.

Seven Kaggle runs failed on code that had never run anywhere. This
one will not be the eighth."
```

---

### Task 4: The evaluation notebook, and the uploads it needs

**Files:**
- Modify: `training/kaggle_paths.py` (`find_code_dir` takes `required`; add
  `EVAL_REQUIRED` and `find_adapter_dir`)
- Modify: `tests/test_kaggle_paths.py` (append four tests)
- Create: `scripts/kaggle_dataset.sh`
- Modify: `scripts/push_kaggle.sh` (use `kaggle_dataset.sh`; take a kernel dir)
- Create: `scripts/push_adapter.sh`
- Modify: `scripts/build_notebook.py` (build both notebooks)
- Create: `evaluation/kernel-metadata.json`
- Create: `evaluation/kaggle_eval.ipynb` (generated)
- Create: `tests/test_kernel_metadata.py`

**Interfaces:**
- Consumes: `evaluate.project_eval_seconds`, the evaluate CLI from Task 2,
  `sources.load(name, *, limit, split, require_rationale, require_single_choice)`.
- Produces:
  - `kaggle_paths.EVAL_REQUIRED: set[str]`
  - `kaggle_paths.find_code_dir(root: Path, required: set[str] = REQUIRED) -> Path`
  - `kaggle_paths.find_adapter_dir(root: Path) -> Path`
  - `bash scripts/kaggle_dataset.sh <slug> "<title>" <dir>` — create or version,
    then block until ready
  - `bash scripts/push_adapter.sh [adapter-dir]` — default `training/outputs/run1`
  - `bash scripts/push_kaggle.sh [kernel-dir]` — default `training`
  - Kaggle kernel `gb1105/qwen3-4b-medical-fine-tune-evaluation`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kaggle_paths.py`, adding `EVAL_REQUIRED, find_adapter_dir`
to its import from `training.kaggle_paths`:

```python
def test_find_code_dir_accepts_a_different_required_set(tmp_path):
    code = tmp_path / "datasets" / "gb1105" / "medical-ft-code"
    code.mkdir(parents=True)
    for stem in EVAL_REQUIRED:
        (code / f"{stem}.py").write_text("")
    assert find_code_dir(tmp_path, EVAL_REQUIRED) == code


def _adapter(directory):
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    return directory


def test_find_adapter_dir_finds_an_adapter_mounted_two_levels_deep(tmp_path):
    adapter = _adapter(tmp_path / "datasets" / "gb1105" / "medical-ft-adapter")
    assert find_adapter_dir(tmp_path) == adapter


def test_find_adapter_dir_prefers_the_final_adapter_over_a_checkpoint(tmp_path):
    final = _adapter(tmp_path / "run1")
    _adapter(tmp_path / "run1" / "checkpoint-541")
    assert find_adapter_dir(tmp_path) == final


def test_find_adapter_dir_ignores_a_config_with_no_weights_beside_it(tmp_path):
    half = tmp_path / "half"
    half.mkdir()
    (half / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="No LoRA adapter"):
        find_adapter_dir(tmp_path)
```

Create `tests/test_kernel_metadata.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@pytest.mark.parametrize("path", ["training/kernel-metadata.json",
                                  "evaluation/kernel-metadata.json"])
def test_kernel_id_matches_the_slug_kaggle_derives_from_the_title(path):
    # Kaggle names a kernel by slugifying its title and ignores a mismatched
    # id. The first push then lands somewhere else and the second 409s.
    meta = json.loads(Path(path).read_text())
    _owner, slug = meta["id"].split("/")
    assert slug == slugify(meta["title"])


def test_the_evaluation_kernel_attaches_code_and_adapter_on_a_private_gpu():
    meta = json.loads(Path("evaluation/kernel-metadata.json").read_text())
    assert set(meta["dataset_sources"]) == {"gb1105/medical-ft-code",
                                            "gb1105/medical-ft-adapter"}
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kaggle_eval.ipynb"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_kaggle_paths.py tests/test_kernel_metadata.py -q`
Expected: FAIL — `ImportError: cannot import name 'EVAL_REQUIRED'` and
`FileNotFoundError: evaluation/kernel-metadata.json`.

- [ ] **Step 3: Extend `training/kaggle_paths.py`**

Below the marker, change `find_code_dir`'s signature and its one use of
`REQUIRED`, and add the new names. The file after the marker becomes:

```python
REQUIRED = {"records", "prompts", "sources", "prepare_data",
            "check_lengths", "budget", "train"}

EVAL_REQUIRED = {"records", "prompts", "sources", "evalcore", "evaluate",
                 "modeling"}


def find_code_dir(root: Path, required: set[str] = REQUIRED) -> Path:
    """Locate the uploaded modules wherever Kaggle mounted them.

    The mount path is not stable: a dataset declared as gb1105/medical-ft-code
    turned up under /kaggle/input/datasets/... rather than at
    /kaggle/input/medical-ft-code. Two runs died on that assumption, so this
    searches for the directory that actually holds the modules instead.
    """
    if not root.exists():
        raise SystemExit(
            "\nSTOP. /kaggle/input does not exist -- no inputs are attached.\n"
            "FIX: sidebar -> + Add Input -> Datasets -> medical-ft-code.")
    candidates = []
    for path in root.rglob("*.py"):
        stems = {p.stem for p in path.parent.glob("*.py")}
        if required <= stems:
            candidates.append(path.parent)
    if not candidates:
        found = sorted(str(p.relative_to(root)) for p in root.rglob("*.py"))[:20]
        tree = sorted(str(p.relative_to(root)) for p in root.rglob("*"))[:30]
        raise SystemExit(
            f"\nSTOP. No directory under {root} contains all of {sorted(required)}.\n"
            f".py files found: {found or 'none'}\n"
            f"First entries under /kaggle/input: {tree}\n"
            "FIX: re-run scripts/push_kaggle.sh, then confirm the "
            "medical-ft-code dataset is attached in the sidebar.")
    return sorted(set(candidates))[0]


def find_adapter_dir(root: Path) -> Path:
    """Locate the uploaded LoRA adapter by content, the same way.

    A directory qualifies when adapter_config.json and adapter_model.safetensors
    sit side by side. A final adapter wins over any checkpoint-* directory, so a
    stray checkpoint cannot be scored in place of the finished run.
    """
    if not root.exists():
        raise SystemExit(
            "\nSTOP. /kaggle/input does not exist -- no inputs are attached.\n"
            "FIX: sidebar -> + Add Input -> Datasets -> medical-ft-adapter.")
    found = sorted({p.parent for p in root.rglob("adapter_config.json")
                    if (p.parent / "adapter_model.safetensors").exists()})
    final = [d for d in found if not d.name.startswith("checkpoint-")]
    if final or found:
        return (final or found)[0]
    tree = sorted(str(p.relative_to(root)) for p in root.rglob("*"))[:30]
    raise SystemExit(
        f"\nSTOP. No LoRA adapter under {root}: no adapter_config.json with "
        "adapter_model.safetensors beside it.\n"
        f"First entries under /kaggle/input: {tree}\n"
        "FIX: run scripts/push_adapter.sh, then attach medical-ft-adapter in "
        "the sidebar.")
```

- [ ] **Step 4: Write `scripts/kaggle_dataset.sh`**

```bash
#!/usr/bin/env bash
# Create or version a private Kaggle Dataset from a directory, then block until
# Kaggle reports it ready.
#
#   bash scripts/kaggle_dataset.sh <slug> "<title>" <dir>
#
# `datasets create` and `datasets version` are asynchronous. A kernel pushed
# before its dataset is ready starts with nothing attached; that cost a run.
set -euo pipefail

SLUG="$1"; TITLE="$2"; SRC="$3"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
USER="$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"

command -v "$KAGGLE" >/dev/null || { echo "kaggle CLI not found at $KAGGLE"; exit 1; }
[ -d "$SRC" ] || { echo "no such directory: $SRC"; exit 1; }

cat > "$SRC/dataset-metadata.json" <<JSON
{"title": "$TITLE", "id": "$USER/$SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON

if "$KAGGLE" datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "==> updating dataset $USER/$SLUG"
  "$KAGGLE" datasets version -p "$SRC" -m "update $(date -u +%FT%TZ)" --dir-mode skip
else
  echo "==> creating dataset $USER/$SLUG"
  "$KAGGLE" datasets create -p "$SRC" --dir-mode skip
fi

echo "==> waiting for $USER/$SLUG to finish processing"
status=""
for _ in $(seq 1 120); do
  status="$("$KAGGLE" datasets status "$USER/$SLUG" 2>&1 || true)"
  case "$status" in
    *ready*) echo "    ready"; exit 0 ;;
    *error*) echo "    processing FAILED: $status"; exit 1 ;;
  esac
  sleep 5
done
echo "    still not ready after 10 minutes: $status"
exit 1
```

- [ ] **Step 5: Rewrite `scripts/push_kaggle.sh` to use it**

```bash
#!/usr/bin/env bash
# Upload the code as a private Kaggle Dataset, then push a notebook.
#
#   bash scripts/push_kaggle.sh              # the training notebook
#   bash scripts/push_kaggle.sh evaluation   # the evaluation notebook
#
# The evaluation notebook also needs the adapter: run scripts/push_adapter.sh
# first. Credentials come from ~/.kaggle/kaggle.json.
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL_DIR="${1:-training}"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
[ -f "$KERNEL_DIR/kernel-metadata.json" ] || {
  echo "no kernel-metadata.json in $KERNEL_DIR"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# Flat, no subdirectories: --dir-mode zip would upload training/ as a zip, and
# whether Kaggle extracts it is not worth discovering during a GPU run. The
# notebook reassembles the package itself.
cp training/*.py training/requirements.txt "$STAGE/"
bash scripts/kaggle_dataset.sh medical-ft-code "Medical Fine-tune Code" "$STAGE"

echo "==> pushing the notebook in $KERNEL_DIR"
"$KAGGLE" kernels push -p "$KERNEL_DIR"

ID="$(python3 -c "import json;print(json.load(open('$KERNEL_DIR/kernel-metadata.json'))['id'])")"
echo
echo "  https://www.kaggle.com/code/$ID"
```

- [ ] **Step 6: Write `scripts/push_adapter.sh`**

```bash
#!/usr/bin/env bash
# Upload the trained LoRA adapter as a private Kaggle Dataset.
#
#   bash scripts/push_adapter.sh [adapter-dir]    # default training/outputs/run1
#
# Only the two files PEFT needs go up. The tokenizer saved beside them is the
# base model's own, which evaluation loads from the base.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:-training/outputs/run1}"
for f in adapter_config.json adapter_model.safetensors; do
  [ -s "$ADAPTER/$f" ] || { echo "missing $ADAPTER/$f"; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" "$STAGE/"
bash scripts/kaggle_dataset.sh medical-ft-adapter "Medical Fine-tune Adapter" "$STAGE"
```

Then: `chmod +x scripts/kaggle_dataset.sh scripts/push_adapter.sh scripts/push_kaggle.sh`
and `bash -n` each of the three to confirm they parse.

- [ ] **Step 7: Write `evaluation/kernel-metadata.json`**

```json
{
  "id": "gb1105/qwen3-4b-medical-fine-tune-evaluation",
  "title": "Qwen3-4B Medical Fine-tune (evaluation)",
  "code_file": "kaggle_eval.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["gb1105/medical-ft-code", "gb1105/medical-ft-adapter"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- [ ] **Step 8: Make `scripts/build_notebook.py` build both notebooks**

Three changes.

(a) Turn the notebook-writing block at the bottom of the file into a function,
and call it for the training notebook:

```python
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
```

(b) Add the evaluation code-fetch cell after `CELL_GET_CODE`. It inlines the same
`FIND_CODE_DIR_SRC`, so `find_adapter_dir` and `EVAL_REQUIRED` come with it:

```python
CELL_GET_CODE_EVAL = ('''# --- 3. Get the code and the adapter from the attached datasets ------------
import os, shlex, shutil, subprocess, sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/ft")
PKG   = WORK / "training"
PKG.mkdir(parents=True, exist_ok=True)

''' + FIND_CODE_DIR_SRC + '''

SRC = find_code_dir(INPUT, EVAL_REQUIRED)
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
```

(c) Add `EVAL_CELLS` after `CELLS`. The hardware cell is reused from the training
notebook by reference, `CELLS[1]`, so the two cannot drift:

```python
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
!pip install -q --no-deps peft
!pip install -q datasets"""),
    ("code", '''# --- 2. Verify the install before spending GPU time on it ------------------
import torch, transformers, peft
print(f"ok | torch {torch.__version__} | transformers {transformers.__version__} "
      f"| peft {peft.__version__}")'''),
    ("code", CELL_GET_CODE_EVAL),
    ("markdown", """## 4. Build the held-out sets

Exactly the sets training was decontaminated against. MedMCQA validation is
loaded with both filters off: training drops rows without an explanation and
rows marked multi-choice, but the benchmark keeps all 4,183, or the score would
not be comparable to any published MedMCQA figure."""),
    ("code", '''import json
from training.sources import load

HOLDOUTS = {
    "medqa": ("test", {}),
    "medmcqa": ("validation", {"require_rationale": False,
                               "require_single_choice": False}),
}
EXPECTED = {"medqa": 1273, "medmcqa": 4183}

for name, (split, flags) in HOLDOUTS.items():
    recs = load(name, limit=0, split=split, **flags)
    if len(recs) != EXPECTED[name]:
        raise SystemExit(
            f"\\nSTOP. {name} {split} gave {len(recs):,} rows, expected "
            f"{EXPECTED[name]:,}.\\nFIX: these must be the exact sets training "
            "was decontaminated against.")
    with open(f"data/holdout_{name}.jsonl", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec.to_dict()) + "\\n")
    print(f"data/holdout_{name}.jsonl  {len(recs):,}")'''),
    ("markdown", """## 5. Smoke pass

32 questions per benchmark through the exact code the full run uses. It proves
the path works on this GPU, measures real throughput, and stops here if the full
run would not fit the session. It also stops if the fine-tune's answers mostly
cannot be parsed, which would mean truncated output rather than a real
result."""),
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
    print(f"{name}: tuned answers unparseable {gen['tuned_unparseable']}/{gen['n']}, "
          f"base {gen['base_unparseable']}/{gen['n']}")
    if gen["tuned_unparseable"] > gen["n"] // 2:
        raise SystemExit(
            f"\\nSTOP. {gen['tuned_unparseable']} of {gen['n']} fine-tuned answers "
            "could not be parsed, which means truncated output, not a result.\\n"
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
```

- [ ] **Step 9: Build both notebooks and validate them**

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/python - <<'PY'
import ast, json
for path in ("training/kaggle_medical.ipynb", "evaluation/kaggle_eval.ipynb"):
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        # Comment out IPython magics and shell escapes at line start only;
        # a blanket replace would also mangle `!=` inside step().
        src = "\n".join("#" + line if line.lstrip().startswith(("!", "%")) else line
                        for line in "".join(cell["source"]).splitlines())
        ast.parse(src, filename=f"{path} cell {i}")
    print(f"{path}: {len(nb['cells'])} cells, every code cell parses")
PY
git diff --stat training/kaggle_medical.ipynb
```

Expected: `wrote training/kaggle_medical.ipynb, 13 cells`, `wrote
evaluation/kaggle_eval.ipynb, 13 cells`, then both "every code cell parses"
lines. The training notebook's diff must be limited to the lines inlined from
`kaggle_paths.py`; inspect it with `git diff training/kaggle_medical.ipynb` and
confirm nothing else changed.

- [ ] **Step 10: Run the full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — every test passes.

Do **not** run any push script. Uploading is the controller's step.

```bash
git add training/kaggle_paths.py tests/test_kaggle_paths.py \
        tests/test_kernel_metadata.py scripts/ evaluation/
git commit -m "Add the evaluation notebook and the uploads it depends on

One Kaggle session: build the exact held-out sets training was cleaned
against, run a 32-question smoke pass through the full code path,
project the runtime from what it measured, and only then score all
5,456 questions against the base model.

The adapter is found by content, like the code, and a final adapter
wins over a stray checkpoint. The kernel id is tested against the slug
Kaggle derives from the title, which is what 409'd a push last time.

Dataset upload and the wait for it to be ready now live in one script
that both pushes use."
```

---

## After this plan: the one Kaggle start

Run by the controller, in this order, only after all four tasks are reviewed:

1. `bash scripts/push_adapter.sh` — uploads `training/outputs/run1`, waits for ready.
2. `bash scripts/push_kaggle.sh evaluation` — refreshes the code dataset, waits,
   pushes the evaluation notebook.
3. Watch it. The smoke pass reports within about 15 minutes; if it clears, the
   full run is the same code on more rows.

## Self-Review

**Spec coverage (section 3).**

| spec requirement | task |
|---|---|
| two held-out sets, never trained on, exact sizes | 4 (built and size-checked in the notebook), 2 (`check_holdout_size` kept) |
| constrained scoring on the full sets | 2 |
| generative scoring on a subsample, lenient extractor | 2 |
| paired, identical prompts, greedy decoding | 2 (one load, same renderer, `do_sample=False`) |
| McNemar, per-subject breakdown, shared scoring module | unchanged `evalcore`, used by 2 |
| results reported for both modes | 4 (results cell) |

**Placeholder scan.** No TBD or TODO. Every code step carries the code; every
test step carries its assertions.

**Type consistency.** `load_model` returns `(model, tok, DeviceChoice)` in Task 1
and is unpacked that way in Tasks 2 and 3. `score_generative` returns
`(letters, texts)` in Task 2 and is unpacked that way in `main`. `find_code_dir`
keeps its default argument, so the training notebook's `find_code_dir(INPUT)`
still resolves. `project_eval_seconds` reads `constrained_s_per_example` and
`generative_s_per_example`, which are exactly the keys `main` writes.

**Review Focus.** Each of the five lines has its test in the owning task:
prompt drift in Tasks 1 and 3; overflow in Task 2; batching in Tasks 2 and 3;
the base check in Task 3; the session projection in Tasks 2 and 4.
