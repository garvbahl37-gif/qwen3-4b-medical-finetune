# Medical Fine-tune: Training & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data, training and evaluation pipeline that fine-tunes
Qwen3-4B on ~40,000 medical examples on a free Kaggle T4 and reports an honest
base-vs-tuned comparison on two held-out exam benchmarks.

**Architecture:** Four dataset loaders normalise into one `Record` type. A
single prompt module builds every prompt, so training and evaluation cannot
drift apart. Preparation filters, decontaminates and splits. Training is Unsloth
4-bit QLoRA with completion-only loss, a step-50 throughput probe that aborts a
run that will not fit the budget, and 200-step checkpoints. Evaluation scores
base and tuned on identical prompts two ways — constrained logits and greedy
generation — and reports McNemar significance.

**Tech Stack:** Python 3.12 (local) / 3.11 (Kaggle), `datasets`, `transformers`,
`unsloth`, `trl`, `peft`, `pytest`. No `scipy`, no `torch` in tested modules.

**Spec:** `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`

## Global Constraints

- Every module starts with `from __future__ import annotations` so 3.11 and 3.12
  agree on annotation evaluation.
- **No `scipy`.** McNemar's exact test uses `math.comb` from the standard library.
- **No `torch` or `unsloth` import at module scope in any file under test.**
  Tests must run on a laptop with no GPU in under 10 seconds. GPU-only code is
  imported inside the function that needs it.
- Seed is `42` everywhere a seed is taken.
- `max_seq` is derived from a measured p99, never hardcoded.
- Training budget is 6 hours; the probe aborts above it. Kaggle allows 9 h per
  GPU session and 30 GPU-hours per week.
- Answer letters are the four capitals `A B C D`.
- Commits name **garvbahl37-gif as sole contributor** — no `Co-Authored-By`
  trailer on any commit.
- All paths in this plan are relative to `FIneTuningMedicalDataset/`.

---

### Task 1: Scaffold, and the shared prompt module

The prompt module is first because everything else depends on it. If training
and evaluation build prompts differently, every number the project reports is
meaningless.

**Files:**
- Create: `training/__init__.py`, `training/prompts.py`, `training/records.py`
- Create: `tests/__init__.py`, `tests/test_prompts.py`
- Create: `requirements-dev.txt`, `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `records.Record` — frozen dataclass, fields `id: str`, `source: str`,
    `kind: str` (`"mcq"` or `"dialogue"`), `question: str`,
    `options: dict[str, str] | None`, `answer: str | None`,
    `rationale: str | None`, `response: str | None`, `subject: str | None`
  - `records.Record.to_dict() -> dict`, `records.Record.from_dict(d) -> Record`
  - `prompts.SYSTEM_MCQ: str`, `prompts.SYSTEM_CHAT: str`
  - `prompts.build_messages(rec: Record, *, with_answer: bool) -> list[dict]`
  - `prompts.target_text(rec: Record) -> str`
  - `prompts.extract_letter(text: str) -> str | None`

- [ ] **Step 1: Create the virtualenv and dev requirements**

```bash
cd FIneTuningMedicalDataset
/Users/garvbahl/Documents/Projects/FIneTuning/.venv/bin/python -m venv .venv
printf 'pytest>=8\ndatasets>=4\n' > requirements-dev.txt
.venv/bin/pip install -q -r requirements-dev.txt
printf '[pytest]\ntestpaths = tests\naddopts = -q\n' > pytest.ini
touch training/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_prompts.py`:

```python
from __future__ import annotations

import pytest

from training.prompts import build_messages, extract_letter, target_text
from training.records import Record

MCQ = Record(
    id="m1", source="medmcqa", kind="mcq",
    question="Which vitamin deficiency causes scurvy?",
    options={"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"},
    answer="B", rationale="Scurvy is caused by a lack of ascorbic acid.",
    response=None, subject="Medicine",
)
CHAT = Record(
    id="c1", source="chatdoctor", kind="dialogue",
    question="I have had a sore throat for three days. What should I do?",
    options=None, answer=None, rationale=None,
    response="A sore throat lasting three days is usually viral.", subject=None,
)


def test_mcq_prompt_lists_every_option_and_hides_the_answer():
    msgs = build_messages(MCQ, with_answer=False)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    for letter, text in MCQ.options.items():
        assert f"{letter}. {text}" in user
    assert "Scurvy is caused" not in user


def test_mcq_target_ends_with_a_parseable_answer_line():
    target = target_text(MCQ)
    assert target.rstrip().endswith("Answer: B")
    assert MCQ.rationale in target


def test_with_answer_appends_the_assistant_turn():
    msgs = build_messages(MCQ, with_answer=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == target_text(MCQ)


def test_dialogue_uses_the_chat_system_prompt_not_the_mcq_one():
    mcq_sys = build_messages(MCQ, with_answer=False)[0]["content"]
    chat_sys = build_messages(CHAT, with_answer=False)[0]["content"]
    assert mcq_sys != chat_sys
    assert target_text(CHAT) == CHAT.response


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C", "C"),
        ("**Answer:** D", "D"),
        ("answer - a", "A"),
        ("The reasoning is long.\n\nAnswer: B", "B"),
        ("Answer: B\nActually, Answer: D", "D"),
        ("(C)", "C"),
        ("C.", "C"),
        ("C) Vitamin D", "C"),
        ("```\nAnswer: A\n```", "A"),
        ("A 45-year-old man presents, most consistent with choice C.", "C"),
        ("A 23-year-old woman presents. The answer is B.", "B"),
        ("The correct option is D.", "D"),
        ("Reasoning about Vitamin C and Hepatitis A.\n\nAnswer: B", "B"),
        ("The correct answer here would be C.", "C"),
        ("Given the findings, C is the correct choice.", "C"),
        ("So my final choice would be C", "C"),
        ("Vitamin D is the best option for this patient's deficiency.", "D"),
        ("I think the answer is A. Wait, reconsidering, C is the best choice.", "C"),
        ("Answer: A. Actually, on reflection, B is correct.", "B"),
        ("Option A is wrong. Option B is incorrect. The correct answer is C.", "C"),
        ("A is not right, B can be excluded, so the best choice is D.", "D"),
        ("I cannot determine this.", None),
        ("", None),
    ],
)
def test_extract_letter_is_lenient_about_format(text, expected):
    assert extract_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Vitamin D deficiency is the most likely cause given the presentation.",
        "This is likely due to Hepatitis B infection based on the serology.",
        "The patient has blood group A and is Rh negative.",
        "I am uncertain, but this could relate to A or B depending on labs.",
        "A 45-year-old man presents with acute chest pain radiating to the jaw.",
        "The correct management of pneumonia requires antibiotics.",
        "Choose wisely when interpreting serology results.",
        "The diagnosis was incorrect, and A does not fit either.",
        "This is not correct: option A is wrong, and there is no clear best fit here.",
    ],
)
def test_extract_letter_refuses_to_invent_an_answer_from_clinical_prose(text):
    # A false positive is worse than None: None scores wrong for base and
    # tuned alike, while a guessed letter is indistinguishable from a real
    # answer in the aggregate accuracy the project reports.
    assert extract_letter(text) is None


def test_record_survives_a_dict_round_trip():
    assert Record.from_dict(MCQ.to_dict()) == MCQ


def test_extract_letter_ignores_letters_inside_words():
    assert extract_letter("Diabetes And Cancer") is None


def test_extract_letter_prefers_an_explicit_answer_line_over_a_stray_letter():
    assert extract_letter("A patient presents.\nAnswer: D") == "D"
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.prompts'`

- [ ] **Step 4: Write `training/records.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Record:
    """One training or evaluation example, normalised across all four sources."""

    id: str
    source: str
    kind: str  # "mcq" | "dialogue"
    question: str
    options: dict[str, str] | None
    answer: str | None
    rationale: str | None
    response: str | None
    subject: str | None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(**d)
```

- [ ] **Step 5: Write `training/prompts.py`**

```python
from __future__ import annotations

import re

from training.records import Record

LETTERS = ("A", "B", "C", "D")

SYSTEM_MCQ = (
    "You are a medical education assistant. Answer the multiple-choice question "
    "by reasoning briefly from the clinical findings, then stating your choice "
    "on a final line in the exact form 'Answer: X'."
)

SYSTEM_CHAT = (
    "You are a medical education assistant. Explain clearly and carefully in "
    "plain language. You are not a substitute for a clinician: do not give a "
    "definitive diagnosis, and advise the person to seek in-person care when "
    "their description warrants it."
)

# CORRECTED AFTER TASK 1 REVIEW (rulings R6, R8). The first version matched
# the first A-D anywhere, returning "A" for "A 45-year-old man ... choice C."
# The second over-corrected and missed "C is the correct choice." A letter
# counts only with evidence: a cue word within 28 characters of a standalone
# letter, in either order.
_CUE = r"(?:answers?|options?|choice|choose|select(?:ed)?|correct|best)"
_CUE_THEN_LETTER = re.compile(
    r"\b" + _CUE + r"\b[^\n]{0,28}?\b([A-D])\b(?![A-Za-z])", re.IGNORECASE)
# Without the reverse direction the extractor misses the base model, whose
# phrasing varies most -- and understating the base score flatters the
# fine-tune, which is the one direction this project must not be wrong in.
_LETTER_THEN_CUE = re.compile(
    r"\b([A-D])\b[^\n]{0,28}?\b" + _CUE + r"\b", re.IGNORECASE)
# A letter the model is rejecting rather than choosing. Without this,
# "option A is wrong" reads as a vote for A -- and elimination reasoning
# ("A is wrong, B is incorrect, the answer is C") is a standard
# chain-of-thought shape, so the mistake would be common.
_NEGATED = re.compile(
    r"^\W{0,3}(?:is|are|was|were|does|do|can)?\s*(?:not\b|n't\b|never\b|wrong\b"
    r"|incorrect\b|excluded\b|ruled out\b|unlikely\b)", re.IGNORECASE)
# A letter standing alone as the final line: "C", "(C)", "C."
_FINAL_BARE = re.compile(r"^\s*\(?([A-D])[).:]?\s*$")
# A final line that opens with a choice marker: "C) Vitamin D"
_FINAL_MARKER = re.compile(r"^\s*\(?([A-D])[).]\s+\S")


def format_question(rec: Record) -> str:
    if rec.kind != "mcq":
        return rec.question
    lines = [rec.question, ""]
    options = rec.options or {}
    lines += [f"{letter}. {options[letter]}" for letter in LETTERS if letter in options]
    return "\n".join(lines)


def target_text(rec: Record) -> str:
    if rec.kind == "dialogue":
        return rec.response or ""
    rationale = (rec.rationale or "").strip()
    return f"{rationale}\n\nAnswer: {rec.answer}" if rationale else f"Answer: {rec.answer}"


def build_messages(rec: Record, *, with_answer: bool) -> list[dict]:
    system = SYSTEM_MCQ if rec.kind == "mcq" else SYSTEM_CHAT
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": format_question(rec)},
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": target_text(rec)})
    return msgs


def extract_letter(text: str) -> str | None:
    """Pull the chosen letter out of a model response.

    Lenient about format, strict about evidence. The base model wraps
    answers in markdown and prose, and penalising formatting rather than
    correctness would distort the base-vs-tuned comparison. But a letter
    guessed from prose that states no answer is worse than no answer at
    all: it is indistinguishable from a real one in the aggregate, while
    None costs both models equally.

    So a letter counts only with evidence: a cue word within 28 characters
    of a standalone letter, in either order, and not one the model is
    rejecting. The candidate that ends LAST wins across both directions,
    because a model that reconsiders states its conclusion last.
    """
    if not text:
        return None
    best: tuple[int, str] | None = None
    for pattern in (_CUE_THEN_LETTER, _LETTER_THEN_CUE):
        for match in pattern.finditer(text):
            if _NEGATED.match(text[match.end(1):match.end(1) + 24]):
                continue
            if best is None or match.end(1) > best[0]:
                best = (match.end(1), match.group(1).upper())
    if best:
        return best[1]
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        for pattern in (_FINAL_BARE, _FINAL_MARKER):
            found = pattern.match(lines[-1])
            if found:
                return found.group(1).upper()
    return None
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -q`
Expected: PASS, 39 passed.

- [ ] **Step 7: Commit**

```bash
git add training/ tests/ requirements-dev.txt pytest.ini
git commit -m "Share one prompt template between training and evaluation

If the two built prompts differently every reported number would be
measuring a different model than the one being served. One module
builds both, and the answer extractor is lenient on purpose: the base
model wraps answers in markdown, and scoring formatting rather than
correctness would flatter the fine-tune."
```

---

### Task 2: Dataset normalisation, with the filters the spec requires

Fetching and normalising are separated so the normalisers can be tested offline
against fixture dictionaries. This is what keeps the suite fast and network-free.

**Files:**
- Create: `training/sources.py`
- Create: `tests/test_sources.py`

**Interfaces:**
- Consumes: `records.Record` from Task 1.
- Produces:
  - `sources.normalise_medmcqa(raw: dict, idx: int, *, require_rationale: bool = True) -> Record | None`
  - `sources.normalise_medqa(raw: dict, idx: int) -> Record | None`
  - `sources.normalise_medical_o1(raw: dict, idx: int) -> Record | None`
  - `sources.normalise_chatdoctor(raw: dict, idx: int) -> Record | None`
  - `sources.MIN_RATIONALE_CHARS: int = 80`
  - `sources.load(name: str, *, limit: int, seed: int = 42, split: str | None = None, require_rationale: bool = True, fetch=None) -> list[Record]`
    — raises `SystemExit` when a non-zero `limit` cannot be met; `limit=0` means
    every surviving row
  - `sources.SOURCES: dict[str, tuple[str, str, str]]` mapping short name to
    `(hf_id, config, default_split)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sources.py`:

```python
from __future__ import annotations

import pytest

from training.records import Record
from training.sources import (
    load,
    normalise_chatdoctor,
    normalise_medical_o1,
    normalise_medmcqa,
    normalise_medqa,
)

GOOD_MEDMCQA = {
    "id": "x1",
    "question": "Chronic urethral obstruction leads to which change?",
    "opa": "Hyperplasia", "opb": "Hypertrophy", "opc": "Atrophy", "opd": "Dysplasia",
    "cop": 2,
    "choice_type": "single",
    "exp": "Chronic obstruction causes hydronephrosis, which over time thins and "
           "atrophies the renal parenchyma through sustained back pressure.",
    "subject_name": "Anatomy",
}


def test_medmcqa_maps_cop_index_to_the_right_letter():
    rec = normalise_medmcqa(GOOD_MEDMCQA, 0)
    assert rec is not None
    assert rec.answer == "C"           # cop=2 -> opc -> "Atrophy"
    assert rec.options["C"] == "Atrophy"
    assert rec.kind == "mcq"
    assert rec.subject == "Anatomy"


def test_medmcqa_drops_multi_choice_rows():
    raw = dict(GOOD_MEDMCQA, choice_type="multi")
    assert normalise_medmcqa(raw, 0) is None


def test_medmcqa_drops_rows_with_no_usable_explanation():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp=None), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp="   "), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, exp="Correct."), 0) is None


def test_medmcqa_keeps_unexplained_rows_when_the_rationale_is_not_required():
    # Evaluation only needs question, options and answer. Filtering the
    # benchmark by explanation-availability would change what it measures.
    raw = dict(GOOD_MEDMCQA, exp=None)
    assert normalise_medmcqa(raw, 0) is None
    rec = normalise_medmcqa(raw, 0, require_rationale=False)
    assert rec is not None and rec.answer == "C" and rec.rationale == ""


def test_medmcqa_drops_rows_with_an_out_of_range_answer_index():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, cop=-1), 0) is None
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, cop=9), 0) is None


def test_medmcqa_drops_rows_with_a_blank_option():
    assert normalise_medmcqa(dict(GOOD_MEDMCQA, opb=""), 0) is None


def test_medqa_uses_its_own_answer_idx():
    raw = {
        "question": "A 23-year-old pregnant woman presents with dysuria.",
        "options": {"A": "Ampicillin", "B": "Ceftriaxone",
                    "C": "Doxycycline", "D": "Nitrofurantoin"},
        "answer": "Nitrofurantoin",
        "answer_idx": "D",
        "meta_info": "step2&3",
    }
    rec = normalise_medqa(raw, 0)
    assert rec is not None and rec.answer == "D"
    assert rec.options["D"] == "Nitrofurantoin"
    assert rec.rationale is None


def test_medqa_drops_rows_that_are_not_four_option():
    raw = {"question": "q", "options": {"A": "a", "B": "b"},
           "answer": "a", "answer_idx": "A", "meta_info": ""}
    assert normalise_medqa(raw, 0) is None


def test_medical_o1_keeps_the_chain_of_thought_as_the_rationale():
    raw = {
        "Question": "What cardiac abnormality explains a paradoxical embolism?",
        "Complex_CoT": "Sudden limb weakness suggests stroke; a swollen calf "
                       "suggests DVT; together they suggest a right-to-left shunt.",
        "Response": "A patent foramen ovale.",
    }
    rec = normalise_medical_o1(raw, 0)
    assert rec is not None and rec.kind == "dialogue"
    assert rec.rationale is not None
    assert "patent foramen ovale" in rec.response


def test_chatdoctor_builds_question_from_patient_input_only():
    raw = {
        "instruction": "If you are a doctor, answer based on the description.",
        "input": "I woke up feeling the room spinning and felt nauseous.",
        "output": "This sounds like benign paroxysmal positional vertigo.",
    }
    rec = normalise_chatdoctor(raw, 0)
    assert rec is not None and rec.kind == "dialogue"
    assert "room spinning" in rec.question
    assert rec.response.startswith("This sounds like")


def test_chatdoctor_drops_empty_turns():
    assert normalise_chatdoctor(
        {"instruction": "x", "input": "", "output": "y"}, 0) is None
    assert normalise_chatdoctor(
        {"instruction": "x", "input": "y", "output": "  "}, 0) is None


def _medmcqa_row(i: int, *, usable: bool = True) -> dict:
    return {
        "id": f"r{i}",
        "question": f"Question number {i}?",
        "opa": "one", "opb": "two", "opc": "three", "opd": "four",
        "cop": i % 4,
        "choice_type": "single" if usable else "multi",
        "exp": "e" * 100,
        "subject_name": "Anatomy",
    }


def _fetch(rows):
    """Stand in for datasets.load_dataset so the sampling tests stay offline."""
    return lambda hf_id, config, split: rows


def test_load_returns_exactly_the_limit_when_enough_rows_survive():
    rows = [_medmcqa_row(i) for i in range(200)]
    got = load("medmcqa", limit=10, fetch=_fetch(rows))
    assert len(got) == 10
    assert all(isinstance(r, Record) for r in got)


def test_load_stops_early_instead_of_scanning_the_whole_source():
    rows = [_medmcqa_row(i) for i in range(500)]
    assert len(load("medmcqa", limit=5, fetch=_fetch(rows))) == 5


def test_load_raises_rather_than_silently_returning_short():
    # Only 8 of 20 survive the filters. Asking for 15 must fail loudly: a
    # training mix that comes up short reports a ratio it does not have.
    rows = [_medmcqa_row(i, usable=i < 8) for i in range(20)]
    with pytest.raises(SystemExit) as excinfo:
        load("medmcqa", limit=15, fetch=_fetch(rows))
    message = str(excinfo.value)
    assert "8" in message
    assert "--medmcqa" in message


def test_load_with_limit_zero_returns_every_surviving_row():
    rows = [_medmcqa_row(i, usable=i % 2 == 0) for i in range(20)]
    assert len(load("medmcqa", limit=0, fetch=_fetch(rows))) == 10


def test_load_is_deterministic_for_a_seed_and_varies_across_seeds():
    rows = [_medmcqa_row(i) for i in range(200)]
    first = [r.id for r in load("medmcqa", limit=10, fetch=_fetch(rows))]
    again = [r.id for r in load("medmcqa", limit=10, fetch=_fetch(rows))]
    other = [r.id for r in load("medmcqa", limit=10, seed=7, fetch=_fetch(rows))]
    assert first == again
    assert first != other


def test_load_threads_require_rationale_through_to_the_normaliser():
    rows = [dict(_medmcqa_row(i), exp="") for i in range(20)]
    with pytest.raises(SystemExit):
        load("medmcqa", limit=5, fetch=_fetch(rows))
    assert len(load("medmcqa", limit=5, require_rationale=False,
                    fetch=_fetch(rows))) == 5
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_sources.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.sources'`

- [ ] **Step 3: Write `training/sources.py`**

```python
from __future__ import annotations

import random

from training.records import Record

LETTERS = ("A", "B", "C", "D")

# Measured on 600 sampled MedMCQA rows: 12.5% have an empty `exp` and a further
# 12.5% have one under 80 characters. Below that length the field says "correct"
# rather than why, and teaches the model nothing.
MIN_RATIONALE_CHARS = 80

SOURCES: dict[str, tuple[str, str, str]] = {
    "medmcqa": ("openlifescienceai/medmcqa", "default", "train"),
    "medqa": ("GBaker/MedQA-USMLE-4-options", "default", "train"),
    "medical_o1": ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train"),
    "chatdoctor": ("lavita/ChatDoctor-HealthCareMagic-100k", "default", "train"),
}


def _clean(value) -> str:
    return str(value).strip() if value is not None else ""


def normalise_medmcqa(raw: dict, idx: int, *,
                      require_rationale: bool = True,
                      require_single_choice: bool = True) -> Record | None:
    # MedMCQA marks a third of its rows choice_type != "single" while still
    # exposing a single `cop` index. Those are the known-noisy rows, so training
    # drops them. Evaluation keeps them: scoring 2,858 of the 4,183 published
    # validation questions would make the number incomparable to every reported
    # MedMCQA figure, and the noise costs base and tuned alike.
    if require_single_choice and _clean(raw.get("choice_type")).lower() != "single":
        return None

    options = {letter: _clean(raw.get(f"op{letter.lower()}")) for letter in LETTERS}
    if not all(options.values()):
        return None

    cop = raw.get("cop")
    if not isinstance(cop, int) or not 0 <= cop < len(LETTERS):
        return None

    rationale = _clean(raw.get("exp"))
    # Training needs an explanation to learn from. Evaluation does not, and
    # filtering the benchmark by explanation-availability would drop a quarter
    # of MedMCQA validation and quietly change what the score means.
    if require_rationale and len(rationale) < MIN_RATIONALE_CHARS:
        return None

    question = _clean(raw.get("question"))
    if not question:
        return None

    return Record(
        id=_clean(raw.get("id")) or f"medmcqa-{idx}",
        source="medmcqa", kind="mcq", question=question, options=options,
        answer=LETTERS[cop], rationale=rationale, response=None,
        subject=_clean(raw.get("subject_name")) or None,
    )


def normalise_medqa(raw: dict, idx: int) -> Record | None:
    options = raw.get("options") or {}
    options = {k: _clean(v) for k, v in options.items()}
    if set(options) != set(LETTERS) or not all(options.values()):
        return None

    answer = _clean(raw.get("answer_idx")).upper()
    if answer not in LETTERS:
        return None

    question = _clean(raw.get("question"))
    if not question:
        return None

    # MedQA ships no explanation, so there is no rationale to teach. The value
    # here is the vignette: long, noisy clinical narrative the other sources lack.
    return Record(
        id=f"medqa-{idx}", source="medqa", kind="mcq", question=question,
        options=options, answer=answer, rationale=None, response=None,
        subject=_clean(raw.get("meta_info")) or None,
    )


def normalise_medical_o1(raw: dict, idx: int) -> Record | None:
    question = _clean(raw.get("Question"))
    response = _clean(raw.get("Response"))
    cot = _clean(raw.get("Complex_CoT"))
    if not question or not response:
        return None
    return Record(
        id=f"medical_o1-{idx}", source="medical_o1", kind="dialogue",
        question=question, options=None, answer=None,
        rationale=cot or None,
        response=f"{cot}\n\n{response}" if cot else response,
        subject=None,
    )


def normalise_chatdoctor(raw: dict, idx: int) -> Record | None:
    patient = _clean(raw.get("input"))
    output = _clean(raw.get("output"))
    if not patient or not output:
        return None
    return Record(
        id=f"chatdoctor-{idx}", source="chatdoctor", kind="dialogue",
        question=patient, options=None, answer=None, rationale=None,
        response=output, subject=None,
    )


_NORMALISERS = {
    "medmcqa": normalise_medmcqa,
    "medqa": normalise_medqa,
    "medical_o1": normalise_medical_o1,
    "chatdoctor": normalise_chatdoctor,
}


def load(name: str, *, limit: int, seed: int = 42, split: str | None = None,
         require_rationale: bool = True, require_single_choice: bool = True,
         fetch=None) -> list[Record]:
    """Fetch a source from the Hub and normalise it.

    `limit=0` means every row that survives the filters, and is what the
    held-out benchmarks use. Any other limit is a requirement rather than
    a ceiling: if the filters cannot produce that many rows this raises,
    because a training mix that quietly comes up short goes on to report
    a mixture ratio the data does not have.

    `fetch` exists so the sampling can be tested without a network; it
    takes (hf_id, config, split) and returns an indexable sequence of raw
    rows. Production callers leave it None.
    """
    hf_id, config, default_split = SOURCES[name]

    if fetch is None:
        from datasets import load_dataset  # imported here so tests stay offline

        def fetch(hf_id: str, config: str, split: str):
            return load_dataset(hf_id, config, split=split)

    ds = fetch(hf_id, config, split or default_split)

    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    normalise = _NORMALISERS[name]
    extra = (
        {"require_rationale": require_rationale,
         "require_single_choice": require_single_choice}
        if name == "medmcqa" else {}
    )
    kept: list[Record] = []
    seen = 0
    for idx in order:
        seen += 1
        rec = normalise(ds[idx], idx, **extra)
        if rec is not None:
            kept.append(rec)
        if limit and len(kept) >= limit:
            break

    rate = len(kept) / seen if seen else 0.0
    print(f"  {name:<12} kept {len(kept):>6,} of {seen:>6,} inspected ({rate:.1%})")

    if limit and len(kept) < limit:
        raise SystemExit(
            f"\nSTOP. {name} yielded only {len(kept):,} usable rows of "
            f"{len(ds):,} inspected, but {limit:,} were requested.\n"
            f"FIX: lower --{name.replace('_', '-')} to {len(kept):,} or below, "
            f"or use a larger split."
        )
    return kept
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_sources.py -q`
Expected: PASS, 19 passed.

- [ ] **Step 5: Verify the normalisers against the real Hub data**

This is the step that catches a schema drift the fixtures cannot. Requires network.

```bash
.venv/bin/python -c "
from training.sources import load
for name in ['medmcqa','medqa','medical_o1','chatdoctor']:
    recs = load(name, limit=50)
    assert len(recs) == 50, (name, len(recs))
    r = recs[0]
    assert r.question and (r.answer or r.response)
print('all four sources normalise against live data')
"
```

Expected: a kept/inspected line per source, then the final line. If a source
reports a kept rate under 40%, stop and inspect — the schema has changed.

- [ ] **Step 6: Commit**

```bash
git add training/sources.py tests/test_sources.py
git commit -m "Normalise four medical sources, dropping what cannot teach

Sampling 600 MedMCQA rows found 12.5% with an empty explanation and
12.5% more under 80 characters, which say 'correct' rather than why.
A third are flagged choice_type != single while still exposing one cop
index; those are the known-noisy rows. All are dropped, and load()
prints the kept rate so a schema change upstream shows up as a cliff
rather than as silently worse training data.

Fetching is split from normalising so the filters test offline."
```

---

### Task 3: Decontamination and the preparation CLI

**Files:**
- Create: `training/prepare_data.py`
- Create: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `sources.load`, `records.Record`.
- Produces:
  - `prepare_data.normalise_question(text: str) -> str`
  - `prepare_data.decontaminate(train: list[Record], holdouts: list[Record]) -> tuple[list[Record], int]`
  - CLI writing `data/train.jsonl`, `data/val.jsonl`, `data/report.json`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_data.py`:

```python
from __future__ import annotations

from training.prepare_data import decontaminate, normalise_question
from training.records import Record


def mk(rid: str, question: str) -> Record:
    return Record(id=rid, source="medmcqa", kind="mcq", question=question,
                  options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer="A",
                  rationale="x" * 100, response=None, subject=None)


def test_normalise_question_ignores_case_punctuation_and_spacing():
    a = normalise_question("Which  vitamin, deficiency causes SCURVY?")
    b = normalise_question("which vitamin deficiency causes scurvy")
    assert a == b


def test_normalise_question_still_separates_genuinely_different_questions():
    assert normalise_question("causes of scurvy") != normalise_question("causes of rickets")


def test_decontaminate_removes_training_rows_that_appear_in_a_holdout():
    train = [mk("1", "Causes of scurvy?"), mk("2", "Causes of rickets?")]
    holdout = [mk("h", "causes of scurvy")]
    kept, removed = decontaminate(train, holdout)
    assert removed == 1
    assert [r.id for r in kept] == ["2"]


def test_decontaminate_is_a_no_op_when_nothing_overlaps():
    train = [mk("1", "a question"), mk("2", "another question")]
    kept, removed = decontaminate(train, [mk("h", "unrelated")])
    assert removed == 0 and len(kept) == 2


def test_decontaminate_removes_duplicates_within_the_training_set_too():
    train = [mk("1", "Causes of scurvy?"), mk("2", "causes of  scurvy")]
    kept, removed = decontaminate(train, [])
    assert len(kept) == 1 and removed == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_prepare_data.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.prepare_data'`

- [ ] **Step 3: Write `training/prepare_data.py`**

```python
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from training.records import Record
from training.sources import load

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalise_question(text: str) -> str:
    """Collapse a question to a comparable key for overlap detection."""
    text = unicodedata.normalize("NFKD", text or "").lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def decontaminate(train: list[Record], holdouts: list[Record]) -> tuple[list[Record], int]:
    """Drop training rows whose question appears in a holdout, or twice in train.

    A benchmark whose answers are in the training set measures memorisation.
    Duplicates inside the training set are dropped for the same reason: they
    silently reweight whatever they duplicate.
    """
    blocked = {normalise_question(r.question) for r in holdouts}
    kept: list[Record] = []
    removed = 0
    for rec in train:
        key = normalise_question(rec.question)
        if key in blocked:
            removed += 1
            continue
        blocked.add(key)
        kept.append(rec)
    return kept, removed


def main() -> None:
    p = argparse.ArgumentParser(description="Build the medical training set.")
    p.add_argument("--medmcqa", type=int, default=14000)
    p.add_argument("--medqa", type=int, default=8000)
    p.add_argument("--medical-o1", type=int, default=6000)
    p.add_argument("--chatdoctor", type=int, default=12000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading held-out benchmarks first, so training can be cleaned against them.")
    holdouts = (
        load("medqa", limit=0, seed=args.seed, split="test")
        + load("medmcqa", limit=0, seed=args.seed, split="validation",
               require_rationale=False, require_single_choice=False)
    )
    print(f"  holdout pool: {len(holdouts):,} questions\n")

    print("Loading training sources.")
    train: list[Record] = []
    for name, want in (
        ("medmcqa", args.medmcqa), ("medqa", args.medqa),
        ("medical_o1", args.medical_o1), ("chatdoctor", args.chatdoctor),
    ):
        if want:
            train += load(name, limit=want, seed=args.seed)

    before = len(train)
    train, removed = decontaminate(train, holdouts)
    print(f"\ndecontamination: dropped {removed:,} of {before:,} "
          f"({removed / before:.2%}) as holdout overlap or duplicate")

    random.Random(args.seed).shuffle(train)
    val, train = train[: args.val_size], train[args.val_size:]

    for name, rows in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        with path.open("w") as fh:
            for rec in rows:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        print(f"wrote {path}  {len(rows):,} rows")

    mix = {}
    for rec in train:
        mix[rec.source] = mix.get(rec.source, 0) + 1
    exam = sum(1 for r in train if r.kind == "mcq")

    report = {
        "train_size": len(train), "val_size": len(val),
        "holdout_pool": len(holdouts),
        "decontaminated_removed": removed, "decontaminated_from": before,
        "mix": mix,
        "exam_fraction": round(exam / len(train), 4) if train else 0.0,
        "seed": args.seed,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmix: {mix}")
    print(f"exam/reasoning {exam / len(train):.0%}, conversational "
          f"{1 - exam / len(train):.0%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_prepare_data.py -q`
Expected: PASS, 5 passed.

- [ ] **Step 5: Smoke-test the CLI at tiny scale**

```bash
.venv/bin/python -m training.prepare_data \
  --medmcqa 40 --medqa 20 --medical-o1 20 --chatdoctor 20 \
  --val-size 10 --out /tmp/medprep
cat /tmp/medprep/report.json
```

Expected: a `decontamination:` line with a real count, then `report.json`
showing `train_size` 90, `val_size` 10 and all four sources in `mix`.

- [ ] **Step 6: Commit**

```bash
git add training/prepare_data.py tests/test_prepare_data.py
git commit -m "Decontaminate the training set against both benchmarks

The holdouts load before the training sources so every training row can
be checked against them. Any row whose normalised question appears in
MedQA test or MedMCQA validation is dropped, and so are duplicates
within training, which silently reweight whatever they duplicate.

The count goes in report.json. A benchmark that cannot report this
number is not measuring generalisation, and saying so afterwards is
worth less than refusing to publish without it."
```

---

### Task 4: Length measurement, so `max_seq` is measured rather than guessed

**Files:**
- Create: `training/check_lengths.py`
- Create: `tests/test_check_lengths.py`

**Interfaces:**
- Consumes: `records.Record`, `prompts.build_messages`.
- Produces:
  - `check_lengths.percentile(values: list[int], q: float) -> int`
  - `check_lengths.summarise(lengths: list[int]) -> dict` with keys
    `n`, `median`, `p90`, `p99`, `max`
  - CLI with `--data`, `--max-seq`, `--drop`

- [ ] **Step 1: Write the failing test**

Create `tests/test_check_lengths.py`:

```python
from __future__ import annotations

from training.check_lengths import percentile, summarise


def test_percentile_picks_the_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.99) == 99
    assert percentile(values, 1.0) == 100


def test_percentile_handles_a_single_value():
    assert percentile([7], 0.99) == 7


def test_percentile_is_order_independent():
    assert percentile([5, 1, 3, 2, 4], 0.5) == percentile([1, 2, 3, 4, 5], 0.5)


def test_summarise_reports_the_fields_the_notebook_reads():
    stats = summarise([10, 20, 30, 40, 50])
    assert stats["n"] == 5
    assert stats["max"] == 50
    assert set(stats) == {"n", "median", "p90", "p99", "max"}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_check_lengths.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.check_lengths'`

- [ ] **Step 3: Write `training/check_lengths.py`**

```python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.prompts import build_messages
from training.records import Record


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile. No numpy, so this runs anywhere."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(lengths: list[int]) -> dict:
    return {
        "n": len(lengths),
        "median": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure token lengths and optionally drop what will not fit.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    p.add_argument("--max-seq", type=int, default=0,
                   help="0 means report only and suggest a value")
    p.add_argument("--drop", action="store_true",
                   help="rewrite --data without examples that exceed --max-seq")
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy; imported only when needed

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]

    lengths = []
    for row in rows:
        msgs = build_messages(Record.from_dict(row), with_answer=True)
        # return_dict=False is load-bearing. transformers 5.x returns a
        # BatchEncoding here, so len() would count its dict keys -- 2 -- rather
        # than tokens. That yields p99=2, a suggested max_seq of 64, and a
        # training run that truncates every example to fragments without error.
        lengths.append(
            len(tok.apply_chat_template(msgs, tokenize=True, return_dict=False)))

    stats = summarise(lengths)
    print(f"tokens  n={stats['n']:,}  median={stats['median']}  "
          f"p90={stats['p90']}  p99={stats['p99']}  max={stats['max']}")

    if not args.max_seq:
        suggested = 64 * math.ceil(stats["p99"] / 64)
        print(f"\nsuggested --max-seq {suggested} (p99 rounded up to a multiple of 64)")
        print(f"it would drop {sum(1 for n in lengths if n > suggested):,} examples")
        return

    over = [i for i, n in enumerate(lengths) if n > args.max_seq]
    print(f"{len(over):,} of {len(rows):,} exceed max-seq {args.max_seq} "
          f"({len(over) / len(rows):.2%})")

    if not over:
        print("nothing to drop")
        return

    if not args.drop:
        # Truncating mid-answer teaches the model to emit unfinished responses,
        # which is a worse failure than losing a few long examples.
        raise SystemExit(
            f"\nSTOP. {len(over):,} examples would be truncated mid-answer.\n"
            f"FIX: re-run with --drop, or raise --max-seq to {stats['max']}.")

    keep = [row for i, row in enumerate(rows) if i not in set(over)]
    with args.data.open("w") as fh:
        for row in keep:
            fh.write(json.dumps(row) + "\n")
    print(f"dropped {len(over):,}; {len(keep):,} remain in {args.data}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_check_lengths.py -q`
Expected: PASS, 4 passed.

- [ ] **Step 5: Measure the real distribution**

Uses the tiny set from Task 3. Downloads a tokenizer on first run.

```bash
.venv/bin/python -m training.check_lengths --data /tmp/medprep/train.jsonl
```

Expected: a `tokens n=90 median=... p99=... max=...` line, then a suggested
`--max-seq`. Record the suggested value — Task 8 feeds it to the notebook.

- [ ] **Step 6: Commit**

```bash
git add training/check_lengths.py tests/test_check_lengths.py
git commit -m "Measure sequence length instead of assuming it

Clinical vignettes plus chain-of-thought run far longer than the 832
tokens the sibling text2sql project measured, and a guessed max_seq is
either wasted memory or a silently truncated answer. This reports the
real distribution and refuses to drop anything without --drop being
asked for explicitly."
```

---

### Task 5: Scoring — accuracy, McNemar, and the paired report

**Files:**
- Create: `training/evalcore.py`
- Create: `tests/test_evalcore.py`

**Interfaces:**
- Consumes: `prompts.extract_letter`, `records.Record`.
- Produces:
  - `evalcore.accuracy(preds: list[str | None], golds: list[str]) -> float`
  - `evalcore.mcnemar_exact(base: list[bool], tuned: list[bool]) -> dict` with
    keys `wins`, `regressions`, `both_correct`, `both_wrong`, `p_value`
  - `evalcore.by_subject(recs, base, tuned) -> dict[str, dict]`
  - `evalcore.paired_report(recs, base_preds, tuned_preds, *, mode: str) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evalcore.py`:

```python
from __future__ import annotations

from training.evalcore import accuracy, by_subject, mcnemar_exact, paired_report
from training.records import Record


def mk(rid, answer, subject):
    return Record(id=rid, source="medmcqa", kind="mcq", question="q",
                  options={"A": "a", "B": "b", "C": "c", "D": "d"},
                  answer=answer, rationale=None, response=None, subject=subject)


def test_accuracy_counts_an_unparseable_answer_as_wrong():
    assert accuracy(["A", None, "B"], ["A", "B", "B"]) == 2 / 3


def test_accuracy_of_an_empty_set_is_zero_not_a_crash():
    assert accuracy([], []) == 0.0


def test_mcnemar_counts_the_four_cells():
    base = [True, True, False, False]
    tuned = [True, False, True, False]
    r = mcnemar_exact(base, tuned)
    assert r["both_correct"] == 1
    assert r["regressions"] == 1   # base right, tuned wrong
    assert r["wins"] == 1          # base wrong, tuned right
    assert r["both_wrong"] == 1


def test_mcnemar_returns_p_one_when_wins_and_regressions_balance():
    r = mcnemar_exact([True, False], [False, True])
    assert r["p_value"] == 1.0


def test_mcnemar_is_significant_when_every_discordant_pair_is_a_win():
    base = [False] * 10
    tuned = [True] * 10
    r = mcnemar_exact(base, tuned)
    assert r["wins"] == 10 and r["regressions"] == 0
    assert r["p_value"] < 0.01


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    r = mcnemar_exact([True, True], [True, True])
    assert r["p_value"] == 1.0


def test_by_subject_breaks_accuracy_out_per_subject():
    recs = [mk("1", "A", "Anatomy"), mk("2", "B", "Anatomy"), mk("3", "C", "Pharmacology")]
    out = by_subject(recs, [True, False, True], [True, True, False])
    assert out["Anatomy"]["n"] == 2
    assert out["Anatomy"]["base"] == 0.5
    assert out["Anatomy"]["tuned"] == 1.0
    assert out["Pharmacology"]["n"] == 1


def test_paired_report_has_everything_the_writeup_quotes():
    recs = [mk("1", "A", "Anatomy"), mk("2", "B", "Anatomy")]
    rep = paired_report(recs, ["A", "A"], ["A", "B"], mode="constrained")
    assert rep["mode"] == "constrained"
    assert rep["n"] == 2
    assert rep["base_accuracy"] == 0.5
    assert rep["tuned_accuracy"] == 1.0
    assert rep["mcnemar"]["wins"] == 1
    assert "Anatomy" in rep["by_subject"]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_evalcore.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.evalcore'`

- [ ] **Step 3: Write `training/evalcore.py`**

```python
from __future__ import annotations

from math import comb

from training.records import Record


def accuracy(preds: list[str | None], golds: list[str]) -> float:
    """An unparseable prediction is wrong, not excluded.

    Dropping them would flatter whichever model rambles more, which is
    usually the base model.
    """
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def mcnemar_exact(base: list[bool], tuned: list[bool]) -> dict:
    """Two-sided exact McNemar on paired outcomes. Standard library only."""
    both_correct = sum(b and t for b, t in zip(base, tuned))
    both_wrong = sum((not b) and (not t) for b, t in zip(base, tuned))
    regressions = sum(b and not t for b, t in zip(base, tuned))
    wins = sum((not b) and t for b, t in zip(base, tuned))

    n = wins + regressions
    if n == 0:
        p = 1.0
    else:
        k = min(wins, regressions)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)

    return {
        "wins": wins, "regressions": regressions,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "discordant": n, "p_value": round(p, 6),
    }


def by_subject(recs: list[Record], base: list[bool], tuned: list[bool]) -> dict:
    buckets: dict[str, dict] = {}
    for rec, b, t in zip(recs, base, tuned):
        key = rec.subject or "unknown"
        bucket = buckets.setdefault(key, {"n": 0, "base_n": 0, "tuned_n": 0})
        bucket["n"] += 1
        bucket["base_n"] += int(b)
        bucket["tuned_n"] += int(t)
    for bucket in buckets.values():
        bucket["base"] = bucket.pop("base_n") / bucket["n"]
        bucket["tuned"] = bucket.pop("tuned_n") / bucket["n"]
    return dict(sorted(buckets.items(), key=lambda kv: -kv[1]["n"]))


def paired_report(recs: list[Record], base_preds: list[str | None],
                  tuned_preds: list[str | None], *, mode: str) -> dict:
    golds = [r.answer or "" for r in recs]
    base_ok = [p == g for p, g in zip(base_preds, golds)]
    tuned_ok = [p == g for p, g in zip(tuned_preds, golds)]
    return {
        "mode": mode,
        "n": len(recs),
        "base_accuracy": accuracy(base_preds, golds),
        "tuned_accuracy": accuracy(tuned_preds, golds),
        "base_unparseable": sum(p is None for p in base_preds),
        "tuned_unparseable": sum(p is None for p in tuned_preds),
        "mcnemar": mcnemar_exact(base_ok, tuned_ok),
        "by_subject": by_subject(recs, base_ok, tuned_ok),
    }
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_evalcore.py -q`
Expected: PASS, 8 passed.

- [ ] **Step 5: Sanity-check McNemar against a known value**

```bash
.venv/bin/python -c "
from training.evalcore import mcnemar_exact
# 12 discordant pairs, 10 wins and 2 regressions.
r = mcnemar_exact([False]*10 + [True]*2, [True]*10 + [False]*2)
print(r)
assert r['wins']==10 and r['regressions']==2
assert abs(r['p_value'] - 0.038574) < 1e-5, r['p_value']
print('exact binomial p matches the closed form')
"
```

Expected: the dict, then the confirmation line.

- [ ] **Step 6: Commit**

```bash
git add training/evalcore.py tests/test_evalcore.py
git commit -m "Score base against tuned in one shared module

Both entry points score through here, so a notebook run and a local run
cannot report subtly different numbers. An unparseable prediction counts
as wrong rather than being excluded: dropping them flatters whichever
model rambles more, which is the base model.

McNemar is the exact binomial rather than the chi-square approximation,
computed with math.comb so the dependency list stays at nothing."
```

---

### Task 6: The evaluation CLI — constrained and generative scoring

**Files:**
- Create: `training/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `evalcore.paired_report`, `prompts.build_messages`, `prompts.extract_letter`.
- Produces:
  - `evaluate.letter_token_ids(tok) -> dict[str, int]`
  - `evaluate.pick_from_logits(logits_row, letter_ids: dict[str, int]) -> str`
  - CLI writing `eval_report.json`

`pick_from_logits` is separated out precisely so it can be tested without a GPU,
using a plain list of floats.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
from __future__ import annotations

from training.evaluate import pick_from_logits


def test_pick_from_logits_returns_the_highest_scoring_letter():
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert pick_from_logits([0.1, 0.2, 5.0, 0.3], ids) == "C"


def test_pick_from_logits_only_considers_the_four_letters():
    # index 7 is the global maximum but is not a letter token
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    logits = [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0]
    assert pick_from_logits(logits, ids) == "B"


def test_pick_from_logits_breaks_ties_deterministically():
    ids = {"A": 0, "B": 1, "C": 2, "D": 3}
    assert pick_from_logits([1.0, 1.0, 1.0, 1.0], ids) == "A"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.evaluate'`

- [ ] **Step 3: Write `training/evaluate.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.evalcore import paired_report
from training.prompts import LETTERS, build_messages, extract_letter
from training.records import Record

BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"


def letter_token_ids(tok) -> dict[str, int]:
    """Token id for each answer letter as it appears after 'Answer: '.

    The leading space matters: most BPE tokenizers give " A" and "A"
    different ids, and scoring the wrong one silently measures noise.
    """
    ids = {}
    for letter in LETTERS:
        encoded = tok.encode(f" {letter}", add_special_tokens=False)
        ids[letter] = encoded[-1]
    return ids


def pick_from_logits(logits_row, letter_ids: dict[str, int]) -> str:
    """Argmax restricted to the four answer letters. Ties go to the earliest."""
    best, best_score = None, None
    for letter in LETTERS:
        score = float(logits_row[letter_ids[letter]])
        if best_score is None or score > best_score:
            best, best_score = letter, score
    return best


def _load(adapter: str | None, max_seq: int):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter or BASE_MODEL,
        max_seq_length=max_seq, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def score_constrained(model, tok, recs: list[Record]) -> list[str]:
    import torch

    ids = letter_token_ids(tok)
    out: list[str] = []
    for rec in recs:
        msgs = build_messages(rec, with_answer=False)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True) + "Answer:"
        batch = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[0, -1]
        out.append(pick_from_logits(logits, ids))
    return out


def score_generative(model, tok, recs: list[Record], max_new_tokens: int = 320
                     ) -> list[str | None]:
    import torch

    out: list[str | None] = []
    for rec in recs:
        msgs = build_messages(rec, with_answer=False)
        batch = tok(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(gen[0][batch["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        out.append(extract_letter(completion))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Score base against tuned, paired.")
    p.add_argument("--adapter", required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seq", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gen-limit", type=int, default=300,
                   help="how many examples also get generative scoring")
    args = p.parse_args()

    recs = [Record.from_dict(json.loads(line))
            for line in args.test.read_text().splitlines() if line]
    recs = [r for r in recs if r.kind == "mcq"]
    if args.limit:
        recs = recs[: args.limit]
    gen_recs = recs[: args.gen_limit]
    print(f"scoring {len(recs):,} constrained, {len(gen_recs):,} generative")

    reports = {}
    preds: dict[str, dict] = {}
    for tag, adapter in (("base", None), ("tuned", args.adapter)):
        model, tok = _load(adapter, args.max_seq)
        print(f"  {tag}: constrained...", flush=True)
        preds.setdefault("constrained", {})[tag] = score_constrained(model, tok, recs)
        print(f"  {tag}: generative...", flush=True)
        preds.setdefault("generative", {})[tag] = score_generative(model, tok, gen_recs)
        del model

    reports["constrained"] = paired_report(
        recs, preds["constrained"]["base"], preds["constrained"]["tuned"],
        mode="constrained")
    reports["generative"] = paired_report(
        gen_recs, preds["generative"]["base"], preds["generative"]["tuned"],
        mode="generative")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"test_set": str(args.test),
                                    "reports": reports}, indent=2))

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
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -q`
Expected: PASS, 3 passed.

- [ ] **Step 5: Confirm the whole suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 78 passed. No test requires a GPU or network.

- [ ] **Step 6: Commit**

```bash
git add training/evaluate.py tests/test_evaluate.py
git commit -m "Score two ways, because they can disagree

Constrained scoring compares the logits of the four letter tokens in one
forward pass. It is deterministic and cheap enough for the full 1,273
and 4,183, so it carries the headline. But it cannot see a model that
knows the answer and cannot state it, and that failure is exactly what
would break the chatbot. Greedy generation on a 300-example subsample
catches it, and both go in the report.

The letter ids encode with a leading space: ' A' and 'A' are different
tokens, and scoring the wrong one measures noise."
```

---

### Task 7: Training, with a probe that aborts a run that will not fit

**Files:**
- Create: `training/train.py`
- Create: `training/budget.py`
- Create: `tests/test_budget.py`

The projection arithmetic lives in its own module so it can be tested without
Unsloth, a GPU, or a six-hour wait.

**Interfaces:**
- Consumes: `records.Record`, `prompts.build_messages`.
- Produces:
  - `budget.project(steps_done: int, seconds_elapsed: int, total_steps: int) -> dict`
    with keys `seconds_per_step`, `projected_seconds`, `remaining_seconds`
  - `budget.check(projection: dict, budget_seconds: int) -> str | None`
    returning `None` if it fits, or the abort message if it does not

- [ ] **Step 1: Write the failing test**

Create `tests/test_budget.py`:

```python
from __future__ import annotations

from training.budget import check, project


def test_project_extrapolates_linearly_from_observed_steps():
    p = project(steps_done=50, seconds_elapsed=100, total_steps=500)
    assert p["seconds_per_step"] == 2.0
    assert p["projected_seconds"] == 1000
    assert p["remaining_seconds"] == 900


def test_project_refuses_to_divide_by_zero_steps():
    p = project(steps_done=0, seconds_elapsed=10, total_steps=100)
    assert p["seconds_per_step"] == 0.0
    assert p["projected_seconds"] == 0


def test_check_passes_a_run_that_fits():
    assert check(project(50, 100, 500), budget_seconds=21600) is None


def test_check_aborts_a_run_that_overruns_and_names_the_fix():
    msg = check(project(50, 2000, 500), budget_seconds=21600)
    assert msg is not None
    assert "batch" in msg.lower()
    assert "5h" in msg or "hours" in msg.lower()


def test_check_is_a_no_op_before_any_step_has_run():
    assert check(project(0, 0, 500), budget_seconds=21600) is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_budget.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.budget'`

- [ ] **Step 3: Write `training/budget.py`**

```python
from __future__ import annotations


def project(steps_done: int, seconds_elapsed: int, total_steps: int) -> dict:
    """Extrapolate total training time from the steps observed so far."""
    if steps_done <= 0:
        return {"seconds_per_step": 0.0, "projected_seconds": 0,
                "remaining_seconds": 0}
    per_step = seconds_elapsed / steps_done
    projected = int(per_step * total_steps)
    return {
        "seconds_per_step": per_step,
        "projected_seconds": projected,
        "remaining_seconds": int(per_step * (total_steps - steps_done)),
    }


def check(projection: dict, budget_seconds: int) -> str | None:
    """Return an abort message if the run will not fit, else None.

    The sibling text2sql notebook asks the operator to watch it/s and
    multiply out before walking away. Nobody does. Failing loudly at
    step 50 costs a minute; discovering it at hour six costs the day
    and a chunk of the weekly GPU quota.
    """
    projected = projection["projected_seconds"]
    if projected <= 0 or projected <= budget_seconds:
        return None
    return (
        f"\nSTOP. Projected training time is {projected / 3600:.1f}h "
        f"against a budget of {budget_seconds / 3600:.1f}h "
        f"({projection['seconds_per_step']:.2f}s/step).\n"
        f"FIX: halve --batch-size and double --grad-accum, which keeps the "
        f"effective batch identical, or cut the training set with the "
        f"--medmcqa / --chatdoctor flags on prepare_data.py."
    )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_budget.py -q`
Expected: PASS, 5 passed.

- [ ] **Step 5: Write `training/train.py`**

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from training.budget import check, project
from training.prompts import build_messages
from training.records import Record

BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

# Qwen3's chat template renders the assistant turn after this marker. Loss is
# masked to everything after it, so the model is never scored on reproducing
# the clinical vignette it was given.
RESPONSE_MARKER = "<|im_start|>assistant\n"
INSTRUCTION_MARKER = "<|im_start|>user\n"


def build_dataset(path: Path, tok):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    texts = [
        tok.apply_chat_template(
            build_messages(Record.from_dict(row), with_answer=True), tokenize=False)
        for row in rows
    ]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    p = argparse.ArgumentParser(description="QLoRA fine-tune on the medical mix.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seq", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--budget-seconds", type=int, default=21600)
    p.add_argument("--probe-steps", type=int, default=50)
    args = p.parse_args()

    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=args.max_seq,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    train_ds = build_dataset(args.data / "train.jsonl", tok)
    val_ds = build_dataset(args.data / "val.jsonl", tok)
    print(f"train {len(train_ds):,}   val {len(val_ds):,}   max_seq {args.max_seq}")

    class Probe(TrainerCallback):
        """Abort at --probe-steps if the run will not fit the budget."""

        def __init__(self) -> None:
            self.started = time.time()

        def on_step_end(self, cfg, state, control, **kw):
            if state.global_step != args.probe_steps:
                return
            proj = project(state.global_step, int(time.time() - self.started),
                           int(state.max_steps))
            print(f"\nprobe: {proj['seconds_per_step']:.2f}s/step, "
                  f"projected {proj['projected_seconds'] / 3600:.2f}h "
                  f"for {int(state.max_steps):,} steps", flush=True)
            message = check(proj, args.budget_seconds)
            if message:
                raise SystemExit(message)
            print("projection fits the budget; continuing\n", flush=True)

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=42,
            output_dir=str(args.out),
            report_to="none",
        ),
        callbacks=[Probe()],
    )

    # Mask the prompt so gradient is spent only on the response.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_MARKER, response_part=RESPONSE_MARKER)

    resume = any(args.out.glob("checkpoint-*")) if args.out.exists() else False
    if resume:
        print(f"resuming from a checkpoint in {args.out}")
    stats = trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "train_stats.json").write_text(json.dumps({
        "train_runtime_seconds": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
        "examples": len(train_ds),
        "max_seq": args.max_seq,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
    }, indent=2))
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the module imports cleanly without a GPU**

Only the argument parser and the pure helpers are exercised; Unsloth is imported
inside `main`, so this must not fail on a laptop.

```bash
.venv/bin/python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('training/train.py').read_text())
top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
names = {a.name.split('.')[0] for n in top for a in n.names}
assert not ({'torch','unsloth','trl','transformers'} & names), names
print('no GPU-only import at module scope:', sorted(names))
"
.venv/bin/python -m pytest -q
```

Expected: the confirmation line, then PASS, 83 passed.

- [ ] **Step 7: Commit**

```bash
git add training/train.py training/budget.py tests/test_budget.py
git commit -m "Abort at step 50 a run that will not fit the budget

The sibling notebook tells the operator to watch it/s and multiply out
before walking away. Nobody does, and this run goes straight to full
scale with no calibration pass, so the arithmetic is automatic and
fatal: at step 50 it extrapolates and raises SystemExit naming the fix.
Failing there costs a minute. Discovering it at hour six costs the day
and a chunk of the weekly GPU quota.

Loss is masked to the response so gradient is not spent learning to
generate clinical vignettes, and checkpoints land every 200 steps so a
dead session resumes instead of restarting.

The projection lives in its own module and is unit-tested, because a
guard nobody can test is a guard nobody should trust."
```

---

### Task 8: The Kaggle notebook and the push script

**Files:**
- Create: `training/kaggle_medical.ipynb`
- Create: `training/kernel-metadata.json`
- Create: `training/requirements.txt`
- Create: `scripts/push_kaggle.sh`

**Interfaces:**
- Consumes: every CLI from Tasks 3-7.
- Produces: a Kaggle kernel at `gb1105/qwen3-4b-medical-fine-tune`.

- [ ] **Step 1: Write `scripts/push_kaggle.sh`**

```bash
#!/usr/bin/env bash
# Upload the training code as a private Kaggle Dataset, then push the notebook.
#
#   bash scripts/push_kaggle.sh
#
# Credentials come from ~/.kaggle/kaggle.json, already present for gb1105.

set -euo pipefail
cd "$(dirname "$0")/.."

KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
SLUG="medical-ft-code"
USER="$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"

command -v "$KAGGLE" >/dev/null || { echo "kaggle CLI not found at $KAGGLE"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/training"
cp training/*.py "$STAGE/training/"
cp training/requirements.txt "$STAGE/training/"

cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "Medical Fine-tune Code",
  "id": "$USER/$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

if "$KAGGLE" datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "==> updating dataset $USER/$SLUG"
  "$KAGGLE" datasets version -p "$STAGE" -m "code update $(date -u +%FT%TZ)" --dir-mode zip
else
  echo "==> creating dataset $USER/$SLUG"
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode zip
fi

echo "==> pushing the notebook"
"$KAGGLE" kernels push -p training

echo
echo "Open it, set Accelerator to GPU T4 x2 and Internet On, then Save & Run All:"
echo "  https://www.kaggle.com/code/$USER/qwen3-4b-medical-fine-tune"
```

- [ ] **Step 2: Write `training/kernel-metadata.json`**

Replace `gb1105` only if `~/.kaggle/kaggle.json` names a different user.

```json
{
  "id": "gb1105/qwen3-4b-medical-fine-tune",
  "title": "Qwen3-4B Medical Fine-tune",
  "code_file": "kaggle_medical.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["gb1105/medical-ft-code"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- [ ] **Step 3: Write `training/requirements.txt`**

```
unsloth
unsloth_zoo
trl
peft
accelerate
bitsandbytes
datasets
```

- [ ] **Step 4: Build the notebook with a script**

Writing `.ipynb` JSON by hand invites a malformed cell that Kaggle rejects only
after upload. Create `scripts/build_notebook.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Qwen3-4B medical fine-tune

~40,000 examples from four medical sources, scored against the base model on
two held-out benchmarks it never saw: MedQA-USMLE test (1,273) and MedMCQA
validation (4,183).

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
    ("markdown", """## 7. Evaluate

Both benchmarks, both scoring modes, paired against the base model on identical
prompts with identical decoding."""),
    ("code", '''# Write the held-out sets exactly as decontamination saw them. MedMCQA loads
# with require_rationale=False: the benchmark needs question, options and
# answer, and filtering it by explanation-availability would drop a quarter of
# it and quietly change what the score means.
import json
from training.sources import load

for name, split, unfiltered, out in (
        ("medqa", "test", False, "data/holdout_medqa.jsonl"),
        ("medmcqa", "validation", True, "data/holdout_medmcqa.jsonl")):
    recs = load(name, limit=0, split=split, require_rationale=not unfiltered)
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r.to_dict()) + "\\n")
    print(f"{out}: {len(recs):,}")'''),
    ("code", '''step(f"python -m training.evaluate --adapter outputs/run1 "
     f"--test data/holdout_medqa.jsonl --max-seq {MAX_SEQ} --gen-limit 300 "
     f"--out outputs/eval_medqa.json")
step(f"python -m training.evaluate --adapter outputs/run1 "
     f"--test data/holdout_medmcqa.jsonl --max-seq {MAX_SEQ} --gen-limit 300 "
     f"--out outputs/eval_medmcqa.json")'''),
    ("code", '''# --- 8. Results, and save everything to the Output panel -------------------
import json, shutil
from pathlib import Path

out = Path("/kaggle/working")
for label, path in (("MedQA-USMLE test", "outputs/eval_medqa.json"),
                    ("MedMCQA validation", "outputs/eval_medmcqa.json")):
    data = json.loads(Path(path).read_text())
    print(f"\\n########## {label} ##########")
    for mode, rep in data["reports"].items():
        m = rep["mcnemar"]
        delta = (rep["tuned_accuracy"] - rep["base_accuracy"]) * 100
        print(f"\\n  --- {mode}  n={rep['n']:,} ---")
        print(f"  base  {rep['base_accuracy']:>7.1%}")
        print(f"  tuned {rep['tuned_accuracy']:>7.1%}   ({delta:+.1f} points)")
        print(f"  wins {m['wins']}  regressions {m['regressions']}  "
              f"p = {m['p_value']:.4f}")
        if m["p_value"] >= 0.05:
            print("  NOT significant at p<0.05 -- indistinguishable from base.")
    shutil.copy(path, out / Path(path).name)

shutil.copy("data/report.json", out / "data_report.json")
shutil.make_archive(str(out / "run1-adapter"), "zip", "outputs/run1")
print("\\nSaved to /kaggle/working -- download from the Output panel.")'''),
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
```

- [ ] **Step 5: Build the notebook and validate it parses**

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/python -c "
import json
nb = json.load(open('training/kaggle_medical.ipynb'))
assert nb['nbformat'] == 4
kinds = [c['cell_type'] for c in nb['cells']]
assert set(kinds) <= {'code','markdown'}, kinds
for c in nb['cells']:
    if c['cell_type'] == 'code':
        compile(''.join(c['source']).replace('%%capture','').replace('!','#'),
                '<cell>', 'exec')
print(f'{len(nb[\"cells\"])} cells, every code cell compiles')
"
```

Expected: `wrote training/kaggle_medical.ipynb, 15 cells` then
`15 cells, every code cell compiles`.

- [ ] **Step 6: Push to Kaggle**

```bash
chmod +x scripts/push_kaggle.sh
bash scripts/push_kaggle.sh
```

Expected: dataset created, kernel pushed, and a URL printed. Confirm with
`~/.local/bin/kaggle kernels list --mine | head -3`.

- [ ] **Step 7: Commit**

```bash
git add training/kaggle_medical.ipynb training/kernel-metadata.json \
        training/requirements.txt scripts/
git commit -m "Ship the run to Kaggle as a dataset plus a notebook

The code uploads as a private Kaggle Dataset rather than being cloned
from GitHub, which keeps this project independent of the text2sql repo.

The notebook is generated by a script and every code cell is compiled
before upload. Handwriting .ipynb JSON invites a malformed cell that
Kaggle only rejects after the upload, and after a queue wait.

step() raises on a non-zero exit: a bare !python returns non-zero
without raising in Jupyter, so the notebook would sail past a dead step
and fail later somewhere confusing."
```

---

## Self-Review

**Spec coverage.**

| spec section | task |
|---|---|
| 1 Data — four sources, counts as flags | 2, 3 |
| 1 Data — measured quality filters | 2 |
| 1 Data — decontamination with reported count | 3 |
| 1 Data — one prompt format | 1 |
| 2 Training — completion-only loss | 7 |
| 2 Training — measured `max_seq`, drop not truncate | 4 |
| 2 Training — probe aborts, 200-step checkpoints | 7 |
| 3 Eval — two held-out sets | 6, 8 |
| 3 Eval — constrained and generative | 6 |
| 3 Eval — McNemar, per-subject, shared module | 5 |
| 6 Kaggle — dataset upload, notebook, T4 guard | 8 |

Sections 4 (serving/safety), 5 (frontend) and the qualitative chat set are
**deliberately out of scope** — they are plans 2 and 3.

**Placeholder scan.** No TBD/TODO. Every code step carries runnable code; every
test step carries real assertions; no step says "similar to Task N".

**Type consistency.** `Record` field names are identical in Tasks 1-7.
`extract_letter` is defined in Task 1 and used in Task 6. `LETTERS` is defined in
`prompts` and imported by `evaluate`; `sources` defines its own local copy on
purpose, so it does not import `prompts` for a four-element tuple.
`build_messages(rec, *, with_answer)` keeps the same signature in Tasks 1, 4, 6, 7.
`project()` / `check()` signatures match between Task 7's tests and its callback.

**Two bugs the review caught, now fixed in place.**

1. *The benchmark was being filtered by explanation-availability.* `load()`
   applied the training filters to MedMCQA validation too, which would have
   dropped roughly a quarter of the 4,183 and made the reported score
   incomparable to any published MedMCQA number. Evaluation needs question,
   options and answer — not an explanation. `require_rationale` now defaults to
   `True` for training and is passed `False` for the holdout.
2. *A "no-op guard" in the notebook was destructive.* A `prepare_data` call with
   every count set to zero would have rewritten `data/train.jsonl` as an empty
   file and then divided by zero computing the decontamination rate. Removed.

The regex, percentile and McNemar implementations were executed against the
assertions in Tasks 1, 4 and 5 before this plan was committed: `extract_letter`
passes all 13 cases, `percentile` all 5, and McNemar returns exactly 0.038574
for 10 wins against 2 regressions.

**One known risk left open.** `RESPONSE_MARKER` in Task 7 assumes Qwen3's chat
template emits `<|im_start|>assistant\n`. If `train_on_responses_only` raises or
silently masks nothing, print one rendered example and correct the marker before
launching the full run. Task 7 Step 5 is where this surfaces.
