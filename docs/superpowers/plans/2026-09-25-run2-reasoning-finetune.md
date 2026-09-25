# Run 2 Reasoning Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen3-4B again, in its own thinking format, on ~11,000 decontaminated examples from eight licensed medical sources, and evaluate it against the base model on four benchmarks in two answer modes, all in one hands-off 12-hour Kaggle session.

**Architecture:** Data is prepared and decontaminated on this Mac (`training/prepare_data_v2.py`) and uploaded frozen as a Kaggle dataset with a content fingerprint. One Kaggle notebook (`run2/kaggle_run2.ipynb`) trains with Unsloth on GPU 0 under a time guard, saves the adapter, then runs two evaluation workers in parallel (fine-tune on GPU 0, base on GPU 1) that score stage by stage until a deadline, and a report step pairs them. Run 1's modules stay as they are; run 2 adds new modules and opt-in flags.

**Tech Stack:** Python 3.12, Unsloth 2026.9.7 + TRL 0.24.0 + transformers 5.5.0 + peft 0.19.1 (Kaggle), datasets, pytest, Kaggle CLI 2.2.4.

**Spec:** `docs/superpowers/specs/2026-09-25-run2-reasoning-finetune-design.md`

## Global Constraints

- Kaggle T4 (compute 7.5: fp16 only, no FlashAttention 2), a 12-hour session, about 30 GPU-hours a week. One push; the session must finish without the laptop.
- Same base model, Qwen3-4B: `unsloth/Qwen3-4B-unsloth-bnb-4bit` for training, `unsloth/Qwen3-4B` (fp16) for evaluation.
- Datasets with an explicit Apache-2.0 or MIT licence only.
- Pinned Kaggle stack: `unsloth==2026.9.7`, `unsloth_zoo==2026.9.6`, `transformers==5.5.0`, `trl==0.24.0`, `peft==0.19.1`.
- Every Python module starts with `from __future__ import annotations`. No `torch`, `transformers`, `peft`, `trl`, `unsloth` or `datasets` import at module scope in any file under test.
- Run 1 stays reproducible: `evaluate.py`, `prepare_data.py`, `sources.py`, `prompts.py` keep their behaviour; `train.py` keeps `--format v1` as its default.
- Commits name garvbahl37-gif as sole contributor, with no `Co-Authored-By` trailer and no "Generated with Claude Code" line.
- At most two agents at a time.

## Review Focus

1. An option block with a gap or a repeated letter (A, B, D) must drop the row, never re-letter it. Pinned in Task 2 (`test_parse_option_lines_needs_a_run_from_a`).
2. A reasoning completion that never closes `</think>` inside the budget must still get a letter, and only one of that question's own options (PubMedQA: A–C). Pinned in Task 1 (`test_final_letter_...`, `test_forced_suffix_...`) and in the Task 5 smoke run (budget 24, PubMedQA letters checked).
3. A worker cut off mid-stage, by the deadline or a crash, must leave a report that pairs only the questions both models scored and states planned against scored. Pinned in Task 5 (`test_build_report_pairs_only_questions_both_models_scored`).
4. A training question sharing 13 consecutive words with a benchmark question must be dropped. Pinned in Task 3 (`test_thirteen_word_overlap_...`).
5. Kaggle mounting an older version of the data must stop the notebook before training. Pinned in Task 6 (`test_data_fingerprint_changes_when_one_file_changes`) and the fetch cell's fingerprint check.

## File map

| File | Status | Responsibility |
|---|---|---|
| `training/records.py` | modify | `Record.reasoning` (the think-block text), default None |
| `training/format_v2.py` | create | run 2 prompts: messages, rendering for training, letter and reasoning prompts, answer extraction |
| `training/sources_v2.py` | create | option parsing, answer matching, one normaliser per new source, benchmark normalisers, benchmark sizes |
| `training/decontam.py` | create | exact + 13-word-overlap index of benchmark questions; cross-source deduper |
| `training/prepare_data_v2.py` | create | the mix, sampling, filters, lengths, `train.jsonl`, `eval_*.jsonl`, `data_report.json` |
| `training/train.py` | modify | `--format v2`, `--group-by-length`, `--stop-after-seconds`, `--no-probe-abort`, `loss_curve.json` |
| `training/evalcore.py` | modify | `paired_diff_ci`, added to `paired_report` |
| `training/eval_worker.py` | create | one model, stage by stage until a deadline, predictions appended per batch |
| `training/eval_report.py` | create | pair the two workers, per stage and pooled, write JSON, print table |
| `training/kaggle_paths.py` | modify | `RUN2_REQUIRED`, `DATA_FILES`, `find_data_dir`, `data_fingerprint` |
| `scripts/build_notebook.py` | modify | `run2/kaggle_run2.ipynb` |
| `run2/kernel-metadata.json` | create | the run 2 kernel |
| `scripts/push_data.sh` | create | upload `data/v2` as `medical-ft-data` |
| `scripts/smoke_eval_v2.py` | create | local end-to-end check on Qwen3-0.6B |
| `.gitignore` | modify | ignore `/data/` |
| tests | create/modify | one test file per new module; additions to `test_evalcore.py`, `test_kaggle_paths.py`, `test_kernel_metadata.py` |

---

### Task 1: The run 2 format

**Files:**
- Modify: `training/records.py`
- Create: `training/format_v2.py`
- Test: `tests/test_format_v2.py`

**Interfaces:**
- Consumes: `training.prompts.SYSTEM_CHAT`, `training.prompts.extract_letter`.
- Produces: `Record.reasoning: str | None = None`; `SYSTEM_MCQ_V2: str`; `option_letters(rec) -> list[str]`; `format_question_v2(rec) -> str`; `short_explanation(text, *, max_sentences=2, max_chars=400) -> str`; `answer_text(rec) -> str`; `v2_messages(rec, *, with_answer: bool) -> list[dict]`; `render_training_text(tok, rec) -> str`; `render_letter_prompt(tok, rec) -> str`; `render_reasoning_prompt(tok, rec) -> str`; `final_letter(final_text, letters) -> str | None`; `forced_suffix(closed: bool) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_format_v2.py`:

```python
from __future__ import annotations

from training.format_v2 import (SYSTEM_MCQ_V2, answer_text, final_letter,
                                forced_suffix, format_question_v2, option_letters,
                                render_letter_prompt, render_reasoning_prompt,
                                render_training_text, short_explanation, v2_messages)
from training.prompts import SYSTEM_CHAT
from training.records import Record


def mcq(**kw) -> Record:
    fields = dict(id="q1", source="medqa", kind="mcq", question="Which drug?",
                  options={"A": "Aspirin", "B": "Heparin", "C": "Warfarin"},
                  answer="B", rationale=None, response=None, subject=None,
                  reasoning=None)
    fields.update(kw)
    return Record(**fields)


def chat(**kw) -> Record:
    fields = dict(id="c1", source="medical_o1", kind="dialogue", question="Why?",
                  options=None, answer=None, rationale=None, response="Because.",
                  subject=None, reasoning=None)
    fields.update(kw)
    return Record(**fields)


class FakeTok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        body = "|".join(f"{m['role']}:{m.get('reasoning_content', '')}:{m['content']}"
                        for m in messages)
        return body + ("|GEN" if kw.get("add_generation_prompt") else "")


def test_record_without_a_reasoning_key_still_loads():
    row = mcq().to_dict()
    row.pop("reasoning")
    assert Record.from_dict(row).reasoning is None


def test_option_letters_are_sorted_and_empty_for_dialogue():
    assert option_letters(mcq(options={"C": "c", "A": "a", "B": "b"})) == ["A", "B", "C"]
    assert option_letters(chat()) == []


def test_format_question_lists_every_option_in_order():
    assert format_question_v2(mcq()) == "Which drug?\n\nA. Aspirin\nB. Heparin\nC. Warfarin"
    assert format_question_v2(chat()) == "Why?"


def test_short_explanation_strips_the_label_and_keeps_two_sentences():
    assert short_explanation("Explanation: One. Two! Three? Four.") == "One. Two!"


def test_short_explanation_caps_length_on_a_word_boundary():
    out = short_explanation("word " * 200, max_chars=50)
    assert len(out) <= 53 and out.endswith("...") and not out.startswith(" ")


def test_answer_text_puts_the_letter_first():
    assert (answer_text(mcq(rationale="Heparin acts fast. It is IV. Extra."))
            == "Answer: B\n\nHeparin acts fast. It is IV.")
    assert answer_text(mcq()) == "Answer: B. Heparin"
    assert answer_text(chat()) == "Because."


def test_v2_messages_carry_reasoning_as_reasoning_content():
    msgs = v2_messages(mcq(reasoning="  think  "), with_answer=True)
    assert msgs[0] == {"role": "system", "content": SYSTEM_MCQ_V2}
    assert msgs[-1]["reasoning_content"] == "think"
    assert "reasoning_content" not in v2_messages(mcq(), with_answer=True)[-1]
    assert v2_messages(chat(), with_answer=False)[0]["content"] == SYSTEM_CHAT
    assert len(v2_messages(chat(), with_answer=False)) == 2


def test_training_text_thinks_only_when_there_is_reasoning():
    tok = FakeTok()
    render_training_text(tok, mcq(reasoning="r"))
    render_training_text(tok, mcq())
    assert [kw["enable_thinking"] for _, kw in tok.calls] == [True, False]
    assert all(kw["tokenize"] is False for _, kw in tok.calls)


def test_letter_prompt_is_non_thinking_and_ends_with_answer_colon():
    tok = FakeTok()
    text = render_letter_prompt(tok, mcq())
    kw = tok.calls[0][1]
    assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
    assert text.endswith("|GENAnswer:")


def test_reasoning_prompt_turns_thinking_on():
    tok = FakeTok()
    render_reasoning_prompt(tok, mcq())
    assert tok.calls[0][1]["enable_thinking"] is True
    assert tok.calls[0][1]["add_generation_prompt"] is True


def test_final_letter_prefers_the_answer_line_and_respects_options():
    assert final_letter("Answer: C\n\nbecause", ["A", "B", "C"]) == "C"
    assert final_letter("Answer: D", ["A", "B", "C"]) is None
    assert final_letter("I think the correct answer is B.", ["A", "B", "C"]) == "B"
    assert final_letter("", ["A", "B"]) is None


def test_forced_suffix_closes_an_open_think_block():
    assert forced_suffix(closed=False) == "\n</think>\n\nAnswer:"
    assert forced_suffix(closed=True) == "\n\nAnswer:"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_format_v2.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'training.format_v2'`.

- [ ] **Step 3: Add `reasoning` to `Record`** — in `training/records.py`, replace the class body's docstring and add the field last:

```python
@dataclass(frozen=True)
class Record:
    """One training or evaluation example, normalised across all sources.

    `reasoning` is the thinking a run 2 training example teaches, rendered
    inside Qwen3's <think> block; None renders an empty block, as run 1 did.
    """

    id: str
    source: str
    kind: str  # "mcq" | "dialogue"
    question: str
    options: dict[str, str] | None
    answer: str | None
    rationale: str | None
    response: str | None
    subject: str | None
    reasoning: str | None = None
```

- [ ] **Step 4: Write `training/format_v2.py`**

```python
from __future__ import annotations

import re

from training.prompts import SYSTEM_CHAT, extract_letter
from training.records import Record

SYSTEM_MCQ_V2 = (
    "You are a medical education assistant. Work through the question, then "
    "give your choice first, on its own line, in the exact form 'Answer: X', "
    "followed by a brief explanation."
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ANSWER_LINE = re.compile(r"Answer:\s*\(?([A-J])\b")


def option_letters(rec: Record) -> list[str]:
    return sorted(rec.options) if rec.options else []


def format_question_v2(rec: Record) -> str:
    if rec.kind != "mcq" or not rec.options:
        return rec.question
    lines = [rec.question, ""]
    lines += [f"{letter}. {rec.options[letter]}" for letter in option_letters(rec)]
    return "\n".join(lines)


def short_explanation(text: str | None, *, max_sentences: int = 2,
                      max_chars: int = 400) -> str:
    """The first sentences of an explanation, to follow the answer line."""
    text = (text or "").strip()
    if text.lower().startswith("explanation:"):
        text = text[len("explanation:"):].strip()
    if not text:
        return ""
    out = " ".join(_SENTENCE_END.split(text)[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return out


def answer_text(rec: Record) -> str:
    """What follows the think block. The letter always comes first, so no
    token cap can cut it off -- run 1's rationale-first answers lost 70 of 300
    MedMCQA letters that way."""
    if rec.kind != "mcq":
        return (rec.response or "").strip()
    head = f"Answer: {rec.answer}"
    explanation = short_explanation(rec.rationale)
    if explanation:
        return f"{head}\n\n{explanation}"
    option = (rec.options or {}).get(rec.answer or "")
    return f"{head}. {option}" if option else head


def v2_messages(rec: Record, *, with_answer: bool) -> list[dict]:
    system = SYSTEM_MCQ_V2 if rec.kind == "mcq" else SYSTEM_CHAT
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": format_question_v2(rec)}]
    if with_answer:
        reply = {"role": "assistant", "content": answer_text(rec)}
        if rec.reasoning:
            reply["reasoning_content"] = rec.reasoning.strip()
        msgs.append(reply)
    return msgs


def render_training_text(tok, rec: Record) -> str:
    """Qwen3's template puts reasoning_content in <think>...</think>; a row
    without reasoning gets the empty block, exactly as run 1 trained."""
    return tok.apply_chat_template(v2_messages(rec, with_answer=True),
                                   tokenize=False,
                                   enable_thinking=bool(rec.reasoning))


def render_letter_prompt(tok, rec: Record) -> str:
    """Thinking off, then 'Answer:' -- the next token is the choice."""
    return tok.apply_chat_template(v2_messages(rec, with_answer=False),
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "Answer:"


def render_reasoning_prompt(tok, rec: Record) -> str:
    """Thinking on: the prompt ends at the assistant turn and the model opens
    its own <think> block."""
    return tok.apply_chat_template(v2_messages(rec, with_answer=False),
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def final_letter(final_text: str, letters: list[str]) -> str | None:
    """The letter stated after the think block, if it is one of the options."""
    found = _ANSWER_LINE.search(final_text or "")
    if found and found.group(1) in letters:
        return found.group(1)
    fallback = extract_letter(final_text or "")
    return fallback if fallback in letters else None


def forced_suffix(closed: bool) -> str:
    """Text that makes the very next token the answer letter (budget forcing).

    An unfinished think block is closed first, in the layout training used."""
    return "\n\nAnswer:" if closed else "\n</think>\n\nAnswer:"
```

- [ ] **Step 5: Run the tests and the suite**

Run: `.venv/bin/python -m pytest tests/test_format_v2.py -q && .venv/bin/python -m pytest -q`
Expected: 13 passed in the new file; the whole suite passes.

- [ ] **Step 6: Commit**

```bash
git add training/records.py training/format_v2.py tests/test_format_v2.py
git commit -m "Add run 2's format: reasoning in the think block, the answer letter first"
```

---

### Task 2: The new sources

**Files:**
- Create: `training/sources_v2.py`
- Test: `tests/test_sources_v2.py`

**Interfaces:**
- Consumes: `training.records.Record`, `training.sources.LETTERS`.
- Produces: `EVAL_SIZES: dict[str, int]`; `MMLU_MEDICAL: tuple[str, ...]`; `PUBMEDQA_OPTIONS`; `is_english(text) -> bool`; `parse_option_lines(lines) -> dict[str, str]`; `split_stem_and_options(text) -> tuple[str, dict]`; `match_option(answer, options) -> str | None`; `stated_letter(text) -> str | None`; normalisers `(raw: dict, idx: int) -> Record | None`: `norm_medreason`, `norm_r1_distill`, `norm_ultramedical`, `norm_medical_o1_reasoning`, `norm_medical_o1_direct`, `norm_reasonmed`, `norm_pubmedqa(raw, idx, *, source="pubmedqa")`, `norm_mmlu`.

- [ ] **Step 1: Write the failing tests** — `tests/test_sources_v2.py`. The fixtures are real rows, shortened (sampled 2026-09-25):

```python
from __future__ import annotations

from training.sources_v2 import (is_english, match_option, norm_medical_o1_direct,
                                 norm_medical_o1_reasoning, norm_medreason, norm_mmlu,
                                 norm_pubmedqa, norm_r1_distill, norm_reasonmed,
                                 norm_ultramedical, parse_option_lines,
                                 split_stem_and_options, stated_letter)

MEDREASON_MCQ = {
    "dataset_name": "medmcqa",
    "question": "Urogenital Diaphragm is made up of the following, except:",
    "options": "Answer Choices:\nA. Deep transverse Perineus\nB. Perinial membrane\n"
               "C. Colle's fascia\nD. Sphincter Urethrae",
    "answer": "Colle's fascia. Explanation: Colle's fascia does not contribute. "
              "It is superficial.",
    "reasoning": "Finding reasoning paths: ...\nConclusion: Colle's fascia.",
}
ULTRA = {
    "id": "u1", "type": "Exam", "answer": "D",
    "conversations": [
        {"from": "human", "value": "A pregnant woman has dysuria. Which is the best "
                                   "treatment?\n\nA. Ampicillin\nB. Ceftriaxone\n"
                                   "C. Doxycycline\nD. Nitrofurantoin"},
        {"from": "gpt", "value": "Nitrofurantoin is safe in pregnancy.\n\n"
                                 "So, the answer is D."},
    ],
}
REASONMED = {
    "instruction": "Please answer the following multiple-choice question:\n"
                   "The most likely poisoning is -?\nA. Mercury\nB. Lead\n"
                   "C. Arsenic\nD. Phosphorus",
    "input": "",
    "output": "Salivation and gum lines suggest a heavy metal. The pattern "
              "points to mercury exposure.",
}
PUBMED = {
    "pubid": 21645374, "question": "Do mitochondria play a role?",
    "final_decision": "yes", "long_answer": "Results depict mitochondrial dynamics.",
    "context": {"contexts": ["Background one.", "Methods two."]},
}


def test_parse_option_lines_needs_a_run_from_a():
    assert parse_option_lines(["Answer Choices:", "A. one", "B. two"]) == {"A": "one", "B": "two"}
    assert parse_option_lines(["A.: 12 mm", "B.: 24 mm"]) == {"A": "12 mm", "B": "24 mm"}
    assert parse_option_lines(["(A) x", "(B) y", "(C) z"]) == {"A": "x", "B": "y", "C": "z"}
    assert parse_option_lines(["B. two", "C. three"]) == {}
    assert parse_option_lines(["A. one", "B. two", "D. four"]) == {}
    assert parse_option_lines(["A. only"]) == {}


def test_split_stem_and_options_reads_trailing_option_lines():
    stem, opts = split_stem_and_options(
        "A 3-year-old girl has a rash.\nWhich drug?\n\nA. Ampicillin\nB. Ceftriaxone\n")
    assert stem == "A 3-year-old girl has a rash.\nWhich drug?"
    assert opts == {"A": "Ampicillin", "B": "Ceftriaxone"}
    assert split_stem_and_options("No options here.") == ("No options here.", {})


def test_match_option_by_prefix_decision_and_longest():
    opts = {"A": "Deep transverse Perineus", "B": "Perinial membrane",
            "C": "Colle's fascia", "D": "Sphincter Urethrae"}
    assert match_option("Colle's fascia. Explanation: it is superficial.", opts) == "C"
    assert match_option("Small cell carcinoma",
                        {"A": "Small cell carcinoma", "B": "Squamous"}) == "A"
    assert match_option("The final decision is: yes. Because...",
                        {"A": "Yes", "B": "No"}) == "A"
    assert match_option("Lead acetate poisoning",
                        {"A": "Lead", "B": "Lead acetate"}) == "B"
    assert match_option("Something else", opts) is None


def test_stated_letter_takes_the_last_and_ignores_words():
    assert stated_letter("maybe the answer is A... no, the answer is C.") == "C"
    assert stated_letter("So, the answer is A.") == "A"
    assert stated_letter("the answer is Aspirin") is None


def test_is_english():
    assert is_english("A patient with fever.")
    assert not is_english("患者发热三天")


def test_medreason_mcq_maps_answer_text_to_its_letter():
    rec = norm_medreason(MEDREASON_MCQ, 7)
    assert (rec.kind, rec.answer, rec.source, rec.subject) == ("mcq", "C", "medreason", "medmcqa")
    assert rec.rationale.startswith("Colle's fascia does not contribute")
    assert rec.reasoning.startswith("Finding reasoning paths")


def test_medreason_keeps_huatuo_as_free_text_and_drops_benchmark_subsets():
    free = norm_medreason({**MEDREASON_MCQ, "dataset_name": "huatuo", "options": "",
                           "answer": "Normal residual volume."}, 1)
    assert free.kind == "dialogue" and free.response == "Normal residual volume."
    assert free.reasoning
    for name in ("MMLU", "pubmedqa", "pubmedqa_artificial", "pubmedqa_unlabeled",
                 "LastHumanity"):
        assert norm_medreason({**MEDREASON_MCQ, "dataset_name": name}, 1) is None


def test_medreason_drops_rows_whose_answer_names_no_single_option():
    assert norm_medreason({**MEDREASON_MCQ, "answer": "None of these"}, 1) is None


def test_ultramedical_keeps_rows_whose_explanation_agrees_with_gold():
    rec = norm_ultramedical(ULTRA, 0)
    assert rec.answer == "D" and rec.options["D"] == "Nitrofurantoin"
    assert rec.reasoning.endswith("the answer is D.")
    assert rec.question == "A pregnant woman has dysuria. Which is the best treatment?"
    assert norm_ultramedical({**ULTRA, "answer": "B"}, 0) is None


def test_r1_distill_and_medical_o1_map_reasoning_and_reply():
    r = norm_r1_distill({"question": "Q?", "reasoning (reasoning_content)": "think",
                         "response (content)": "reply"}, 3)
    assert (r.kind, r.reasoning, r.response, r.source) == ("dialogue", "think", "reply", "r1_distill")
    raw = {"Question": "Q?", "Complex_CoT": "cot", "Response": "resp"}
    o = norm_medical_o1_reasoning(raw, 4)
    assert (o.reasoning, o.response) == ("cot", "resp")
    d = norm_medical_o1_direct(raw, 4)
    assert d.reasoning is None and d.source == "medical_o1_direct" and d.id != o.id


def test_reasonmed_reads_the_option_its_conclusion_names():
    rec = norm_reasonmed(REASONMED, 0)
    assert rec.question == "The most likely poisoning is -?" and rec.answer == "A"
    assert norm_reasonmed({**REASONMED, "output": "Mercury or lead could both fit."}, 0) is None
    assert norm_reasonmed({**REASONMED, "output": "Long reasoning... the answer is B."}, 0).answer == "B"


def test_pubmedqa_includes_the_abstract_and_three_options():
    rec = norm_pubmedqa(PUBMED, 0)
    assert rec.question == ("Context: Background one. Methods two.\n\n"
                            "Question: Do mitochondria play a role?")
    assert rec.options == {"A": "yes", "B": "no", "C": "maybe"} and rec.answer == "A"
    assert rec.id == "pubmedqa-21645374"
    assert norm_pubmedqa(PUBMED, 0, source="pubmedqa_artificial").source == "pubmedqa_artificial"
    assert norm_pubmedqa({**PUBMED, "final_decision": "unsure"}, 0) is None


def test_mmlu_maps_choices_to_letters():
    rec = norm_mmlu({"question": "Q?", "subject": "anatomy",
                     "choices": ["a", "b", "c", "d"], "answer": 2}, 5)
    assert rec.answer == "C" and rec.options["C"] == "c"
    assert (rec.subject, rec.source, rec.id) == ("anatomy", "mmlu_medical", "mmlu-anatomy-5")
    assert norm_mmlu({"question": "Q?", "subject": "anatomy",
                      "choices": ["a", "b"], "answer": 0}, 5) is None
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sources_v2.py -q`
Expected: collection error, `No module named 'training.sources_v2'`.

- [ ] **Step 3: Write `training/sources_v2.py`**

```python
from __future__ import annotations

import re

from training.records import Record
from training.sources import LETTERS

# Confirmed 2026-09-25 from the Hub: MedQA-USMLE test, MedMCQA validation (all
# rows), PubMedQA pqa_labeled, and the six MMLU medical subjects' test splits.
EVAL_SIZES = {"medqa": 1_273, "medmcqa": 4_183, "pubmedqa": 1_000, "mmlu_medical": 1_089}
MMLU_MEDICAL = ("anatomy", "clinical_knowledge", "college_biology",
                "college_medicine", "medical_genetics", "professional_medicine")
PUBMEDQA_OPTIONS = {"A": "yes", "B": "no", "C": "maybe"}

# MedReason sub-sources. 'MMLU' may come from MMLU's test split and 'pubmedqa'
# IS PubMedQA's labeled set -- both are benchmarks here. Its other PubMedQA rows
# ask about an abstract they do not include, and 'LastHumanity' is Humanity's
# Last Exam.
MEDREASON_MCQ = ("medqa", "medmcqa", "MedXpertQA")
MEDREASON_FREE = ("huatuo",)

_OPTION = re.compile(r"^\s*\(?([A-J])\s*[.):]\s*:?\s*(\S.*?)\s*$")
_DECISION = re.compile(r"final decision is:?\s*\**\s*(yes|no|maybe)\b", re.IGNORECASE)
_STATED = re.compile(r"[Aa]nswer is:?\s*\**\s*\(?([A-J])\b")
_WORD = re.compile(r"[a-z0-9]+")
_REASONMED_PREAMBLE = re.compile(
    r"^\s*please answer the following multiple-choice question:?\s*", re.IGNORECASE)


def _clean(value) -> str:
    return str(value).strip() if value is not None else ""


def _norm(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def is_english(text: str, *, min_ascii: float = 0.97) -> bool:
    """A cheap language screen: MedReason's 'huatuo' rows are Chinese-origin."""
    if not text:
        return False
    return sum(ch.isascii() for ch in text) / len(text) >= min_ascii


def parse_option_lines(lines: list[str]) -> dict[str, str]:
    """'A. x' / 'A) x' / '(A) x' / 'A.: x' lines -> options. The letters must
    run A, B, C... with no gap: a gap means a lost option, and re-lettering
    would silently change which answer is right."""
    opts: dict[str, str] = {}
    for line in lines:
        found = _OPTION.match(line)
        if found and found.group(1) not in opts:
            opts[found.group(1)] = found.group(2)
    letters = sorted(opts)
    if len(letters) < 2 or letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return {}
    return opts


def split_stem_and_options(text: str) -> tuple[str, dict[str, str]]:
    """Options written as the text's trailing 'A. ...' lines."""
    lines = (text or "").rstrip().splitlines()
    i = len(lines)
    while i > 0 and (_OPTION.match(lines[i - 1]) or not lines[i - 1].strip()):
        i -= 1
    options = parse_option_lines(lines[i:])
    if not options:
        return (text or "").strip(), {}
    return "\n".join(lines[:i]).strip(), options


def match_option(answer: str, options: dict[str, str]) -> str | None:
    """The one option an answer text names; None when it names none or several."""
    decision = _DECISION.search(answer or "")
    if decision:
        want = decision.group(1).lower()
        hits = [k for k, v in options.items() if _norm(v) == want]
        return hits[0] if len(hits) == 1 else None
    said = _norm(answer)
    if not said:
        return None
    hits = [k for k, v in options.items()
            if _norm(v) and (said == _norm(v) or said.startswith(_norm(v) + " "))]
    if len(hits) > 1:
        longest = max(len(_norm(options[k])) for k in hits)
        hits = [k for k in hits if len(_norm(options[k])) == longest]
    return hits[0] if len(hits) == 1 else None


def stated_letter(text: str) -> str | None:
    """The last 'answer is X' in a text. Case-sensitive on the letter, so
    'the answer is a combination' is not read as A."""
    found = _STATED.findall(text or "")
    return found[-1] if found else None


def norm_medreason(raw: dict, idx: int) -> Record | None:
    name = _clean(raw.get("dataset_name"))
    question = _clean(raw.get("question"))
    reasoning = _clean(raw.get("reasoning"))
    answer = _clean(raw.get("answer"))
    if not (question and reasoning and answer) or not is_english(question + reasoning):
        return None
    if name in MEDREASON_FREE:
        return Record(id=f"medreason-{idx}", source="medreason", kind="dialogue",
                      question=question, options=None, answer=None, rationale=None,
                      response=answer, subject=name, reasoning=reasoning)
    if name not in MEDREASON_MCQ:
        return None
    options = parse_option_lines(_clean(raw.get("options")).splitlines())
    letter = match_option(answer, options) if options else None
    if letter is None:
        return None
    explanation = answer.split("Explanation:", 1)[1] if "Explanation:" in answer else ""
    return Record(id=f"medreason-{idx}", source="medreason", kind="mcq",
                  question=question, options=options, answer=letter,
                  rationale=_clean(explanation) or None, response=None,
                  subject=name, reasoning=reasoning)


def norm_r1_distill(raw: dict, idx: int) -> Record | None:
    question = _clean(raw.get("question"))
    reasoning = _clean(raw.get("reasoning (reasoning_content)"))
    response = _clean(raw.get("response (content)"))
    if not (question and reasoning and response) or not is_english(question + response):
        return None
    return Record(id=f"r1_distill-{idx}", source="r1_distill", kind="dialogue",
                  question=question, options=None, answer=None, rationale=None,
                  response=response, subject=None, reasoning=reasoning)


def norm_ultramedical(raw: dict, idx: int) -> Record | None:
    conversation = raw.get("conversations") or []
    if len(conversation) < 2:
        return None
    stem, options = split_stem_and_options(_clean(conversation[0].get("value")))
    explanation = _clean(conversation[-1].get("value"))
    answer = _clean(raw.get("answer")).upper()
    if not (stem and explanation and options) or answer not in options:
        return None
    if stated_letter(explanation) != answer or not is_english(stem + explanation):
        return None
    return Record(id=_clean(raw.get("id")) or f"ultramedical-{idx}",
                  source="ultramedical", kind="mcq", question=stem, options=options,
                  answer=answer, rationale=None, response=None,
                  subject=_clean(raw.get("type")) or None, reasoning=explanation)


def norm_medical_o1_reasoning(raw: dict, idx: int) -> Record | None:
    question = _clean(raw.get("Question"))
    cot = _clean(raw.get("Complex_CoT"))
    response = _clean(raw.get("Response"))
    if not (question and cot and response):
        return None
    return Record(id=f"medical_o1-{idx}", source="medical_o1", kind="dialogue",
                  question=question, options=None, answer=None, rationale=None,
                  response=response, subject=None, reasoning=cot)


def norm_medical_o1_direct(raw: dict, idx: int) -> Record | None:
    question = _clean(raw.get("Question"))
    response = _clean(raw.get("Response"))
    if not (question and response):
        return None
    return Record(id=f"medical_o1_direct-{idx}", source="medical_o1_direct",
                  kind="dialogue", question=question, options=None, answer=None,
                  rationale=None, response=response, subject=None, reasoning=None)


def norm_reasonmed(raw: dict, idx: int) -> Record | None:
    """ReasonMed states no gold field. Its card describes multi-agent checking
    of every trace; the answer is the letter the trace states, else the one
    option its conclusion names. A trace naming several options is dropped."""
    instruction = _REASONMED_PREAMBLE.sub("", _clean(raw.get("instruction")))
    extra = _clean(raw.get("input"))
    stem, options = split_stem_and_options(f"{instruction}\n{extra}" if extra else instruction)
    output = _clean(raw.get("output"))
    if not (options and stem and output) or not is_english(stem + output):
        return None
    letter = stated_letter(output)
    if letter not in options:
        tail = f" {_norm(output[-800:])} "
        hits = [k for k, v in options.items() if _norm(v) and f" {_norm(v)} " in tail]
        letter = hits[0] if len(hits) == 1 else None
    if letter is None:
        return None
    return Record(id=f"reasonmed-{idx}", source="reasonmed", kind="mcq",
                  question=stem, options=options, answer=letter, rationale=None,
                  response=None, subject=None, reasoning=output)


def norm_pubmedqa(raw: dict, idx: int, *, source: str = "pubmedqa") -> Record | None:
    """The abstract goes in with the question: PubMedQA asks about it, and
    without it the question is guesswork."""
    passages = [p for p in ((raw.get("context") or {}).get("contexts") or []) if p]
    question = _clean(raw.get("question"))
    by_text = {text: letter for letter, text in PUBMEDQA_OPTIONS.items()}
    letter = by_text.get(_clean(raw.get("final_decision")).lower())
    if not passages or not question or letter is None:
        return None
    return Record(id=f"{source}-{raw.get('pubid', idx)}", source=source, kind="mcq",
                  question="Context: " + " ".join(passages) + f"\n\nQuestion: {question}",
                  options=dict(PUBMEDQA_OPTIONS), answer=letter,
                  rationale=_clean(raw.get("long_answer")) or None, response=None,
                  subject=None)


def norm_mmlu(raw: dict, idx: int) -> Record | None:
    choices = [_clean(c) for c in (raw.get("choices") or [])]
    answer = raw.get("answer")
    question = _clean(raw.get("question"))
    if (len(choices) != 4 or not all(choices) or not question
            or not isinstance(answer, int) or not 0 <= answer < 4):
        return None
    subject = _clean(raw.get("subject")) or "unknown"
    return Record(id=f"mmlu-{subject}-{idx}", source="mmlu_medical", kind="mcq",
                  question=question, options=dict(zip(LETTERS, choices)),
                  answer=LETTERS[answer], rationale=None, response=None,
                  subject=subject)
```

- [ ] **Step 4: Run the tests and the suite**

Run: `.venv/bin/python -m pytest tests/test_sources_v2.py -q && .venv/bin/python -m pytest -q`
Expected: 13 passed in the new file; the suite passes.

- [ ] **Step 5: Commit**

```bash
git add training/sources_v2.py tests/test_sources_v2.py
git commit -m "Normalise run 2's sources, keeping only rows whose answer is certain"
```

---

### Task 3: Decontamination and the data build

**Files:**
- Create: `training/decontam.py`, `training/prepare_data_v2.py`
- Modify: `.gitignore` (add `/data/`)
- Test: `tests/test_decontam.py`, `tests/test_prepare_data_v2.py`

**Interfaces:**
- Consumes: `training.prepare_data.normalise_question`, `training.check_lengths.percentile`/`summarise`, `training.format_v2.v2_messages`, Task 2's normalisers, `training.sources.load`, `training.sources.normalise_medmcqa`/`normalise_medqa`.
- Produces: `EvalIndex(questions, n=13).hits(question) -> "exact" | "ngram" | None`; `Deduper().first_time(question) -> bool`; `MixEntry(name, kind, target)`; `MIX`; `collect(entry, rows, normalise, *, index, deduper, measure, cap, max_candidates) -> (list[(Record, int)], dict)`; `choose_max_seq(lengths, *, cap) -> int`; `summarise_mix(recs) -> dict`; CLI `python -m training.prepare_data_v2 --out data/v2` writing `train.jsonl`, `eval_medqa.jsonl`, `eval_medmcqa.jsonl`, `eval_pubmedqa.jsonl`, `eval_mmlu_medical.jsonl`, `data_report.json` (keys include `max_seq`, `train_size`, `reasoning_fraction`, `mcq_fraction`, `sources`).

- [ ] **Step 1: Write the failing tests** — `tests/test_decontam.py`:

```python
from __future__ import annotations

from training.decontam import Deduper, EvalIndex, ngrams

VIGNETTE = ("a 45 year old man presents with crushing chest pain radiating to the "
            "left arm and jaw for two hours")


def test_exact_match_after_normalisation():
    index = EvalIndex(["What is the most likely diagnosis?"])
    assert index.hits("what is the MOST likely diagnosis") == "exact"
    assert index.hits("A different question entirely") is None


def test_thirteen_word_overlap_is_caught_but_short_text_needs_an_exact_match():
    index = EvalIndex([VIGNETTE])
    assert index.hits("Case: " + VIGNETTE + " What next?") == "ngram"
    assert index.hits("crushing chest pain radiating to the left arm") is None


def test_ngrams_of_short_text_is_empty():
    assert ngrams("one two three", 13) == set()


def test_deduper_flags_the_second_occurrence():
    d = Deduper()
    assert d.first_time("Q one?") and not d.first_time("q ONE") and d.first_time("Q two?")
```

and `tests/test_prepare_data_v2.py`:

```python
from __future__ import annotations

from training.decontam import Deduper, EvalIndex
from training.prepare_data_v2 import MIX, MixEntry, choose_max_seq, collect, summarise_mix
from training.records import Record


def rec(i, *, kind="mcq", reasoning="r", q=None) -> Record:
    return Record(id=f"r{i}", source="s", kind=kind, question=q or f"question number {i}",
                  options={"A": "a", "B": "b"} if kind == "mcq" else None,
                  answer="A" if kind == "mcq" else None, rationale=None,
                  response=None if kind == "mcq" else "resp", subject=None,
                  reasoning=reasoning)


def test_collect_stops_at_target_and_counts_each_rejection():
    def normalise(raw, idx):
        i = raw["i"]
        if i == 0:
            return None
        if i == 1:
            return rec(1, kind="dialogue")
        if i == 2:
            return rec(2, q="blocked question")
        if i in (3, 4):
            return rec(i, q="same")
        return rec(i)

    kept, stats = collect(
        MixEntry("s", "mcq", 3), [(i, {"i": i}) for i in range(20)], normalise,
        index=EvalIndex(["blocked question"]), deduper=Deduper(),
        measure=lambda r: 999 if r.id == "r5" else 10, cap=100, max_candidates=100)
    assert [r.id for r, _ in kept] == ["r3", "r6", "r7"]
    assert stats == {"target": 3, "seen": 8, "unusable": 1, "other_kind": 1,
                     "contaminated_exact": 1, "contaminated_ngram": 0,
                     "duplicate": 1, "too_long": 1, "kept": 3}


def test_collect_gives_up_after_max_candidates():
    kept, stats = collect(MixEntry("s", None, 10), [(i, {}) for i in range(50)],
                          lambda raw, idx: None, index=EvalIndex([]), deduper=Deduper(),
                          measure=lambda r: 1, cap=10, max_candidates=5)
    assert kept == [] and stats["seen"] == 5 and stats["unusable"] == 5


def test_choose_max_seq_rounds_p99_up_to_64_and_caps():
    assert choose_max_seq([100] * 99 + [1000], cap=3072) == 128
    assert choose_max_seq(list(range(1, 5001)), cap=3072) == 3072


def test_summarise_mix_reports_fractions():
    s = summarise_mix([rec(1), rec(2, reasoning=None), rec(3, kind="dialogue")])
    assert s["reasoning_fraction"] == round(2 / 3, 4)
    assert s["mcq_fraction"] == round(2 / 3, 4)
    assert s["by_source"] == {"s": 3}


def test_the_mix_is_three_quarters_reasoning_and_eleven_thousand_rows():
    reasoning = {"medreason", "r1_distill", "ultramedical", "medical_o1", "reasonmed"}
    total = sum(e.target for e in MIX)
    assert total == 11_000
    assert sum(e.target for e in MIX if e.name in reasoning) == 8_250
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_decontam.py tests/test_prepare_data_v2.py -q`
Expected: collection errors, modules not found.

- [ ] **Step 3: Write `training/decontam.py`**

```python
from __future__ import annotations

from training.prepare_data import normalise_question

NGRAM = 13


def ngrams(norm_text: str, n: int = NGRAM) -> set[int]:
    words = norm_text.split()
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


class EvalIndex:
    """Every benchmark question, for exact and 13-word-overlap lookups.

    Exact matching alone misses a benchmark vignette copied with a new first
    line, and MedReason and ReasonMed both draw on the same exam banks the
    benchmarks come from. Thirteen consecutive words is the long-standing
    contamination test; questions shorter than that need an exact match.
    """

    def __init__(self, questions: list[str], n: int = NGRAM) -> None:
        self.n = n
        self.exact: set[str] = set()
        self.grams: set[int] = set()
        for question in questions:
            key = normalise_question(question)
            self.exact.add(key)
            self.grams |= ngrams(key, n)

    def hits(self, question: str) -> str | None:
        key = normalise_question(question)
        if key in self.exact:
            return "exact"
        if len(key.split()) >= self.n and ngrams(key, self.n) & self.grams:
            return "ngram"
        return None


class Deduper:
    """The same question from two sources would silently double its weight."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def first_time(self, question: str) -> bool:
        key = normalise_question(question)
        if key in self.seen:
            return False
        self.seen.add(key)
        return True
```

- [ ] **Step 4: Write `training/prepare_data_v2.py`**

```python
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from training import sources
from training import sources_v2 as sv2
from training.check_lengths import percentile, summarise
from training.decontam import Deduper, EvalIndex
from training.format_v2 import v2_messages
from training.records import Record


@dataclass(frozen=True)
class MixEntry:
    name: str
    kind: str | None
    target: int

    @property
    def label(self) -> str:
        return f"{self.name}:{self.kind}" if self.kind else self.name


# 75% reasoning (8,250) and 25% direct (2,750): Unsloth's and Qwen's guidance
# for keeping Qwen3's thinking. 11,000 rows at ~925 tokens is about 4.5 hours
# at run 1's measured T4 throughput.
MIX: tuple[MixEntry, ...] = (
    MixEntry("medreason", "mcq", 2200),
    MixEntry("medreason", "dialogue", 1100),
    MixEntry("r1_distill", None, 1650),
    MixEntry("ultramedical", None, 1650),
    MixEntry("medical_o1", None, 1100),
    MixEntry("reasonmed", None, 550),
    MixEntry("medmcqa", None, 800),
    MixEntry("medqa", None, 550),
    MixEntry("pubmedqa_artificial", None, 750),
    MixEntry("medical_o1_direct", None, 650),
)

# name: (hf_id, config, split, streaming, reversed order). The big three are
# streamed through a shuffle buffer rather than downloaded whole. The direct
# medical-o1 slice reads the same shuffled order from the other end, so it
# never meets the reasoning slice's rows.
HF: dict[str, tuple[str, str | None, str, bool, bool]] = {
    "medreason": ("UCSC-VLAA/MedReason", None, "train", False, False),
    "r1_distill": ("FreedomIntelligence/Medical-R1-Distill-Data", "en", "train", False, False),
    "ultramedical": ("TsinghuaC3I/UltraMedical", None, "train", True, False),
    "medical_o1": ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", False, False),
    "medical_o1_direct": ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", "train", False, True),
    "reasonmed": ("lingshu-medical-mllm/ReasonMed", None, "train", True, False),
    "pubmedqa_artificial": ("qiaojin/PubMedQA", "pqa_artificial", "train", True, False),
    "medmcqa": ("openlifescienceai/medmcqa", "default", "train", False, False),
    "medqa": ("GBaker/MedQA-USMLE-4-options", "default", "train", False, False),
}

NORMALISERS = {
    "medreason": sv2.norm_medreason,
    "r1_distill": sv2.norm_r1_distill,
    "ultramedical": sv2.norm_ultramedical,
    "medical_o1": sv2.norm_medical_o1_reasoning,
    "medical_o1_direct": sv2.norm_medical_o1_direct,
    "reasonmed": sv2.norm_reasonmed,
    "pubmedqa_artificial": partial(sv2.norm_pubmedqa, source="pubmedqa_artificial"),
    "medmcqa": sources.normalise_medmcqa,
    "medqa": sources.normalise_medqa,
}

STAT_KEYS = ("target", "seen", "unusable", "other_kind", "contaminated_exact",
             "contaminated_ngram", "duplicate", "too_long", "kept")


def collect(entry: MixEntry, rows, normalise, *, index: EvalIndex, deduper: Deduper,
            measure, cap: int, max_candidates: int):
    """Take rows in order until `entry.target` survive every filter.

    Filters run cheapest first, and every rejection is counted, so the data
    report shows exactly why a source came up short instead of hiding it.
    """
    stats = dict.fromkeys(STAT_KEYS, 0)
    stats["target"] = entry.target
    kept: list[tuple[Record, int]] = []
    for idx, raw in rows:
        if len(kept) >= entry.target or stats["seen"] >= max_candidates:
            break
        stats["seen"] += 1
        rec = normalise(raw, idx)
        if rec is None:
            stats["unusable"] += 1
            continue
        if entry.kind and rec.kind != entry.kind:
            stats["other_kind"] += 1
            continue
        hit = index.hits(rec.question)
        if hit:
            stats[f"contaminated_{hit}"] += 1
            continue
        if not deduper.first_time(rec.question):
            stats["duplicate"] += 1
            continue
        n_tokens = measure(rec)
        if n_tokens > cap:
            stats["too_long"] += 1
            continue
        kept.append((rec, n_tokens))
    stats["kept"] = len(kept)
    return kept, stats


def choose_max_seq(lengths: list[int], *, cap: int) -> int:
    return min(cap, 64 * math.ceil(percentile(lengths, 0.99) / 64))


def summarise_mix(recs: list[Record]) -> dict:
    n = len(recs) or 1
    by_source: dict[str, int] = {}
    for r in recs:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return {"by_source": dict(sorted(by_source.items())),
            "reasoning_fraction": round(sum(bool(r.reasoning) for r in recs) / n, 4),
            "mcq_fraction": round(sum(r.kind == "mcq" for r in recs) / n, 4)}


def iter_rows(name: str, *, seed: int):
    from datasets import load_dataset  # network; imported only when building

    hf_id, config, split, streaming, reverse = HF[name]
    if streaming:
        ds = load_dataset(hf_id, config, split=split, streaming=True)
        yield from enumerate(ds.shuffle(seed=seed, buffer_size=10_000))
        return
    ds = load_dataset(hf_id, config, split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    if reverse:
        order.reverse()
    for i in order:
        yield i, ds[i]


def load_eval_sets() -> dict[str, list[Record]]:
    from datasets import load_dataset

    sets = {
        "medqa": sources.load("medqa", limit=0, split="test"),
        "medmcqa": sources.load("medmcqa", limit=0, split="validation",
                                require_rationale=False, require_single_choice=False),
        "pubmedqa": [r for i, row in enumerate(
                         load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"))
                     if (r := sv2.norm_pubmedqa(row, i))],
        "mmlu_medical": [r for subject in sv2.MMLU_MEDICAL
                         for i, row in enumerate(load_dataset("cais/mmlu", subject, split="test"))
                         if (r := sv2.norm_mmlu(row, i))],
    }
    wrong = {k: len(v) for k, v in sets.items() if len(v) != sv2.EVAL_SIZES[k]}
    if wrong:
        raise SystemExit(f"\nSTOP. Benchmark sizes {wrong} differ from {sv2.EVAL_SIZES}.\n"
                         "FIX: a dataset on the Hub has changed; check its revision.")
    return sets


def write_jsonl(path: Path, recs: list[Record]) -> None:
    with path.open("w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec.to_dict()) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build run 2's training set and benchmark files.")
    p.add_argument("--out", type=Path, default=Path("data/v2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="unsloth/Qwen3-4B")
    p.add_argument("--cap", type=int, default=3072, help="longest example kept, in tokens")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every target (dry runs)")
    p.add_argument("--candidates-factor", type=int, default=8)
    p.add_argument("--show", type=int, default=0, help="print N rendered examples per source")
    args = p.parse_args()

    from transformers import AutoTokenizer  # heavy; imported only when building

    args.out.mkdir(parents=True, exist_ok=True)
    print("Benchmarks first, so training can be cleaned against them.")
    evals = load_eval_sets()
    for name, recs in evals.items():
        write_jsonl(args.out / f"eval_{name}.jsonl", recs)
        print(f"  eval_{name}.jsonl  {len(recs):,}")
    index = EvalIndex([r.question for recs in evals.values() for r in recs])

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def measure(rec: Record) -> int:
        return len(tok.apply_chat_template(
            v2_messages(rec, with_answer=True), tokenize=True, return_dict=False,
            enable_thinking=bool(rec.reasoning)))

    deduper = Deduper()
    kept_all: list[tuple[Record, int]] = []
    report_sources: dict[str, dict] = {}
    print("\nTraining sources.")
    for entry in MIX:
        target = max(1, round(entry.target * args.scale))
        scaled = MixEntry(entry.name, entry.kind, target)
        kept, stats = collect(
            scaled, iter_rows(entry.name, seed=args.seed), NORMALISERS[entry.name],
            index=index, deduper=deduper, measure=measure, cap=args.cap,
            max_candidates=target * args.candidates_factor)
        report_sources[scaled.label] = stats
        short = "" if stats["kept"] >= target else f"   SHORT by {target - stats['kept']:,}"
        print(f"  {scaled.label:<22} kept {stats['kept']:>6,} of {stats['seen']:>7,} seen"
              f" (contaminated {stats['contaminated_exact'] + stats['contaminated_ngram']:,},"
              f" duplicate {stats['duplicate']:,}, too long {stats['too_long']:,}){short}")
        for rec, n_tokens in kept[: args.show]:
            text = tok.apply_chat_template(v2_messages(rec, with_answer=True), tokenize=False,
                                           enable_thinking=bool(rec.reasoning))
            print(f"\n----- {scaled.label} {rec.id} ({n_tokens} tokens) -----\n"
                  f"{text[:1500]}\n...\n{text[-400:]}")
        kept_all += kept

    lengths = [n for _, n in kept_all]
    max_seq = choose_max_seq(lengths, cap=args.cap)
    final = [rec for rec, n in kept_all if n <= max_seq]
    random.Random(args.seed).shuffle(final)
    write_jsonl(args.out / "train.jsonl", final)

    report = {
        "version": "run2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "tokenizer": args.tokenizer, "cap": args.cap,
        "scale": args.scale, "max_seq": max_seq,
        "length": summarise([n for _, n in kept_all if n <= max_seq]),
        "train_size": len(final),
        "dropped_over_max_seq": len(kept_all) - len(final),
        **summarise_mix(final),
        "sources": report_sources,
        "eval_sizes": {name: len(recs) for name, recs in evals.items()},
    }
    (args.out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ntrain.jsonl {len(final):,} rows | max_seq {max_seq} | "
          f"reasoning {report['reasoning_fraction']:.0%} | "
          f"multiple choice {report['mcq_fraction']:.0%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Ignore the data directory** — append to `.gitignore`:

```
# run 2 data, uploaded to Kaggle by scripts/push_data.sh, never committed
/data/
```

- [ ] **Step 6: Run the tests and the suite**

Run: `.venv/bin/python -m pytest tests/test_decontam.py tests/test_prepare_data_v2.py -q && .venv/bin/python -m pytest -q`
Expected: 9 passed in the new files; the suite passes.

- [ ] **Step 7: A small real build** (network): `.venv/bin/python -m training.prepare_data_v2 --out /tmp/v2dry --scale 0.02 --show 1`
Expected: the four `eval_*` counts print as 1,273 / 4,183 / 1,000 / 1,089; every source keeps its scaled target (for example `medreason:mcq kept 44`); one rendered example per source shows `<think>\n...\n</think>\n\nAnswer: X` for reasoning rows and `<think>\n\n</think>\n\n` for direct rows. Read every printed example; a wrong mapping stops the plan here.

- [ ] **Step 8: The full build**: `.venv/bin/python -m training.prepare_data_v2 --out data/v2`
Expected: `train.jsonl` near 11,000 rows, `max_seq` at most 3,072, reasoning fraction near 0.75, and any SHORT source understood before continuing.

- [ ] **Step 9: Commit** (code only; `data/` is ignored)

```bash
git add training/decontam.py training/prepare_data_v2.py tests/test_decontam.py tests/test_prepare_data_v2.py .gitignore
git commit -m "Build run 2's data locally: nine sources, decontaminated, measured"
```

---

### Task 4: Training flags for run 2

**Files:**
- Modify: `training/train.py`
- Test: `tests/test_train_v2.py`

**Interfaces:**
- Consumes: `training.format_v2.render_training_text`.
- Produces: `should_stop(*, elapsed: float, limit: int) -> bool`; `loss_curve(log_history) -> list[dict]`; `build_texts(rows, tok, fmt) -> list[str]`; CLI flags `--format {v1,v2}` (default v1), `--group-by-length`, `--stop-after-seconds N` (0 = off), `--probe-abort/--no-probe-abort` (default abort); outputs `loss_curve.json` beside `train_stats.json`, which gains `format`, `lr`, `group_by_length`, `steps`, `max_steps`, `stopped_early`.

- [ ] **Step 1: Write the failing tests** — `tests/test_train_v2.py`:

```python
from __future__ import annotations

from training.train import build_texts, loss_curve, should_stop


class FakeTok:
    def apply_chat_template(self, messages, **kw):
        return f"{kw.get('enable_thinking')}|{messages[-1]['content']}"


ROW = {"id": "q", "source": "s", "kind": "mcq", "question": "Q?",
       "options": {"A": "a", "B": "b"}, "answer": "B", "rationale": None,
       "response": None, "subject": None}


def test_should_stop_only_with_a_positive_limit():
    assert not should_stop(elapsed=10_000, limit=0)
    assert not should_stop(elapsed=99, limit=100)
    assert should_stop(elapsed=100, limit=100)


def test_loss_curve_keeps_logged_losses_only():
    history = [{"loss": 2.5, "epoch": 0.01, "step": 10, "learning_rate": 1e-4},
               {"train_runtime": 5.0},
               {"loss": 2.0, "epoch": 0.02, "step": 20, "learning_rate": 9e-5}]
    assert loss_curve(history) == [
        {"step": 10, "epoch": 0.01, "loss": 2.5, "lr": 1e-4},
        {"step": 20, "epoch": 0.02, "loss": 2.0, "lr": 9e-5}]


def test_build_texts_v2_thinks_only_for_rows_with_reasoning():
    rows = [dict(ROW, reasoning="think"), dict(ROW)]
    assert build_texts(rows, FakeTok(), "v2") == ["True|Answer: B. b", "False|Answer: B. b"]


def test_build_texts_v1_is_run_1s_format():
    assert build_texts([dict(ROW)], FakeTok(), "v1") == ["None|Answer: B"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_train_v2.py -q`
Expected: `ImportError: cannot import name 'build_texts'`.

- [ ] **Step 3: Edit `training/train.py`**

Add the import after `from training.budget import check, project`:

```python
from training.format_v2 import render_training_text
```

Replace `build_dataset` with:

```python
def should_stop(*, elapsed: float, limit: int) -> bool:
    return limit > 0 and elapsed >= limit


def loss_curve(log_history: list[dict]) -> list[dict]:
    return [{"step": h.get("step"), "epoch": h.get("epoch"), "loss": h["loss"],
             "lr": h.get("learning_rate")}
            for h in log_history if "loss" in h]


def build_texts(rows: list[dict], tok, fmt: str) -> list[str]:
    recs = [Record.from_dict(row) for row in rows]
    if fmt == "v2":
        return [render_training_text(tok, rec) for rec in recs]
    return [tok.apply_chat_template(build_messages(rec, with_answer=True), tokenize=False)
            for rec in recs]


def build_dataset(path: Path, tok, fmt: str = "v1"):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return Dataset.from_dict({"text": build_texts(rows, tok, fmt)})
```

Add the flags after `--probe-steps`:

```python
    p.add_argument("--format", choices=("v1", "v2"), default="v1",
                   help="v2: run 2's thinking format (training.format_v2)")
    p.add_argument("--group-by-length", action="store_true",
                   help="batch similar lengths together instead of packing")
    p.add_argument("--stop-after-seconds", type=int, default=0,
                   help="stop cleanly and save once training has run this long (0: off)")
    p.add_argument("--probe-abort", action=argparse.BooleanOptionalAction, default=True,
                   help="abort at --probe-steps when the projection exceeds --budget-seconds")
```

Change the dataset line to `train_ds = build_dataset(args.data / "train.jsonl", tok, args.format)`.

In `Probe.on_step_end`, replace the last three lines (`message = check(...)` through the final print) with:

```python
            message = check(proj, args.budget_seconds)
            if message and args.probe_abort:
                raise SystemExit(message)
            if message:
                print(f"probe warning:{message}\nthe --stop-after-seconds guard "
                      "ends training in time instead", flush=True)
            else:
                print("projection fits the budget; continuing\n", flush=True)
```

After the `Probe` class, add:

```python
    class StopAtBudget(TrainerCallback):
        """End training cleanly once --stop-after-seconds have passed.

        Run 1's probe aborted a whole session at step 50. This keeps the
        steps already trained: the adapter is saved and the report says the
        schedule was cut short."""

        def __init__(self) -> None:
            self.started = time.time()
            self.fired = False

        def on_step_end(self, cfg, state, control, **kw):
            if self.fired or not should_stop(elapsed=time.time() - self.started,
                                             limit=args.stop_after_seconds):
                return
            self.fired = True
            print(f"\ntime guard: {args.stop_after_seconds / 3600:.2f}h reached at step "
                  f"{state.global_step} of {state.max_steps}; stopping and saving",
                  flush=True)
            control.should_training_stop = True
            control.should_save = True

    stopper = StopAtBudget()
```

In `config_kwargs`, after `report_to="none",` add:

```python
        # transformers 5 replaced group_by_length with this field.
        **({"train_sampling_strategy": "group_by_length"} if args.group_by_length else {}),
```

Change `callbacks=[Probe()]` to `callbacks=[Probe(), stopper]`.

Replace the `train_stats.json` block with:

```python
    (args.out / "loss_curve.json").write_text(
        json.dumps(loss_curve(trainer.state.log_history), indent=2))
    (args.out / "train_stats.json").write_text(json.dumps({
        "format": args.format,
        "train_runtime_seconds": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
        "examples": len(train_ds),
        "max_seq": args.max_seq,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
        "lr": args.lr,
        "group_by_length": args.group_by_length,
        "steps": trainer.state.global_step,
        "max_steps": trainer.state.max_steps,
        "stopped_early": stopper.fired,
    }, indent=2))
```

- [ ] **Step 4: Run the tests and the suite**

Run: `.venv/bin/python -m pytest tests/test_train_v2.py -q && .venv/bin/python -m pytest -q`
Expected: 4 passed; the suite passes.

- [ ] **Step 5: Confirm the config field under the pinned stack** (scratch venv with transformers 5.5.0 and trl 0.24.0):

Run: `<pinned-venv>/bin/python -c "from trl import SFTConfig; c = SFTConfig(output_dir='/tmp/x', train_sampling_strategy='group_by_length', max_length=64); print(c.train_sampling_strategy)"`
Expected: `group_by_length`.

- [ ] **Step 6: Commit**

```bash
git add training/train.py tests/test_train_v2.py
git commit -m "Let training run the thinking format, group lengths, and stop in time"
```

---

### Task 5: Evaluation workers and the report

**Files:**
- Modify: `training/evalcore.py`, `tests/test_evalcore.py`
- Create: `training/eval_worker.py`, `training/eval_report.py`, `scripts/smoke_eval_v2.py`
- Test: `tests/test_eval_worker.py`, `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `training.evaluate.resolve_eos_ids`, `sha256_of_file`; `training.modeling.load_model`, `DEFAULT_BASE`; Task 1's rendering and extraction; `training.sources_v2.EVAL_SIZES`.
- Produces: `paired_diff_ci(base_ok, tuned_ok, *, z=1.959964) -> {"diff", "low", "high"}` (and `paired_report` gains `"diff_ci"`); `STAGES`, `REASONING_SIZES`, `parse_stages`, `stage_order`, `fits_before`, `load_benchmark`, `pick_letter`; CLI `python -m training.eval_worker --role {base,tuned} [--adapter DIR] --eval-dir D --out-dir O [--deadline EPOCH] [--think-budget 1536] [--batch-size 16] [--seed 1234] [--stages ...] [--limit N] [--no-size-check] [--device ...]` writing `O/<role>/<mode>__<bench>.jsonl` and `O/<role>/meta.json`; `build_report(eval_dir, bench_dir) -> dict`, `format_table(report) -> str`; CLI `python -m training.eval_report --eval-dir O --bench-dir D --out FILE`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_evalcore.py`:

```python
from training.evalcore import paired_diff_ci


def test_paired_diff_ci_matches_the_paired_wald_formula():
    base = [True] * 50 + [False] * 50
    tuned = [True] * 40 + [False] * 10 + [True] * 30 + [False] * 20
    ci = paired_diff_ci(base, tuned)
    assert ci["diff"] == 0.2
    half = 1.959964 * ((40 - 20 ** 2 / 100) / 100 ** 2) ** 0.5
    assert abs(ci["low"] - (0.2 - half)) < 1e-6 and abs(ci["high"] - (0.2 + half)) < 1e-6


def test_paired_diff_ci_of_nothing_is_zero():
    assert paired_diff_ci([], []) == {"diff": 0.0, "low": 0.0, "high": 0.0}
```

`tests/test_eval_worker.py`:

```python
from __future__ import annotations

import pytest

from training.eval_worker import STAGES, fits_before, parse_stages, pick_letter, stage_order
from training.records import Record


def recs(n: int) -> list[Record]:
    return [Record(id=f"q{i}", source="s", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject=None) for i in range(n)]


def test_letter_stages_keep_every_question_in_file_order():
    assert [r.id for r in stage_order(recs(5), "letter", "medqa", 1)] == [f"q{i}" for i in range(5)]


def test_reasoning_order_is_a_seeded_shuffle_cut_to_size():
    a = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    b = stage_order(recs(2000), "reasoning", "medmcqa", 7)
    assert [r.id for r in a] == [r.id for r in b] and len(a) == 1000
    assert [r.id for r in a] != [f"q{i}" for i in range(1000)]
    assert len(stage_order(recs(1273), "reasoning", "medqa", 7)) == 1273


def test_fits_before():
    assert fits_before(100.0, 50.0, 50.0) and not fits_before(100.0, 50.0, 51.0)


def test_parse_stages():
    assert parse_stages("letter:medqa,reasoning:pubmedqa") == [("letter", "medqa"),
                                                               ("reasoning", "pubmedqa")]
    assert parse_stages("") == list(STAGES)
    with pytest.raises(SystemExit):
        parse_stages("letter:nope")


def test_pick_letter_restricts_to_the_questions_options_and_breaks_ties_early():
    row = {10: 1.0, 11: 5.0, 12: 5.0, 13: 9.0}
    ids = {"A": 10, "B": 11, "C": 12, "D": 13}
    assert pick_letter(row, ids, ["A", "B", "C"]) == "B"
    assert pick_letter(row, ids, ["A", "B", "C", "D"]) == "D"
    with pytest.raises(FloatingPointError):
        pick_letter({10: float("nan"), 11: 0.0}, ids, ["A", "B"])
```

`tests/test_eval_report.py`:

```python
from __future__ import annotations

import json

from training.eval_report import build_report, format_table
from training.records import Record


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_build_report_pairs_only_questions_both_models_scored(tmp_path):
    recs = [Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject="s") for i in range(3)]
    write(tmp_path / "bench" / "eval_medqa.jsonl", [r.to_dict() for r in recs])
    ev = tmp_path / "eval"
    for role in ("base", "tuned"):
        write(ev / role / "meta.json", [])
        (ev / role / "meta.json").write_text(json.dumps({"seed": 1, "limit": 0}))
    write(ev / "base" / "letter__medqa.jsonl", [{"id": "q0", "letter": "B"},
                                                {"id": "q1", "letter": "A"}])
    write(ev / "tuned" / "letter__medqa.jsonl", [{"id": "q0", "letter": "A"},
                                                 {"id": "q1", "letter": "A"},
                                                 {"id": "q2", "letter": "A"}])
    write(ev / "base" / "reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A", "forced": i == 0, "closed": i != 0,
            "new_tokens": 10 * (i + 1)} for i in range(3)])
    write(ev / "tuned" / "reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A", "forced": False, "closed": True,
            "new_tokens": 5} for i in range(3)])

    report = build_report(ev, tmp_path / "bench")
    letter = report["stages"]["letter:medqa"]
    assert (letter["planned"], letter["scored"]) == (3, 2)
    assert (letter["base_accuracy"], letter["tuned_accuracy"]) == (0.5, 1.0)
    reasoning = report["stages"]["reasoning:medqa"]
    assert reasoning["base_counts"] == {"forced": 1, "closed": 2, "median_new_tokens": 20}
    assert set(report["pooled"]) == {"letter", "reasoning"}
    assert "letter:medmcqa" not in report["stages"]
    assert "letter:medqa" in format_table(report)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_evalcore.py tests/test_eval_worker.py tests/test_eval_report.py -q`
Expected: import errors for `paired_diff_ci`, `training.eval_worker`, `training.eval_report`.

- [ ] **Step 3: Add the interval to `training/evalcore.py`** — insert before `paired_report`, and add `"diff_ci": paired_diff_ci(base_ok, tuned_ok),` to the dict `paired_report` returns (after `"mcnemar"`):

```python
def paired_diff_ci(base_ok: list[bool], tuned_ok: list[bool], *,
                   z: float = 1.959964) -> dict:
    """95% interval for tuned-minus-base accuracy on the same questions.

    The paired Wald form: its variance comes only from the questions the two
    models disagree on, which is also what McNemar's test counts."""
    n = len(base_ok)
    if n == 0:
        return {"diff": 0.0, "low": 0.0, "high": 0.0}
    wins = sum((not b) and t for b, t in zip(base_ok, tuned_ok))
    losses = sum(b and (not t) for b, t in zip(base_ok, tuned_ok))
    diff = (wins - losses) / n
    variance = max(0.0, (wins + losses) - (wins - losses) ** 2 / n) / n ** 2
    half = z * variance ** 0.5
    return {"diff": round(diff, 6), "low": round(diff - half, 6),
            "high": round(diff + half, 6)}
```

- [ ] **Step 4: Write `training/eval_worker.py`**

```python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from training.evaluate import resolve_eos_ids, sha256_of_file
from training.format_v2 import (final_letter, forced_suffix, option_letters,
                                render_letter_prompt, render_reasoning_prompt)
from training.modeling import DEFAULT_BASE, load_model
from training.records import Record
from training.sources_v2 import EVAL_SIZES

# Letter choice on everything first (fast, comparable with run 1), then
# reasoning in priority order: a deadline cuts from the end.
STAGES: tuple[tuple[str, str], ...] = (
    ("letter", "medqa"), ("letter", "medmcqa"), ("letter", "pubmedqa"),
    ("letter", "mmlu_medical"),
    ("reasoning", "medqa"), ("reasoning", "mmlu_medical"),
    ("reasoning", "pubmedqa"), ("reasoning", "medmcqa"),
)
REASONING_SIZES = {"medqa": 0, "mmlu_medical": 0, "pubmedqa": 500, "medmcqa": 1000}
# Qwen's recommended thinking-mode sampling. Greedy decoding in thinking mode
# degrades answers and loops, per both Qwen and Unsloth.
SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
ALL_LETTERS = tuple("ABCDEFGHIJ")
# Seconds per batch before the first one is measured.
FIRST_GUESS = {"letter": 15.0, "reasoning": 150.0}


def parse_stages(text: str) -> list[tuple[str, str]]:
    if not text:
        return list(STAGES)
    out = []
    for item in text.split(","):
        mode, _, bench = item.strip().partition(":")
        if (mode, bench) not in STAGES:
            raise SystemExit(f"\nSTOP. Unknown stage {item!r}; known: "
                             f"{[f'{m}:{b}' for m, b in STAGES]}")
        out.append((mode, bench))
    return out


def stage_order(recs: list[Record], mode: str, bench: str, seed: int) -> list[Record]:
    """The questions a stage scores, in scoring order. Reasoning stages use a
    fixed seeded shuffle, so a prefix cut by the deadline is a random sample,
    and both models see the same questions in the same order."""
    if mode == "letter":
        return list(recs)
    order = list(recs)
    random.Random(f"{seed}-{bench}").shuffle(order)
    size = REASONING_SIZES.get(bench, 0)
    return order[:size] if size else order


def fits_before(deadline: float, now: float, est_seconds: float) -> bool:
    return now + est_seconds <= deadline


def load_benchmark(path: Path, bench: str, *, check_size: bool = True) -> list[Record]:
    recs = [Record.from_dict(json.loads(line))
            for line in path.read_text().splitlines() if line]
    if check_size and len(recs) != EVAL_SIZES[bench]:
        raise SystemExit(f"\nSTOP. {path} holds {len(recs):,} questions, expected "
                         f"{EVAL_SIZES[bench]:,}.\nFIX: rebuild data/v2 and re-upload it.")
    bad = [r.id for r in recs if r.kind != "mcq" or r.answer not in (r.options or {})]
    if bad:
        raise SystemExit(f"\nSTOP. {len(bad)} unscorable questions in {path}: {bad[:5]}")
    return recs


def letter_token_ids(tok) -> dict[str, int]:
    return {L: tok.encode(f" {L}", add_special_tokens=False)[-1] for L in ALL_LETTERS}


def pick_letter(row, ids: dict[str, int], letters: list[str]) -> str:
    """Argmax over this question's own letters; ties go to the earliest.
    Non-finite scores mean fp16 overflowed and the result must not be used."""
    scores = {L: float(row[ids[L]]) for L in letters}
    if not all(math.isfinite(s) for s in scores.values()):
        raise FloatingPointError(f"non-finite letter logits {scores}")
    return max(letters, key=lambda L: (scores[L], -letters.index(L)))


def letters_for_prompts(model, tok, prompts, letter_sets, ids) -> list[str]:
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    positions = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        last = model(**enc, position_ids=positions, logits_to_keep=1,
                     use_cache=False).logits[:, -1, :]
    return [pick_letter(row, ids, letters) for row, letters in zip(last, letter_sets)]


def reason_batch(model, tok, recs, *, ids, think_id, stop_ids, budget, seed) -> list[dict]:
    """Think, then answer. A completion that states no letter -- the budget
    ran out mid-thought, or it never wrote 'Answer: X' -- is budget-forced:
    the think block is closed if needed, 'Answer:' is appended, and the
    letter is read from the logits. Both models get the same treatment."""
    import torch

    prompts = [render_reasoning_prompt(tok, r) for r in recs]
    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=budget,
                             pad_token_id=tok.pad_token_id, **SAMPLING)
    width = enc["input_ids"].shape[1]
    rows, to_force = [], []
    for i, (rec, seq) in enumerate(zip(recs, out)):
        new = seq[width:].tolist()
        while new and new[-1] in stop_ids:
            new.pop()
        closed = think_id in new
        letter = None
        if closed:
            after = new[len(new) - new[::-1].index(think_id):]
            letter = final_letter(tok.decode(after, skip_special_tokens=True),
                                  option_letters(rec))
        text = tok.decode(new, skip_special_tokens=False)
        rows.append({"id": rec.id, "letter": letter, "closed": closed,
                     "forced": letter is None, "new_tokens": len(new), "text": text})
        if letter is None:
            to_force.append(i)
    if to_force:
        forced = letters_for_prompts(
            model, tok,
            [prompts[i] + rows[i]["text"] + forced_suffix(rows[i]["closed"]) for i in to_force],
            [option_letters(recs[i]) for i in to_force], ids)
        for i, letter in zip(to_force, forced):
            rows[i]["letter"] = letter
    return rows


def _split_on_oom(fn, group):
    """A long evaluation can fragment GPU memory; halve the batch and retry
    rather than lose the stage."""
    import torch

    try:
        return fn(group)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(group) == 1:
            raise
        mid = len(group) // 2
        return _split_on_oom(fn, group[:mid]) + _split_on_oom(fn, group[mid:])


def run_stage(model, tok, mode, recs, out_path: Path, *, batch_size, budget, seed,
              deadline, rates, ids, think_id, stop_ids) -> tuple[str, int]:
    written = 0
    with out_path.open("w") as fh:
        for start in range(0, len(recs), batch_size):
            group = recs[start:start + batch_size]
            if not fits_before(deadline, time.time(), rates.get(mode, FIRST_GUESS[mode])):
                return "deadline", written
            began = time.time()
            if mode == "letter":
                def fn(g):
                    letters = letters_for_prompts(
                        model, tok, [render_letter_prompt(tok, r) for r in g],
                        [option_letters(r) for r in g], ids)
                    return [{"id": r.id, "letter": L} for r, L in zip(g, letters)]
            else:
                def fn(g, _start=start):
                    return reason_batch(model, tok, g, ids=ids, think_id=think_id,
                                        stop_ids=stop_ids, budget=budget,
                                        seed=seed * 1_000_003 + _start)
            rows = _split_on_oom(fn, group)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            written += len(rows)
            took = time.time() - began
            rates[mode] = took if mode not in rates else 0.8 * rates[mode] + 0.2 * took
    return "done", written


def main() -> None:
    p = argparse.ArgumentParser(description="Score one model, stage by stage, until a deadline.")
    p.add_argument("--role", choices=("base", "tuned"), required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0: none")
    p.add_argument("--think-budget", type=int, default=1536)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--stages", default="")
    p.add_argument("--limit", type=int, default=0, help="questions per stage (smoke runs)")
    p.add_argument("--no-size-check", action="store_true")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = p.parse_args()

    import peft
    import torch
    import transformers

    if args.role == "tuned" and not args.adapter:
        raise SystemExit("\nSTOP. --role tuned needs --adapter.")
    adapter = args.adapter if args.role == "tuned" else None
    stages = parse_stages(args.stages)
    deadline = args.deadline if args.deadline > 0 else float("inf")
    out_dir = args.out_dir / args.role
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"role": args.role, "base": args.base, "adapter": adapter,
            "seed": args.seed, "think_budget": args.think_budget,
            "batch_size": args.batch_size, "deadline": args.deadline,
            "limit": args.limit, "started": time.time(), "status": "running",
            "stages": {}}

    def save_meta() -> None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    save_meta()
    began = time.time()
    model, tok, choice = load_model(args.base, adapter, device=args.device)
    meta["load_seconds"] = round(time.time() - began, 1)
    meta["environment"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__, "device": choice.device, "dtype": choice.dtype_name,
        "device_name": (torch.cuda.get_device_name(0) if choice.device == "cuda"
                        else choice.device)}
    if adapter:
        meta["adapter_sha256"] = sha256_of_file(Path(adapter) / "adapter_model.safetensors")
    ids = letter_token_ids(tok)
    think_id = tok.convert_tokens_to_ids("</think>")
    stop_ids = resolve_eos_ids(getattr(model.generation_config, "eos_token_id", None),
                               tok.eos_token_id)
    if tok.pad_token_id is not None:
        stop_ids.add(tok.pad_token_id)
    save_meta()

    rates: dict[str, float] = {}
    benches: dict[str, list[Record]] = {}
    try:
        for mode, bench in stages:
            if bench not in benches:
                benches[bench] = load_benchmark(args.eval_dir / f"eval_{bench}.jsonl", bench,
                                                check_size=not args.no_size_check)
            recs = stage_order(benches[bench], mode, bench, args.seed)
            if args.limit:
                recs = recs[: args.limit]
            key = f"{mode}:{bench}"
            began = time.time()
            status, written = run_stage(
                model, tok, mode, recs, out_dir / f"{mode}__{bench}.jsonl",
                batch_size=args.batch_size, budget=args.think_budget, seed=args.seed,
                deadline=deadline, rates=rates, ids=ids, think_id=think_id,
                stop_ids=stop_ids)
            meta["stages"][key] = {"planned": len(recs), "scored": written,
                                   "status": status,
                                   "seconds": round(time.time() - began, 1)}
            save_meta()
            print(f"{args.role} {key}: {written:,}/{len(recs):,} {status} "
                  f"in {time.time() - began:.0f}s", flush=True)
            if status == "deadline":
                meta["status"] = "deadline"
                break
        else:
            meta["status"] = "finished"
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["finished"] = time.time()
        save_meta()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `training/eval_report.py`**

```python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from training.eval_worker import STAGES, load_benchmark, stage_order
from training.evalcore import paired_report
from training.records import Record


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {row["id"]: row for row in rows}


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pair(recs: list[Record], base_rows: dict, tuned_rows: dict):
    kept = [r for r in recs if r.id in base_rows and r.id in tuned_rows]
    return (kept, [base_rows[r.id]["letter"] for r in kept],
            [tuned_rows[r.id]["letter"] for r in kept])


def reasoning_counts(rows: dict, kept: list[Record]) -> dict:
    picked = [rows[r.id] for r in kept]
    tokens = [row.get("new_tokens", 0) for row in picked]
    return {"forced": sum(bool(row.get("forced")) for row in picked),
            "closed": sum(bool(row.get("closed")) for row in picked),
            "median_new_tokens": statistics.median(tokens) if tokens else 0}


def build_report(eval_dir: Path, bench_dir: Path) -> dict:
    """Pair the two workers stage by stage, on the questions both scored."""
    metas = {role: read_meta(eval_dir / role / "meta.json") for role in ("base", "tuned")}
    seed = metas["tuned"].get("seed", metas["base"].get("seed", 1234))
    limit = metas["tuned"].get("limit") or metas["base"].get("limit") or 0
    benches: dict[str, list[Record]] = {}
    stages: dict[str, dict] = {}
    pooled_in: dict[str, tuple[list, list, list]] = {"letter": ([], [], []),
                                                     "reasoning": ([], [], [])}
    for mode, bench in STAGES:
        base_rows = read_rows(eval_dir / "base" / f"{mode}__{bench}.jsonl")
        tuned_rows = read_rows(eval_dir / "tuned" / f"{mode}__{bench}.jsonl")
        if not base_rows and not tuned_rows:
            continue
        if bench not in benches:
            benches[bench] = load_benchmark(bench_dir / f"eval_{bench}.jsonl", bench,
                                            check_size=False)
        planned = stage_order(benches[bench], mode, bench, seed)
        if limit:
            planned = planned[:limit]
        kept, base_letters, tuned_letters = pair(planned, base_rows, tuned_rows)
        rep = paired_report(kept, base_letters, tuned_letters, mode=mode)
        rep.update({"benchmark": bench, "planned": len(planned), "scored": len(kept)})
        if mode == "reasoning":
            rep["base_counts"] = reasoning_counts(base_rows, kept)
            rep["tuned_counts"] = reasoning_counts(tuned_rows, kept)
        stages[f"{mode}:{bench}"] = rep
        for bucket, values in zip(pooled_in[mode], (kept, base_letters, tuned_letters)):
            bucket.extend(values)
    pooled = {}
    for mode, (kept, base_letters, tuned_letters) in pooled_in.items():
        if kept:
            rep = paired_report(kept, base_letters, tuned_letters, mode=mode)
            rep.pop("by_subject", None)
            pooled[mode] = rep
    return {"stages": stages, "pooled": pooled, "models": metas}


def format_table(report: dict) -> str:
    lines = [f"{'stage':<24}{'scored':>13}{'base':>8}{'tuned':>8}{'change':>8}"
             f"{'p':>10}  95% CI of the change"]
    rows = list(report["stages"].items())
    rows += [(f"pooled {mode}", rep) for mode, rep in report["pooled"].items()]
    for key, rep in rows:
        ci = rep["diff_ci"]
        scored = f"{rep.get('scored', rep['n']):,}/{rep.get('planned', rep['n']):,}"
        change = (rep["tuned_accuracy"] - rep["base_accuracy"]) * 100
        lines.append(f"{key:<24}{scored:>13}{rep['base_accuracy']:>8.1%}"
                     f"{rep['tuned_accuracy']:>8.1%}{change:>+8.1f}"
                     f"{rep['mcnemar']['p_value']:>10.4f}"
                     f"  [{ci['low'] * 100:+.1f}, {ci['high'] * 100:+.1f}]")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Pair base and tuned predictions into a report.")
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = build_report(args.eval_dir, args.bench_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(format_table(report))
    for role, meta in report["models"].items():
        print(f"{role}: {meta.get('status', 'missing')}", meta.get("error", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and the suite**

Run: `.venv/bin/python -m pytest tests/test_evalcore.py tests/test_eval_worker.py tests/test_eval_report.py -q && .venv/bin/python -m pytest -q`
Expected: every new test passes; the suite passes.

- [ ] **Step 7: Write `scripts/smoke_eval_v2.py`**

```python
"""Run 2's evaluation, end to end, on a small model before any GPU time.

Qwen3-0.6B with a random LoRA adapter on CPU. Every stage runs with four
questions and a 24-token thinking budget, so the budget-forcing path runs too;
then the report pairs the two workers. Also checks the real Qwen3 template
renders run 2's training format.
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
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.eval_worker import STAGES, letter_token_ids
    from training.format_v2 import render_training_text
    from training.records import Record

    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="smoke-v2-"))
    tok = AutoTokenizer.from_pretrained(TINY)

    # 1. The real template renders both training formats.
    four = {"A": "Aspirin", "B": "Heparin", "C": "Warfarin", "D": "Insulin"}
    thinking = Record(id="t", source="s", kind="mcq", question="Which anticoagulant is IV?",
                      options=four, answer="B", rationale=None, response=None,
                      subject=None, reasoning="Heparin is given IV.")
    direct = Record(**{**thinking.to_dict(), "reasoning": None})
    text = render_training_text(tok, thinking)
    assert "<think>\nHeparin is given IV.\n</think>\n\nAnswer: B" in text, text[-160:]
    assert "<think>\n\n</think>\n\nAnswer: B" in render_training_text(tok, direct)
    think_id = tok.convert_tokens_to_ids("</think>")
    assert tok.decode([think_id], skip_special_tokens=False) == "</think>"
    ids = letter_token_ids(tok)
    assert len(set(ids.values())) == 10, ids
    print("ok  training format, </think> token and ten distinct letter tokens")

    # 2. Benchmarks: four questions each; PubMedQA keeps its three options.
    questions = [("Which vitamin deficiency causes scurvy?", "C"),
                 ("Which organ secretes insulin?", "B"),
                 ("Which bone is in the thigh?", "A"),
                 ("Which chamber pumps blood to the body?", "D")]
    for bench in ("medqa", "medmcqa", "mmlu_medical"):
        rows = [Record(id=f"{bench}-{i}", source=bench, kind="mcq", question=q,
                       options=four, answer=a, rationale=None, response=None,
                       subject="Smoke").to_dict() for i, (q, a) in enumerate(questions)]
        (work / f"eval_{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pubmed = [Record(id=f"pubmedqa-{i}", source="pubmedqa", kind="mcq",
                     question=f"Context: A trial.\n\nQuestion: Does drug {i} work?",
                     options={"A": "yes", "B": "no", "C": "maybe"}, answer="A",
                     rationale=None, response=None, subject=None).to_dict()
              for i in range(4)]
    (work / "eval_pubmedqa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pubmed))

    # 3. A random adapter, so the fine-tune genuinely differs from the base.
    torch.manual_seed(42)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16,
                                                 target_modules=TARGETS, lora_dropout=0.0))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    adapter = work / "adapter"
    peft_model.save_pretrained(adapter)
    del peft_model, base

    # 4. Both workers, every stage, then the report.
    common = [sys.executable, "-m", "training.eval_worker", "--base", TINY,
              "--eval-dir", str(work), "--out-dir", str(work / "eval"),
              "--think-budget", "24", "--batch-size", "2", "--limit", "4",
              "--no-size-check", "--device", "cpu"]
    for role in ("tuned", "base"):
        extra = ["--role", role] + (["--adapter", str(adapter)] if role == "tuned" else [])
        subprocess.run(common + extra, check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "training.eval_report", "--eval-dir",
                    str(work / "eval"), "--bench-dir", str(work), "--out",
                    str(work / "report.json")], check=True, cwd=ROOT)

    report = json.loads((work / "report.json").read_text())
    assert set(report["stages"]) == {f"{m}:{b}" for m, b in STAGES}, report["stages"].keys()
    for key, rep in report["stages"].items():
        assert rep["scored"] == rep["planned"] == 4, (key, rep["scored"], rep["planned"])
    assert report["stages"]["reasoning:medqa"]["tuned_counts"]["forced"] >= 1
    for role in ("base", "tuned"):
        assert report["models"][role]["status"] == "finished", report["models"][role]
        rows = [json.loads(line) for line in
                (work / "eval" / role / "reasoning__pubmedqa.jsonl").read_text().splitlines()]
        assert {r["letter"] for r in rows} <= {"A", "B", "C"}, rows
    print("ok  every stage paired 4/4, budget forcing used, PubMedQA letters within A-C")
    print(f"\nSMOKE V2 PASSED in {time.time() - started:.0f}s  ({work})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the smoke test in both environments**

Run: `.venv/bin/python scripts/smoke_eval_v2.py`
Then in the pinned venv, with torchao absent: `<pinned-venv>/bin/python scripts/smoke_eval_v2.py`
Expected: `SMOKE V2 PASSED` both times. A failure here is fixed before anything else.

- [ ] **Step 9: Commit**

```bash
git add training/evalcore.py training/eval_worker.py training/eval_report.py scripts/smoke_eval_v2.py tests/test_evalcore.py tests/test_eval_worker.py tests/test_eval_report.py
git commit -m "Evaluate run 2 in both answer modes, stage by stage until a deadline"
```

---

### Task 6: The Kaggle notebook and uploads

**Files:**
- Modify: `training/kaggle_paths.py`, `scripts/build_notebook.py`, `tests/test_kaggle_paths.py`, `tests/test_kernel_metadata.py`
- Create: `run2/kernel-metadata.json`, `scripts/push_data.sh`

**Interfaces:**
- Consumes: every earlier task; `scripts/kaggle_dataset.sh`; `scripts/push_kaggle.sh <kernel-dir>`.
- Produces: `RUN2_REQUIRED`, `DATA_FILES`, `find_data_dir(root) -> Path`, `data_fingerprint(dir) -> str`; `run2/kaggle_run2.ipynb` (built only when `data/v2/data_report.json` exists); `bash scripts/push_data.sh [dir]`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kaggle_paths.py`:

```python
from training.kaggle_paths import DATA_FILES, RUN2_REQUIRED, data_fingerprint, find_data_dir


def test_find_data_dir_requires_every_data_file(tmp_path):
    d = tmp_path / "datasets" / "gb1105" / "medical-ft-data"
    d.mkdir(parents=True)
    for name in DATA_FILES[:-1]:
        (d / name).write_text("x")
    with pytest.raises(SystemExit):
        find_data_dir(tmp_path)
    (d / DATA_FILES[-1]).write_text("{}")
    assert find_data_dir(tmp_path) == d


def test_data_fingerprint_changes_when_one_file_changes(tmp_path):
    for name in DATA_FILES:
        (tmp_path / name).write_text(name)
    before = data_fingerprint(tmp_path)
    (tmp_path / "train.jsonl").write_text("changed")
    assert data_fingerprint(tmp_path) != before


def test_every_run2_module_exists():
    root = Path(__file__).resolve().parents[1] / "training"
    assert all((root / f"{name}.py").exists() for name in RUN2_REQUIRED)
```

(Add `from pathlib import Path` to that file's imports.) In `tests/test_kernel_metadata.py`, add `"run2/kernel-metadata.json"` to both parametrize lists, and append:

```python
def test_the_run2_kernel_attaches_code_and_data_on_a_private_gpu():
    meta = json.loads(Path("run2/kernel-metadata.json").read_text())
    assert meta["id"] == "gb1105/qwen3-4b-medical-fine-tune-run-2"
    assert meta["dataset_sources"] == ["gb1105/medical-ft-code", "gb1105/medical-ft-data"]
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kaggle_run2.ipynb"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_kaggle_paths.py tests/test_kernel_metadata.py -q`
Expected: import errors and a missing metadata file.

- [ ] **Step 3: Extend `training/kaggle_paths.py`** — below the marker, after `EVAL_REQUIRED`:

```python
RUN2_REQUIRED = {"records", "prompts", "sources", "sources_v2", "format_v2",
                 "evalcore", "evaluate", "modeling", "train", "budget",
                 "eval_worker", "eval_report", "kaggle_paths"}

DATA_FILES = ("train.jsonl", "eval_medqa.jsonl", "eval_medmcqa.jsonl",
              "eval_pubmedqa.jsonl", "eval_mmlu_medical.jsonl", "data_report.json")
```

and at the end of the file:

```python
def find_data_dir(root: Path) -> Path:
    """Locate the uploaded run 2 data by content, like find_code_dir."""
    if not root.exists():
        raise SystemExit(
            "\nSTOP. /kaggle/input does not exist -- no inputs are attached.\n"
            "FIX: sidebar -> + Add Input -> Datasets -> medical-ft-data.")
    found = sorted({p.parent for p in root.rglob("train.jsonl")
                    if all((p.parent / name).exists() for name in DATA_FILES)})
    if found:
        return found[0]
    tree = sorted(str(p.relative_to(root)) for p in root.rglob("*"))[:30]
    raise SystemExit(
        f"\nSTOP. No directory under {root} holds all of {list(DATA_FILES)}.\n"
        f"First entries under /kaggle/input: {tree}\n"
        "FIX: run scripts/push_data.sh, then attach medical-ft-data in the sidebar.")


def data_fingerprint(directory: Path) -> str:
    """The same idea as code_fingerprint, for the frozen data files."""
    import hashlib

    digest = hashlib.sha256()
    for name in DATA_FILES:
        digest.update(name.encode() + b"\0" + (directory / name).read_bytes() + b"\0")
    return digest.hexdigest()[:16]
```

- [ ] **Step 4: Create `run2/kernel-metadata.json`**

```json
{
  "id": "gb1105/qwen3-4b-medical-fine-tune-run-2",
  "title": "Qwen3-4B Medical Fine-tune Run 2",
  "code_file": "kaggle_run2.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "gb1105/medical-ft-code",
    "gb1105/medical-ft-data"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- [ ] **Step 5: Create `scripts/push_data.sh`** (and `chmod +x`)

```bash
#!/usr/bin/env bash
# Upload run 2's prepared data as a private Kaggle Dataset.
#
#   bash scripts/push_data.sh [data-dir]    # default data/v2
#
# Build it first: python -m training.prepare_data_v2 --out data/v2
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:-data/v2}"
FILES=(train.jsonl eval_medqa.jsonl eval_medmcqa.jsonl eval_pubmedqa.jsonl
       eval_mmlu_medical.jsonl data_report.json)
for f in "${FILES[@]}"; do
  [ -s "$DATA/$f" ] || { echo "missing $DATA/$f -- run training.prepare_data_v2 first"; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do cp "$DATA/$f" "$STAGE/"; done
bash scripts/kaggle_dataset.sh medical-ft-data "Medical Fine-tune Data" "$STAGE"
```

- [ ] **Step 6: Add the run 2 notebook to `scripts/build_notebook.py`.** Import the data helpers next to `code_fingerprint`:

```python
from training.kaggle_paths import code_fingerprint, data_fingerprint
```

Then, after `EVAL_CELLS` and before `write_notebook`, add:

```python
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
    proc = subprocess.Popen(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
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

# A failing command returns non-zero without raising in Jupyter. No shell:
# every command here is built from constants, so it splits into a plain list.
def step(cmd: str, env: dict | None = None):
    print(f"$ {cmd}\\n", flush=True)
    p = subprocess.run(shlex.split(cmd), env={**os.environ, **(env or {})})
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
```

At the end of the file, after the two existing `write_notebook` calls:

```python
if (RUN2_DATA / "data_report.json").exists():
    write_notebook(run2_cells(data_fingerprint(RUN2_DATA)), Path("run2/kaggle_run2.ipynb"))
else:
    print("skipped run2/kaggle_run2.ipynb: no data/v2 (run training.prepare_data_v2)")
```

- [ ] **Step 7: Run the tests and the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: every test passes.

- [ ] **Step 8: Build and validate the notebooks**

Run: `.venv/bin/python scripts/build_notebook.py`, then parse every code cell of `run2/kaggle_run2.ipynb` with `ast.parse` (skipping `%` and `!` lines); grep it for the current `code_fingerprint(Path("training"))` and `data_fingerprint(Path("data/v2"))`; and `bash -n scripts/push_data.sh`.
Expected: `wrote run2/kaggle_run2.ipynb, 12 cells`; all cells parse; both fingerprints appear once each.

Then rehearse the verify cell's torchao repair in the pinned venv: install `torchao==0.10.0`, run the cell's `LORA_CHECK` snippet (it must fail naming torchao), uninstall torchao, and run it again (it must pass). This is exactly the path cell 2 takes on Kaggle.

- [ ] **Step 9: Commit**

```bash
git add training/kaggle_paths.py scripts/build_notebook.py scripts/push_data.sh run2/ tests/test_kaggle_paths.py tests/test_kernel_metadata.py training/kaggle_medical.ipynb evaluation/kaggle_eval.ipynb
git commit -m "Add run 2's one-session Kaggle notebook and its data upload"
```

---

### Task 7: Independent review before the launch

- [ ] **Step 1:** Generate a review package of the whole range (the commit before Task 1 to HEAD) and dispatch ONE reviewer on the most capable model. Brief it with the spec, this plan, the diff, and the Review Focus list. Ask it to check against the installed pinned stack rather than memory; to confirm the data report and five rendered examples look right; and to say plainly whether the notebook is safe for its one Kaggle start.
- [ ] **Step 2:** Fix every Critical or Important finding. Re-run the suite and both smoke runs, rebuild the notebooks, and commit.

### Task 8: The launch and the record

- [ ] **Step 1:** `bash scripts/push_data.sh` (waits until Kaggle serves exactly the uploaded files).
- [ ] **Step 2:** `bash scripts/push_kaggle.sh run2` (rebuilds the notebooks, uploads the code, pushes the kernel, and fails on a push error).
- [ ] **Step 3:** Confirm `kaggle kernels status` shows RUNNING after 10 minutes, past setup and into training. Start the background poller.
- [ ] **Step 4:** Save the notebook exactly as pushed to `results/run2/kaggle_run2_notebook.ipynb`. Update `TODO.md` and the README status, then commit and push to GitHub.
- [ ] **Step 5:** When the kernel completes, download its output into `results/run2/`. Record the result honestly in `TODO.md`, the README and `docs/evaluation.md`, then commit and push.
