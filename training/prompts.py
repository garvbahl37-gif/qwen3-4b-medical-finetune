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

# Leniency is anchored to answer-indicating language rather than "the first
# A-D anywhere". The base model wraps answers in prose, and the earlier
# unanchored form returned "A" for "A 45-year-old man ... choice C." —
# medical vignettes open that way, and "Vitamin D" / "Hepatitis B" /
# "blood group A" are everywhere in this domain.
_ANSWER_LINE = re.compile(
    r"(?:answer|option|choice|select(?:ed)?|correct)\b[^A-Za-z0-9\n]{0,10}"
    r"(?:(?:is|was|:)[^A-Za-z0-9\n]{0,5})?([A-D])(?![A-Za-z])",
    re.IGNORECASE,
)
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
    None costs both models equally. So the fallback only fires on the
    last non-empty line, and only when that line is itself a choice.

    The last explicit match wins, because a model that reconsiders states
    its conclusion last.
    """
    if not text:
        return None
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].upper()
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        for pattern in (_FINAL_BARE, _FINAL_MARKER):
            found = pattern.match(lines[-1])
            if found:
                return found.group(1).upper()
    return None
