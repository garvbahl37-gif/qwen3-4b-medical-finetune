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

# Matches "Answer: C", "**Answer:** C", "answer - a". The letter must not be
# followed by another letter, so "Answer: About" does not yield "A".
_ANSWER_LINE = re.compile(
    r"answer\s*\**\s*[:\-–]\s*\**\s*([A-D])(?![A-Za-z])", re.IGNORECASE
)
# A letter standing alone as a choice marker: "(C)", "C.", "C)", " C ".
_BARE_LETTER = re.compile(r"(?:^|[\s(\[])([A-D])(?=[\s.):\]]|$)")


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

    Deliberately lenient. The base model wraps answers in prose and markdown,
    and penalising formatting rather than correctness would distort the
    base-vs-tuned comparison. The last explicit answer line wins, because a
    model that reconsiders states its conclusion last.
    """
    if not text:
        return None
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].upper()
    bare = _BARE_LETTER.findall(text)
    return bare[0].upper() if bare else None
