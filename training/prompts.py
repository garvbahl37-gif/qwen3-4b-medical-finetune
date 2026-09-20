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

# Cue words that mark a stated answer. Leniency is anchored to these rather
# than to "the first A-D anywhere": the unanchored form returned "A" for
# "A 45-year-old man ... choice C.", and medical vignettes open that way.
_CUE = r"(?:answers?|options?|choice|choose|select(?:ed)?|correct|best)"
# Forward: cue then letter ("the correct option is D").
_CUE_THEN_LETTER = re.compile(
    r"\b" + _CUE + r"\b[^\n]{0,28}?\b([A-D])\b(?![A-Za-z])", re.IGNORECASE)
# Reverse: letter then cue ("C is the correct choice"). Without this the
# extractor misses the base model, whose phrasing varies most -- and
# understating the base score would flatter the fine-tune.
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
