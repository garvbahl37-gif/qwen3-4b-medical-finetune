from __future__ import annotations

import re

from training.prompts import SYSTEM_CHAT, extract_letter
from training.records import Record

SYSTEM_MCQ_V2 = (
    "You are a medical education assistant. Work through the question, then "
    "give your choice first, on its own line, in the exact form 'Answer: X', "
    "followed by a brief explanation."
)

_ANSWER_LINE = re.compile(r"Answer:\s*\(?([A-J])\b")
# A sentence ends at . ! or ? followed by space and a capital -- unless the
# word before it is an abbreviation ("Dr.", "p. 373", "i.e.").
_SENTENCE_BREAK = re.compile(r"[.!?]\s+(?=[A-Z(\[\"'])")
_ABBREVIATIONS = {"ans", "dr", "vs", "fig", "ref", "etc", "approx", "no", "mr", "mrs",
                  "st", "p", "pp", "i.e", "e.g", "viz", "cf"}
# MedMCQA explanations open by restating the answer: "Ans. is 'c' i.e., ...".
_LEADING_ANSWER = re.compile(
    r"^\s*ans(?:wer)?\s*[.:]?\s*(?:is\s*)?[-:]?\s*['\"(\[]?[a-e]['\")\]]?\s*"
    r"(?:i\s*\.?\s*e\s*\.?)?\s*[,:;\-]?\s*", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    out, start = [], 0
    for brk in _SENTENCE_BREAK.finditer(text):
        words = text[start:brk.start()].split()
        last = words[-1].lower() if words else ""
        if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        out.append(text[start:brk.start() + 1].strip())
        start = brk.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


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
    text = _LEADING_ANSWER.sub("", text, count=1).strip()
    if not text:
        return ""
    out = " ".join(_sentences(text)[:max_sentences]).strip()
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


def v2_messages(rec: Record, *, with_answer: bool, think: bool | None = None) -> list[dict]:
    """The conversation, with the mode stated in the user turn.

    Every prompt ends with Qwen3's own switch, /think or /no_think -- the
    control the Qwen3 report trained its hybrid model with. Run 2 left the
    switch out, and the model learned to guess the mode from the question's
    style: it skipped thinking on 971 of 1,273 MedQA questions. A training
    row thinks when it has reasoning; a prompt says what it asks for.
    """
    if think is None:
        think = bool(rec.reasoning)
    system = SYSTEM_MCQ_V2 if rec.kind == "mcq" else SYSTEM_CHAT
    switch = "/think" if think else "/no_think"
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"{format_question_v2(rec)}\n\n{switch}"}]
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
    return tok.apply_chat_template(v2_messages(rec, with_answer=False, think=False),
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "Answer:"


def render_reasoning_prompt(tok, rec: Record) -> str:
    """Thinking on: the prompt ends at the assistant turn and the model opens
    its own <think> block."""
    return tok.apply_chat_template(v2_messages(rec, with_answer=False, think=True),
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
