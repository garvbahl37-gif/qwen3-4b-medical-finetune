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
