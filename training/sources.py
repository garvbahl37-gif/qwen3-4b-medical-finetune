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
                      require_rationale: bool = True) -> Record | None:
    # MedMCQA marks a third of its rows choice_type != "single" while still
    # exposing a single `cop` index. Those are the known-noisy rows; drop them.
    if _clean(raw.get("choice_type")).lower() != "single":
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
         require_rationale: bool = True) -> list[Record]:
    """Fetch a source from the Hub and normalise it. Requires network."""
    from datasets import load_dataset  # imported here so tests stay offline

    hf_id, config, default_split = SOURCES[name]
    ds = load_dataset(hf_id, config, split=split or default_split)

    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    normalise = _NORMALISERS[name]
    extra = {"require_rationale": require_rationale} if name == "medmcqa" else {}
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
    return kept
