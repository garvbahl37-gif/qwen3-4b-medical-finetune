from __future__ import annotations

from dataclasses import asdict, dataclass


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

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(**d)
