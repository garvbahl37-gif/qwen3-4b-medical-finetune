from __future__ import annotations

from collections import Counter

from training.prepare_data import normalise_question

NGRAM = 13
# A training row counts as a copy of a benchmark question when it contains at
# least this share of that question's 13-word sequences.
MIN_SHARE = 0.5


def ngrams(norm_text: str, n: int = NGRAM) -> set[int]:
    words = norm_text.split()
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


class EvalIndex:
    """Every benchmark question, for exact and 13-word-overlap lookups.

    Exact matching alone misses a benchmark vignette copied with a new first
    line, and MedReason and ReasonMed both draw on the same exam banks the
    benchmarks come from. But a single shared 13-word run is not evidence of
    a copy here: USMLE vignettes are templated, and on real data any-overlap
    flagged 31% of MedQA *training* questions against the MedQA test set on
    stock phrases such as "vital signs are within normal limits physical
    examination is unremarkable which of". A copy shares most of a question's
    13-word runs; a stock phrase shares one or two. So a row is contaminated
    when it holds at least MIN_SHARE of some benchmark question's runs.
    Questions shorter than 13 words need an exact match.
    """

    def __init__(self, questions: list[str], n: int = NGRAM,
                 min_share: float = MIN_SHARE) -> None:
        self.n = n
        self.min_share = min_share
        self.exact: set[str] = set()
        self.postings: dict[int, list[int]] = {}
        self.sizes: list[int] = []
        for j, question in enumerate(questions):
            key = normalise_question(question)
            self.exact.add(key)
            grams = ngrams(key, n)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings.setdefault(gram, []).append(j)

    def hits(self, question: str) -> str | None:
        key = normalise_question(question)
        if key in self.exact:
            return "exact"
        shared: Counter[int] = Counter()
        for gram in ngrams(key, self.n):
            for j in self.postings.get(gram, ()):
                shared[j] += 1
        if any(count >= self.min_share * self.sizes[j] for j, count in shared.items()):
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
