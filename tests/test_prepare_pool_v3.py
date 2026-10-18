from __future__ import annotations

from training.prepare_pool_v3 import split_pool
from training.records import Record


def mcq(i: int, source: str = "medqa", question: str | None = None) -> Record:
    return Record(id=f"{source}-{i}", source=source, kind="mcq",
                  question=question or f"{source} question number {i} about a patient",
                  options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer="A",
                  rationale=None, response=None, subject=None)


def test_split_pool_keeps_dev_and_pool_apart_and_renames_ids():
    medqa = [mcq(i) for i in range(10)]
    medmcqa = [mcq(i, "medmcqa") for i in range(10)]
    out, stats = split_pool({"medqa": medqa, "medmcqa": medmcqa}, eval_questions=[],
                            dev_size={"medqa": 2, "medmcqa": 3},
                            pool_size={"medqa": 5, "medmcqa": 4})
    assert [len(out[k]) for k in ("dev_medqa", "dev_medmcqa", "pool")] == [2, 3, 9]
    ids = [r.id for k in out for r in out[k]]
    assert len(ids) == len(set(ids))
    # training-split ids must never look like benchmark ids ("medqa-12")
    assert all(r.id.startswith(("medqatrain-", "medmcqatrain-")) for k in out for r in out[k])
    assert stats["medqa"]["dev"] == 2 and stats["medqa"]["pool"] == 5


def test_benchmark_copies_and_duplicates_are_dropped():
    text = ("a 45 year old man presents with crushing chest pain radiating to the left arm "
            "and jaw for two hours with sweating and nausea")
    medqa = [mcq(0, question=text), mcq(1, question=text), mcq(2)]
    out, stats = split_pool({"medqa": medqa, "medmcqa": []}, eval_questions=[text],
                            dev_size={"medqa": 1, "medmcqa": 0},
                            pool_size={"medqa": 5, "medmcqa": 0})
    assert stats["medqa"]["contaminated"] == 2
    assert [r.question for r in out["dev_medqa"]] == [mcq(2).question]
    assert out["pool"] == []


def test_a_pool_question_copying_a_dev_question_is_dropped():
    text = ("a 30 year old woman has fever and a new heart murmur after a dental procedure "
            "two weeks ago and splinter haemorrhages")
    medqa = [mcq(0, question=text), mcq(1, question=text + " now"), mcq(2)]
    out, stats = split_pool({"medqa": medqa, "medmcqa": []}, eval_questions=[],
                            dev_size={"medqa": 1, "medmcqa": 0},
                            pool_size={"medqa": 5, "medmcqa": 0})
    assert out["dev_medqa"][0].question == text
    assert [r.question for r in out["pool"]] == [mcq(2).question]
    assert stats["medqa"]["near_dev"] == 1
