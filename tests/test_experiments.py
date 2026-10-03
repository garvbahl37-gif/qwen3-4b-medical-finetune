from __future__ import annotations

import csv
import json
from pathlib import Path

from training.experiments import COLUMNS, rows_for
from training.experiments_config import E0
from training.records import Record

PHASE_13 = ["experiment_id", "git_commit", "dataset_version", "model", "quantization",
            "lora_rank", "lora_alpha", "learning_rate", "epochs", "batch_size",
            "gradient_accumulation", "max_seq_length", "warmup_ratio", "weight_decay",
            "train_loss", "validation_loss", "medqa_reasoning", "medqa_letter", "medmcqa",
            "pubmedqa", "mmlu_medical", "general_score", "safety_score",
            "median_thinking_length", "notes"]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def bench(tmp_path: Path) -> Path:
    recs = [Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None,
                   response=None, subject="s") for i in range(4)]
    write(tmp_path / "bench" / "eval_medqa.jsonl", [r.to_dict() for r in recs])
    return tmp_path / "bench"


def test_the_csv_starts_with_phase_13s_columns_in_order():
    assert COLUMNS[:len(PHASE_13)] == PHASE_13


def test_scores_are_recomputed_on_questions_every_model_answered(tmp_path):
    b = bench(tmp_path)
    write(tmp_path / "p/base/reasoning__medqa.jsonl",
          [{"id": f"q{i}", "letter": "A", "sample": s, "think_tokens": 100} for s in (0, 1)
           for i in range(4)])
    write(tmp_path / "p/ft/reasoning__medqa.jsonl",
          [{"id": "q0", "letter": "B", "sample": 0, "new_tokens": 7},
           {"id": "q1", "letter": "A", "sample": 0, "new_tokens": 9}])
    (tmp_path / "train_stats.json").write_text(json.dumps({"train_loss": 0.123456}))
    exp = {"experiment_id": "x", "git_commit": "abc", "training": {"lora_rank": 16},
           "artifacts": {"train_stats": "train_stats.json"},
           "evaluation": {"format": "v2_predictions", "protocol": "p", "models": {
               "base": {"label": "Base", "dirs": ["p/base"], "quantization": "fp16"},
               "ft": {"label": "FT", "dirs": ["p/ft", "p/missing"], "trained": True}}}}
    base, ft = rows_for(exp, tmp_path, b)
    # the fine-tune answered q0 and q1 only, so both are scored on those two
    assert base["medqa_reasoning"] == 1.0 and ft["medqa_reasoning"] == 0.5
    assert base["n_medqa_reasoning"] == ft["n_medqa_reasoning"] == 2
    assert base["medqa_reasoning_any_correct"] == 1.0 and ft["medqa_reasoning_any_correct"] == ""
    assert base["thinking_length_kind"] == "think_tokens" and base["median_thinking_length"] == 100
    assert ft["thinking_length_kind"].startswith("new_tokens") and ft["median_thinking_length"] == 7
    assert ft["train_loss"] == 0.1235 and ft["lora_rank"] == 16
    assert base["train_loss"] == "" and base["lora_rank"] == "" and base["quantization"] == "fp16"


def test_missing_predictions_leave_scores_empty(tmp_path):
    b = bench(tmp_path)
    exp = {"experiment_id": "E", "evaluation": {"format": "v2_predictions", "models": {
        "base": {"label": "A", "dirs": ["nowhere"]}}}}
    (row,) = rows_for(exp, tmp_path, b)
    assert row["medqa_reasoning"] == "" and row["medqa_letter"] == ""


def test_run_1s_evaluation_json_is_read_per_role(tmp_path):
    b = bench(tmp_path)
    preds = [{"id": f"q{i}", "answer": "A",
              "constrained": {"base": "A", "tuned": "A" if i else "B"}} for i in range(4)]
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "eval_medqa.json").write_text(json.dumps({"predictions": preds}))
    exp = {"experiment_id": "run1", "evaluation": {"format": "v1_eval_json", "models": {
        "base": {"dir": "r1", "role": "base"}, "run1": {"dir": "r1", "role": "tuned"}}}}
    base, tuned = rows_for(exp, tmp_path, b)
    assert (base["medqa_letter"], tuned["medqa_letter"]) == (1.0, 0.75)


def test_e0s_record_lists_a_directory_for_every_scheduled_worker():
    exp = json.loads(Path("experiments/E0/experiment.json").read_text())
    dirs = {d for m in exp["evaluation"]["models"].values() for d in m["dirs"]}
    for session, gpus in E0["sessions"].items():
        for gpu, queue in gpus.items():
            for model, _ in queue:
                assert (f"results/e0/{session}/e0_{session}_predictions/gpu{gpu}/{model}"
                        in dirs), (session, gpu, model)


def test_every_record_names_its_commit_and_data():
    for path in Path("experiments").glob("*/experiment.json"):
        exp = json.loads(path.read_text())
        assert exp["experiment_id"] == path.parent.name
        assert exp["dataset_version"]
        if exp["experiment_id"] != "E0":
            assert exp["git_commit"]


def test_the_committed_csv_has_the_current_columns():
    with Path("experiments/results.csv").open() as fh:
        assert next(csv.reader(fh)) == COLUMNS
