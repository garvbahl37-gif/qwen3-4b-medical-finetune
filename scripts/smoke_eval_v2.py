"""Run 2's evaluation, end to end, on a small model before any GPU time.

Qwen3-0.6B with a random LoRA adapter on CPU. Every stage runs with four
questions and a 24-token thinking budget, so the budget-forcing path runs too;
then the report pairs the two workers. Also checks the real Qwen3 template
renders run 2's training format.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TINY = "Qwen/Qwen3-0.6B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.eval_worker import STAGES, letter_token_ids
    from training.format_v2 import render_training_text
    from training.records import Record

    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="smoke-v2-"))
    tok = AutoTokenizer.from_pretrained(TINY)

    # 1. The real template renders both training formats.
    four = {"A": "Aspirin", "B": "Heparin", "C": "Warfarin", "D": "Insulin"}
    thinking = Record(id="t", source="s", kind="mcq", question="Which anticoagulant is IV?",
                      options=four, answer="B", rationale=None, response=None,
                      subject=None, reasoning="Heparin is given IV.")
    direct = Record(**{**thinking.to_dict(), "reasoning": None})
    text = render_training_text(tok, thinking)
    assert "<think>\nHeparin is given IV.\n</think>\n\nAnswer: B" in text, text[-160:]
    assert "<think>\n\n</think>\n\nAnswer: B" in render_training_text(tok, direct)
    think_id = tok.convert_tokens_to_ids("</think>")
    assert tok.decode([think_id], skip_special_tokens=False) == "</think>"
    ids = letter_token_ids(tok)
    assert len(set(ids.values())) == 10, ids
    print("ok  training format, </think> token and ten distinct letter tokens")

    # 2. Benchmarks: four questions each; PubMedQA keeps its three options.
    questions = [("Which vitamin deficiency causes scurvy?", "C"),
                 ("Which organ secretes insulin?", "B"),
                 ("Which bone is in the thigh?", "A"),
                 ("Which chamber pumps blood to the body?", "D")]
    for bench in ("medqa", "medmcqa", "mmlu_medical"):
        rows = [Record(id=f"{bench}-{i}", source=bench, kind="mcq", question=q,
                       options=four, answer=a, rationale=None, response=None,
                       subject="Smoke").to_dict() for i, (q, a) in enumerate(questions)]
        (work / f"eval_{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pubmed = [Record(id=f"pubmedqa-{i}", source="pubmedqa", kind="mcq",
                     question=f"Context: A trial.\n\nQuestion: Does drug {i} work?",
                     options={"A": "yes", "B": "no", "C": "maybe"}, answer="A",
                     rationale=None, response=None, subject=None).to_dict()
              for i in range(4)]
    (work / "eval_pubmedqa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pubmed))

    # 3. A random adapter, so the fine-tune genuinely differs from the base.
    torch.manual_seed(42)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16,
                                                 target_modules=TARGETS, lora_dropout=0.0))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    adapter = work / "adapter"
    peft_model.save_pretrained(adapter)
    del peft_model, base

    # 4. Both workers, every stage, then the report.
    common = [sys.executable, "-m", "training.eval_worker", "--base", TINY,
              "--eval-dir", str(work), "--out-dir", str(work / "eval"),
              "--think-budget", "24", "--batch-size", "2", "--limit", "4",
              "--no-size-check", "--device", "cpu"]
    for role in ("tuned", "base"):
        extra = ["--role", role] + (["--adapter", str(adapter)] if role == "tuned" else [])
        subprocess.run(common + extra, check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "training.eval_report", "--eval-dir",
                    str(work / "eval"), "--bench-dir", str(work), "--out",
                    str(work / "report.json")], check=True, cwd=ROOT)

    report = json.loads((work / "report.json").read_text())
    assert set(report["stages"]) == {f"{m}:{b}" for m, b in STAGES}, report["stages"].keys()
    for key, rep in report["stages"].items():
        assert rep["scored"] == rep["planned"] == 4, (key, rep["scored"], rep["planned"])
    assert report["stages"]["reasoning:medqa"]["tuned_counts"]["forced"] >= 1
    for role in ("base", "tuned"):
        assert report["models"][role]["status"] == "finished", report["models"][role]
        rows = [json.loads(line) for line in
                (work / "eval" / role / "reasoning__pubmedqa.jsonl").read_text().splitlines()]
        assert {r["letter"] for r in rows} <= {"A", "B", "C"}, rows
    print("ok  every stage paired 4/4, budget forcing used, PubMedQA letters within A-C")
    print(f"\nSMOKE V2 PASSED in {time.time() - started:.0f}s  ({work})")


if __name__ == "__main__":
    main()
