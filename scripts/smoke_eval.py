"""End-to-end smoke test of the evaluation path, on a tiny real Qwen3.

Runs the exact code the Kaggle evaluation runs -- modeling.load_model, the
batched scorers, the adapter switch, evaluate.py's CLI and report -- against
Qwen3-0.6B, which shares Qwen3-4B's tokenizer and chat template, with a small
random LoRA adapter. Seven Kaggle runs failed on code that had never executed
anywhere; this makes sure the evaluation cannot fail that way.

    .venv/bin/python scripts/smoke_eval.py
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
TRAINED_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

QUESTIONS = [
    ("Deficiency of which vitamin causes scurvy?",
     ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"], "B"),
    ("Which organ produces insulin?",
     ["Liver", "Kidney", "Pancreas", "Spleen"], "C"),
    ("The commonest cause of community-acquired pneumonia is:",
     ["Streptococcus pneumoniae", "Klebsiella", "Pseudomonas", "Legionella"], "A"),
    ("First-line treatment of anaphylaxis is:",
     ["Oral antihistamine", "Intramuscular adrenaline", "IV hydrocortisone",
      "Nebulised salbutamol"], "B"),
    ("The normal adult resting heart rate is:",
     ["20-40", "40-60", "60-100", "100-140"], "C"),
    ("Peaked T waves on an ECG suggest:",
     ["Hypokalaemia", "Hyperkalaemia", "Hyponatraemia", "Hypocalcaemia"], "B"),
    ("Koplik spots are seen in:",
     ["Measles", "Mumps", "Rubella", "Chickenpox"], "A"),
    ("The antidote for paracetamol overdose is:",
     ["Naloxone", "Flumazenil", "N-acetylcysteine", "Atropine"], "C"),
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.evaluate import score_constrained
    from training.modeling import load_model, release_memory
    from training.prompts import render_inference_prompt
    from training.records import Record

    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="medsmoke-"))
    recs = [
        Record(id=f"smoke-{i}", source="fixture", kind="mcq", question=q,
               options=dict(zip("ABCD", opts)), answer=a, rationale=None,
               response=None, subject="Smoke")
        for i, (q, opts, a) in enumerate(QUESTIONS)
    ]
    test_file = work / "fixtures.jsonl"
    test_file.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))

    # 1. The real template renders exactly the trained prefix plus the cue.
    tok = AutoTokenizer.from_pretrained(TINY)
    rendered = render_inference_prompt(tok, recs[0], answer_prefix=True)
    assert rendered.endswith(TRAINED_PREFIX + "Answer:"), repr(rendered[-80:])
    print("ok  the prompt ends with the trained prefix and 'Answer:'")

    # 2. A small random LoRA adapter, so 'tuned' genuinely differs from 'base'.
    torch.manual_seed(42)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=TARGETS, lora_dropout=0.0))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    adapter = work / "adapter"
    peft_model.save_pretrained(adapter)
    del peft_model, base
    print(f"ok  a random LoRA adapter saved to {adapter}")

    # 3. Batch size must not change a single prediction.
    model, tok, choice = load_model(TINY, str(adapter), device="cpu")
    one = score_constrained(model, tok, recs, batch_size=1)
    four = score_constrained(model, tok, recs, batch_size=4)
    assert one == four, f"batching changed predictions: {one} vs {four}"
    print(f"ok  batch 1 and batch 4 agree on all {len(one)}: {''.join(one)}")

    # 4. disable_adapter() must score the true base, nothing in between.
    with model.disable_adapter():
        via_disable = score_constrained(model, tok, recs, batch_size=4)
    del model
    release_memory(choice.device)
    plain, plain_tok, _ = load_model(TINY, None, device="cpu")
    via_plain = score_constrained(plain, plain_tok, recs, batch_size=4)
    assert via_disable == via_plain, (
        f"disabled adapter differs from a plain base: {via_disable} vs {via_plain}")
    print("ok  the disabled adapter scores identically to a freshly loaded base")
    del plain
    release_memory("cpu")

    # 5. The CLI end to end, exactly as the notebook calls it.
    out = work / "report.json"
    subprocess.run(
        [sys.executable, "-m", "training.evaluate", "--base", TINY,
         "--adapter", str(adapter), "--test", str(test_file), "--out", str(out),
         "--gen-limit", "4", "--max-new-tokens", "48", "--batch-size", "4",
         "--device", "cpu"],
        check=True, cwd=Path(__file__).resolve().parents[1])
    report = json.loads(out.read_text())
    assert report["reports"]["constrained"]["n"] == 8
    assert report["reports"]["generative"]["n"] == 4
    assert {"load_s", "constrained_s_per_example",
            "generative_s_per_example"} <= set(report["timing"])
    assert len(report["samples"]) == 4
    print("ok  evaluate.py end to end: both modes, timing and samples reported")

    print(f"\nSMOKE PASSED in {time.time() - started:.0f}s  ({work})")


if __name__ == "__main__":
    main()
