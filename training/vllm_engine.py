"""vLLM generation for the evaluation worker (--engine vllm).

Hugging Face `generate` on a T4 spent 22.9 s per MedQA question at a 1,536-token
budget in run 3. vLLM's paged KV cache and continuous batching keep the GPU
busy across questions of different lengths. Its docs list compute capability
7.5 (the T4) as supported, and Unsloth's Qwen3-4B GRPO notebook runs it on a
Kaggle T4. This module only generates: prompts, letter reading and row
records come from training.eval_worker, so both engines score identically.

Two choices keep it comparable to the transformers path:
- prompts go in as token ids, tokenised without special tokens, exactly as
  the transformers path tokenises them;
- a letter is read by a one-token greedy step restricted to the question's
  own option letters, the vLLM form of eval_worker.pick_letter.
"""
from __future__ import annotations

import json
from pathlib import Path

from training.eval_worker import SAMPLING, row_from_tokens
from training.format_v2 import forced_suffix, option_letters, render_reasoning_prompt

# Prompts longer than this are not expected (PubMedQA abstracts run to about
# 900 tokens); the context also has to hold the forced answer after a full
# thinking budget.
PROMPT_ROOM = 1536


def lora_rank(adapter: str) -> int:
    return int(json.loads((Path(adapter) / "adapter_config.json").read_text())["r"])


def make_llm(base: str, adapter: str | None, *, budget: int, seed: int, load_in_4bit: bool,
             gpu_memory_utilization: float, enforce_eager: bool, tensor_parallel: int = 1):
    """AWQ and GPTQ checkpoints (the teachers) carry their quantization in
    their config, which vLLM reads; only bitsandbytes is named here."""
    from vllm import LLM

    kw = dict(model=base, tokenizer=base, dtype="half", seed=seed,
              max_model_len=budget + PROMPT_ROOM, tensor_parallel_size=tensor_parallel,
              gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager)
    if adapter:
        kw.update(enable_lora=True, max_lora_rank=lora_rank(adapter), max_loras=1)
    if load_in_4bit:
        kw["quantization"] = "bitsandbytes"
    return LLM(**kw)


def lora_request(adapter: str | None):
    if not adapter:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest("adapter", 1, adapter)


def _token_prompts(tok, texts: list[str]) -> list[dict]:
    return [{"prompt_token_ids": tok(t, add_special_tokens=False)["input_ids"]} for t in texts]


def vllm_letters(llm, tok, prompts: list[str], letter_sets: list[list[str]],
                 ids: dict[str, int], lora=None) -> list[str]:
    """The highest-scoring of each question's own letters after its prompt."""
    from vllm import SamplingParams

    params = [SamplingParams(max_tokens=1, temperature=0.0,
                             allowed_token_ids=[ids[L] for L in letters])
              for letters in letter_sets]
    outs = llm.generate(_token_prompts(tok, prompts), params, lora_request=lora,
                        use_tqdm=False)
    back = {v: k for k, v in ids.items()}
    return [back[out.outputs[0].token_ids[0]] for out in outs]


def vllm_reason(llm, tok, recs, *, ids, think_id: int, stop_ids: set[int], budget: int,
                seeds: list[int], lora=None) -> list[dict]:
    """Think, then answer, with the same budget forcing as reason_batch: a
    completion that states no letter gets its think block closed and
    'Answer:' appended, and the letter is read from the next token."""
    from vllm import SamplingParams

    prompts = [render_reasoning_prompt(tok, r) for r in recs]
    params = [SamplingParams(temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                             top_k=SAMPLING["top_k"], max_tokens=budget, seed=s,
                             skip_special_tokens=False)
              for s in seeds]
    outs = llm.generate(_token_prompts(tok, prompts), params, lora_request=lora,
                        use_tqdm=False)
    rows = [row_from_tokens(rec, list(out.outputs[0].token_ids), tok, think_id=think_id,
                            stop_ids=stop_ids, seed=s)
            for rec, out, s in zip(recs, outs, seeds)]
    forced = [i for i, row in enumerate(rows) if row["letter"] is None]
    if forced:
        letters = vllm_letters(
            llm, tok, [prompts[i] + rows[i]["text"] + forced_suffix(rows[i]["closed"])
                       for i in forced],
            [option_letters(recs[i]) for i in forced], ids, lora)
        for i, letter in zip(forced, letters):
            rows[i]["letter"] = letter
    return rows
