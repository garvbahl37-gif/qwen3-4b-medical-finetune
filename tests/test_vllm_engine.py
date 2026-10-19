"""vLLM needs a CUDA GPU, so it is replaced here by the smallest stand-in that
records what it was asked. Everything else is the real code path."""
from __future__ import annotations

import sys
import types

import pytest

from training.records import Record

THINK = 99


class Params:
    def __init__(self, **kw):
        self.kw = kw


class Out:
    def __init__(self, ids):
        self.outputs = [types.SimpleNamespace(token_ids=ids)]


class FakeLLM:
    """Answers each request from a script keyed by request number."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append((prompts, params, lora_request))
        return [Out(self.script(p, q)) for p, q in zip(prompts, params)]


class Tok:
    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text[-5:]]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("</think>" if i == THINK else chr(i) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return messages[-1]["content"]


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=Params))


def mcq(i):
    return Record(id=f"q{i}", source="medqa", kind="mcq", question=f"Q{i}",
                  options={"A": "a", "B": "b", "C": "c"}, answer="A", rationale=None,
                  response=None, subject=None)


IDS = {L: 200 + n for n, L in enumerate("ABCDEFGHIJ")}


def test_letters_restrict_the_greedy_step_to_each_questions_options():
    from training.vllm_engine import vllm_letters

    llm = FakeLLM(lambda p, q: [q.kw["allowed_token_ids"][-1]])
    out = vllm_letters(llm, Tok(), ["x", "y"], [["A", "B"], ["A", "B", "C"]], IDS)
    assert out == ["B", "C"]
    params = llm.calls[0][1]
    assert all(p.kw["max_tokens"] == 1 and p.kw["temperature"] == 0.0 for p in params)


def test_reason_scores_like_the_transformers_path_and_forces_unfinished_thoughts():
    from training.vllm_engine import vllm_reason

    def script(prompt, params):
        if params.kw["max_tokens"] == 1:            # the forcing step
            return [IDS["C"]]
        if params.kw["seed"] == 11:                 # finished: thinks, closes, answers
            return [ord("h"), THINK] + [ord(c) for c in "\n\nAnswer: B"] + [7]
        return [ord("h")] * 4                       # ran out of budget mid-thought

    llm = FakeLLM(script)
    rows = vllm_reason(llm, Tok(), [mcq(0), mcq(1)], ids=IDS, think_id=THINK, stop_ids={7},
                       budget=64, seeds=[11, 12], lora="L")
    assert [(r["letter"], r["closed"], r["forced"], r["seed"]) for r in rows] == [
        ("B", True, False, 11), ("C", False, True, 12)]
    first = llm.calls[0]
    assert all("prompt_token_ids" in p for p in first[0]) and first[2] == "L"
    assert [p.kw["seed"] for p in first[1]] == [11, 12]
    assert first[1][0].kw["temperature"] == 0.6 and first[1][0].kw["top_k"] == 20
    assert len(llm.calls) == 2 and len(llm.calls[1][0]) == 1   # only the cut one is forced


def test_lora_rank_comes_from_the_adapter_config(tmp_path):
    from training.vllm_engine import lora_rank

    (tmp_path / "adapter_config.json").write_text('{"r": 16}')
    assert lora_rank(str(tmp_path)) == 16


def test_make_llm_splits_a_large_teacher_across_gpus(monkeypatch):
    from training.vllm_engine import make_llm

    seen = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=lambda **kw: seen.update(kw) or "llm", SamplingParams=Params))
    make_llm("Qwen/Qwen3-30B-A3B-GPTQ-Int4", None, budget=4096, seed=1, load_in_4bit=False,
             gpu_memory_utilization=0.9, enforce_eager=False, tensor_parallel=2)
    assert seen["tensor_parallel_size"] == 2 and seen["dtype"] == "half"
    assert "quantization" not in seen      # GPTQ/AWQ are read from the checkpoint's config
