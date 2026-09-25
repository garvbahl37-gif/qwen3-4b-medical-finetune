from __future__ import annotations

import pytest

from training.train import build_texts, loss_curve, should_stop


class FakeTok:
    """Renders like Qwen3: reasoning inside <think> when there is some."""

    def apply_chat_template(self, messages, **kw):
        reasoning = messages[-1].get("reasoning_content", "")
        return f"{kw.get('enable_thinking')}|<think>{reasoning}</think>|{messages[-1]['content']}"


class DroppingTok:
    """A template that silently ignores reasoning_content."""

    def apply_chat_template(self, messages, **kw):
        return f"<think></think>|{messages[-1]['content']}"


ROW = {"id": "q", "source": "s", "kind": "mcq", "question": "Q?",
       "options": {"A": "a", "B": "b"}, "answer": "B", "rationale": None,
       "response": None, "subject": None}


def test_should_stop_only_with_a_positive_limit():
    assert not should_stop(elapsed=10_000, limit=0)
    assert not should_stop(elapsed=99, limit=100)
    assert should_stop(elapsed=100, limit=100)


def test_loss_curve_keeps_logged_losses_only():
    history = [{"loss": 2.5, "epoch": 0.01, "step": 10, "learning_rate": 1e-4},
               {"train_runtime": 5.0},
               {"loss": 2.0, "epoch": 0.02, "step": 20, "learning_rate": 9e-5}]
    assert loss_curve(history) == [
        {"step": 10, "epoch": 0.01, "loss": 2.5, "lr": 1e-4},
        {"step": 20, "epoch": 0.02, "loss": 2.0, "lr": 9e-5}]


def test_build_texts_v2_thinks_only_for_rows_with_reasoning():
    rows = [dict(ROW, reasoning="think"), dict(ROW)]
    assert build_texts(rows, FakeTok(), "v2") == ["True|<think>think</think>|Answer: B. b",
                                                  "False|<think></think>|Answer: B. b"]


def test_build_texts_v1_is_run_1s_format():
    assert build_texts([dict(ROW)], FakeTok(), "v1") == ["None|<think></think>|Answer: B"]


def test_build_texts_stops_when_the_template_drops_reasoning():
    with pytest.raises(SystemExit):
        build_texts([dict(ROW, reasoning="a long enough reasoning trace")], DroppingTok(), "v2")
