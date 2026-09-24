from __future__ import annotations

import pytest

from training.modeling import DEFAULT_BASE, choose_device


def test_cuda_is_preferred_and_runs_in_float16():
    # The T4 has no native bfloat16; Unsloth reported "Bfloat16 = FALSE" on it.
    choice = choose_device(None, cuda=True, mps=True)
    assert (choice.device, choice.dtype_name) == ("cuda", "float16")


def test_apple_silicon_uses_mps_in_bfloat16():
    choice = choose_device(None, cuda=False, mps=True)
    assert (choice.device, choice.dtype_name) == ("mps", "bfloat16")


def test_cpu_falls_back_to_float32():
    choice = choose_device(None, cuda=False, mps=False)
    assert (choice.device, choice.dtype_name) == ("cpu", "float32")


def test_an_explicit_device_wins_when_it_is_available():
    assert choose_device("cpu", cuda=True, mps=True).device == "cpu"


def test_requesting_an_unavailable_device_stops_loudly():
    with pytest.raises(SystemExit, match="cuda"):
        choose_device("cuda", cuda=False, mps=True)


def test_an_unknown_device_name_stops_loudly():
    with pytest.raises(SystemExit, match="tpu"):
        choose_device("tpu", cuda=True, mps=True)


def test_the_default_base_is_full_precision_qwen3_4b():
    # The adapter was trained on Unsloth's 4-bit copy; LoRA deltas apply to
    # the unquantised weights unchanged, and those are what gets served.
    assert DEFAULT_BASE == "unsloth/Qwen3-4B"
    assert "bnb" not in DEFAULT_BASE
