from __future__ import annotations

from dataclasses import dataclass

# The adapter was trained on Unsloth's 4-bit copy of Qwen3-4B. Evaluation and
# serving load the full-precision weights instead: a LoRA adapter's deltas
# apply to the unquantised base unchanged, and scoring the full-precision model
# is scoring what actually gets served.
DEFAULT_BASE = "unsloth/Qwen3-4B"

_DTYPES = {"cuda": "float16", "mps": "bfloat16", "cpu": "float32"}


@dataclass(frozen=True)
class DeviceChoice:
    device: str
    dtype_name: str


def choose_device(requested: str | None, *, cuda: bool, mps: bool) -> DeviceChoice:
    """Pick where to run and at what precision. Pure, so it is testable.

    CUDA runs float16: the T4 has no native bfloat16. MPS runs bfloat16, which
    Apple silicon supports and which is Qwen3's native precision. CPU runs
    float32, because half precision on CPU is slow where it works at all.
    """
    if requested is not None and requested not in _DTYPES:
        raise SystemExit(
            f"\nSTOP. Unknown device {requested!r}; expected one of "
            f"{sorted(_DTYPES)}.")
    available = {"cuda": cuda, "mps": mps, "cpu": True}
    if requested is not None and not available[requested]:
        raise SystemExit(
            f"\nSTOP. --device {requested} was requested but is not available "
            "on this machine.")
    device = requested or ("cuda" if cuda else "mps" if mps else "cpu")
    return DeviceChoice(device, _DTYPES[device])


def load_model(base: str, adapter: str | None, *, device: str | None = None):
    """Load the base weights, wrap them with a LoRA adapter if given.

    Returns (model, tokenizer, DeviceChoice). With an adapter the model is a
    PeftModel, whose disable_adapter() context yields the untouched base --
    which is how evaluation scores both models from a single load.

    The tokenizer pads on the LEFT. Batched scoring reads the logits at the
    final position; with right padding that position is a pad token for every
    sequence shorter than the longest in the batch, and scores noise.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    choice = choose_device(device, cuda=torch.cuda.is_available(),
                           mps=torch.backends.mps.is_available())

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=getattr(torch, choice.dtype_name))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(choice.device)
    model.eval()
    return model, tok, choice


def release_memory(device: str) -> None:
    """Return freed memory to the device after the caller drops its model.

    `del model` alone leaves the CUDA allocator holding the freed blocks, so a
    following load can fail on a card that objectively has room.
    """
    import gc

    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
