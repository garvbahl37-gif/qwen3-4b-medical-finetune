from __future__ import annotations

from dataclasses import dataclass

# The adapter was trained on Unsloth's 4-bit copy of Qwen3-4B. Evaluation and
# serving load the full-precision weights instead: a LoRA adapter's deltas
# apply to the unquantised base unchanged, and scoring the full-precision model
# is scoring what actually gets served.
DEFAULT_BASE = "unsloth/Qwen3-4B"
# The exact 4-bit copy run 1-3 trained on. Scoring the adapter on it is the
# control for "does the adapter behave differently on the base it learned on".
TRAINING_BASE_4BIT = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

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


def four_bit_problem(device: str, quantization_config: dict | None) -> str | None:
    """Why a 4-bit load cannot go ahead, or None. Pure, so it is testable.

    Only a pre-quantized checkpoint is accepted: quantizing the fp16 weights
    here would not reproduce Unsloth's copy, which leaves some layers
    unquantized (its llm_int8_skip_modules)."""
    if device != "cuda":
        return f"4-bit loading needs CUDA (bitsandbytes); this run is on {device}."
    if not quantization_config or quantization_config.get("quant_method") != "bitsandbytes":
        return ("the base is not a pre-quantized bitsandbytes checkpoint; pass "
                f"--base {TRAINING_BASE_4BIT}.")
    return None


def load_model(base: str, adapter: str | None, *, device: str | None = None,
               load_in_4bit: bool = False):
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

    if load_in_4bit:
        from transformers import AutoConfig

        quant = getattr(AutoConfig.from_pretrained(base), "quantization_config", None)
        if hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        problem = four_bit_problem(choice.device, quant)
        if problem:
            raise SystemExit(f"\nSTOP. {problem}")
        # A quantized model cannot be moved after loading; it is placed on
        # the GPU as it loads.
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16,
                                                     device_map={"": 0})
        # The checkpoint asks for bfloat16 compute, which the T4 lacks;
        # Unsloth trained in float16 there, so the control computes in float16.
        for module in model.modules():
            if hasattr(module, "compute_dtype") and hasattr(module, "quant_state"):
                module.compute_dtype = torch.float16
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=getattr(torch, choice.dtype_name))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if not load_in_4bit:
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
