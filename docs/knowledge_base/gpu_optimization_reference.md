# GPU and Kaggle reference

## Hardware ([measured] run 3 log)

- **GPUs.** Kaggle "GPU T4 x2": 2 × Tesla T4. Unsloth reports "Max memory:
  14.562 GB", "Bfloat16 = FALSE", xformers 0.0.35, FA2 = False.
- **Image.** torch 2.10.0+cu128. The pinned stack is unsloth 2026.9.7,
  unsloth_zoo 2026.9.6, transformers 5.5.0, trl 0.24.0 and peft 0.19.1.
- **Session.** 12 hours. Every notebook stops work 40 minutes before the
  limit (`DEADLINE`).
- **Quota.** About 30 GPU-hours a week (Kaggle's stated quota; not measured
  here).

## Measured speeds

| What | Setting | Speed | Source |
|---|---|---|---|
| QLoRA training, rank 64 | 13,617 rows, max_seq 2,496, batch 2 × 8 | 852 steps in 5.0 h (17,984 s) | run 3 train_stats |
| Trainable parameters, rank 64 | | 132,120,576 of 4.15 B (3.18%) | run 3 log |
| Letter mode (transformers, fp16, batch 16) | | 0.08–0.41 s per question | run 3 meta |
| Thinking mode, base (transformers, batch 16, budget 1,536) | MedQA | 22.9 s per question | run 3 meta |
| Thinking mode, run 3 adapter | MedQA | 9.0 s per question | run 3 meta |
| Model load (fp16, from cache) | | 11.6–15.8 s | run 3 meta |
| vLLM 0.32.0 on a T4 (fp16, chunks of 192, budget 4,096, Triton attention) | E2 pool generation, 2 GPUs | 4,608 completions in about 4.45 h, roughly 7 s per completion per GPU; median thinking 748 tokens, p90 1,923 | results/e2/attempt1 |
| vLLM install (uv, `--torch-backend=auto`) and engine start | | install 209 s; engine start about 90 s, of which 33 s is compilation | results/e2/attempt1 log |

## Memory arithmetic ([code]/[measured] model config)

- **Weights.** Qwen3-4B fp16 is about 8.05 GB.
- **KV cache.** 2 × 36 layers × 8 KV heads × 128 dims × 2 bytes ≈ 147 KB per
  token.
- **transformers at a 3,072-token budget.** A batch of 16 needs about 8.7 GB
  of KV cache, which does not fit beside the weights. A batch of 8 needs
  about 4.3 GB, which does. Hence `reasoning_batch_size` 8 (E0, E2 eval) and
  6 at 4,096 (E2 generation, transformers fallback).
- **vLLM at `gpu_memory_utilization` 0.90.** That is about 13.1 GB, leaving
  roughly 4 GB of KV cache, or about 27k tokens. That is several
  full-length sequences, and more short ones, through paged attention.
- **vLLM backends.** vLLM's docs list compute capability 7.5 as supported;
  FlashAttention needs 8.0 or higher, while Triton and Flex attention list
  "any" **[docs]**. vLLM must be told `dtype="half"`, because a T4 has no
  bf16 (`vllm_engine.make_llm`).

## Rules learned the hard way (see [known_failures](known_failures.md))

- **Download each base model once** before parallel workers start, then set
  `HF_HUB_OFFLINE=1`. Run 3 lost a session to a download race.
- **Read forced answers two at a time.** Retry an out-of-memory error outside
  the `except` block, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Keep vLLM in its own virtual environment.** It pins its own torch and
  transformers. Install with `uv pip install vllm --torch-backend=auto` so
  torch matches the driver.
- **Kaggle lists dataset files 20 per page by default.** The upload check
  must pass `--page-size 200` (fixed 2026-10-19).
- **Use time guards everywhere.** Training uses `--stop-after-seconds`,
  workers use `--deadline`, and every phase is smoke-run on two questions
  before the real run.
