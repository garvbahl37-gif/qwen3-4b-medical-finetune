# Research sources

Fetched on 2026-10-19 by two research agents with capped fetches. Most were
read from abstract or documentation pages only; the quotes are the fetch
tool's summaries and may be paraphrased. "Applies" rates how directly a
source bears on a 4B hybrid-thinking model on a T4.

## Model and framework documentation

| Source | URL | What we use from it | Applies |
|---|---|---|---|
| Qwen/Qwen3-4B model card | https://huggingface.co/Qwen/Qwen3-4B | thinking and non-thinking sampling; never greedy; `/think` `/no_think`; `enable_thinking` behaviour; history without thinking | high |
| Qwen3-4B tokenizer_config.json | https://huggingface.co/Qwen/Qwen3-4B/raw/main/tokenizer_config.json | how `reasoning_content` renders; no `{% generation %}` | high |
| Qwen3 Technical Report, arXiv 2505.09388 (May 2025) | https://arxiv.org/abs/2505.09388 | thinking budget; small models built from the flagships (abstract only; stage details unverified) | medium |
| TRL v0.24.0 SFT docs | https://raw.githubusercontent.com/huggingface/trl/v0.24.0/docs/source/sft_trainer.md | `assistant_only_loss` needs `{% generation %}`; completion-only loss for prompt-completion data | high |
| TRL `trl-training` skill (main, 1.17.0.dev0) | `third_party/trl/skills/trl-training/SKILL.md` | trainers and dataset formats; GRPO's generation-batch rule; `use_vllm`, colocate vs server | medium (newer than our 0.24.0) |
| Unsloth Qwen3 guide | https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune | "Use 75% reasoning and 25% non-reasoning … to … retain its reasoning capabilities" | high |
| Unsloth notebooks index; Qwen3 (4B) GRPO notebook | https://unsloth.ai/docs/get-started/unsloth-notebooks; https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO.ipynb | a Kaggle T4 GRPO notebook for Qwen3-4B with vLLM `fast_inference`, max_seq 2,048, lora_rank 32, gpu_memory_utilization 0.9 (num_generations and runtime not read) | high, for a later GRPO experiment |
| vLLM GPU installation | https://docs.vllm.ai/en/latest/getting_started/installation/gpu/ | "compute capability 7.5 or higher (e.g., T4 …)" | high |
| vLLM attention backends | https://docs.vllm.ai/en/latest/design/attention_backends/ | FlashAttention needs ≥ 8.0; Triton and Flex attention work on any GPU | high |
| vLLM bitsandbytes | https://docs.vllm.ai/en/latest/features/quantization/bnb/ | bnb quantization is supported (LoRA on bnb not stated) | low |

## Papers

| Paper | ID, date | Finding | Applies |
|---|---|---|---|
| AlphaMed: Beyond Distillation … Minimalist Rule-Based RL | 2505.17952, May 2025 | GRPO with a binary MCQ reward on MedQA train plus sampled MedMCQA (19,178 questions), 300 steps. Llama-3.1-8B MedQA 58.72 → 76.19. 8 × A800, full fine-tuning. No controlled SFT-vs-RL ablation. | medium: right recipe, wrong compute |
| m1: test-time scaling for medical reasoning | 2504.00869, Apr 2025 | best budget near 4K tokens, worse beyond; "Wait" forcing unreliable; knowledge, not depth, is the limit | medium |
| STaR | 2203.14465, Mar 2022 | fine-tuning on own rationales that reach the right answer improves reasoning | high in principle |
| RFT (Scaling relationship … rejection sampling) | 2308.01825, Aug 2023 | gains grow with distinct reasoning paths | high in principle |
| ReST-EM (Beyond Human Data) | 2312.06585, Dec 2023 | sample, filter by feedback, fine-tune, repeat | high in principle |
| SFT Memorizes, RL Generalizes | 2501.17161, Jan 2025 | SFT generalises worse out of distribution than RL (synthetic tasks) | low–medium |
| LoRA Learns Less and Forgets Less | 2405.09673, May 2024 | LoRA keeps out-of-domain performance better than full fine-tuning | medium |
| HuatuoGPT-o1 | 2412.18925, Dec 2024 | verifier-guided reasoning SFT, then RL with verifier rewards; RL adds gains | medium |
| MedReason | 2504.00993, Apr 2025 | knowledge-graph reasoning SFT; DeepSeek-Distill-8B +7.7% | medium (the closest to an already-reasoning base) |
| ReasonMed | 2506.09513, Jun 2025 | 370K verified paths; detailed CoT plus concise summaries fine-tune best | medium |

**No source found reports Qwen3-4B medical benchmark scores**, or an SFT
result on a hybrid-thinking model. Our run 3 is the only direct evidence.

## Not verified this session

Med-R1, MedVLM-R1, Open-Medical-R1, UltraMedical's paper, m1's numbers, any
MedQA self-consistency result, the Qwen3 report's training stages, whether
the 2507 models keep `/think`, vLLM LoRA on bnb models, and the Unsloth GRPO
notebook's runtime.
