# Qwen3-4B technical reference

The model is Qwen3-4B, the April 2025 hybrid-thinking release. We use
Unsloth's copies: `unsloth/Qwen3-4B` (fp16 for evaluation and generation) and
`unsloth/Qwen3-4B-unsloth-bnb-4bit` (NF4 for QLoRA training). The 2507
releases are different models and none of this has been checked against them.

## Tokens and template

| Fact | Evidence |
|---|---|
| `<think>` = 151667, `</think>` = 151668, `<|im_end|>` (EOS) = 151645 | [measured] tokenizer, 2026-10-03 |
| `generation_config.json` stop ids: [151645, 151643] | [measured] `GenerationConfig.from_pretrained`, 2026-10-19 |
| A literal `</think>` inside text tokenizes **as the special token** | [measured] a ReasonMed training row rendered 2 × 151667 and 2 × 151668 |
| The template renders `reasoning_content` as `<think>\n…\n</think>\n\n` before the content, only for assistant turns after the last user query | [docs] tokenizer_config.json; [code] `require_reasoning_rendered` checks it on Kaggle |
| `enable_thinking=False` adds an empty `<think>\n\n</think>\n\n` to the generation prompt | [docs] model card / template |
| The template has no `{% generation %}` markers | [docs] tokenizer_config.json |
| `/think` and `/no_think` are not in the template | [docs] tokenizer_config.json |
| `/think` and `/no_think` in a user or system turn switch the mode (the latest wins) when `enable_thinking=True`, and have no effect when it is False | [docs] model card |

## Generation settings

| Setting | Value | Evidence |
|---|---|---|
| Thinking mode | temperature 0.6, top-p 0.95, top-k 20, min-p 0 | [docs] model card |
| Non-thinking mode | temperature 0.7, top-p 0.8, top-k 20 | [docs] model card |
| Greedy decoding | "DO NOT use greedy decoding": it causes degradation and repetition | [docs] model card |
| Output length | 32,768 tokens recommended for most queries | [docs] model card |
| This project's thinking mode | Qwen's thinking settings; budget 1,536 (runs 2–3), 3,072 (E0, E2 eval), 4,096 (E2 generation) | [code] `eval_worker.SAMPLING`, `experiments_config` |
| This project's letter mode | thinking off, prompt ends "Answer:", argmax over the question's own letters | [code] |
| Multi-turn history should keep only final answers, not thinking | [docs] model card. Not relevant to single-turn MCQ. |

## Measured behaviour of this model (MedQA test, 1,273 questions)

| Mode | Base | Run 3 adapter | Evidence |
|---|---|---|---|
| Thinking, budget 1,536 | 69.7% | 60.2% | [measured] `results/run3/` |
| Letter, no thinking | 57.0% | 58.4% | [measured] |
| Median new tokens (thinking + answer) | 1,036 | 403 | [measured] |
| Base thinking, p90 of new tokens | at the 1,536 cap | — | [measured] (`reports/DATASET_AUDIT.md`) |

At 1,536 tokens the base model's long thoughts are cut. Its true accuracy and
length at a larger budget are **[unknown]** until E0 or E2's base evaluation
runs at 3,072.

## How Qwen3 was trained

The technical report (arXiv 2505.09388) describes a thinking mode, a
non-thinking mode, a "thinking budget", and small models built from the
flagships' knowledge **[paper, abstract only]**. The stage-by-stage details
(long-CoT cold start, reasoning RL, thinking fusion, strong-to-weak
distillation) were **not** verified this session. The PDF did not parse; read
arxiv.org/html/2505.09388 before relying on them.
