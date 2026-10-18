# Skill inventory (discovered 2026-10-19)

The search covered:
- `~/.claude/skills/`: `avoid-ai-writing`, and a synced folder of no relevance;
- the installed plugins under `~/.claude/plugins/cache/`;
- the repository, which has no `.claude/skills/` and had no `CLAUDE.md`
  until this session;
- the huggingface/trl clone (`third_party/trl`: `skills/trl-training`,
  `.agents/skills/update-paper-index`, `AGENTS.md`).

Only skills that bear on this project are listed.

| Skill | Source | What it contributes | Used? | Output |
|---|---|---|---|---|
| `trl-training` | huggingface/trl `skills/` (1.17.0.dev0) | trainer and dataset-format map; `assistant_only_loss` and packing fields; GRPO generation-batch divisibility; `use_vllm` colocate vs server | read in full | fine_tuning_reference; GRPO notes for H5. Our pinned TRL is 0.24.0, so version-specific facts were checked against the v0.24.0 docs. |
| TRL `AGENTS.md` | huggingface/trl | "tests exercise real objects … mock only what cannot run" | read | tests use real tokenizers and data; only vLLM (needs CUDA) and Kaggle are stood in for |
| `update-paper-index` | huggingface/trl `.agents/skills` | paper entries for TRL PRs | read; not applicable (we do not contribute to TRL) | — |
| superpowers:dispatching-parallel-agents | plugin | one agent per independent domain, self-contained prompts | used | two research agents (Qwen3/TRL/vLLM facts; medical literature), with fetch caps after the user asked for frugality |
| superpowers:test-driven-development | plugin | failing test first, then code | used | pool, shard, row builder, vLLM engine, self_traces, run_plan E2, majority vote: 294 tests |
| superpowers:verification-before-completion | plugin | evidence before claims | used | local rehearsals on Qwen3-0.6B; Kaggle status checked before reporting |
| superpowers:writing-plans, brainstorming | plugin | design approval before work | consulted; the approval gate was skipped because the user asked for continuous execution | — |
| superpowers:systematic-debugging | plugin | root cause before fix | applied to the Kaggle listing stall (pagination found by comparing counts) | fix in `kaggle_dataset.sh` |
| anthropic-skills:deep-research | built-in | multi-agent research | not used: it fans out beyond the user's one-to-two-agent cap | — |
| claude-mem:* (learn-codebase, smart-explore) | plugin | codebase exploration and memory | not used: the repository was already mapped in this session | — |
| code-review, simplify | built-in | diff review | available for the next review pass | — |

**Unavailable this session:** context7 (documentation lookups) needs
authorisation in `/mcp`, so documentation came from WebFetch instead.
