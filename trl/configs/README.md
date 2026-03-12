# Config Layout

This directory contains runnable YAML configs grouped by purpose.

Folders:

- `train/`
  - Main training configs.
- `eval/`
  - Standalone evaluation-only configs.
- `comparison/`
  - Side-by-side comparison configs for raw baseline, vanilla GRPO, and the current method.

Commonly used configs:

- Main method training:
  - `train/codegrpo_train_qwen7b_vllm_mbpp_small.yaml`
- Vanilla GRPO training:
  - `train/codegrpo_train_qwen7b_vllm_mbpp_vanilla_grpo.yaml`
- Raw standalone eval:
  - `eval/codegrpo_eval_qwen7b_raw_mbpp.yaml`

Comparison suite:

- `comparison/codegrpo_method_mbpp.yaml`
- `comparison/vanilla_grpo_multiround_k2_mbpp.yaml`
- `comparison/vanilla_grpo_single_round_k8_mbpp.yaml`
- `comparison/raw_qwen7b_eval_mbpp.yaml`

Recommended use:

1. Use `train/` for day-to-day training runs.
2. Use `eval/` for standalone evaluation commands.
3. Use `comparison/` when running matched ablations.
