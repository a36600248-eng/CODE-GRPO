# MBPP Comparison Suite: Dual-GPU Server

This folder contains the official dual-GPU server-mode comparison configs.

Methods:

- `raw_qwen7b_eval_mbpp.yaml`
  - Raw base-model standalone evaluation.
- `vanilla_grpo_multiround_k2_mbpp.yaml`
  - Vanilla GRPO, multi-round repair, `K=2`.
- `vanilla_grpo_single_round_k8_mbpp.yaml`
  - Vanilla GRPO, single-round wide sampling, `K=8`.
- `codegrpo_method_mbpp.yaml`
  - Current full CodeGRPO method.

All train configs in this folder assume:

- dual-GPU server sync (`vllm_mode: server`)
- BF16 + LoRA
- `vllm_sync_steps: 5`
- lightweight but useful rollout trace retention:
  - keep at most 8 train traces
  - prefer error / high-soft-reward-failure samples
  - keep full prompt/output text for the selected traces
