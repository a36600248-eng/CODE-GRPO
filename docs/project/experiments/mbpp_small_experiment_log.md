# MBPP Small Experiment Log

## Reset Notice

This log was reset on `2026-03-14`.

All earlier MBPP experiment records based on the old single-card / colocate vLLM training path are considered invalid for method comparison and checkpoint selection.

Invalidated scope:

- single-card `colocate` training runs
- `4bit + LoRA` online rollout conclusions on the old path
- embedded eval conclusions derived from the stale-weight path
- any experiment notes that assumed the old rollout/eval chain was trustworthy

Reason:

- the old path was shown to suffer from stale or otherwise untrustworthy online weight usage
- later direct verification confirmed that the trusted route is the dual-GPU official `vllm server` sync path instead

Current trusted baseline for new experiment records:

- dual-GPU
- `vllm_mode: server`
- BF16 + LoRA
- `vllm_sync_steps: 5`
- configs under [trl/configs/comparison/server_2gpu](d:/MY-GRPO/trl/configs/comparison/server_2gpu)

New experiment records should start below this line.
