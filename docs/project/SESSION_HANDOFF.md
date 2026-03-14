# Session Handoff

## Current Status

This file was reset on `2026-03-14` after invalidating the old single-card / colocate experiment path.

Only the dual-GPU official `vllm server` sync route should be treated as trustworthy for new training conclusions.

## Trusted Training Route

- repo root: `d:\MY-GRPO`
- server project root: `/root/autodl-tmp/CODE-GRPO/trl`
- conda env: `codegrpo`
- model: `/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct`
- training route:
  - dual-GPU
  - `vllm_mode: server`
  - BF16 + LoRA
  - `vllm_sync_steps: 5`

## Trusted Config Folder

- [trl/configs/comparison/server_2gpu](d:/MY-GRPO/trl/configs/comparison/server_2gpu)

Main files:

1. [codegrpo_method_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/server_2gpu/codegrpo_method_mbpp.yaml)
2. [vanilla_grpo_multiround_k2_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/server_2gpu/vanilla_grpo_multiround_k2_mbpp.yaml)
3. [vanilla_grpo_single_round_k8_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/server_2gpu/vanilla_grpo_single_round_k8_mbpp.yaml)
4. [raw_qwen7b_eval_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/server_2gpu/raw_qwen7b_eval_mbpp.yaml)

## Key Validation Result

The main stale-weight suspicion was directly checked and resolved on the trusted route:

- a real training rollout trace at `global_step=20`
- and standalone generation from `checkpoint-20`
- with the same rendered prompt and `temperature=0`
- matched exactly

Interpretation:

- rollout on the trusted dual-GPU server route did read updated weights
- old single-card colocate conclusions should not be reused

## Invalidated Old Route

Do not use old conclusions from:

- single-card `colocate`
- `4bit + LoRA` online rollout on the old path
- stale embedded eval runs
- old notes in the previous MBPP experiment log

Reference:

- [docs/project/experiments/mbpp_small_experiment_log.md](d:/MY-GRPO/docs/project/experiments/mbpp_small_experiment_log.md)

## Common Server Commands

Single config:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
bash /root/autodl-tmp/CODE-GRPO/run_dual_gpu_server.sh \
  configs/comparison/server_2gpu/vanilla_grpo_multiround_k2_mbpp.yaml \
  42 \
  8010
```

Full comparison suite:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
bash /root/autodl-tmp/CODE-GRPO/run_all_server_2gpu.sh 42 8010
```

## Logging / Trace Defaults

Current defaults are intentionally limited:

- external dependency logs are reduced to `WARNING`
- train traces are capped at `8`
- selected traces keep full prompt/output text
- trace selection prefers error cases and high-soft-reward failures

## Next Work

- run new comparison experiments only from the trusted `server_2gpu` config folder
- ignore old single-card results for method claims
