# Session Handoff

## Purpose

This file is a compact handoff for a new session. It captures the current project state, server paths, active configs, recent experiment conclusions, and the next decisions to make.

## Local Repo

- Repo root: `d:\MY-GRPO`
- Main docs:
  - [algorithm.md](d:/MY-GRPO/docs/project/algorithm.md)
  - [constraint.txt](d:/MY-GRPO/docs/project/constraint.txt)
  - [project README](d:/MY-GRPO/docs/project/README.md)
  - [experiment log](d:/MY-GRPO/docs/project/experiments/mbpp_small_experiment_log.md)

## Server Paths

- Project root on server: `/root/autodl-tmp/CODE-GRPO/trl`
- TensorBoard root: `/root/tf-logs/CODE-GRPO`
- Main training outputs:
  - current method: `/root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo`
  - raw eval: `/root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo_eval`
  - comparison runs may use separate `runs_*` directories depending on config
- Model path:
  - `/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct`

## Environment

- Conda env: `codegrpo`
- Common run prelude:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## Current Config Layout

- Train configs: [trl/configs/train](d:/MY-GRPO/trl/configs/train)
- Eval configs: [trl/configs/eval](d:/MY-GRPO/trl/configs/eval)
- Comparison configs: [trl/configs/comparison](d:/MY-GRPO/trl/configs/comparison)
- Config index: [trl/configs/README.md](d:/MY-GRPO/trl/configs/README.md)

## Main Active Config

File:
- [codegrpo_train_qwen7b_vllm_mbpp_small.yaml](d:/MY-GRPO/trl/configs/train/codegrpo_train_qwen7b_vllm_mbpp_small.yaml)

Current important settings:
- model: `Qwen2.5-Coder-7B-Instruct`
- 4bit + LoRA
- train dataset: `trl/data/mbpp_sanitized_codegrpo_train.jsonl`
- eval dataset: `trl/data/mbpp_sanitized_codegrpo_validation.jsonl`
- `num_train_epochs: 2`
- `eval_strategy: steps`
- `eval_steps: 60`
- `learning_rate: 1e-6`
- training temperature: `0.7`
- eval temperature: `0.0`
- training rollout:
  - `K=2`
  - `T_max=2`
  - `N_max=4`
  - `M_audit=3`
- eval rollout:
  - code-only single trajectory
  - no audits
  - `eval_T_max_override=5`
- best model selection:
  - `load_best_model_at_end: true`
  - `metric_for_best_model: eval_pass_at_1`

## Data Split

Current usable sanitized MBPP total: `404`

Current split:
- train: `283`
- validation: `61`
- test: `60`

Notes:
- Validation/test are relatively small for distinguishing tiny gains.
- Current train/eval selection uses `train + validation` style development, but final untouched test reporting has not yet been finalized.

## Current Method State

### Generation / Training

- Main generation now uses fenced python blocks, not `<CODE>`.
- Main generation parser and prompt have been stabilized for fenced code.
- Empty generation handling and parser robustness were improved earlier.
- Training still uses the full method:
  - code generation
  - logic audit
  - exec audit
  - soft reward
  - final_reason stage when applicable

### Eval

- Eval is intentionally simplified:
  - code-only
  - single trajectory
  - no logic audit
  - no exec audit
  - no tree branching
- Primary eval metrics:
  - `eval_pass_at_1_round_1`
  - `eval_pass_at_1_round_2`
  - ...
  - `eval_pass_at_1_round_5`
  - `eval_pass_at_1`
- `best_pass_rate_*` are auxiliary and should not be treated as the main result.

## TensorBoard / Logging

- TensorBoard writes under `/root/tf-logs/CODE-GRPO`
- Review bundles are slimmed down and should be the default artifact to inspect.
- Current logs were intentionally reduced to avoid excessive noise.
- Important TensorBoard tags:
  - `eval/pass_at_1_round_1`
  - `eval/pass_at_1_round_2`
  - `eval/pass_at_1_round_5`
  - `train/window/mean_R_code`
  - `train/window/mean_R_reason`
  - `train/window/mean_R_soft_effective`

## Comparison Suite

Configs:
- current method:
  - [codegrpo_method_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/codegrpo_method_mbpp.yaml)
- vanilla GRPO, multiround K=2:
  - [vanilla_grpo_multiround_k2_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/vanilla_grpo_multiround_k2_mbpp.yaml)
- vanilla GRPO, single-round K=8:
  - [vanilla_grpo_single_round_k8_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/vanilla_grpo_single_round_k8_mbpp.yaml)
- raw Qwen7B eval:
  - [raw_qwen7b_eval_mbpp.yaml](d:/MY-GRPO/trl/configs/comparison/raw_qwen7b_eval_mbpp.yaml)
- comparison notes:
  - [comparison README](d:/MY-GRPO/trl/configs/comparison/README.md)

Interpretation:
- `vanilla K=2 multiround` is a structure baseline.
- `vanilla K=8 single-round` is a large-sampling baseline.
- The user wants to compare all methods with `3` seeds eventually.

## Key Experimental Conclusions So Far

1. The current method and vanilla K=2 have both shown weak or unstable eval gains.
2. This suggests the issue is likely not only the custom method design.
3. Training-side rewards move, but often oscillate rather than trend upward.
4. Eval is more trustworthy than train-side reward curves.
5. The strongest current suspicion is the training recipe, especially:
   - low learning rate (`1e-6`)
   - possibly insufficiently effective optimization signal
6. Before changing method design again, K=8 baseline should be finished and compared.

## Important Clarifications

- `step` is optimizer update count, not equal to number of fresh rollouts.
- In GRPO, one batch of generated samples can be reused for multiple optimizer steps.
- K=8 single-round therefore produces many more optimizer steps than a naive "one rollout per question" intuition suggests.

## Immediate Next Steps

1. Finish the ongoing K=8 single-round run and inspect its eval.
2. Compare:
   - raw baseline
   - vanilla K=2
   - vanilla K=8
   - current method
3. If K=8 also fails to improve, prioritize testing a higher learning rate before changing method structure.
4. Later, run 3 seeds per method and compare `mean +- std` externally.

## Current Risks / Open Questions

- Is `learning_rate=1e-6` too small for all methods?
- Are validation/test splits too small to resolve small gains?
- Is the current recipe simply not strong enough to move eval despite reward movement?
- Is K=8 also flat on eval? If yes, the problem is likely recipe-wide rather than method-specific.

## Files Worth Reading First in a New Session

1. [docs/project/algorithm.md](d:/MY-GRPO/docs/project/algorithm.md)
2. [docs/project/experiments/mbpp_small_experiment_log.md](d:/MY-GRPO/docs/project/experiments/mbpp_small_experiment_log.md)
3. [trl/configs/train/codegrpo_train_qwen7b_vllm_mbpp_small.yaml](d:/MY-GRPO/trl/configs/train/codegrpo_train_qwen7b_vllm_mbpp_small.yaml)
4. [trl/configs/comparison/README.md](d:/MY-GRPO/trl/configs/comparison/README.md)

