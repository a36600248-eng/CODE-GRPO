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


## 2026-03-21 Soft-Only Single-Round GRPO (Seed 42)

Run:

- train run id: `20260321_201811__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`
- local review bundle: `D:\BroswerDownload\20260321_201811__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm\review_bundle`
- config family: `codegrpo_single_round_zero_pass_soft_reward.yaml`
- training mode: single-round, soft reward on, aux SFT off, pseudo-multiround off, question prior off

Important results:

- training coverage fixed: `60` train rollout rows cover `60` unique questions (previous sampler bug no longer present)
- train last step:
  - `mean_pass_rate = 0.8125`
  - `mean_R_code = 0.8125`
  - `advantage/code_zero_rate = 0.5`
  - `zero_pass_soft_trigger_rate = 0.1875`
- train averages across rollout rows:
  - `mean_pass_rate ~= 0.5965`
  - `mean zero_pass_soft_trigger_rate ~= 0.3563`
  - `31 / 60` train rollout rows had `zero_pass_soft_trigger_rate > 0`
  - `15 / 60` train rollout rows were all-zero-pass
- online eval at step `60`:
  - `eval_pass_at_1 = 0.6000`
  - `eval_mean_pass_rate = 0.5833`
  - `eval_mean_R_code = 0.6170`
  - `eval_best_pass_rate_overall = 0.7111`
  - best observed online `eval_best_pass_rate_overall = 0.7667`

Interpretation:

- this run is usable for comparison; the earlier repeated-question sampler pathology is fixed
- zero-pass soft reward is actively participating in training rather than being a dead feature
- sample-level evidence shows zero-pass sibling groups are being ranked by soft reward (not all tied)
- remaining missing piece for clean comparison is to run the matching `pure GRPO` and `raw eval` baselines under the same server path
