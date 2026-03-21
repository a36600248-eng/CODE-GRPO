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


## 2026-03-21 Pure GRPO Standalone Eval (Seed 42)

Run:

- eval run id: `20260321_211715__test__train_out__json-mbpp_sanitized_codegrpo___hf`
- local review bundle: `D:\BroswerDownload\20260321_211715__test__train_out__json-mbpp_sanitized_codegrpo___hf\review_bundle`
- source adapter: `runs_single_round_pure_grpo_seed42/train/20260321_205414__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm/train_out`
- eval backend note: standalone adapter eval fell back from vLLM server mode to HF backend by design because dynamic LoRA requests are not supported in server mode

Important results:

- `eval_pass_at_1 = 0.5667`
- `eval_best_pass_rate_overall = 0.6111`
- `eval_mean_pass_rate = 0.6111`
- `eval_mean_R_code = 0.6455`
- `eval_generation_format_ok_rate = 0.9000`
- `eval_zero_pass_soft_trigger_rate = 0.3333`
- `eval_soft_lift = 0.0344`

Current comparison status:

- pure GRPO standalone eval is now available and trustworthy
- the latest soft-only result already looked healthy during training (`eval_pass_at_1 = 0.6000` at step 60), but it is currently an online eval from the train run, not yet a matching standalone post-train eval bundle
- therefore the current reading is only preliminary:
  - soft-only appears slightly better on `pass_at_1` (`0.6000` vs `0.5667`)
  - pure GRPO currently has a stronger standalone `mean_pass_rate` / `mean_R_code` readout than the soft-only online eval snapshot
- do not finalize the method comparison until raw base eval and soft-only standalone eval are both present in the same review-bundle style

## 2026-03-21 Raw Base Model Standalone Eval (Seed 42)

Run:

- eval run id: `20260321_213814__test__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`
- local review bundle: `D:\BroswerDownload\20260321_213814__test__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm\review_bundle`
- source model: base `Qwen2.5-Coder-7B-Instruct`
- eval backend: `vllm server`

Important results:

- `eval_pass_at_1 = 0.6000`
- `eval_best_pass_rate_overall = 0.6389`
- `eval_mean_pass_rate = 0.6389`
- `eval_mean_R_code = 0.6755`
- `eval_generation_format_ok_rate = 0.8667`
- `eval_zero_pass_soft_trigger_rate = 0.3000`
- `eval_soft_lift = 0.0366`
- `rollout_row_count = 30`

Current three-way reading:

- raw base standalone currently looks stronger than pure GRPO standalone on `mean_pass_rate` / `mean_R_code`
- raw base standalone matches the current soft-only online eval on `pass_at_1` (`0.6000`), but soft-only currently shows a higher online `best_pass_rate_overall` (`0.7111`)
- therefore the current evidence does not yet support "60-step training clearly beats the base model" on this small eval slice
- the clean next comparison point is still a matching soft-only standalone eval bundle; until then, treat soft-vs-pure-vs-raw as provisional rather than final
