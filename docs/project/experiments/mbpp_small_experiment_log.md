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

## 2026-03-21 Soft-Only Single-Round GRPO Medium (Seed 42)

Run:

- train run id: `20260321_231904__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`
- local review bundle: `D:\BroswerDownload\20260321_231904__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm\review_bundle`
- config family: `codegrpo_single_round_zero_pass_soft_reward_mbpp_medium.yaml`
- training mode: single-round, soft reward on, aux SFT off, pseudo-multiround off, question prior off
- notable config changes vs smoke: `max_steps=240`, `max_train_samples=240`, `max_eval_samples=61`, `eval_steps=60`, `max_completion_length_code=256`

Important results:

- training coverage remains healthy: `240` train rollout rows cover `240` unique questions
- train averages across rollout rows:
  - `mean_pass_rate ~= 0.5455`
  - `mean zero_pass_soft_trigger_rate ~= 0.4052`
  - `127 / 240` train rollout rows had `zero_pass_soft_trigger_rate > 0`
  - `96 / 240` train rollout rows were all-pass
  - `64 / 240` train rollout rows were all-zero-pass
- train last step:
  - `mean_pass_rate = 0.4250`
  - `mean_R_code = 0.4690`
  - `advantage/code_zero_rate = 0.2`
  - `zero_pass_soft_trigger_rate = 0.5750`
  - `soft_lift = 0.0440`
- online eval trajectory:
  - step `60`: `pass_at_1 = 0.4918`, `mean_pass_rate = 0.5307`, `mean_R_code = 0.5671`, `best_pass_rate_overall = 0.6434`
  - step `120`: `pass_at_1 = 0.5738`, `mean_pass_rate = 0.5485`, `mean_R_code = 0.5818`, `best_pass_rate_overall = 0.6995`
  - step `180`: `pass_at_1 = 0.5738`, `mean_pass_rate = 0.5587`, `mean_R_code = 0.5929`, `best_pass_rate_overall = 0.6803`
  - step `240`: `pass_at_1 = 0.4590`, `mean_pass_rate = 0.5432`, `mean_R_code = 0.5798`, `best_pass_rate_overall = 0.7158`

Interpretation:

- enlarging the run increases soft-reward participation and reduces the chance that the result is dominated by trivial sampling noise
- however, the current signal shape is mixed:
  - greedy `pass_at_1` improves from step 60 to 120/180, then drops sharply by step 240
  - `best_pass_rate_overall` continues to improve and reaches its best value at step 240
- this pattern is more consistent with "soft reward improves candidate ranking / best-of-K search quality" than with "soft reward steadily improves single-sample greedy correctness"
- the run does not currently justify training to the end of 240 steps; based on online eval, step 120 to 180 looks healthier than step 240
- the next clean comparison should use the same `61`-example eval scale for raw base and any pure-GRPO medium baseline

## 2026-03-22 Soft-Only Medium Standalone Eval (Seed 42)

Run:

- eval run ids:
  - `20260322_000656__test__train_out__json-mbpp_sanitized_codegrpo___hf`
  - `20260322_010924__test__train_out__json-mbpp_sanitized_codegrpo___hf`
- local review bundles:
  - `D:\BroswerDownload\20260322_000656__test__train_out__json-mbpp_sanitized_codegrpo___hf\review_bundle`
  - `D:\BroswerDownload\20260322_010924__test__train_out__json-mbpp_sanitized_codegrpo___hf\review_bundle`
- source adapter: `runs_single_round_soft_grpo_medium_seed42/train/20260321_231904__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm/train_out`
- eval backend note: standalone adapter eval fell back from vLLM server mode to HF backend by design
- config family: `codegrpo_single_round_zero_pass_soft_reward_mbpp_medium.yaml`

Important results:

- `eval_pass_at_1 = 0.4754`
- `eval_best_pass_rate_overall = 0.5068`
- `eval_mean_pass_rate = 0.5068`
- `eval_mean_R_code = 0.5455`
- `eval_generation_format_ok_rate = 0.7869`
- `eval_zero_pass_soft_trigger_rate = 0.4426`
- `eval_soft_lift = 0.0387`
- `rollout_row_count = 61`

Interpretation:

- the two standalone eval bundles are effectively duplicate reruns of the same soft-medium adapter; they are not pure-GRPO medium results
- compared with the soft-medium online eval checkpoints from the train run, the standalone readout is materially worse:
  - online best region (`step 120` to `180`) had `pass_at_1 ~= 0.5738`
  - standalone post-train eval drops to `0.4754`
- this means the current soft-medium run does not hold up well as a final post-train checkpoint, even though its mid-training online eval looked better
- the current medium-scale conclusion is therefore: soft reward helps training-time ranking / best-of-K behavior more than it helps the final standalone greedy checkpoint
- a true `pure GRPO medium` standalone bundle is still missing from the local archive and should not be inferred from these two soft-medium eval runs


## 2026-03-22 Pure GRPO Medium Standalone Eval (Seed 42, Corrected)

Run:

- eval run id: `20260322_005617__test__train_out__json-mbpp_sanitized_codegrpo___hf`
- local review bundle: `D:\BroswerDownload\20260322_005617__test__train_out__json-mbpp_sanitized_codegrpo___hf\review_bundle`
- source adapter: `runs_single_round_pure_grpo_medium_seed42/train/20260322_001802__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm/train_out`
- eval backend note: standalone adapter eval fell back from vLLM server mode to HF backend by design
- config family: `codegrpo_single_round_pure_grpo_mbpp_medium.yaml`

Important results:

- `eval_pass_at_1 = 0.4754`
- `eval_best_pass_rate_overall = 0.5068`
- `eval_mean_pass_rate = 0.5068`
- `eval_mean_R_code = 0.5455`
- `eval_generation_format_ok_rate = 0.7869`
- `eval_zero_pass_soft_trigger_rate = 0.4426`
- `eval_soft_lift = 0.0386`
- `rollout_row_count = 61`

Final medium-scale comparison:

- `raw base` full eval (`61` examples):
  - `pass_at_1 = 0.4918`
  - `mean_pass_rate = 0.5178`
  - `best_pass_rate_overall = 0.5178`
- `pure medium` standalone:
  - `pass_at_1 = 0.4754`
  - `mean_pass_rate = 0.5068`
  - `best_pass_rate_overall = 0.5068`
- `soft medium` standalone:
  - `pass_at_1 = 0.4754`
  - `mean_pass_rate = 0.5068`
  - `best_pass_rate_overall = 0.5068`

Interpretation:

- on the final standalone checkpoint comparison, `soft medium` and `pure medium` are effectively tied
- both are slightly below the raw base model on this `61`-example eval slice
- therefore the current evidence supports a narrow but important conclusion:
  - in this setup, soft reward clearly affects training-time ranking behavior, but that advantage does not survive into a better final standalone greedy checkpoint
  - removing soft reward also does not fix the problem; pure medium ends up in essentially the same place
- the practical reading is that the main bottleneck is no longer "does soft reward participate" but "why do train-time gains fail to translate into final standalone checkpoint gains"

## 2026-03-22 Pure GRPO Medium Train Run (Seed 42)

Run:

- train run id: `20260322_001802__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`
- local review bundle: `D:\BroswerDownload\20260322_001802__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm\review_bundle`
- config family: `codegrpo_single_round_pure_grpo_mbpp_medium.yaml`
- training mode: single-round, soft reward off, aux SFT off, pseudo-multiround off, question prior off

Important results:

- training coverage remains healthy: `240` train rollout rows cover `240` unique questions
- train averages across rollout rows:
  - `mean_pass_rate ~= 0.5484`
  - `mean zero_pass_soft_trigger_rate = 0.0`
  - `95 / 240` train rollout rows were all-pass
  - `66 / 240` train rollout rows were all-zero-pass
- train last step:
  - `mean_pass_rate = 0.3500`
  - `mean_R_code = 0.3500`
  - `advantage/code_zero_rate = 0.8`
- online eval trajectory:
  - step `60`: `pass_at_1 = 0.4754`, `mean_pass_rate = 0.5292`, `mean_R_code = 0.5381`, `best_pass_rate_overall = 0.6831`
  - step `120`: `pass_at_1 = 0.4590`, `mean_pass_rate = 0.5437`, `mean_R_code = 0.5528`, `best_pass_rate_overall = 0.6544`
  - step `180`: `pass_at_1 = 0.5082`, `mean_pass_rate = 0.5427`, `mean_R_code = 0.5518`, `best_pass_rate_overall = 0.6557`
  - step `240`: `pass_at_1 = 0.5410`, `mean_pass_rate = 0.5492`, `mean_R_code = 0.5601`, `best_pass_rate_overall = 0.6585`

Interpretation:

- compared with soft-medium training, pure-medium is more stable late in training on greedy `pass_at_1`
- soft-medium has stronger best-of-K style behavior during training (`best_pass_rate_overall` is consistently higher and peaks at `0.7158`), while pure-medium finishes with a stronger final online `pass_at_1` (`0.5410` vs `0.4590`)
- however, both runs lose that apparent online advantage when converted to standalone post-train eval, and both end near `pass_at_1 = 0.4754`
- this suggests the current gap is not simply "soft vs no-soft"; the bigger issue is the mismatch between online training-time eval and final standalone checkpoint quality
