# TODO

## Now

- Sync the latest reward-logic changes to the remote server before the next formal run.
- Restart the formal `server_2gpu` run from a clean state after syncing:
  - `raw_eval`
  - `codegrpo`
  - `vanilla_k2`
  - `vanilla_k8`
- For the next `CodeGRPO` run, verify from `rollout_summary_rank0.jsonl` that:
  - `mean_R_soft_raw` now tracks real logic matches instead of logic format bonus
  - `mean_R_soft_match_raw` is no longer silently decoupled from code soft reward
  - `logic_format_ok_rate` and `exec_format_ok_rate` are non-trivial
  - `nonzero_A_code_rate` improves over the previous run
- After the next run completes, collect the four `review_bundle` directories and compare:
  - `eval_pass_at_1`
  - `eval_best_pass_rate_overall`
  - `eval_pass_at_1_round_1`
  - `eval_pass_at_1_round_2`
  - `eval_compile_ok_rate`
  - `eval_generation_format_ok_rate`
  - train/eval runtime and throughput

## Reward Fixes

- Keep the current reward split:
  - code soft reward only comes from `logic_match`
  - logic-audit and exec-audit rewards are independent from code reward
- Keep logic-audit as:
  - no code soft reward when `logic_match == 0`
  - format-invalid logic audit gets no main logic reward
  - answer match gets reward
  - confirmed match (`code pass && logic match`) gets an extra bonus
- Keep exec-audit as:
  - format-invalid exec audit gets no main exec reward
  - exec reward stays on the reason branch only
- Keep code reward as:
  - code format invalid does not hard-zero everything
  - `pass_rate` stays intact
  - compile/soft auxiliary rewards are strongly down-scaled when generation format is invalid

## Next Config

- Keep `CodeGRPO` formal config on the lighter reason-weight setting:
  - `beta_reason: 0.5`
  - `M_audit: 1`
  - `code_compile_reward_scale: 0.05`
  - `code_format_reward_scale: 0.05`
  - `logic_format_reward_scale: 0.05`
  - `logic_match_reward_scale: 1.0`
  - `logic_confirmed_bonus: 0.25`
- Keep the main fair comparison on train-time depth:
  - `CodeGRPO eval_T_max_override: 2`
  - `vanilla_k2 eval_T_max_override: 2`
- Keep these unchanged for now:
  - `T_max: 2`
  - `vllm_sync_steps: 5`
  - `learning_rate: 1e-6`

## Speed

- Profile where wall time is going:
  - vLLM rollout / audit generation on GPU0
  - training forward/backward on GPU1
  - weight sync frequency
- Try a safer speed pass first:
  - keep sparse trace retention
  - keep `trace_store_full_text: false`
  - revisit `generation_batch_size` for formal `CodeGRPO` and `vanilla_k2`
  - revisit `vllm_sync_steps`
- Investigate a real async pipeline:
  - GPU0 pre-generates next rollout batch
  - GPU1 trains on the current batch
  - overlap rollout and optimization instead of strict alternation
- After the reward fix is validated, revisit:
  - `generation_batch_size`
  - `vllm_sync_steps`
  - audit completion length, especially whether short audit length is suppressing format validity

## Fairness

- Keep the current comparison as the first reference:
  - `CodeGRPO` vs `vanilla_k2` as the main fair comparison
  - `CodeGRPO` vs `vanilla_k8` as a larger-sampling reference, not a strict equal-budget baseline
- Add a stricter fair baseline later:
  - match total generation budget
  - or match total sampled tokens / total model calls
- Keep `raw_eval` as the lower-bound anchor, not as a training fairness baseline.
- Recheck later whether to report an extra "longer inference budget" table separately from the main fair comparison.

## Data

- After the next stable formal run finishes, expand the evaluation set.
- Prepare a larger held-out evaluation split for more stable conclusions.
- Keep the current small evaluation split as a fast development/debug set.
- Add multi-seed evaluation after the larger eval split is ready.
- Verify whether logic/exec prompt parsing is still too strict on a small manually inspected sample before scaling up.

## Logging

- Keep TensorBoard enabled for the formal runs.
- Keep high-signal metrics only in console/TensorBoard public logs.
- Continue hiding pass-related `*_std` metrics from console and TensorBoard.
- Preserve enough sparse traces in `review_bundle` for post-run diagnosis.
- For the next run, inspect these fields first before drawing conclusions:
  - `logic_format_ok_rate`
  - `exec_format_ok_rate`
  - `mean_R_soft_raw`
  - `mean_R_soft_match_raw`
  - `logic_confirmed_rate`
  - `nonzero_A_code_rate`
  - `nonzero_A_reason_rate`
