# MBPP Small Experiment Log

This file tracks the parameter groups used for `codegrpo_train_qwen7b_vllm_mbpp_small.yaml`,
the observed issues from the resulting run, and the rationale for the next group.

## Group A

- Config file: `trl/codegrpo_train_qwen7b_vllm_mbpp_small.yaml`
- Dataset: `trl/data/mbpp_sanitized_codegrpo_small64.jsonl`
- Representative run:
  - `20260310_154839__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Key tree hyperparameters

- `K = 2`
- `K_reason = 1`
- `T_max = 2`
- `N_max = 4`
- `M_audit = 2`
- `M_retry = 1`
- `lambda_soft = 0.2`
- `code_format_reward_scale = 0.08`
- `max_completion_length = 80`

### Main observations

1. Pipeline is healthy.
   - Search, audit, confirmed gating, and final one-shot `reason-only` all triggered on real MBPP samples.

2. Execution audit is the strongest and most stable signal.
   - `exec_format_ok_rate` stayed near `1.0` for most steps.

3. Logic training signal is still sparse.
   - `logic_confirmed_rate` stayed low on average.
   - The logic branch is not dead, but confirmed logic samples are still relatively rare.

4. Soft reward can lift wrong code too much.
   - Some questions had `mean_pass_rate = 0.0` while `mean_R_soft_raw` stayed very high.
   - This means unconfirmed logic matches are still giving the code branch a noticeable boost through `R_soft`.

5. Main generation format remains weaker than execution audit format.
   - `generation_format_ok_rate` was much lower than `exec_format_ok_rate`.

### Risks inferred from Group A

- If `lambda_soft` stays too high, wrong code can get overly favorable `R_code`.
- With only `M_audit = 2`, audit variance is still high on MBPP.
- Main generation format pressure is not yet strong enough.

## Group B

### Goal

Make code reward more conservative, make audit estimates slightly more reliable, and push main generation format a bit harder without changing the framework itself.

### Parameter changes

- `M_audit: 2 -> 3`
- `lambda_soft: 0.2 -> 0.12`
- `code_format_reward_scale: 0.08 -> 0.12`

### Expected effect

1. `lambda_soft` down:
   - Reduce the chance that wrong code gets a relatively high `R_code` from lucky logic matches.
   - Keep `R_soft` useful, but make hard pass signal dominate more clearly.

2. `M_audit` up:
   - Make logic/exec audit less noisy.
   - Make a single lucky hit matter less.

3. `code_format_reward_scale` up:
   - Increase pressure on the model to return strict `<CODE>...</CODE>` format.

### What to check after Group B

1. `mean_pass_rate`
2. `mean_R_soft_raw`
3. `logic_confirmed_rate`
4. `generation_format_ok_rate`
5. `final_reason_node_count`

### Success criteria for Group B

- Fewer cases where `pass_rate = 0` but `R_code` is still relatively high.
- `generation_format_ok_rate` improves.
- `logic_confirmed_rate` does not collapse.
- `final_reason` still triggers on some samples.

### Representative run

- `20260310_162134__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. Search performance improved modestly.
   - `mean_pass_rate` increased from about `0.1733` to about `0.2283`.
   - `best_pass_rate_overall` mean increased from about `0.38` to about `0.4733`.

2. `final_reason` triggered more often.
   - `final_reason_node_count` mean increased from about `0.16` to about `0.28`.
   - `final_reason` appeared on `14/50` steps instead of `8/50`.

3. Confirmed logic training signal improved slightly, but is still sparse.
   - `logic_confirmed_rate` mean increased from about `0.11` to about `0.1283`.
   - This is better, but still low relative to the total number of audit samples.

4. The main problem remains: soft reward is still too optimistic for some wrong-code cases.
   - There are still many samples with `mean_pass_rate = 0` while `mean_R_soft_raw` is near `1.0`.
   - Representative questions: `mbpp_train_733`, `mbpp_train_743`, `mbpp_train_605`, `mbpp_train_728`.
   - Lowering `lambda_soft` from `0.2` to `0.12` reduced the effect somewhat, but did not remove the pattern.

5. Main generation format did not improve.
   - `generation_format_ok_rate` moved from about `0.455` down to about `0.425`.
   - Increasing `code_format_reward_scale` alone was not enough to materially improve strict `<CODE>` compliance.

6. Execution audit remains the most reliable signal.
   - `exec_format_ok_rate` stayed high, around `0.9617`.
   - The execution branch is still the healthiest training signal in this setup.

7. The final one-shot `reason-only` stage is now validated on real MBPP samples, but its quality is mixed.
   - Good case:
     - `mbpp_train_628`: final reason stage improved `R_reason` from about `0.6833` to `1.0`.
   - Weak cases:
     - `mbpp_train_619`: final reason stage stayed at `R_reason = 0.05`.
     - `mbpp_train_742`: final reason stage stayed at `R_reason = 0.05`.
   - This means the stage is functioning, but unified post-pass reasoning does not yet consistently repair reasoning mistakes.

### Interpretation

Group B was a partial improvement, not a full fix.

- Good:
  - More solved samples.
  - More `final_reason` activations.
  - Slightly more confirmed logic training.

- Still bad:
  - Wrong-code samples can still get very high raw soft reward.
  - Main generation format remains weak.
  - Final post-pass reasoning is inconsistent across problems.

### Recommended next direction (not applied yet)

If another parameter group is tested, the next most defensible move is:

- keep `M_audit = 3`
- reduce `lambda_soft` again, e.g. `0.12 -> 0.08`
- keep the main structure unchanged
- do **not** increase tree size yet

Reason:

- The dominant remaining issue is still overly generous soft reward on unsolved code.
- The bottleneck is no longer basic pipeline health.
- Increasing search size before soft reward is better calibrated would add noise faster than it adds useful signal.

## Group C

### Goal

Further reduce optimistic soft-reward lift on unsolved code while preserving the existing tree/search setup.

### Parameter changes

- `lambda_soft: 0.12 -> 0.08`
- `soft_reward_ineligible_scale: 0.3 -> 0.2`

### Why this group

1. Lower `lambda_soft`
   - Make raw logic agreement less able to lift `R_code` when the code still fails tests.
   - Keep soft reward as a tie-breaker, not a near-primary driver.

2. Lower `soft_reward_ineligible_scale`
   - Reduce soft-reward contribution further in weaker eligibility situations.
   - This is a conservative calibration step, not a framework change.

### What to check after Group C

1. Whether `pass_rate = 0` samples still frequently show very high `R_code`
2. Whether `mean_R_soft_raw - mean_pass_rate` narrows
3. Whether `logic_confirmed_rate` stays at least stable
4. Whether `final_reason_node_count` remains non-trivial
5. Whether `mean_pass_rate` drops too much
