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

### Representative run

- `20260310_164541__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. The optimistic soft-reward lift was reduced.
   - `mean_R_soft_raw` dropped from about `0.5221` to about `0.3922`.
   - Typical wrong-code `R_code` values on high-soft cases also dropped.
   - Example:
     - `mbpp_train_611` still had strong raw logic hits on some audit cases, but wrong-code nodes now stayed around `R_code ≈ 0.116 ~ 0.275`, instead of earlier runs where similar cases could climb much higher.

2. The cost was large: pass-related metrics fell noticeably.
   - `mean_pass_rate` dropped from about `0.2283` to about `0.1133`.
   - `best_pass_rate_overall` mean dropped from about `0.4733` to about `0.2733`.
   - `pass_hits` dropped from `26/50` to `18/50`.

3. Confirmed logic and final-reason coverage also fell.
   - `logic_confirmed_rate` mean dropped from about `0.1283` to about `0.0633`.
   - `final_reason_node_count` mean dropped from about `0.28` to about `0.12`.
   - `final_reason_hits` dropped from `14/50` to `6/50`.

4. Main generation format did not meaningfully improve, but it also did not collapse.
   - `generation_format_ok_rate` moved from about `0.425` to about `0.44`.
   - This is basically flat; Group C did not solve the main-generation formatting issue.

5. Execution audit remained the strongest signal.
   - `exec_format_ok_rate` remained high at about `0.9517`.
   - However, some weaker late steps showed that execution quality can also soften if code quality drops too much.

### Interpretation

Group C over-corrected.

- Good:
  - It reduced the soft-reward optimism problem.
  - Wrong code is less likely to be lifted by raw logic agreement alone.

- Bad:
  - It also suppressed useful exploration too much.
  - Fewer questions reached strong code states.
  - Fewer questions reached `final_reason`.
  - Confirmed logic training became even sparser.

### Recommendation after Group C

Do **not** keep tightening soft reward.

The better next move is:

- revert `lambda_soft` upward slightly, but not back to Group B
- keep `M_audit = 3`
- keep tree size unchanged

Most reasonable next target:

- `lambda_soft: 0.08 -> 0.10`
- keep `soft_reward_ineligible_scale = 0.2`

Reason:

- Group B was too optimistic.
- Group C was too conservative.
- The current evidence suggests the useful region is probably between them.

## Group D

### Goal

Test the midpoint between Group B and Group C without changing the tree or audit structure.

### Parameter changes

- `lambda_soft: 0.08 -> 0.10`
- keep `M_audit = 3`
- keep `soft_reward_ineligible_scale = 0.2`
- keep tree/search settings unchanged

### Why this group

1. Group B was strong on pass metrics, but soft reward still looked too optimistic.
2. Group C reduced optimism, but harmed pass rate and final-reason coverage too much.
3. The most defensible next step is to test the midpoint rather than keep tightening or fully reverting.

### What to check after Group D

1. `mean_pass_rate`
2. `best_pass_rate_overall`
3. `mean_R_soft_raw`
4. `logic_confirmed_rate`
5. `final_reason_node_count`

### Desired outcome

- Better pass metrics than Group C
- Less soft-reward optimism than Group B
- Final-reason coverage remains active

### Representative run

- `20260310_224208__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. Group D is the best balance so far.
   - `mean_pass_rate` recovered from about `0.1133` in Group C to about `0.2067`.
   - `best_pass_rate_overall` recovered from about `0.2733` in Group C to about `0.4467`.
   - This is close to Group B, but without returning all the way to Group B's optimistic behavior.

2. Soft-reward optimism is lower than Group B, though not fully eliminated.
   - `mean_R_soft_raw` sits around `0.4629`, between Group B (`~0.5221`) and Group C (`~0.3922`).
   - Typical wrong-code `R_code` values are also in a middle range.
   - Example:
     - `mbpp_train_610` wrong-code nodes stayed around `R_code ≈ 0.10 ~ 0.26`.
     - This is clearly less inflated than the earlier more optimistic groups.

3. Confirmed logic coverage is also back to a healthier level.
   - `logic_confirmed_rate` rose to about `0.125`, essentially back near Group B (`~0.1283`) and much better than Group C (`~0.0633`).

4. Final-reason coverage is active again.
   - `final_reason_node_count` mean is about `0.24`, much better than Group C (`~0.12`) and close to Group B (`~0.28`).
   - `final_reason_hits` rose to `12/50`.

5. Main generation format is still a weakness.
   - `generation_format_ok_rate` is about `0.435`, almost unchanged from Group B and only slightly above Group C.
   - Group D did not solve strict `<CODE>` formatting.

6. Execution audit remains the strongest signal.
   - `exec_format_ok_rate` stayed high at about `0.9583`.
   - This is still the most reliable training signal in the current setup.

7. Some overly positive soft-reward cases remain.
   - Examples still exist with `pass_rate = 0` and fairly high `mean_R_soft_raw`, such as:
     - `mbpp_train_733`
     - `mbpp_train_640`
   - So the optimism issue is reduced, not fully solved.

### Interpretation

Group D is the strongest configuration among A/B/C/D so far.

- Better than Group C on pass metrics and final-reason coverage.
- Less optimistic than Group B.
- Similar to Group B on confirmed logic coverage.

This is the first group that looks like a practical default rather than just an exploratory checkpoint.

### Current recommendation

Keep Group D as the working baseline for now.

Do not change the tree yet.
If another test is needed, the next adjustment should be small and targeted, for example:

- keep `lambda_soft = 0.10`
- keep `M_audit = 3`
- keep `soft_reward_ineligible_scale = 0.2`
- increase training length modestly before changing reward again

Reason:

- The reward calibration now looks usable.
- The main remaining weaknesses are training stability/coverage and main-generation formatting, not a broken reward balance.

## Group D+

### Goal

Keep the current best reward balance and test whether a longer run improves coverage and stability.

### Parameter changes

- `max_steps: 50 -> 100`
- keep all Group D reward and tree settings unchanged

### Why this group

1. Group D is currently the best-balanced configuration.
2. The remaining question is no longer "is reward balance broken?" but "does a longer run improve coverage?"
3. The cleanest next experiment is to extend training length without adding new reward confounders.

### What to check after Group D+

1. `mean_pass_rate`
2. `best_pass_rate_overall`
3. `logic_confirmed_rate`
4. `final_reason_node_count`
5. `generation_format_ok_rate`
6. Whether late-stage metrics are more stable than in the 50-step runs

### Representative run

- `20260310_234414__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. The 100-step run did not collapse; it continued to improve in the second half.
   - Overall:
     - `mean_pass_rate ≈ 0.2033`
     - `best_pass_rate_overall ≈ 0.4433`
     - `logic_confirmed_rate ≈ 0.1242`
     - `final_reason_node_count ≈ 0.25`
   - These are very close to the 50-step Group D results, which means the configuration remains stable.

2. The second half is better than the first half.
   - First 50 steps:
     - `mean_pass_rate ≈ 0.1717`
     - `best_pass_rate_overall ≈ 0.3867`
     - `logic_confirmed_rate ≈ 0.0917`
     - `final_reason_node_count ≈ 0.20`
     - `generation_format_ok_rate ≈ 0.40`
   - Second 50 steps:
     - `mean_pass_rate ≈ 0.235`
     - `best_pass_rate_overall ≈ 0.50`
     - `logic_confirmed_rate ≈ 0.1567`
     - `final_reason_node_count ≈ 0.30`
     - `generation_format_ok_rate ≈ 0.465`

3. This means longer training is useful at the current setting.
   - The run is not just oscillating around the same behavior.
   - Coverage and confirmation both improve later in training.

4. Main generation format is still not fully solved, but it improves slightly over time.
   - It remains the weakest major metric.
   - However, it is better in the second half than in the first half.

5. The remaining main issue is still soft-reward optimism on some unsolved cases.
   - Late-stage examples still appear where:
     - `mean_pass_rate = 0`
     - `mean_R_soft_raw` is still high
   - This means the issue is reduced but not gone.

6. Execution audit remains strong and stable.
   - `exec_format_ok_rate ≈ 0.97`
   - This is still the most reliable branch in the current system.

### Interpretation

Group D+ confirms that:

- Group D is not just a short-run artifact.
- The current reward balance is usable over a longer run.
- Training longer is currently more valuable than rewriting reward terms again.

### Current recommendation

Keep Group D reward settings as the default baseline.

If the next step is a parameter change, it should be small and targeted:

- preferably focus on main-generation formatting
- not on another major reward rebalance

If the next step is a training-scale change, it is now reasonable to:

- keep the same reward settings
- try a modest dataset-size increase or a modest additional training-length increase

Reason:

- The current setup is finally showing sustained improvement over time.
- The main remaining weakness is not catastrophic reward balance; it is formatting and residual soft-reward optimism on a subset of hard unsolved cases.

## Code update before next run

### Goal

Harden main-generation formatting without touching reward balance.

### Code changes

1. Main generation now uses an opening `<CODE>` prefill at the end of the prompt.
   - The model is instructed to continue with code body directly and close with `</CODE>` once.

2. Main generation parsing is now separated from audit parsing.
   - `generation_outside_noise_chars` is used only for main code generation.
   - `format_outside_noise_chars` remains for logic / execution audit parsing.

3. Main generation output is truncated at the first `</CODE>` before parsing.
   - This reduces trailing prose contamination.

4. Main generation format check is stricter.
   - It now expects a single code block or a valid prefilled close-tag completion.

### What to check after the next run

1. `generation_format_ok_rate`
2. `mean_pass_rate`
3. Whether `exec_format_ok_rate` / `logic_confirmed_rate` stay stable
4. Whether code outputs stop carrying extra prose in traces
