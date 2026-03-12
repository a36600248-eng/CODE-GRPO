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
     - `mbpp_train_611` still had strong raw logic hits on some audit cases, but wrong-code nodes now stayed around `R_code â‰?0.116 ~ 0.275`, instead of earlier runs where similar cases could climb much higher.

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
     - `mbpp_train_610` wrong-code nodes stayed around `R_code â‰?0.10 ~ 0.26`.
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
     - `mean_pass_rate â‰?0.2033`
     - `best_pass_rate_overall â‰?0.4433`
     - `logic_confirmed_rate â‰?0.1242`
     - `final_reason_node_count â‰?0.25`
   - These are very close to the 50-step Group D results, which means the configuration remains stable.

2. The second half is better than the first half.
   - First 50 steps:
     - `mean_pass_rate â‰?0.1717`
     - `best_pass_rate_overall â‰?0.3867`
     - `logic_confirmed_rate â‰?0.0917`
     - `final_reason_node_count â‰?0.20`
     - `generation_format_ok_rate â‰?0.40`
   - Second 50 steps:
     - `mean_pass_rate â‰?0.235`
     - `best_pass_rate_overall â‰?0.50`
     - `logic_confirmed_rate â‰?0.1567`
     - `final_reason_node_count â‰?0.30`
     - `generation_format_ok_rate â‰?0.465`

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
   - `exec_format_ok_rate â‰?0.97`
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

### Follow-up run after format hardening

### Representative run

- `20260311_003647__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. Main-generation format got much worse, not better.
   - `generation_format_ok_rate` dropped to about `0.1525`.
   - This is far below the earlier Group D+ baseline (`~0.4325`).

2. Pass-related metrics also dropped.
   - `mean_pass_rate` dropped to about `0.1617`.
   - `best_pass_rate_overall` dropped to about `0.3767`.
   - Unlike the earlier 100-step run, the second half became worse than the first half.

3. The root cause is visible in traces: the model is still returning fenced code blocks.
   - Example traces:
     - `mbpp_train_635_rank0_000022`
     - `mbpp_train_644_rank0_000087`
     - `mbpp_train_725_rank0_000026`
   - Typical raw output is:
     - ````python ... ````
   - So the stricter parser is correctly marking format invalid, but the current "prefill `<CODE>` in user prompt" strategy is not actually steering the model into the desired format.

4. This means the latest hardening was directionally correct, but the implementation choice was too naive.
   - Strict parsing exposed the real problem.
   - The current prefill method is not a true assistant-side prefill; it is only text appended inside the user prompt.
   - Qwen still falls back to its default fenced-code behavior.

### Interpretation

Do not keep the current format-hardening result as the new baseline.

What this run proves:

- The earlier `generation_format_ok_rate ~0.43` was partly relying on looser handling.
- Simply appending `<CODE>` inside the user message is not enough to force strict code-only output.
- The next fix should target generation control more directly, rather than making parsing even stricter.

### Recommended next direction

1. Keep the stricter generation metric so the failure remains visible.
2. Replace the current pseudo-prefill with a stronger generation-control mechanism:
   - align the main-generation protocol with the model prior by using fenced Python code blocks, or
   - implement a true assistant-prefill path if the chat template supports it.
3. Do not rebalance rewards again before fixing this generation-format bottleneck.

## Code update after format-hardening rollback

### Goal

Align the main-generation format with the model's observed prior instead of forcing `<CODE>`.

### Code changes

1. Main generation now expects exactly one fenced Python block:
   - opening fence: ```python
   - closing fence: ```

2. Main-generation parsing now accepts fenced code as the primary valid format.

3. Main-generation training samples are stored in the same fenced-code format.
   - This keeps prompt format and training target format aligned.

4. Logic / execution audit formats remain unchanged.
   - They still use tagged outputs because their format compliance is already high.

### What to check after the next run

1. `generation_format_ok_rate` should recover materially from the failed strict-`<CODE>` attempt.
2. `mean_pass_rate` should at least return toward the earlier Group D+ baseline.
3. Traces should show fenced code without extra prose outside the block.

### Representative run

- `20260311_014047__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. The fenced-code switch fixed the main-generation format collapse.
   - `generation_format_ok_rate` recovered to about `0.4575`.
   - This is not only far above the failed strict-`<CODE>` run (`~0.1525`), it is also slightly above the earlier Group D+ baseline (`~0.4325`).

2. Pass-related metrics also recovered and improved.
   - `mean_pass_rate` rose to about `0.2354`.
   - `best_pass_rate_overall` rose to about `0.5058`.
   - Both are better than the earlier Group D+ baseline.

3. The second half is again healthier than the first half.
   - First 50 steps:
     - `mean_pass_rate ~ 0.2267`
     - `best_pass_rate_overall ~ 0.48`
     - `generation_format_ok_rate ~ 0.44`
   - Second 50 steps:
     - `mean_pass_rate ~ 0.2442`
     - `best_pass_rate_overall ~ 0.5317`
     - `generation_format_ok_rate ~ 0.475`
   - This restores the "later training is better" pattern that was lost in the failed strict-`<CODE>` run.

4. Traces confirm that the new main-generation format matches the model prior.
   - Representative traces:
     - `mbpp_train_618_rank0_000077`
     - `mbpp_train_725_rank0_000069`
   - Their successful main-generation outputs are clean fenced Python blocks with no extra prose.

5. Remaining generation-format failures are now a more meaningful failure mode.
   - Example:
     - `mbpp_train_733_rank0_000038`
   - Here the output is still fenced Python, but the block is cut off before completion.
   - So the current failures are no longer mainly "wrong wrapper format"; they are more often incomplete generations or empty outputs.

6. Soft-reward optimism is still present on some unsolved cases, but it is not the dominant blocker.
   - Examples still include:
     - `mbpp_train_733`
     - `mbpp_train_605`
     - `mbpp_train_611`
     - `mbpp_train_640`
   - However, the overall system is now back in a healthy operating region.

### Interpretation

The fenced-code change should be kept.

This run shows:

- The earlier main-generation formatting problem was primarily a format-protocol mismatch with the model prior.
- Using fenced Python for main generation is materially better than forcing `<CODE>`.
- The main-generation bottleneck has shifted from "wrapper mismatch" to "incomplete / empty generations" on a subset of cases.

### Current recommendation

Keep the current fenced-code main-generation protocol as the new baseline.

Do not change reward balance yet.

If the next change is needed, it should target one of these:

1. incomplete fenced-code generations on hard cases
2. empty-output generations
3. residual soft-reward optimism on a few stubborn unsolved samples

## Next run setup

### Parameter change

- `max_completion_length: 80 -> 96`
- `max_steps: 100 -> 60`

### Why

Representative failed traces now show a different failure mode than before:

- the model often uses the correct fenced-Python wrapper,
- but some hard cases produce incomplete code blocks or empty outputs,
- e.g. truncated binary-search style code in `mbpp_train_733`.

The current implementation still uses a shared generation length budget across main generation and audit generation in the colocated vLLM path, so this is a conservative global bump rather than a split-length backend change.

### What to check after the next run

1. whether incomplete fenced-code failures reduce
2. whether `generation_format_ok_rate` improves further
3. whether `mean_pass_rate` and `best_pass_rate_overall` continue to improve
4. whether memory stays stable on the current A800 run

### Runtime note

This run is intentionally shortened to 60 steps because the current objective is targeted diagnosis:

- check whether longer completions reduce truncated fenced-code generations,
- without paying the full 100-step cost before the trend is clear.

### Representative run

- `20260311_073802__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. `max_completion_length = 96` did not produce a clear improvement over the fenced-code baseline.
   - `generation_format_ok_rate ~ 0.4583`, which is essentially flat relative to the previous fenced-code run (`~0.4575`).
   - `mean_pass_rate ~ 0.225`, slightly below the previous fenced-code run (`~0.2354`).
   - `best_pass_rate_overall ~ 0.50`, essentially flat relative to the previous run (`~0.5058`).

2. The shorter 60-step run does not show the same "second half gets better" pattern as the stronger 100-step fenced-code run.
   - First half:
     - `mean_pass_rate ~ 0.2528`
     - `best_pass_rate_overall ~ 0.5778`
   - Second half:
     - `mean_pass_rate ~ 0.1972`
     - `best_pass_rate_overall ~ 0.4222`
   - This suggests the 60-step run is noisier and less informative than the 100-step run.

3. Truncated / empty generation is still present, but it is no longer the dominant failure mode in sampled traces.
   - Good examples:
     - `mbpp_train_618_rank0_000051`
     - `mbpp_train_625_rank0_000054`
     - `mbpp_train_627_rank0_000028`
   - These show correct fenced Python outputs with `generation_format_ok = True`.
   - Remaining hard failures now still include:
     - empty outputs
     - occasional incomplete code
   - But the 96-token increase alone did not obviously unlock a new performance regime.

4. Memory remained stable on the current A800 run.
   - No instability or OOM regression was observed in this shortened 96-token run.

5. `final_reason` remains active.
   - `final_reason_node_count ~ 0.35`
   - This is actually higher than the previous fenced-code run, though the shorter run makes this harder to interpret as a stable trend.

### Interpretation

The fenced-code switch remains the important fix.

The `96`-token increase by itself is not yet a convincing improvement:

- it did not clearly raise format rate,
- it did not clearly raise pass metrics,
- and the shorter 60-step run is less reliable for trend judgment than the earlier 100-step run.

### Current recommendation

Do not treat `max_completion_length = 96` as a confirmed win yet.

At this point, the stronger baseline remains:

- fenced-code main generation
- Group D reward settings
- longer training horizon for trend judgment

If the next change is needed, it should probably not be another blind length increase.

## Group E plan: split code vs audit length

### Planned change

- `max_steps: 60 -> 100`
- shared `max_completion_length: 96 -> 128`
- `max_completion_length_code: 128`
- `max_completion_length_audit: 64`

### Why this is the current best next experiment

1. Main generation is still the only path that shows clear completion-length pressure on hard MBPP tasks.
2. Logic / execution audit format rates are already stable enough that they do not need the same token budget.
3. A split budget should give more room to code generation without wasting the same budget on audit prompts.
4. The next run should return to `100` steps so the trend is comparable to the stronger earlier baseline.

### What to check after the next run

1. whether incomplete fenced-code failures decrease
2. whether `generation_format_ok_rate` improves without hurting audit format rates
3. whether `mean_pass_rate` and `best_pass_rate_overall` improve over the current fenced-code baseline
4. whether memory remains stable under code `128` / audit `64`

### Representative run

- `20260311_092615__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. The split `128 / 64 / 100` setup did not improve the main generation bottleneck.
   - `mean_pass_rate ~ 0.2242`
   - `best_pass_rate_overall ~ 0.4867`
   - `generation_format_ok_rate ~ 0.4800`
   - Relative to the stronger fenced-code baseline, these numbers are flat to slightly worse.

2. The systematic "one sibling is usable, one sibling is empty" pattern still remains.
   - Representative traces:
     - `mbpp_train_616_rank0_000003`
     - `mbpp_train_733_rank0_000047`
     - `mbpp_train_737_rank0_000008`
   - In these traces, one node often has a valid fenced Python block while the sibling has `raw_output = ""` and `generation_format_ok = false`.
   - This keeps `generation_format_ok_rate` pinned near `0.5` on many questions.

3. Audit quality regressed under `max_completion_length_audit = 64`.
   - `logic_format_ok_rate ~ 0.8258`
   - `exec_format_ok_rate ~ 0.9000`
   - These are clearly below the earlier fenced-code runs, where logic / execution audit format rates were both materially higher.

4. The run got weaker in the second half rather than stronger.
   - First half:
     - `mean_pass_rate ~ 0.2667`
     - `best_pass_rate_overall ~ 0.5800`
     - `logic_confirmed_rate ~ 0.1650`
   - Second half:
     - `mean_pass_rate ~ 0.1817`
     - `best_pass_rate_overall ~ 0.3933`
     - `logic_confirmed_rate ~ 0.0950`
   - This is a negative trend compared with the more stable Group D fenced-code baseline.

5. Soft-reward optimism is still present on several zero-pass samples.
   - Representative hard samples:
     - `mbpp_train_733`
     - `mbpp_train_611`
     - `mbpp_train_640`
   - These still show `pass_rate = 0` with relatively high `mean_R_soft_raw`.

6. `final_reason` remains functional.
   - `final_reason_node_count ~ 0.31`
   - So this run did not break the main workflow; it simply failed to improve the core main-generation bottleneck.

### Interpretation

The split-length idea in this specific form is not a win.

- Giving code generation `128` tokens did not remove the empty-sibling pattern.
- Giving audit only `64` tokens made logic / execution audit worse.
- The main-generation problem now looks less like pure length pressure and more like a sampling / generation-path issue that still produces empty siblings.

### Current recommendation

Do not keep the current `128 / 64 / 100` setup as the new baseline.

The next change should likely be:

1. restore audit length to the stronger previous range, and
2. investigate the empty-output sibling behavior directly rather than only giving code generation more tokens.

## Group F plan: restore audit budget and stabilize main generation

### Planned change

- keep `max_completion_length_code = 128`
- restore `max_completion_length_audit: 64 -> 96`
- add code-only sampling overrides:
  - `generation_temperature_code = 0.7`
  - `generation_top_p_code = 0.95`
  - `generation_min_new_tokens_code = 16`
- retry empty main-generation outputs once:
  - `generation_empty_retry_count = 1`

### Why this is the current best next experiment

1. The latest `128 / 64 / 100` run showed that audit quality regressed under the shorter audit budget.
2. The main-generation bottleneck is no longer best explained by pure length pressure; many failures are empty sibling outputs.
3. Empty `raw_output` means the issue is upstream of the parser, so reward changes are not the right next move.
4. A small main-generation sampling stabilization is a narrower, more justified intervention than another reward change.

### What to check after the next run

1. whether the repeated `generation_format_ok_rate ~ 0.5` pattern improves
2. whether empty-sibling traces become less common
3. whether logic / execution audit format rates recover after restoring audit length
4. whether `mean_pass_rate` and `best_pass_rate_overall` return to or exceed the stronger fenced-code baseline

### Representative run

- `20260311_102038__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. This is a real improvement over the previous `128 / 64 / 100` run.
   - `mean_pass_rate ~ 0.4525`
   - `best_pass_rate_overall ~ 0.5533`
   - `generation_format_ok_rate ~ 0.8900`
   - `logic_format_ok_rate ~ 0.9867`
   - `exec_format_ok_rate ~ 0.9800`
   - `logic_confirmed_rate ~ 0.2608`
   - `final_reason_node_count ~ 0.55`

2. The repeated empty-sibling pattern is no longer the dominant failure mode.
   - Main-generation format rate is now far above the old `~0.5` ceiling.
   - The code-only sampling overrides plus one-shot empty-output retry appear to have worked.

3. Audit quality recovered after restoring audit length.
   - `logic_format_ok_rate` and `exec_format_ok_rate` both returned to the stronger earlier range.
   - This confirms that `max_completion_length_audit = 64` was too aggressive for this setup.

4. The run improves in the second half instead of degrading.
   - First half:
     - `mean_pass_rate ~ 0.3933`
     - `best_pass_rate_overall ~ 0.5200`
     - `logic_confirmed_rate ~ 0.2250`
     - `final_reason_node_count ~ 0.50`
   - Second half:
     - `mean_pass_rate ~ 0.5117`
     - `best_pass_rate_overall ~ 0.5867`
     - `logic_confirmed_rate ~ 0.2967`
     - `final_reason_node_count ~ 0.60`

5. A new residual issue appeared in some fenced-code generations: duplicated fence fragments.
   - Representative trace:
     - `mbpp_train_635_rank0_000005`
   - Example raw output:
     - ````python ... ```python ````
   - These cases are currently marked as generation-format-valid but become `SYNTAX_ERROR` after code extraction, so the parser/stop logic is still too permissive for this failure mode.

6. Soft-reward optimism still exists on a few hard zero-pass samples.
   - Representative cases:
     - `mbpp_train_624`
     - `mbpp_train_608`
     - `mbpp_train_743`
     - `mbpp_train_611`
   - These still show high `mean_R_soft_raw` even when `pass_rate = 0`.
   - However, this is no longer the main blocker in the current setup.

### Interpretation

This run is the strongest MBPP-small result so far.

The current combination appears substantially better:

- code length `128`
- audit length restored to `96`
- stabilized main-generation sampling
- one retry for empty main-generation outputs

The next highest-value fix is no longer about empty siblings. It is about sanitizing malformed duplicated fenced-code suffixes so they do not slip through as generation-format-valid syntax errors.

### Current recommendation

Keep this configuration as the working baseline for now.

If the next code change is needed, it should focus on:

1. stricter/smarter fenced-code parsing or stopping for duplicated fence fragments, and
2. only after that, revisiting the remaining soft-reward optimism on a few stubborn zero-pass questions.

## Eval Tracking Upgrade

To reduce noise from comparing training first-half vs second-half summaries, the MBPP-small config now enables fixed-step evaluation:

- `eval_strategy: steps`
- `eval_steps: 20`
- `seed: 42`
- `data_seed: 42`

Rationale:

- training-rollout summaries are high variance because sampled questions, audit cases, and sibling generations vary by step;
- fixed eval on the held-out split is a better signal for whether code pass rate and format quality are actually improving;
- future comparisons should prioritize `eval_mean_pass_rate`, `eval_best_pass_rate_overall`, and `eval_generation_format_ok_rate` over train-half vs train-half comparisons.

## Latest observed run after parser tightening

### Representative run

- `20260311_143920__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Observed results

1. The run remains healthy but is not stronger than the previous best baseline.
   - `mean_pass_rate ~ 0.4183`
   - `best_pass_rate_overall ~ 0.5267`
   - `generation_format_ok_rate ~ 0.8650`
   - `logic_format_ok_rate ~ 0.9800`
   - `exec_format_ok_rate ~ 0.9808`
   - `logic_confirmed_rate ~ 0.2667`
   - `final_reason_node_count ~ 0.49`

2. Logic / execution audit remain stable.
   - Audit format rates stay high.
   - `final_reason` still triggers in real training, so the one-shot reason-only branch remains active.
   - terminal logic backprop remains active on some solved steps (`logic_confirmed_rate > 0`, non-zero terminal bonus on selected steps).

3. Main-generation behavior is still slightly noisier than the strongest previous run.
   - Some steps still show weakened `generation_format_ok_rate`.
   - The old â€œzero-pass but high soft rewardâ€?pattern still appears on a few hard MBPP items:
     - `mbpp_train_733`
     - `mbpp_train_623`
     - `mbpp_train_644`
     - `mbpp_train_743`
     - `mbpp_train_608`

4. This run should be interpreted as a mild regression / sampling-variance outcome, not as a framework failure.
   - Core workflow remains intact.
   - Reward branches still function.
   - The remaining issue is reward calibration / generation stability on a subset of hard zero-pass problems.

### Important note

This reviewed run still does **not** include `eval_results.json`, so the newly enabled fixed-step eval configuration has not yet been validated by a completed bundle.

Future comparisons should stop using â€œfirst half vs second half of train logsâ€?as the primary basis and instead rely on:

- `eval_mean_pass_rate`
- `eval_best_pass_rate_overall`
- `eval_generation_format_ok_rate`

Implementation note:

- after enabling step-based eval, `per_device_eval_batch_size` must be divisible by `num_generations_eval`;
- the config was adjusted from `per_device_eval_batch_size = 1` to `2` to satisfy the GRPO eval batching constraint.

## Fixed-eval validation run

### Representative run

- `20260311_152858__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Train-side summary

1. The training run completed successfully and remained broadly healthy.
   - `mean_pass_rate ~ 0.3768`
   - `best_pass_rate_overall ~ 0.4619`
   - `generation_format_ok_rate ~ 0.8929`
   - `logic_format_ok_rate ~ 0.9815`
   - `exec_format_ok_rate ~ 0.9792`
   - `logic_confirmed_rate ~ 0.1893`
   - `final_reason_node_count ~ 0.4429`

2. Train-side performance is somewhat weaker than the strongest previous baseline run.
   - This is not a framework regression.
   - It is consistent with training variance plus the fact that train-rollout statistics depend on sampled questions and audit cases.

3. The old soft-reward optimism pattern still appears on some hard zero-pass problems.
   - Representative cases:
     - `mbpp_train_733`
     - `mbpp_train_623`
     - `mbpp_train_644`
     - `mbpp_train_743`
     - `mbpp_train_608`

### Fixed eval observations

The main value of this run is that fixed-step eval is now confirmed to work.

Observed eval checkpoints:

- step 20:
  - `eval_mean_pass_rate ~ 0.1771`
  - `eval_best_pass_rate_overall ~ 0.3333`
  - `eval_generation_format_ok_rate ~ 0.8125`
  - `eval_pass_at_2 ~ 0.25`

- step 40:
  - `eval_mean_pass_rate ~ 0.1354`
  - `eval_best_pass_rate_overall ~ 0.2083`
  - `eval_generation_format_ok_rate ~ 0.9063`
  - `eval_pass_at_1 ~ 0.125`
  - `eval_pass_at_2 ~ 0.125`

- step 60:
  - `eval_mean_pass_rate ~ 0.1667`
  - `eval_best_pass_rate_overall ~ 0.3333`
  - `eval_generation_format_ok_rate ~ 0.8438`
  - `eval_pass_at_1 ~ 0.125`
  - `eval_pass_at_2 ~ 0.125`

- step 80:
  - `eval_mean_pass_rate ~ 0.0729`
  - `eval_best_pass_rate_overall ~ 0.0833`
  - `eval_generation_format_ok_rate ~ 0.8438`
  - `eval_pass_at_1 ~ 0.0`
  - `eval_pass_at_2 ~ 0.0`

- step 100:
  - `eval_mean_pass_rate ~ 0.1667`
  - `eval_best_pass_rate_overall ~ 0.2917`
  - `eval_generation_format_ok_rate ~ 0.7813`
  - `eval_pass_at_1 ~ 0.0`
  - `eval_pass_at_2 ~ 0.25`

### Interpretation

1. Fixed eval confirms that the current setup is still noisy.
   - The eval curve is not monotonic.
   - Step 80 is a clear trough, followed by partial recovery at step 100.

2. The new eval path is still much more trustworthy than comparing â€œfirst half vs second halfâ€?of train-rollout logs.
   - Future model-quality comparisons should be based on fixed eval checkpoints first.

3. The current run should not be treated as a definitive improvement or collapse.
   - It is a usable reference run for eval-enabled analysis.
   - It shows that current variance is still high enough that single-run conclusions must be made cautiously.

### Artifact note

- eval metrics are present in `trainer_events_rank0.txt`.
- the bundle did not include a separate `eval_results.json`, so the current source of truth for step-based eval remains the trainer text log.

## Re-read of latest fixed-eval run

### Representative run

- `20260311_152858__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Consolidated findings

1. Train-side behavior still looks healthy, but train metrics are materially more optimistic than fixed eval.
   - Train aggregate remains decent:
     - `mean_pass_rate ~ 0.3768`
     - `best_pass_rate_overall ~ 0.4619`
     - `generation_format_ok_rate ~ 0.8929`
   - However fixed eval at step 100 is much weaker:
     - `eval_mean_pass_rate ~ 0.1667`
     - `eval_best_pass_rate_overall ~ 0.2917`
     - `eval_generation_format_ok_rate ~ 0.7813`

2. Fixed eval confirms that current variance is still high.
   - Eval checkpoints across the same run:
     - step 20: `eval_mean_pass_rate ~ 0.1771`
     - step 40: `eval_mean_pass_rate ~ 0.1354`
     - step 60: `eval_mean_pass_rate ~ 0.1667`
     - step 80: `eval_mean_pass_rate ~ 0.0729`
     - step 100: `eval_mean_pass_rate ~ 0.1667`
   - This is not a monotonic improvement curve.
   - Step 80 is a clear trough; step 100 partially recovers but does not establish a strong upward trend.

3. Logic / execution audit still look technically sound.
   - On fixed eval:
     - `eval_logic_format_ok_rate` stays around `1.0`
     - `eval_exec_format_ok_rate` stays around `1.0`
   - The remaining weakness is not parser failure on audit branches.
   - The main weakness remains code search / code generation quality on held-out samples.

4. Soft-reward optimism remains visible on held-out zero-pass problems.
   - Representative fixed-eval items:
     - `mbpp_train_728`
     - `mbpp_train_623`
     - `mbpp_train_644`
     - `mbpp_train_746`
   - These can still show `pass_rate = 0` with non-trivial `mean_R_soft_raw`, confirming that soft reward is still helping some wrong code branches too much.

5. Fixed eval is now the primary comparison source of truth.
   - Future tuning decisions should prioritize:
     - `eval_mean_pass_rate`
     - `eval_best_pass_rate_overall`
     - `eval_generation_format_ok_rate`
   - Train first-half vs second-half comparisons are still useful as side evidence, but they should no longer drive main conclusions.

### Current interpretation

- The framework is stable.
- The audit branches are stable.
- The remaining problem is not basic pipeline correctness but training variance plus residual soft-reward over-optimism on a subset of hard zero-pass items.
- Current setup is usable for further tuning, but single-run train metrics should not be treated as reliable evidence of general improvement without fixed-eval confirmation.

## Eval-design note

### User-side evaluation criterion proposal

The current full eval still replays the rollout tree, including audit and `final_reason` logic. This is useful for diagnosing whether reward branches are alive, but it is not the cleanest primary model-quality metric.

For primary evaluation, the better design is:

1. Eval should be code-only.
   - Do not run logic audit or execution audit during main eval.
   - The purpose of those branches during training is to improve code success, not to become the final reported metric.

2. Eval should use a single trajectory rather than a branching tree.
   - If training allows up to `T_max = N` code rounds, eval should follow one multi-round repair trajectory:
     - round 1 generate one code candidate
     - if failed, round 2 repair once
     - if failed, round 3 repair once
     - and so on

3. Report roundwise solve probability directly.
   - If maximum code rounds is 3, the main eval metrics should be:
     - round-1 `pass@1`
     - round-2 cumulative solve rate
     - round-3 cumulative solve rate
   - Equivalently:
     - probability solved within 1 round
     - probability solved within 2 rounds
     - probability solved within 3 rounds

### Interpretation

This is a more reasonable primary evaluation target than the current heavy full-rollout eval, because it measures what ultimately matters:

- whether the model can write correct code immediately;
- whether it can repair code within a bounded number of rounds;
- and how solve probability improves as more repair rounds are allowed.

### Practical recommendation

Keep two eval modes:

1. `full eval`
   - current rollout-tree eval with audit branches;
   - use only for reward / algorithm diagnostics.

2. `light eval`
   - code-only, single-trajectory, no audit;
   - use as the main comparable metric across experiments.

## Eval switch note

- The representative run `20260311_152858__...` was still produced by the old full-rollout eval path.
- Its `eval_logic_format_ok_rate`, `eval_exec_format_ok_rate`, and `eval_final_reason_*` metrics confirm that it was not yet using the new code-only single-trajectory eval.
- After this note, the default eval path and the MBPP small config were aligned to:
  - `eval_code_only_single_trajectory = true`
  - `num_generations_eval = 1`
  - `per_device_eval_batch_size = 1`
- Future eval interpretation should therefore focus on:
  - `eval_pass_at_1_round_1`
  - `eval_pass_at_1_round_2`
  - ...
  - `eval_pass_at_1_round_T_max`
  - and the final cumulative `eval_pass_at_1`

## Train-length and reward-window decision

- The next comparison run should use `max_steps = 200`, not `100`.
- Rationale:
  - `100` steps are enough to expose obvious failures;
  - but they are still short for judging stable trend direction in this rollout-heavy RL setup.
- `200` steps remain cheap enough on the current 7B setup while giving a more informative curve.

### Windowed reward logging

Train-side window logging was added for easier TensorBoard analysis.

- A new config field was introduced:
  - `reward_window_bins = N`
- Window interval is computed as:
  - `ceil(max_steps / N)`
- For the current MBPP small config:
  - `max_steps = 200`
  - `reward_window_bins = 10`
  - so one reward window is logged every `20` steps

The window metrics are:

- `window/mean_R_code`
- `window/mean_R_reason`
- `window/mean_R_soft_effective`
- `window/steps`
- `window/end_step`

This does not change optimization. It only adds a lower-variance train-side reward summary to complement eval.

## Group H: first valid code-only eval + reward-window run

Representative run:
- `20260311_165502__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### What this run validates

1. The new eval path is now correct.
   - Eval no longer reports audit-branch metrics such as `eval_logic_format_ok_rate` or `eval_exec_format_ok_rate`.
   - Eval now reports only code-centric metrics:
     - `eval_pass_at_1_round_1`
     - `eval_pass_at_1_round_2`
     - `eval_pass_at_1`
     - `eval_generation_format_ok_rate`
     - `eval_best_pass_rate_overall`
   - This is the first run that should be treated as the new primary comparable eval baseline.

2. Window reward logging is working.
   - Every `20` train steps, TensorBoard/logs now emit:
     - `window/mean_R_code`
     - `window/mean_R_reason`
     - `window/mean_R_soft_effective`
     - `window/steps`
     - `window/end_step`
   - This provides a lower-variance train-side trend summary than raw per-step reward.

### Eval findings

Fixed code-only eval checkpoints:

- step 20:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.875`
  - `eval_pass_at_1 ~ 0.5`
- step 40:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.8125`
  - `eval_pass_at_1 ~ 0.5`
- step 60:
  - `eval_mean_pass_rate ~ 0.4167`
  - `eval_best_pass_rate_overall ~ 0.4583`
  - `eval_generation_format_ok_rate ~ 0.875`
  - `eval_pass_at_1 ~ 0.375`
- step 80:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.875`
  - `eval_pass_at_1 ~ 0.5`
- step 100:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.75`
  - `eval_pass_at_1 ~ 0.5`
- step 120:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.875`
  - `eval_pass_at_1 ~ 0.5`
- step 140:
  - `eval_mean_pass_rate ~ 0.7083`
  - `eval_best_pass_rate_overall ~ 0.7083`
  - `eval_generation_format_ok_rate ~ 0.9375`
  - `eval_pass_at_1 ~ 0.625`
- step 160:
  - `eval_mean_pass_rate ~ 0.4583`
  - `eval_best_pass_rate_overall ~ 0.4583`
  - `eval_generation_format_ok_rate ~ 0.8125`
  - `eval_pass_at_1 ~ 0.375`
- step 180:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.9375`
  - `eval_pass_at_1 ~ 0.5`
- step 200:
  - `eval_mean_pass_rate ~ 0.5833`
  - `eval_best_pass_rate_overall ~ 0.5833`
  - `eval_generation_format_ok_rate ~ 0.875`
  - `eval_pass_at_1 ~ 0.5`

Interpretation:

1. Eval is now much easier to interpret than the old full-rollout eval.
   - It is directly reporting code solve probability, not audit-branch behavior.

2. There is still substantial variance.
   - The best checkpoint in this run is around step `140`, not the final step.
   - Final-step eval is respectable, but not the peak.

3. `eval_pass_at_1_round_1` and `eval_pass_at_1_round_2` are equal throughout this run.
   - This means the second repair round did not increase strict solve probability on the fixed eval split in this run.
   - So the current eval improvement is still coming mainly from first-round code quality, not from extra repair-round wins on the held-out set.

### Window reward findings

Windowed train rewards:

- step 20:
  - `window/mean_R_code ~ 0.8171`
  - `window/mean_R_reason ~ 0.5917`
  - `window/mean_R_soft_effective ~ 0.7581`
- step 40:
  - `window/mean_R_code ~ 0.5639`
  - `window/mean_R_reason ~ 0.4885`
  - `window/mean_R_soft_effective ~ 0.3403`
- step 60:
  - `window/mean_R_code ~ 0.6035`
  - `window/mean_R_reason ~ 0.4948`
  - `window/mean_R_soft_effective ~ 0.4890`
- step 80:
  - `window/mean_R_code ~ 0.6373`
  - `window/mean_R_reason ~ 0.5167`
  - `window/mean_R_soft_effective ~ 0.6542`
- step 100:
  - `window/mean_R_code ~ 0.4336`
  - `window/mean_R_reason ~ 0.5037`
  - `window/mean_R_soft_effective ~ 0.4008`
- step 120:
  - `window/mean_R_code ~ 0.4245`
  - `window/mean_R_reason ~ 0.5600`
  - `window/mean_R_soft_effective ~ 0.4577`
- step 140:
  - `window/mean_R_code ~ 0.7994`
  - `window/mean_R_reason ~ 0.5560`
  - `window/mean_R_soft_effective ~ 0.5998`
- step 160:
  - `window/mean_R_code ~ 0.5552`
  - `window/mean_R_reason ~ 0.5256`
  - `window/mean_R_soft_effective ~ 0.5174`
- step 180:
  - `window/mean_R_code ~ 0.7336`
  - `window/mean_R_reason ~ 0.6117`
  - `window/mean_R_soft_effective ~ 0.5855`
- step 200:
  - `window/mean_R_code ~ 0.5921`
  - `window/mean_R_reason ~ 0.3850`
  - `window/mean_R_soft_effective ~ 0.4615`

Interpretation:

1. Windowed reward logs are useful and should be kept.
   - They expose trend changes much more cleanly than raw per-step reward.

2. `R_code` and `R_soft_effective` still move together strongly.
   - This is consistent with residual soft-reward influence on code reward.

3. Residual soft-reward optimism is still visible late in training.
   - Representative late train steps still show:
     - `mean_pass_rate = 0`
     - `mean_R_soft_effective ~ 0.84`
   - So even with cleaner eval, the underlying optimism issue has not disappeared.

### Current interpretation

- This run successfully validates the new eval design and the new reward-window logging.
- The code-only eval is a better primary comparison target than the old full-rollout eval.
- The current 7B setup is trainable and can reach `eval_pass_at_1 ~ 0.625` at its best checkpoint in this run.
- However, final-step performance is not yet stable enough to assume monotonic improvement.
- The main remaining issues are:
  - residual soft-reward optimism on some zero-pass train questions;
  - second repair round not yet improving held-out cumulative solve rate.

### Practical recommendation

For future tuning:

1. Use code-only eval as the main decision metric.
2. Keep reward-window logging enabled.
3. Compare checkpoints by peak eval around the training horizon, not only final-step eval.
4. Do not conclude that `T_max = 2` repair is helping eval until `eval_pass_at_1_round_2` rises above `eval_pass_at_1_round_1`.

## Dataset-and-horizon reset

The previous MBPP-small runs were still too small to draw stable trend conclusions:

- train was effectively only about `56` examples after the tiny split;
- eval was only about `8` examples;
- `eval_pass_at_1` therefore moved in large `12.5%` increments.

This makes both:
- train-half vs train-half comparisons,
- and fixed-eval checkpoint comparisons,

too noisy for confident tuning decisions.

### New baseline decision

Switch the working configuration to:

- train set: `mbpp_sanitized_codegrpo_train.jsonl` (`120` examples)
- eval set: `mbpp_sanitized_codegrpo_validation.jsonl` (`43` examples)
- `max_steps = 300`
- `eval_steps = 50`

Tree/search hyperparameters remain unchanged for now:

- `T_max = 2`
- `N_max = 4`
- `K = 2`
- `K_reason = 1`

### Rationale

1. The next bottleneck is statistical stability, not search depth.
2. `T_max = 2` should not be increased yet, because current code-only eval still shows:
   - `eval_pass_at_1_round_2 == eval_pass_at_1_round_1`
   - so the second repair round is not yet buying additional held-out solves.
3. Increasing train/eval size and horizon should make:
   - reward windows,
   - code-only eval checkpoints,
   - and trend comparisons
   meaningfully more reliable.

## Run: 20260311_222240 (train120 / eval43 / 300 steps / code-only eval)

This is the first larger-train, larger-eval run with the new code-only eval protocol that is large enough to judge trend quality with less noise.

### Main results

Fixed eval checkpoints:

- step 50:
  - `eval_pass_at_1_round_1 = 0.5116`
  - `eval_pass_at_1_round_2 = 0.5116`
  - `eval_pass_at_1 = 0.5116`
  - `eval_generation_format_ok_rate = 0.8605`
- step 100:
  - `eval_pass_at_1_round_1 = 0.5581`
  - `eval_pass_at_1_round_2 = 0.5814`
  - `eval_pass_at_1 = 0.5814`
  - `eval_generation_format_ok_rate = 0.8488`
- step 150:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`
  - `eval_generation_format_ok_rate = 0.8140`
- step 200:
  - `eval_pass_at_1_round_1 = 0.5581`
  - `eval_pass_at_1_round_2 = 0.5581`
  - `eval_pass_at_1 = 0.5581`
  - `eval_generation_format_ok_rate = 0.8023`
- step 250:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`
  - `eval_generation_format_ok_rate = 0.8140`
- step 300:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`
  - `eval_generation_format_ok_rate = 0.8256`

Window reward snapshots (`reward_window_bins = 10`, so each window is `30` steps):

- step 30:
  - `window/mean_R_code ~ 0.6489`
  - `window/mean_R_reason ~ 0.4532`
  - `window/mean_R_soft_effective ~ 0.5061`
- step 60:
  - `window/mean_R_code ~ 0.6921`
  - `window/mean_R_reason ~ 0.5612`
  - `window/mean_R_soft_effective ~ 0.6558`
- step 90:
  - `window/mean_R_code ~ 0.6737`
  - `window/mean_R_reason ~ 0.4304`
  - `window/mean_R_soft_effective ~ 0.4565`
- step 120:
  - `window/mean_R_code ~ 0.4632`
  - `window/mean_R_reason ~ 0.4104`
  - `window/mean_R_soft_effective ~ 0.4782`
- step 150:
  - `window/mean_R_code ~ 0.6371`
  - `window/mean_R_reason ~ 0.5287`
  - `window/mean_R_soft_effective ~ 0.5800`
- step 180:
  - `window/mean_R_code ~ 0.5640`
  - `window/mean_R_reason ~ 0.5860`
  - `window/mean_R_soft_effective ~ 0.4618`
- step 210:
  - `window/mean_R_code ~ 0.6640`
  - `window/mean_R_reason ~ 0.6418`
  - `window/mean_R_soft_effective ~ 0.6359`
- step 240:
  - `window/mean_R_code ~ 0.7354`
  - `window/mean_R_reason ~ 0.5189`
  - `window/mean_R_soft_effective ~ 0.3819`
- step 270:
  - `window/mean_R_code ~ 0.7622`
  - `window/mean_R_reason ~ 0.4901`
  - `window/mean_R_soft_effective ~ 0.4919`
- step 300:
  - `window/mean_R_code ~ 0.5650`
  - `window/mean_R_reason ~ 0.4815`
  - `window/mean_R_soft_effective ~ 0.3918`

### Interpretation

1. The larger `eval` split is already more useful than the old `8`-example holdout.
   - The curve still moves, but it no longer jumps in meaningless `12.5%` increments.

2. Best held-out checkpoint in this run is step `100`.
   - `eval_pass_at_1 = 0.5814`
   - later checkpoints do not improve on that.
   - This confirms that selecting by best eval checkpoint is necessary; final-step eval is not a safe proxy.

3. The second repair round now shows a small but real gain at least at one checkpoint.
   - step `100` has:
     - `eval_pass_at_1_round_1 = 0.5581`
     - `eval_pass_at_1_round_2 = 0.5814`
   - This is the first meaningful sign that the second code-repair round can add held-out solves.
   - However, the effect is not yet stable across later checkpoints.

4. Code generation format is acceptable but still not fully stable.
   - `eval_generation_format_ok_rate` stays in about the `0.80 ~ 0.86` range.
   - So the main-generation formatting problem is much improved, but not completely solved.

5. Residual soft-reward optimism remains.
   - Representative zero-pass but high-soft examples still appear:
     - `mbpp_train_733`
     - `mbpp_train_788`
     - `mbpp_train_623`
     - `mbpp_train_743`
     - `mbpp_train_786`
   - Typical pattern:
     - `mean_pass_rate = 0`
     - `mean_R_soft_effective >= 0.92`
     - `R_code` still lifted into roughly `0.20 ~ 0.32`

6. Reward windows are useful but still too noisy to replace fixed eval.
   - They show phases and oscillation clearly.
   - They should remain diagnostic signals, not the primary model-selection metric.

### Current recommendation

1. Keep code-only eval as the primary decision metric.
2. Keep reward-window logging enabled.
3. Select checkpoints by peak `eval_pass_at_1`, not final-step eval.
4. Do not yet increase search depth.
   - The second round has shown only a small, unstable benefit.
5. The next tuning target, if any, should still be residual soft-reward optimism rather than audit formatting.

## Schedule switch: step-driven -> epoch-driven

The training schedule is now switched to an explicit epoch form:

- `shuffle_dataset = true` remains enabled
- each epoch traverses the training questions once in shuffled order
- training control is now:
  - `num_train_epochs = 2`
  - `max_steps = -1`
- eval control is now:
  - `eval_strategy = "epoch"`

Rationale:

1. This matches the intended interpretation better than truncating by raw global step count.
2. It guarantees that each epoch covers the full train set once before reshuffling for the next epoch.
3. It makes later comparisons cleaner:
   - epoch 1 vs epoch 2
   - eval after epoch 1 vs eval after epoch 2

Important note:

- The GRPO trainer still uses its internal prompt repetition/reuse mechanism for generation caching across steps.
- So optimizer-step accounting is still more complex than plain SFT.
- But at the dataset traversal level, the schedule is now epoch-driven rather than max-step-driven.

## Baseline-and-repeat eval addition

The config now adds two evaluation-oriented controls:

- `run_base_model_baseline_eval = true`
- `eval_repeat_count = 3`

Meaning:

1. Before PEFT/RL training starts, the raw base model is evaluated once on the eval split using the same code-only eval harness.
   - Metrics are saved as `baseline_eval_results.json`.

2. Each code-only eval now repeats every validation question multiple times with different eval seeds and averages the resulting metrics.
   - This is intended to reduce evaluation variance from stochastic sampling.

Rationale:

1. We need a fair baseline under the exact same local harness before comparing RL checkpoints.
2. External MBPP pass@1 numbers are not directly comparable unless prompt, decoding, and harness match.
3. Repeated eval is a cleaner way to reduce noise than trying to over-interpret single noisy eval passes.

## Eval cadence adjustment

The train loop remains epoch-driven, but eval is switched back to step-based checkpoints.

Current schedule:

- `num_train_epochs = 3`
- `max_steps = -1`
- `eval_strategy = "steps"`
- `eval_steps = 60`

Rationale:

1. Epoch-driven training is still useful because it guarantees full train-set coverage before reshuffling.
2. But epoch-only eval is too sparse when the total epoch count is small.
   - With only `3` epochs, we would only get `3` validation points.
3. Step-based eval gives a denser trend line while keeping the underlying train schedule epoch-based.
4. `eval_steps = 60` is a compromise:
   - frequent enough to observe trend changes,
   - not so frequent that eval dominates runtime.

## Dataset expansion: train163 / eval241

The working dataset split is now expanded to:

- train: `mbpp_sanitized_codegrpo_trainval.jsonl` (`163` examples)
- eval: `mbpp_sanitized_codegrpo_test.jsonl` (`241` examples)

Rationale:

1. The previous `train=120 / eval=43` setup was still small enough that held-out trends could remain noisy.
2. A larger eval set should make:
   - `eval_pass_at_1_round_1`
   - `eval_pass_at_1_round_2`
   - `eval_pass_at_1`
   more statistically meaningful.
3. The user explicitly prefers the larger `241`-example eval despite the added runtime cost, because the main bottleneck is still training rather than validation throughput.

Expected tradeoff:

- eval will be noticeably slower than before;
- but the resulting curve should be much more trustworthy than the smaller `43`-example eval.

## Follow-up interpretation after the 300-step larger-split run

### Code-only eval checkpoints every 50 steps

From the step-driven run on train `120` / eval `43`, the fixed code-only eval changed as follows:

- step 50:
  - `eval_pass_at_1_round_1 = 0.5116`
  - `eval_pass_at_1_round_2 = 0.5116`
  - `eval_pass_at_1 = 0.5116`
- step 100:
  - `eval_pass_at_1_round_1 = 0.5581`
  - `eval_pass_at_1_round_2 = 0.5814`
  - `eval_pass_at_1 = 0.5814`
- step 150:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`
- step 200:
  - `eval_pass_at_1_round_1 = 0.5581`
  - `eval_pass_at_1_round_2 = 0.5581`
  - `eval_pass_at_1 = 0.5581`
- step 250:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`
- step 300:
  - `eval_pass_at_1_round_1 = 0.5349`
  - `eval_pass_at_1_round_2 = 0.5349`
  - `eval_pass_at_1 = 0.5349`

### Current bottlenecks

1. First-round solve rate is still below the rough external target that has been informally cited (`~0.61` for MBPP pass@1).
   - In our own code-only eval, the best first-round point observed so far is only about `0.5581`.
   - So first-round code quality is still the main gap.

2. Second-round repair is only weakly helpful.
   - It gave a real but small gain at step `100`:
     - `0.5581 -> 0.5814`
   - But that gain did not remain stable at later checkpoints.
   - This means repair is not useless, but it is not yet a strong source of improvement.

3. Residual soft-reward optimism still exists.
   - Some train questions still show:
     - `pass_rate = 0`
     - but `mean_R_soft_effective` remains very high.
   - So the soft reward is still able to lift bad code more than desired on some examples.

4. The main-generation formatting problem is no longer the dominant issue.
   - Formatting is now much healthier than before.
   - The bigger remaining issue is code quality, not parser compliance.

### Recommended next direction

1. Do not increase search depth yet.
   - `T_max = 2` is not the clear bottleneck yet.
   - First-round generation quality is the bigger problem.

2. Do not aggressively retune the reward again yet.
   - The reward is imperfect, but it is no longer the first thing blocking progress.

3. If comparing against an external MBPP first-round pass@1 number, the next clean experiment should be:
   - evaluate the raw base model under this exact same code-only eval harness,
   - then compare the best RL checkpoint to that baseline.
   - Without that, we still do not know how much of the current number comes from the base model itself.

4. Training can remain epoch-driven, but eval should preferably stay step-based if trend visibility is important.
   - Epoch-driven train coverage is cleaner.
   - But eval only once per epoch gives too few checkpoints for diagnosis.

### Infra note

- Eval trace dumping is now disabled by default.
  - Reason: code-only eval no longer needs per-question JSON dumps in normal runs.
  - This keeps review bundles smaller and avoids mixing train trace diagnostics with eval artifacts.
- Eval console logging is now also reduced.
  - Code-only eval no longer prints per-question `[REWARD]` or `[EVAL_TREE]` lines by default.
  - Console output should now focus on checkpoint-level eval metrics instead of rollout internals.

## 2026-03-12 run: trainval-163 / test-241 / 3 epochs / baseline-enabled

Run:
- `20260312_003117__train__qwen2.5-coder-7b-instruct__json-mbpp_sanitized_codegrpo___vllm`

### Run shape

- Training split: `163` MBPP trainval questions
- Eval split: `241` MBPP test questions
- Train schedule: `3` epochs
- Effective trainer steps per epoch: about `326`
- Total trainer steps: `978`
- Eval cadence: every `60` steps
- Eval repeat count: `3`

### Raw base-model baseline under the same harness

From `baseline_eval_results.json`:

- `eval_pass_at_1_round_1 = 0.4813`
- `eval_pass_at_1_round_2 = 0.5201`
- `eval_pass_at_1 = 0.5201`
- `eval_generation_format_ok_rate = 0.8396`

This is important because it shows the current harness itself is already fairly strong for the raw Qwen2.5-Coder-7B-Instruct baseline. Earlier comparisons without this baseline were too noisy.

### Eval checkpoints

- step `60`
  - `round_1 = 0.4965`
  - `round_2 = 0.5173`
  - `final = 0.5173`
- step `120`
  - `round_1 = 0.4841`
  - `round_2 = 0.5076`
  - `final = 0.5076`
- step `180`
  - `round_1 = 0.4924`
  - `round_2 = 0.5187`
  - `final = 0.5187`
- step `240`
  - `round_1 = 0.4703`
  - `round_2 = 0.5159`
  - `final = 0.5159`
- step `300`
  - `round_1 = 0.4896`
  - `round_2 = 0.5270`
  - `final = 0.5270`
- step `360` (best checkpoint)
  - `round_1 = 0.5090`
  - `round_2 = 0.5436`
  - `final = 0.5436`
- step `420`
  - `round_1 = 0.4855`
  - `round_2 = 0.5214`
  - `final = 0.5214`
- step `480`
  - `round_1 = 0.5021`
  - `round_2 = 0.5339`
  - `final = 0.5339`
- step `540`
  - `round_1 = 0.4813`
  - `round_2 = 0.5187`
  - `final = 0.5187`
- step `600`
  - `round_1 = 0.5035`
  - `round_2 = 0.5367`
  - `final = 0.5367`
- step `660`
  - `round_1 = 0.4952`
  - `round_2 = 0.5270`
  - `final = 0.5270`
- step `720`
  - `round_1 = 0.4855`
  - `round_2 = 0.5214`
  - `final = 0.5214`
- step `780`
  - `round_1 = 0.5090`
  - `round_2 = 0.5367`
  - `final = 0.5367`
- step `840`
  - `round_1 = 0.5007`
  - `round_2 = 0.5311`
  - `final = 0.5311`
- step `900`
  - `round_1 = 0.4869`
  - `round_2 = 0.5173`
  - `final = 0.5173`
- step `960`
  - `round_1 = 0.5007`
  - `round_2 = 0.5297`
  - `final = 0.5297`

### Interpretation

1. RL is helping, but only modestly.
   - Best checkpoint (`step 360`) improves over raw base:
     - round 1: `0.4813 -> 0.5090` (`+0.0277`)
     - final: `0.5201 -> 0.5436` (`+0.0235`)
   - This is a real gain, but much smaller than earlier impression-based conclusions.

2. Second-round repair is now consistently useful, but the gain is still small.
   - At most checkpoints:
     - `round_2 > round_1`
   - The typical gain is about `+0.02 ~ +0.04`
   - This is much better than the earlier stage where round 2 often added nothing.

3. The best checkpoint appears early relative to total training length.
   - Best point is around `step 360`
   - Later checkpoints oscillate around that level instead of clearly improving
   - This strongly suggests we should select by best eval checkpoint rather than by final step.

4. Eval formatting is stable enough and is no longer the limiting factor.
   - `eval_generation_format_ok_rate` stays in roughly the `0.84 ~ 0.86` band.

5. Remaining problem is not parser stability; it is still code quality and soft-reward optimism.
   - There are still training cases with:
     - `pass_rate = 0`
     - but `mean_R_soft_effective` very high
   - So the residual optimism issue is still real, though it is no longer a catastrophic blocker.

### Practical takeaways

1. Keep the code-only eval harness.
   - It is now finally informative.

2. Do not judge progress by the final checkpoint.
   - Best checkpoint selection is necessary.

3. Do not increase tree depth yet.
   - Round 2 already adds a little value now.
   - The bigger gain target is still round-1 code quality.

4. The current training schedule is probably longer than necessary.
   - Since the best checkpoint is already around `step 360`, a shorter run may be sufficient for iteration speed.

## Eval repeat statistics and reporting update

### Why this was added

The fixed eval now uses:
- `241` held-out questions
- repeated evaluation runs per checkpoint (`eval_repeat_count`)

Flattening all repeats into one mean is numerically valid, but it hides repeat-to-repeat variance. For analysis and paper-style reporting, the more useful quantity is:
- mean across eval repeats
- standard deviation across eval repeats

### What is now logged

For code-only eval metrics, the trainer now records:
- `eval_pass_at_1_round_k`
- `eval_pass_at_1_round_k_std`

and likewise for the other eval-side metrics produced by the code-only trajectory evaluator.

### Plotting update

`review_bundle/plots/eval_pass_at_1_by_round.png` now uses:
- trainer step / eval checkpoint on the x-axis
- one line per repair horizon (`<= round k`)
- shaded bands for `mean +/- std` when eval repeats are enabled

### Important evaluation note

At this stage, the `241`-question MBPP test split is being used as model-selection eval during training. That is acceptable for internal iteration, but it means this split is no longer an untouched final test set in the strict academic sense.

## Logging and review bundle slimming

To keep future runs readable and easier to share:

- trainer step logs are now recorded every `5` steps instead of every step
- duplicated plain-text trainer logs are disabled by default
- train rollout traces are dumped sparsely instead of effectively every step
- rollout summary jsonl no longer duplicates the same eval metrics in nested form
- `review_bundle` now keeps only the compact essentials:
  - `trainer_events_rank*.jsonl`
  - `rollout_summary_rank*.jsonl`
  - result json files
  - generated plots
  - a very small sampled set of traces

Verbose runtime logs and duplicated text logs still remain in the full run directory when needed for debugging, but they are no longer copied into `review_bundle` by default.

## Follow-up config decision after the baseline-aligned run

### Best-checkpoint policy

The next training runs should not keep `save_strategy = "no"`.

Reason:
- We now know the best model is not the final one.
- In the latest larger-split run, the best eval point was around `step 360`, while later checkpoints oscillated below it.

So the config is updated to:
- save every eval point
- keep only a small number of checkpoints
- reload the best model at the end

Target metric:
- `eval_pass_at_1`

### Eval-round extension

Training still uses:
- `T_max = 2`

But eval now gets an independent override:
- `eval_T_max_override = 5`

Reason:
- The method is explicitly multi-round.
- Even if training only exposes two repair rounds, code-only eval should also measure whether the learned policy remains useful under longer repair horizons.
- This is especially important for comparison with vanilla GRPO, because the current method may not dominate on first-round pass@1 but may still benefit more from multi-round repair.

The intended readout now becomes:
- `eval_pass_at_1_round_1`
- `eval_pass_at_1_round_2`
- ...
- `eval_pass_at_1_round_5`

These are cumulative solve rates within `<= r` repair rounds.

## Plot exports for paper-style inspection

The review bundle now auto-generates:

- `plots/train_reward_curves.png`
  - training-side curves for:
    - `R_code`
    - `R_reason`
    - `R_soft_effective`
- `plots/eval_pass_at_1_by_round.png`
  - eval curves for:
    - `pass@1 within <= round 1`
    - `pass@1 within <= round 2`
    - ...
    - up to the configured eval repair horizon

This is acceptable for paper-style reporting if:
- the eval set is fixed,
- the metric definition is explicitly stated,
- and final claims are backed by multiple seeds (mean/std), not just one run.

For internal iteration this plotting setup is sufficient and much easier to read than raw jsonl logs.
## Eval Policy Update (2026-03-12)
- Eval now uses a single deterministic run per checkpoint: `eval_repeat_count = 1`.
- Code-only eval generation now uses `eval_generation_temperature_code = 0.0`.
- This makes checkpoint selection cleaner and shifts variance estimation to external multi-seed runs.
- Going forward, report `mean ¡À std` across independent experiment seeds instead of repeated eval sampling within one run.

## Training Length Update (2026-03-12)
- num_train_epochs is reduced from 3 to 2.
- Reason: previous larger runs tended to peak early (roughly around the old mid-training region) and then oscillate rather than improve monotonically.
- Since best-checkpoint loading is already enabled, extending training far beyond the typical peak is mostly slowing iteration rather than improving model quality.
- The new target is to keep total training closer to the previously observed best-checkpoint region while making multi-seed runs cheaper.



## 2026-03-12 - Eval metric prefix fix

- Fixed a regression where eval metrics were being prefixed twice (`eval_eval_*`).
- Root cause: `CodeGRPOTrainer.log()` already prefixes eval metrics with `eval_`, and `CodeGRPOTrainer.evaluate()` was adding `eval_` again when merging `_last_eval_metrics`.
- Impact before fix:
  - `metric_for_best_model: eval_pass_at_1` could not be found.
  - training crashed at the first checkpoint-selection point.
- Impact after fix:
  - best-checkpoint selection works again with `metric_for_best_model: eval_pass_at_1`
  - future runs should report a single `eval_` prefix only
