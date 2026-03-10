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
