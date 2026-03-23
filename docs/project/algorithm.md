## Current Algorithm

This project now keeps only the current main algorithm path:

1. Single-round code GRPO.
2. Each problem generates a primary batch of `K=num_generations` code candidates in one round, with optional bounded retry only for degenerate retry paths such as `M_retry` or `undiff_retry`.
3. Hard reward is unit-test pass rate.
4. Candidates can receive a bounded soft reward in addition to hard reward.
5. Optional `code + input -> output/error` auxiliary training uses a separate auxiliary SFT loss.
6. Optional pseudo-multiround is implemented at the trainer/data-pool level, not as tree-search reasoning.
7. Optional question-level EMA prior reweights sampling and iterative-node priority.

Legacy logic-audit, exec-audit, final-reason, and heavy multiround search are no longer part of the active algorithm.

### Main Rollout

For each problem:

1. Build one code-generation prompt from the problem text plus optional recent iterative history.
2. Generate `K` primary code candidates.
3. Run unit tests for each candidate.
4. Compute `hard_reward = pass_cnt / test_count`.
5. If bounded soft reward is enabled, compute soft reward on the selected diagnostic set.
6. Use the final reward for sibling-group GRPO advantage computation.

### Final Reward

For every candidate:

- `final_reward = hard_reward + bounded soft reward + bounded compile/format shaping`

where:

- `beta < 1 / test_count`
- the soft term is `beta * effective_normalized_soft_reward`
- if no valid diagnostic evidence is available, `effective_normalized_soft_reward = 0`
- if the sample is soft-reward-ineligible, `effective_normalized_soft_reward` first applies the configured ineligible scaling before `beta`

This keeps soft reward strictly below genuine hard-reward progress and avoids treating `no evidence` as a positive signal.

### Bounded Soft Reward

The bounded soft reward path compares:

- `problem + input -> oracle output` confidence
- `code + input -> oracle output` confidence

and uses their average log-probability delta across a bounded diagnostic set.

The raw delta is clipped and mapped into `[0, 1]`, then scaled by `beta`.

Important implementation detail:

- if all diagnostic comparisons are skipped or invalid, the soft reward path contributes `0` rather than a neutral positive midpoint

### Code-IO Auxiliary Training

This auxiliary branch is optional and does not change the main reward definition.

For selected execution cases from both correct and incorrect candidates, the model receives:

- input: `code + input`
- target: `actual output` or `error type`

This branch uses a separate supervised loss on the auxiliary targets. It exists to improve program-behavior modeling and make bounded soft reward more reliable.

### Pseudo-Multiround

Pseudo-multiround is a trainer-side pool, not a search tree.

If a problem rollout produces no fully solved candidate, the trainer keeps up to `pseudo_iterative_select_count` iterative nodes:

- best pass candidate
- best soft candidate
- one novelty-oriented candidate

Later steps alternate between original problems and iterative nodes, but only after an optional warmstart phase that forces every original question to be rolled out once. After warmstart, the final source choice depends on original-problem keep probability, iterative-node priority, and optional forced-original sampling. The pool is bounded by capacity and TTL, and solved questions prune stale iterative nodes to reduce off-policy drift.

Iterative-node priority is repair-shaped rather than monotonic in reward:

- mid-pass candidates are preferred over nearly solved candidates
- zero-pass candidates can still enter through a smaller positive soft-reward bonus
- question prior weight multiplies iterative-node priority at insertion time

### Question-Level EMA Prior

This optional feature keeps a per-question moving estimate of:

- recent code success
- recent reason-signal strength

It does not change reward. It only changes:

- original-problem keep probability
- iterative-node priority
- weighted sampler refresh when enabled

The `too_hard` bucket is gated by a minimum observation count before it can activate. This avoids aggressively downweighting questions during the cold-start phase when the model has only seen them a small number of times.

### Evaluation Semantics

The active eval path is code-only single-trajectory eval:

- one actual repair trajectory is executed per eval sample
- each executed round generates exactly one repair candidate, then either stops on success or continues with the next repair round
- `pass_at_1_round_r` and `best_pass_rate_round_r` reflect only the rounds that were actually executed
- when `eval_code_only_single_trajectory=true`, `eval_T_max_override` controls the maximum number of consecutive single-trajectory repair rounds

### Original-Problem Recovery

Pseudo-multiround can optionally keep hard problems from sinking forever:

- original-problem keep probability can receive an age-based recovery bonus
- a small fraction of rollout slots can be forced to stay on original problems

This keeps difficult questions periodically returning to on-policy exploration without forcing every iterative node or every hard question to be retried immediately. When warmstart coverage is enabled, pseudo-multiround does not begin until every original question has been rolled out once as an original problem.

### Retained Configs

The current runnable configs are:

- `trl/configs/comparison/server_2gpu_smoke/raw_qwen7b_eval_mbpp.yaml`
- `trl/configs/comparison/server_2gpu_smoke/codegrpo_single_round_zero_pass_soft_reward.yaml`
- `trl/configs/comparison/server_2gpu_smoke/codegrpo_pseudo_multiround_zero_pass_soft_reward.yaml`

### Removed From Mainline

The following are intentionally removed from the active path:

- logic audit training
- exec audit training
- final-reason stage
- frozen-reason rollout variants
- terminal logic backprop
- heavy tree-search multiround repair
- legacy comparison configs for old methods

### Deferred TODOs

- Improve iterative-history compression for large `N`. The current summary keeps the latest failure in detail but compresses older rounds too aggressively (`status_counts`, one repeated failed input, compile-failure flag). Follow-up design should preserve more trial-and-error signal, especially multiple important failed cases, ineffective repair attempts, and coarse temporal progression across rounds, without blowing up prompt length.
