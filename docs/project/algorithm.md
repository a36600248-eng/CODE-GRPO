## Current Algorithm

This project now keeps only the current main algorithm path:

1. Single-round code GRPO.
2. Each problem generates `K=num_generations` code candidates once.
3. Hard reward is unit-test pass rate.
4. Zero-pass candidates optionally receive a bounded soft reward.
5. Optional `code + input -> output/error` auxiliary training uses a separate auxiliary SFT loss.
6. Optional pseudo-multiround is implemented at the trainer/data-pool level, not as tree-search reasoning.
7. Optional question-level EMA prior reweights sampling and iterative-node priority.

Legacy logic-audit, exec-audit, final-reason, and heavy multiround search are no longer part of the active algorithm.

### Main Rollout

For each problem:

1. Build one code-generation prompt from the problem text plus optional recent iterative history.
2. Generate `K` code candidates.
3. Run unit tests for each candidate.
4. Compute `hard_reward = pass_cnt / test_count`.
5. If `pass_cnt == 0` and zero-pass soft reward is enabled, compute soft reward.
6. Use the final reward for sibling-group GRPO advantage computation.

### Final Reward

If `pass_cnt > 0`:

- `final_reward = hard_reward + bounded compile/format shaping`

If `pass_cnt == 0`:

- `final_reward = beta * normalized_soft_reward`

where `beta < 1 / test_count`, so soft reward only ranks zero-pass samples and cannot outrank genuine hard reward progress.

### Zero-Pass Soft Reward

Zero-pass soft reward compares:

- `problem + input -> oracle output` confidence
- `code + input -> oracle output` confidence

and uses their average log-probability delta across a bounded diagnostic set.

The raw delta is clipped and mapped into `[0, 1]`, then scaled by `beta`.

### Code-IO Auxiliary Training

This auxiliary branch is optional and does not change the main reward definition.

For selected execution cases from both correct and incorrect candidates, the model receives:

- input: `code + input`
- target: `actual output` or `error type`

This branch uses a separate supervised loss on the auxiliary targets. It exists to improve program-behavior modeling and make zero-pass soft reward more reliable.

### Pseudo-Multiround

Pseudo-multiround is a trainer-side pool, not a search tree.

If a problem rollout produces no fully solved candidate, the trainer keeps up to `pseudo_iterative_select_count` iterative nodes:

- best pass candidate
- best soft candidate
- one novelty-oriented candidate

Later steps alternate between original problems and iterative nodes, but only after an optional warmstart phase that forces every original question to be rolled out once. After warmstart, the final source choice depends on original-problem keep probability, iterative-node priority, and optional forced-original sampling. The pool is bounded by capacity and TTL, and solved questions prune stale iterative nodes to reduce off-policy drift.

### Question-Level EMA Prior

This optional feature keeps a per-question moving estimate of:

- recent code success
- recent reason-signal strength

It does not change reward. It only changes:

- original-problem keep probability
- iterative-node priority
- weighted sampler refresh when enabled

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

