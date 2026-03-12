# TODO

## Logging cleanup
- [done] Silence per-step reward-loss lines during baseline eval / eval-only passes.
- Keep eval progress bar and final summary metrics, but stop printing per-example zero-loss messages.

## Experiment follow-up
- Run 3 seeds with the current config and aggregate `mean +- std` for:
  - `best eval_pass_at_1_round_1`
  - `best eval_pass_at_1_round_2`
  - `best eval_pass_at_1`
- After multi-seed runs, decide whether soft-reward optimism still needs another ablation.

## Checkpoint workflow
- Verify `load_best_model_at_end` and `metric_for_best_model=eval_pass_at_1` match the saved best checkpoint in `trainer_state.json` for the next completed run.

