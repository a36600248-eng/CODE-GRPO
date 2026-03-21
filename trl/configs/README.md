# Config Layout

This directory now emphasizes the current Code-GRPO mainline.

Active comparison configs:

- `comparison/server_2gpu_smoke/raw_qwen7b_eval_mbpp.yaml`
- `comparison/server_2gpu_smoke/codegrpo_single_round_zero_pass_soft_reward.yaml`
- `comparison/server_2gpu_smoke/codegrpo_pseudo_multiround_zero_pass_soft_reward.yaml`

Use the top-level helper scripts for the two training variants:

- `run_zero_pass_soft_reward_server_2gpu.sh`
- `run_pseudo_multiround_zero_pass_soft_reward_server_2gpu.sh`
