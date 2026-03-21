## Current Commands

Raw eval:

```bash
python -m trl.cli.main code_grpo_eval \
  --config configs/comparison/server_2gpu_smoke/raw_qwen7b_eval_mbpp.yaml
```

Single-round zero-pass soft reward training:

```bash
bash run_zero_pass_soft_reward_server_2gpu.sh
```

Pseudo-multiround zero-pass soft reward training:

```bash
bash run_pseudo_multiround_zero_pass_soft_reward_server_2gpu.sh
```
