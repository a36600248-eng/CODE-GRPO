# MBPP Comparison Suite

This folder contains the comparison configs for the current Code-GRPO study.

Files:

- `codegrpo_method_mbpp.yaml`
  - Current full method.
- `vanilla_grpo_multiround_k2_mbpp.yaml`
  - Vanilla GRPO, multi-round repair, `K=2`, no audit / no soft reward / no reason loss.
- `vanilla_grpo_single_round_k8_mbpp.yaml`
  - Vanilla GRPO, single-round wide sampling, `K=8`, no audit / no soft reward / no reason loss.
- `raw_qwen7b_eval_mbpp.yaml`
  - Standalone raw Qwen2.5-Coder-7B-Instruct evaluation on the same MBPP eval harness.

Recommended comparison order:

1. Raw base eval
2. Vanilla GRPO multi-round K=2
3. Vanilla GRPO single-round K=8
4. Current method

Commands:

Raw base eval:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m trl.cli.main code_grpo_eval --config configs/comparison/raw_qwen7b_eval_mbpp.yaml --seed 42 --data_seed 42
```

Vanilla GRPO multi-round K=2:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m trl.cli.main code_grpo --config configs/comparison/vanilla_grpo_multiround_k2_mbpp.yaml --seed 42 --data_seed 42
```

Vanilla GRPO single-round K=8:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m trl.cli.main code_grpo --config configs/comparison/vanilla_grpo_single_round_k8_mbpp.yaml --seed 42 --data_seed 42
```

Current method:

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m trl.cli.main code_grpo --config configs/comparison/codegrpo_method_mbpp.yaml --seed 42 --data_seed 42
```

For multi-seed experiments, change both `--seed` and `--data_seed` together.

