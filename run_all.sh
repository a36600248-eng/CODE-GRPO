#!/bin/bash
set -e

################################
# 参数检查
################################
if [ -z "$1" ]; then
    echo "Usage: bash run_all.sh <seed>"
    exit 1
fi

SEED=$1

################################
# 环境初始化
################################
echo "===== Initializing Environment ====="

cd ~/autodl-tmp/CODE-GRPO/trl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Environment ready."
echo "Running comparison suite with seed=$SEED"

################################
# 当前方法训练
################################
echo ""
echo "=============================="
echo "Running: Current Method"
echo "=============================="

python -m trl.cli.main code_grpo \
--config configs/comparison/codegrpo_method_mbpp.yaml \
--seed $SEED \
--data_seed $SEED


################################
# Vanilla GRPO 多轮 K=2
################################
echo ""
echo "=============================="
echo "Running: Vanilla GRPO Multi-round K=2"
echo "=============================="

python -m trl.cli.main code_grpo \
--config configs/comparison/vanilla_grpo_multiround_k2_mbpp.yaml \
--seed $SEED \
--data_seed $SEED


################################
# Vanilla GRPO 单轮 K=8
################################
echo ""
echo "=============================="
echo "Running: Vanilla GRPO Single-round K=8"
echo "=============================="

python -m trl.cli.main code_grpo \
--config configs/comparison/vanilla_grpo_single_round_k8_mbpp.yaml \
--seed $SEED \
--data_seed $SEED


################################
# 原始 Qwen7B 评测
################################
echo ""
echo "=============================="
echo "Running: Raw Qwen7B Evaluation"
echo "=============================="

python -m trl.cli.main code_grpo_eval \
--config configs/comparison/raw_qwen7b_eval_mbpp.yaml \
--seed $SEED \
--data_seed $SEED


################################
# 实验完成
################################
echo ""
echo "=============================="
echo "All experiments finished for seed=$SEED"
echo "=============================="

################################
# 自动关机
################################
echo "Server will shutdown in 1 minute..."
shutdown -h +1
