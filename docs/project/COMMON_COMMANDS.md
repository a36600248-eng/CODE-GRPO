# 常用指令速查

只保留当前可信路线：

- 双卡
- `vllm_mode: server`
- BF16 + LoRA
- `vllm_sync_steps: 5`

服务器默认目录：

- 项目：`~/autodl-tmp/CODE-GRPO/trl`
- 环境：`codegrpo`
- 模型：`/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct`

## 1. 运行四方法 smoke

先跑一个很快的总体验证，顺序是：

1. `raw_eval`
2. `CodeGRPO`
3. `vanilla K=2`
4. `vanilla K=8`

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
bash /root/autodl-tmp/CODE-GRPO/run_smoke_server_2gpu.sh 42 8010
```

## 2. 运行正式四方法总套件

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
bash /root/autodl-tmp/CODE-GRPO/run_all_server_2gpu.sh 42 8010
```

## 3. 单独跑 raw eval

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo_eval \
  --config configs/comparison/server_2gpu/raw_qwen7b_eval_mbpp.yaml \
  --seed 42 \
  --data_seed 42 \
  --vllm_server_base_url http://127.0.0.1:8010
```

## 4. 单独跑正式 CodeGRPO

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
  --config configs/comparison/server_2gpu/codegrpo_method_mbpp.yaml \
  --seed 42 \
  --data_seed 42 \
  --vllm_server_base_url http://127.0.0.1:8010
```

## 5. 单独跑正式 vanilla K=2

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
  --config configs/comparison/server_2gpu/vanilla_grpo_multiround_k2_mbpp.yaml \
  --seed 42 \
  --data_seed 42 \
  --vllm_server_base_url http://127.0.0.1:8010
```

## 6. 单独跑正式 vanilla K=8

```bash
cd ~/autodl-tmp/CODE-GRPO/trl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegrpo

CUDA_VISIBLE_DEVICES=1 python -m trl.cli.main code_grpo \
  --config configs/comparison/server_2gpu/vanilla_grpo_single_round_k8_mbpp.yaml \
  --seed 42 \
  --data_seed 42 \
  --vllm_server_base_url http://127.0.0.1:8010
```

## 7. 查看 smoke 四个结果

```bash
python - <<'PY'
import json, os, glob

roots = {
    "raw_eval": "/root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo_eval_server_2gpu_smoke/test/*",
    "codegrpo": "/root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo_server_2gpu_smoke/train/*",
    "vanilla_k2": "/root/autodl-tmp/CODE-GRPO/trl/runs_vanilla_grpo_k2_server_2gpu_smoke/train/*",
    "vanilla_k8": "/root/autodl-tmp/CODE-GRPO/trl/runs_vanilla_grpo_k8_server_2gpu_smoke/train/*",
}

for name, pattern in roots.items():
    runs = sorted(glob.glob(pattern), reverse=True)
    if not runs:
        print(name, "NO_RUN")
        continue
    run = runs[0]
    path = os.path.join(run, "logs", "trainer_events_rank0.jsonl")
    best = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                logs = row.get("logs", {})
                if "eval_pass_at_1" in logs:
                    best = {
                        "step": row.get("step"),
                        "eval_pass_at_1": logs.get("eval_pass_at_1"),
                        "eval_best_pass_rate_overall": logs.get("eval_best_pass_rate_overall"),
                        "eval_generation_format_ok_rate": logs.get("eval_generation_format_ok_rate"),
                    }
    print(name, run)
    print(best if best is not None else "NO_EVAL_METRICS")
PY
```

## 8. 查看某次 run 的 eval 指标

把 `RUN_ROOT` 换成你的 run 目录。

```bash
RUN_ROOT=/root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo_server_2gpu/train/<run_id>

python - <<'PY'
import json, os
run = os.environ["RUN_ROOT"]
path = os.path.join(run, "logs", "trainer_events_rank0.jsonl")
want_steps = {20, 40, 60, 80, 100, 120}
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        logs = row.get("logs", {})
        if row.get("step") in want_steps and "eval_pass_at_1" in logs:
            print("step", row["step"])
            print("  eval_pass_at_1 =", logs.get("eval_pass_at_1"))
            print("  eval_best_pass_rate_overall =", logs.get("eval_best_pass_rate_overall"))
            print("  eval_generation_format_ok_rate =", logs.get("eval_generation_format_ok_rate"))
PY
```

## 9. 查看 server 日志

正式套件：

```bash
tail -n 120 /root/autodl-tmp/CODE-GRPO/trl/vllm_server_8010.log
```

smoke：

```bash
tail -n 120 /root/autodl-tmp/CODE-GRPO/trl/vllm_server_smoke_8010.log
```

## 10. 查看最新 run

CodeGRPO 正式：

```bash
ls -dt /root/autodl-tmp/CODE-GRPO/trl/runs_codegrpo_server_2gpu/train/* | head -1
```

vanilla K=2 正式：

```bash
ls -dt /root/autodl-tmp/CODE-GRPO/trl/runs_vanilla_grpo_k2_server_2gpu/train/* | head -1
```

vanilla K=8 正式：

```bash
ls -dt /root/autodl-tmp/CODE-GRPO/trl/runs_vanilla_grpo_k8_server_2gpu/train/* | head -1
```

## 11. 清旧 server 进程

```bash
pkill -f "trl.cli.main vllm-serve" || true
pkill -f "VLLM::EngineCore" || true
sleep 2
nvidia-smi
```

## 12. 看 GPU 占用

```bash
nvidia-smi
```

## 13. 只保留可信正式结果时可删的旧 runs

```bash
cd ~/autodl-tmp/CODE-GRPO/trl

rm -rf runs_codegrpo
rm -rf runs_eval_compare_midscale_hf
rm -rf runs_vanilla_grpo_k2
rm -rf runs_vanilla_grpo_k2_hardmix
rm -rf runs_vanilla_grpo_k2_midscale
rm -rf runs_vanilla_grpo_k8
rm -f vllm_server_*.log
```

## 14. 当前可信配置目录

正式：

- `configs/comparison/server_2gpu`

smoke：

- `configs/comparison/server_2gpu_smoke`
