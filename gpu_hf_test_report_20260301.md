# HF+GPU 最小训练测试报告（不使用 vLLM）

时间：2026-03-01
机器：Windows + RTX 3050 Laptop (4GB)
目标：在本地 GPU 上验证 CodeGRPO pipeline（含参数更新路径）

## 1. 最终可用环境

新建环境：`codegrpo-gpu`

关键版本：
- torch: `2.4.1+cu118`
- transformers: `4.56.2`
- accelerate: `1.12.0`
- datasets: `4.6.1`
- peft: `0.18.1`
- numpy: `1.26.4`
- trl: editable (`d:\MY-GRPO\trl`)

GPU 检测：
- `torch.cuda.is_available() == True`
- device: `NVIDIA GeForce RTX 3050 Laptop GPU`

## 2. 为兼容当前 torch 版本所做的最小代码补丁

文件：`trl/trl/models/utils.py`

改动：
- 对 `FSDPModule` 增加兼容回退：若 `torch.distributed.fsdp.FSDPModule` 不存在，则回退为 `FSDP`。
- 该改动仅影响兼容导入与类型判断，不改变算法逻辑。

## 3. 训练命令（成功）

```powershell
D:\anaconda3\envs\codegrpo-gpu\python.exe -m trl.scripts.code_grpo `
  --config codegrpo_train_8bit.yaml `
  --model_name_or_path distilgpt2 `
  --backend hf `
  --use_vllm false `
  --use_cpu false `
  --load_in_8bit false `
  --max_steps 1 `
  --do_eval false `
  --logging_steps 1 `
  --save_strategy no `
  --report_to none
```

说明：
- 明确禁用 vLLM。
- 使用 HF backend + GPU。
- 使用 `distilgpt2` 避开旧 `.bin` 权重在部分 torch 版本下的安全限制触发。

## 4. 成功输出要点

命令退出码：`0`

关键日志：
- `CodeGRPO training completed.`
- `Model saved to ./tmp_codegrpo_train_out.`
- 训练统计已打印：`train_runtime/train_loss/step_time` 等

## 5. 额外备注

- 之前直接用 `sshleifer/tiny-gpt2` 会触发 transformers 对 `torch.load` 的安全限制（与 `.bin` 权重有关）。
- 当前可复现路径：`codegrpo-gpu` + `distilgpt2` + `backend=hf` + `use_vllm=false`。
