# Conda 环境配置记录（2026-03-01）

## 1) 新环境

- 名称：`codegrpo-local`
- 路径：`D:\anaconda3\envs\codegrpo-local`
- Python：`3.11`

创建命令：

```powershell
conda create -y -n codegrpo-local python=3.11
```

## 2) 已安装依赖

执行过的安装命令：

```powershell
conda run -n codegrpo-local python -m pip install --upgrade pip setuptools wheel
conda run -n codegrpo-local python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n codegrpo-local python -m pip install -r d:\MY-GRPO\trl\requirements.txt
conda run -n codegrpo-local python -m pip install -e d:\MY-GRPO\trl
conda run -n codegrpo-local python -m pip install "transformers>=4.56.2,<5.0.0" --upgrade
```

版本核验（当前）：

- `torch==2.10.0+cpu`
- `transformers==4.57.6`
- `trl` 可导入（editable 指向 `d:\MY-GRPO\trl`）
- `accelerate/datasets/peft/bitsandbytes/tensorboard/pytest` 已安装

## 3) vLLM 安装结果

已尝试：

```powershell
conda run -n codegrpo-local python -m pip install vllm
```

结果：失败（Windows 本地构建 wheel 失败）。

结论：当前机器可用于 **HF backend** 训练链路；`vllm` 建议在 Linux + CUDA 服务器使用。

## 4) 切换与使用

激活：

```powershell
conda activate codegrpo-local
```

训练示例（HF backend）：

```powershell
cd d:\MY-GRPO\trl
python -m trl.scripts.code_grpo --config codegrpo_train_8bit.yaml --backend hf --use_vllm false
```

## 5) 已补充的环境文件

- `d:\MY-GRPO\environment.codegrpo-local.yml`（本地可直接复用）
- `d:\MY-GRPO\environment.codegrpo-vllm-linux.yml`（服务器 Linux + vLLM 预置）

## 6) 额外说明

`--help` 在当前 Windows 控制台上会触发编码异常（输出含非 GBK 字符），不影响带 `--config` 的实际训练命令执行。

## 7) 本地自检（已执行）

执行命令：

```powershell
cd d:\MY-GRPO\trl
D:\anaconda3\envs\codegrpo-local\python.exe -m trl.scripts.code_grpo --config codegrpo_test_8bit_ascii.yaml --backend hf --use_vllm false
```

结果：命令成功退出，输出 `CodeGRPO test mode completed.`，并返回 `eval_*` 指标字典（说明 pipeline 在该环境可运行）。
