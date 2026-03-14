## MBPP Smoke Suite: Dual-GPU Server

This folder contains a fast smoke-test comparison suite for the trusted dual-GPU
official vLLM server sync path.

Purpose:

- verify the end-to-end pipeline still works
- get four rough results quickly
- keep the method definitions aligned with the official configs

Order:

1. `raw_qwen7b_eval_mbpp.yaml`
2. `codegrpo_method_mbpp.yaml`
3. `vanilla_grpo_multiround_k2_mbpp.yaml`
4. `vanilla_grpo_single_round_k8_mbpp.yaml`

Smoke-specific changes versus the official configs:

- `report_to: none`
- no checkpoint saving
- short training (`max_steps: 10`)
- a single eval near the end (`eval_steps: 10`)
- no rollout trace dumping
- shorter completion lengths for faster turnaround

These configs are for pipeline validation and rough comparison only, not final
reported results.
