<!-- Modified for COPC. -->
# COPC

**Coupled Off-Policy Correction for Asynchronous LLM Reinforcement Learning**

This repository contains the official implementation of **COPC**, an asynchronous actor–critic method for large language model reinforcement learning.

## Overview

Asynchronous RL improves training throughput by separating rollout generation from policy optimization. The resulting policy lag affects both the importance weights in the actor objective and the advantages estimated from older trajectories.

COPC combines **two-sided token-level importance-ratio masking** in the actor objective with **two-sided clipped-ratio weighting of TD residuals** for return and advantage estimation. The paper analyzes how these corrections interact through the bias and variance of the actor update, and studies their joint effect on learning stability and performance.

## What is included

- **Multi-turn RL environments:** tool-integrated mathematical reasoning with ReTool and search-augmented reasoning with Search-R1.
- **Training pipelines:** model preparation, critic pretraining, and fully asynchronous RL with checkpoint resume.
- **Baselines and ablations:** PPO (sync and async), GRPO, DAPO, SAO, V-trace, and AReaL, alongside COPC without return correction and COPC masking with V-trace targets.

## Experiments

| Environment | Model | Documentation |
| --- | --- | --- |
| ReTool | Qwen3-4B-Instruct-2507 | [Setup and training](examples/retool/README.md) |
| Search-R1 | Llama-3.2-3B Search SFT | [Setup and training](examples/search-r1/README.md) |

## Repository layout

```text
slime/       Training objectives and asynchronous rollout implementation
examples/    ReTool and Search-R1 environments
scripts/     Model preparation, critic pretraining, and RL launchers
docs/        Experiment configuration
```

The installed distribution is named `copc-rl` and retains the upstream `slime` Python namespace.

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{hu2026copc,
  title         = {{COPC}: Coupled Off-Policy Correction for Asynchronous LLM Reinforcement Learning},
  author        = {Zicheng Hu and Zhijian Zhou and Xuan Zhang and Yuchen Liu and Cheng Chen and Yuan Li and Qi Gu and Yan Feng and Hongyan Hao and Chao Qu},
  year          = {2026},
  eprint        = {2610.09597},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.09597}
}
```

## Acknowledgements

This codebase builds on [slime](https://github.com/THUDM/slime), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and [SGLang](https://github.com/sgl-project/sglang). We thank their developers and contributors for open-sourcing these projects.
