<!-- Modified for COPC. -->
# ReTool: tool-integrated mathematical reasoning

Start from Qwen3-4B-Instruct-2507 and ReTool SFT data, such as `JoeYing/ReTool-SFT`.

## Quick start

Run from the repository root with Python >= 3.10, PyTorch, Megatron-LM, and SGLang installed. Set `MEGATRON_ROOT` if Megatron-LM is not importable. RL defaults use six rollout GPUs and two training GPUs; choose a fresh `OUTPUT_DIR` for each run.

```bash
python -m pip install -e .
python -m pip install -r examples/retool/requirements.txt

# 1. Train SFT and export checkpoints.
MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507 \
TRAIN_DATA=/path/to/retool-sft.jsonl OUTPUT_DIR=/path/to/retool-sft \
  bash scripts/retool/sft.sh

# 2. Pretrain the critic with the SFT actor frozen.
export MODEL_DIR=/path/to/retool-sft/hf
export MODEL_TORCH_DIST=/path/to/retool-sft/megatron
export TRAIN_DATA=/path/to/math-rl.jsonl
OUTPUT_DIR=/path/to/retool-critic bash scripts/retool/pretrain.sh

# 3. Train COPC from the SFT actor and pretrained critic.
export CRITIC_LOAD_DIR=/path/to/retool-critic/critic_megatron CRITIC_STEP=49
OUTPUT_DIR=/path/to/retool-copc bash scripts/retool/train.sh copc
```

For GRPO or DAPO, keep the model/data exports, skip critic pretraining and the `CRITIC_*` exports, and replace `copc` with `grpo` or `dapo`.

Replace `copc` with `ppo-sync` to select synchronous PPO.

## Data and tools

RL JSONL/Parquet rows require `prompt` (raw question text or chat messages, without a rendered chat template) and `label` (the answer string):

```json
{"prompt": [{"role": "user", "content": "What is 1 + 1?"}], "label": "2"}
```

The rollout provides a Python `code_interpreter` tool; its dependencies are installed by the requirements command above.

The RL launcher enables background model checkpoint writes by default (`ASYNC_SAVE=1`, which adds `--async-save`). Asynchronous modes also save complete training checkpoints: set `ASYNC_RESUME` to the previous run's `resume/` directory and choose a new `OUTPUT_DIR` to resume. See [Experiment configuration](../../docs/experiments.md) for details and training options.
