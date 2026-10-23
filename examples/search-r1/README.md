<!-- Modified for COPC. -->
# Search-R1: search-augmented reasoning

Start from a trained Llama-3.2-3B Search SFT checkpoint with its original tokenizer and tool chat template. Run commands from the repository root. Training requires Python >= 3.10, PyTorch, Megatron-LM, and SGLang; set `MEGATRON_ROOT` if Megatron-LM is not importable. COPC and PPO default to six rollout GPUs and two training GPUs.

## Retrieval service

Use a separate Python >= 3.10 environment and allow about 170 GiB of free disk space for asset preparation:

```bash
python -m pip install -r examples/search-r1/requirements-retrieval.txt
python examples/search-r1/retrieval/prepare_assets.py --root /path/to/search-assets
python examples/search-r1/retrieval/server.py \
  --assets /path/to/search-assets --host 127.0.0.1 --port 8000 \
  --device cpu --cpu-threads 8
```

The preparation command downloads and verifies the Wikipedia corpus, E5 encoder, and FAISS index. Leave the server running during critic pretraining and RL. For remote workers, bind it to a reachable interface and replace `127.0.0.1` in `RETRIEVAL_SERVICE_URL` below with the service host's address.

## Training

In the training environment, convert the SFT checkpoint to Megatron format, pretrain the critic, and start RL. Use a new `OUTPUT_DIR` for each stage:

```bash
python -m pip install -e .
python -m pip install 'aiohttp>=3.9,<4'

# 1. Convert the Search SFT checkpoint.
export MODEL_DIR=/path/to/search-sft
OUTPUT_DIR=/path/to/search-converted bash scripts/search-r1/sft.sh

# 2. Pretrain the critic with the SFT actor frozen.
export MODEL_TORCH_DIST=/path/to/search-converted/megatron
export TRAIN_DATA=/path/to/search-rl.parquet
export RETRIEVAL_SERVICE_URL=http://127.0.0.1:8000/retrieve
OUTPUT_DIR=/path/to/search-critic bash scripts/search-r1/pretrain.sh

# 3. Train COPC from the SFT actor and pretrained critic.
export CRITIC_LOAD_DIR=/path/to/search-critic/critic_megatron CRITIC_STEP=49
OUTPUT_DIR=/path/to/search-copc bash scripts/search-r1/train.sh copc
```

For GRPO or DAPO, keep the model/data/retrieval exports, skip critic pretraining and the `CRITIC_*` exports, and replace `copc` with `grpo` or `dapo`.

Replace `copc` with `ppo-sync` to select synchronous PPO.

## Data

RL JSONL/Parquet rows contain:

- `prompt`: the original question as a user message, e.g. `[{"role": "user", "content": "Who wrote Hamlet?"}]`.
- `reward_model`: answer aliases, e.g. `{"style": "rule", "ground_truth": {"target": ["William Shakespeare", "Shakespeare"]}}`.
- `tools`: a list containing the SFT model's `search` function schema with a `query_list` array of strings.

Supply raw messages and preserve the tool schema; the launcher applies the checkpoint's chat template. Final answers use `\boxed{...}`.

The RL launcher enables background model checkpoint writes by default (`ASYNC_SAVE=1`, which adds `--async-save`). Asynchronous modes also save complete training checkpoints: set `ASYNC_RESUME` to the previous run's `resume/` directory and choose a new `OUTPUT_DIR` to resume. See [Experiment configuration](../../docs/experiments.md) for details and training options.
