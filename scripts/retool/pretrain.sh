#!/usr/bin/env bash
# Pretrain the critic from the SFT model with a frozen actor.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
case "${1:-}" in
    --help|-h)
        cat <<'HELP'
Usage: bash scripts/retool/pretrain.sh
Required environment:
  MODEL_DIR         SFT Hugging Face checkpoint and tokenizer.
  MODEL_TORCH_DIST  Matching native SFT checkpoint (SFT output megatron/).
  TRAIN_DATA        RL prompt/label JSONL or Parquet dataset.
  OUTPUT_DIR        Fresh output directory; produces critic_megatron/.
Optional environment:
  NUM_STEPS         Critic-only rollouts (default 200; two updates per rollout).
  SAVE_INTERVAL     Checkpoint interval (default 10).
  TRAIN_GPUS        Training GPUs on one node (default 4).
  ROLLOUT_GPUS      Generation GPUs on the same node (default 4).
  MEGATRON_ROOT     Megatron-LM checkout if it is not already importable.
  RAY_ADDRESS       Existing Ray cluster; otherwise start a local runtime.
  DRY_RUN=1         Print commands without preparing files or starting training.
The frozen actor is republished at save boundaries for complete checkpoints.
This stage starts fresh; ASYNC_RESUME is not supported before an actor update.
HELP
        exit 0 ;;
esac
[[ $# == 0 ]] || { echo "Unknown argument; use --help." >&2; exit 2; }
require_env MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR
normalize_paths MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR
[[ -z "${ASYNC_RESUME:-}" ]] || { echo "This fresh critic stage does not support ASYNC_RESUME before an actor update." >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR already exists; choose a fresh directory." >&2; exit 2; }
NUM_STEPS=${NUM_STEPS:-200}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL:-$SAVE_INTERVAL}
for value in "$NUM_STEPS" "$SAVE_INTERVAL" "$UPDATE_WEIGHTS_INTERVAL" "$TRAIN_GPUS" "$ROLLOUT_GPUS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Steps, intervals and GPU counts must be positive integers." >&2; exit 2; }
done
(( NUM_STEPS % UPDATE_WEIGHTS_INTERVAL == 0 )) || {
    echo "NUM_STEPS must be divisible by UPDATE_WEIGHTS_INTERVAL for the final checkpoint." >&2; exit 2;
}
export MAX_TURNS=${MAX_TURNS:-8} MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-8}
export RETOOL_REWARD_MODE=math_verify
load_model_args
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "$MODEL_TORCH_DIST/latest_checkpointed_iteration.txt" ]] || {
        echo "MODEL_TORCH_DIST must be the native SFT checkpoint directory." >&2; exit 2;
    }
fi
ROLE_CONFIG="$OUTPUT_DIR/roles.yaml"
launch mkdir -p -- "$OUTPUT_DIR"
launch "$PYTHON" - "$ROLE_CONFIG" "$MODEL_TORCH_DIST" "$OUTPUT_DIR" <<'PY_ROLES'
import json
import sys
from pathlib import Path
config_path, model_path, output_path = sys.argv[1:]
output = Path(output_path)
config = {"megatron": [
    {"name": "default", "role": "critic", "overrides": {
        "lr": 5e-6, "lr_warmup_iters": 10,
        "load": model_path, "no_load_optim": True, "no_load_rng": True,
        "finetune": True, "use_checkpoint_opt_param_scheduler": False,
        "save": str(output / "critic_megatron"),
    }},
    {"name": "default", "role": "actor", "overrides": {
        "lr": 1e-6, "save": str(output / "actor_megatron"),
    }},
]}
Path(config_path).write_text(json.dumps(config, indent=2) + "\n")
PY_ROLES

# Full checkpoints require publishing the frozen actor at save boundaries.
TRAIN_ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS"
    --megatron-config-path "$ROLE_CONFIG"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$MODEL_DIR" --load "$MODEL_TORCH_DIST" --ref-load "$MODEL_TORCH_DIST"
    --finetune --no-load-optim --no-load-rng --start-rollout-id 0
    --save "$OUTPUT_DIR/actor_megatron" --save-interval "$SAVE_INTERVAL" --ckpt-format torch_dist
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --prompt-data "$TRAIN_DATA" --input-key prompt --label-key label --apply-chat-template
    --rollout-shuffle --rollout-seed 42 --seed 1234 --reward-key score
    --num-rollout "$NUM_STEPS" --rollout-batch-size 256 --n-samples-per-prompt 16
    --rollout-max-prompt-len 2048 --rollout-max-response-len 16384 --rollout-temperature 1
    --rollout-num-gpus "$ROLLOUT_GPUS" --fully-async-max-staleness 5
    --global-batch-size 2048 --num-steps-per-rollout 2 --balance-data --log-passrate
    --advantage-estimator ppo --gamma 1 --lambd 1 --value-clip 0.2 --normalize-advantages
    --num-critic-only-steps "$NUM_STEPS" --update-weights-interval "$UPDATE_WEIGHTS_INTERVAL"
    --eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0 --kl-coef 0 --kl-loss-coef 0
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98
    --tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 32768
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.85
    --sglang-router-disable-circuit-breaker --sglang-server-concurrency 256
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash
    --custom-generate-function-path generate_with_retool.generate
    --custom-rm-path generate_with_retool.reward_func
)
launch_training "train_async.py" "${TRAIN_ARGS[@]}"
