#!/usr/bin/env bash
# Fit the Search critic from the supplied Search SFT backbone; freeze the actor.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Search critic warmup (actor frozen throughout):
  Set MODEL_DIR, MODEL_TORCH_DIST, TRAIN_DATA, OUTPUT_DIR, and RETRIEVAL_SERVICE_URL.
  bash scripts/search-r1/pretrain.sh
Defaults: 50 updates, 512 prompts x 5 responses, batch 2560, 4 training + 4 rollout GPUs.
Optional: DRY_RUN=1, NUM_ROLLOUT, SAVE_INTERVAL, UPDATE_WEIGHTS_INTERVAL,
          CRITIC_LR, CRITIC_LR_WARMUP_ITERS, ACTOR_GPUS, ROLLOUT_GPUS, MEGATRON_ROOT.
Output: OUTPUT_DIR/critic_megatron; 50 updates end at iter_0000049.
This fresh critic-only stage does not support ASYNC_RESUME.
USAGE
    exit 0
fi
[[ $# -eq 0 ]] || { echo "Configure this stage with environment variables; no positional arguments are accepted." >&2; exit 2; }
export MODEL_ARCH=llama
require_env MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR RETRIEVAL_SERVICE_URL
normalize_paths MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR

[[ -z "${ASYNC_RESUME:-}" ]] || { echo "This fresh critic-only stage does not support ASYNC_RESUME: the frozen actor has no Adam updates." >&2; exit 2; }
NUM_ROLLOUT=${NUM_ROLLOUT:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
# Full checkpoints require publishing the frozen actor at save boundaries.
UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL:-$SAVE_INTERVAL}
CRITIC_LR=${CRITIC_LR:-5e-6}
CRITIC_LR_WARMUP_ITERS=${CRITIC_LR_WARMUP_ITERS:-10}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
for value in "$NUM_ROLLOUT" "$SAVE_INTERVAL" "$UPDATE_WEIGHTS_INTERVAL" "$ACTOR_GPUS" "$ROLLOUT_GPUS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Step counts, intervals, and GPU allocations must be positive integers." >&2; exit 2; }
done
[[ "$CRITIC_LR_WARMUP_ITERS" =~ ^[0-9]+$ ]] || { echo "CRITIC_LR_WARMUP_ITERS must be nonnegative." >&2; exit 2; }
(( CRITIC_LR_WARMUP_ITERS < NUM_ROLLOUT )) || { echo "Critic LR warmup must be shorter than NUM_ROLLOUT." >&2; exit 2; }
(( NUM_ROLLOUT % UPDATE_WEIGHTS_INTERVAL == 0 )) || { echo "NUM_ROLLOUT must end at a natural weight-publication boundary." >&2; exit 2; }
(( 2560 % ACTOR_GPUS == 0 )) || { echo "The fixed 2560-sample batch must be divisible by ACTOR_GPUS." >&2; exit 2; }
export SEARCH_PROTOCOL=llama3
export SEARCH_MAX_TURNS=${SEARCH_MAX_TURNS:-8}
export SEARCH_TOPK=${SEARCH_TOPK:-3}
export SEARCH_MAX_DOC_LENGTH=${SEARCH_MAX_DOC_LENGTH:-200}
export SEARCH_REWARD_MODE=${SEARCH_REWARD_MODE:-grpo}
export SEARCH_RETRIEVAL_FAILURE_MODE=${SEARCH_RETRIEVAL_FAILURE_MODE:-observation}
export PYTHONPATH="$REPO_ROOT/examples/search-r1:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
load_model_args
check_search_service
ROLE_CONFIG="$OUTPUT_DIR/megatron_roles.json"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON" - "$MODEL_DIR" "$MODEL_TORCH_DIST" "$TRAIN_DATA" "$OUTPUT_DIR" "$CRITIC_LR" "$CRITIC_LR_WARMUP_ITERS" <<'PYCODE'
import json
import math
from pathlib import Path
import sys
model, native, data, output = (Path(value).expanduser().resolve() for value in sys.argv[1:5])
for source in (model, native, data):
    if not source.exists():
        raise SystemExit("Missing input: " + str(source))
    if output == source or output in source.parents or source in output.parents:
        raise SystemExit("OUTPUT_DIR must be separate from all training inputs")
tracker = native / "latest_checkpointed_iteration.txt"
if not tracker.is_file():
    raise SystemExit("MODEL_TORCH_DIST must include latest_checkpointed_iteration.txt")
iteration = tracker.read_text().strip()
if iteration == "release":
    native_checkpoint = native / "release"
elif iteration.isascii() and iteration.isdigit():
    native_checkpoint = native / ("iter_%07d" % int(iteration))
else:
    raise SystemExit("Invalid native SFT checkpoint tracker")
if not all((native_checkpoint / name).is_file() for name in (".metadata", "metadata.json", "common.pt")) or not any(native_checkpoint.rglob("*.distcp")):
    raise SystemExit("MODEL_TORCH_DIST does not contain a complete converted torch_dist checkpoint")
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit("OUTPUT_DIR must be absent or empty for fresh critic initialization")
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "llama" or config.get("num_hidden_layers") != 28 or config.get("hidden_size") != 3072:
    raise SystemExit("This Search stage expects the Llama-3.2-3B Search SFT model")
tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
template_path = model / "chat_template.jinja"
template = template_path.read_text() if template_path.is_file() else tokenizer_config.get("chat_template", "")
if not isinstance(template, str) or not all(marker in template for marker in (
        "<|start_header_id|>", "<|eot_id|>", "ipython", "parameters", "16 Sep 2026")):
    raise SystemExit("Preserve the Search SFT tool chat template and its fixed date (16 Sep 2026)")
learning_rate = float(sys.argv[5])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit("CRITIC_LR must be finite and positive")
roles = {"megatron": [{"name": "search_critic", "role": "critic", "overrides": {
    "load": str(native), "save": str(output / "critic_megatron"), "lr": learning_rate,
    "lr_warmup_iters": int(sys.argv[6]), "finetune": True, "no_load_optim": True,
    "no_load_rng": True, "use_checkpoint_opt_param_scheduler": False,
}}]}
output.mkdir(parents=True, exist_ok=True)
with (output / "megatron_roles.json").open("x") as stream:
    json.dump(roles, stream, indent=2, allow_nan=False)
    stream.write("\n")
PYCODE
fi
printf 'Search critic pretraining: %s updates; actor frozen; critic LR=%s, LR warmup=%s; full save every %s rollouts.\n' \
    "$NUM_ROLLOUT" "$CRITIC_LR" "$CRITIC_LR_WARMUP_ITERS" "$SAVE_INTERVAL"
launch_training "$REPO_ROOT/train_async.py" \
    "${MODEL_ARGS[@]}" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "$ACTOR_GPUS" --num-gpus-per-node "$((ACTOR_GPUS + ROLLOUT_GPUS))" \
    --hf-checkpoint "$MODEL_DIR" --load "$MODEL_TORCH_DIST" --ref-load "$MODEL_TORCH_DIST" \
    --megatron-config-path "$ROLE_CONFIG" --finetune --no-load-optim --no-load-rng --start-rollout-id 0 \
    --save "$OUTPUT_DIR/actor_megatron" --save-interval "$SAVE_INTERVAL" --async-save --async-checkpoint \
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 512 --n-samples-per-prompt 5 \
    --global-batch-size 2560 --num-steps-per-rollout 1 \
    --num-critic-only-steps "$NUM_ROLLOUT" --update-weights-interval "$UPDATE_WEIGHTS_INTERVAL" \
    --fully-async-max-staleness 16 \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --custom-generate-function-path generate_with_search.generate \
    --custom-rm-path search_reward.reward_func --reward-key score --eval-reward-key em \
    --prompt-data "$TRAIN_DATA" --input-key prompt --label-key reward_model \
    --apply-chat-template --tool-key tools --eval-tool-key tools \
    --apply-chat-template-kwargs '{"date_string":"16 Sep 2026"}' \
    --rollout-shuffle --rollout-seed 42 --seed 1234 \
    --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine 1 \
    --rollout-max-prompt-len 512 --rollout-max-response-len 8192 \
    --rollout-max-context-len 8704 --seq-length 8704 \
    --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 \
    --balance-data --calculate-per-token-loss \
    --advantage-estimator ppo --gamma 1 --lambd 1 --value-clip 0.2 \
    --normalize-advantages --use-rollout-logprobs --kl-coef 0 --entropy-coef 0 \
    --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --lr-warmup-iters 0 \
    --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 8704 --offload-train \
    --sglang-dtype bfloat16 --sglang-mem-fraction-static 0.6 --sglang-server-concurrency 128 \
    --sglang-context-length 8704 --sglang-max-prefill-tokens 8704 \
    --sglang-disable-cuda-graph --sglang-watchdog-timeout 1800 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash
