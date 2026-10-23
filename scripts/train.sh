#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() {
    cat <<'HELP'
Usage: bash scripts/train.sh <retool|search-r1> [copc|copc-wo-rc|copc-vtrace|ppo|ppo-sync|grpo|dapo|sao|vtrace|areal] [training flags...]
Required: MODEL_DIR (SFT HF), MODEL_TORCH_DIST, TRAIN_DATA, OUTPUT_DIR.
Critic algorithms also require CRITIC_LOAD_DIR; CRITIC_STEP defaults to 49.
Search-R1 additionally requires RETRIEVAL_SERVICE_URL (the /retrieve endpoint).
Optional: NUM_STEPS, TRAIN_GPUS, ROLLOUT_GPUS, FULLY_ASYNC_MAX_STALENESS,
          ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE, NUM_STEPS_PER_ROLLOUT,
          ACTOR_LR, CRITIC_LR, SAVE_INTERVAL, UPDATE_WEIGHTS_INTERVAL, ASYNC_SAVE.
NUM_STEPS counts rollouts; updates per rollout = sampled batch / GLOBAL_BATCH_SIZE.
Defaults: ReTool 120 updates, Search-R1 450; AReaL uses 60/225 rollouts x 2 updates.
FULLY_ASYNC_MAX_STALENESS defaults to 8 for ReTool and 14 for Search-R1 in async modes.
SAVE_INTERVAL and UPDATE_WEIGHTS_INTERVAL count rollouts. ASYNC_SAVE defaults to 1.
Default saves: every 10 optimizer updates (10 rollouts, or 5 for AReaL).
COPC: EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28 RHO_CLIP=0.2 RHO_CLIP_HIGH=0.28.
copc-wo-rc: standard PPO GAE + COPC actor masking; keeps COPC sampling and normalization.
copc-vtrace: V-trace target + COPC actor masking; advantage normalization enabled.
ppo-sync: synchronous PPO; default 6 rollout GPUs + 2 training GPUs.
VTRACE_CLIP_ONLY=1 with vtrace selects copc-vtrace.
AReaL: AREAL_IS_WEIGHT_MAX=5, policy clipping EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28.
DAPO: GRPO advantages, EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28, token loss, dynamic sampling; no critic.
V-trace: VALUE_CLIP=0.2 VTRACE_RHO_CLIP=1 VTRACE_C_CLIP=1 VTRACE_PG_RHO_CLIP=1.
Both V-trace entries enable advantage normalization.
Use ASYNC_RESUME=/path/to/run/resume to restore a complete checkpoint in async modes.
Use DRY_RUN=1 to print commands without starting Ray, GPUs or writing outputs.
Use RAY_ADDRESS to connect to an existing Ray cluster; otherwise start local Ray.
HELP
}
if [[ $# == 0 || ${1:-} == --help || ${2:-} == --help ]]; then usage; exit 0; fi
TASK=$1; shift
ALGO=${ALGO:-copc}
if [[ $# -gt 0 && $1 != --* ]]; then ALGO=$1; shift; fi
case "$TASK" in retool|search-r1) ;; *) usage >&2; exit 2 ;; esac
case "$ALGO" in
    copc|copc-wo-rc|copc-vtrace|ppo|ppo-sync|grpo|dapo|sao|vtrace|areal) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ -n ${VTRACE_CLIP_ONLY:-} ]]; then
    [[ $VTRACE_CLIP_ONLY == 0 || $VTRACE_CLIP_ONLY == 1 ]] || {
        echo "VTRACE_CLIP_ONLY must be 0 or 1." >&2; exit 2;
    }
    if [[ $VTRACE_CLIP_ONLY == 1 ]]; then
        [[ $ALGO == vtrace || $ALGO == copc-vtrace ]] || {
            echo "VTRACE_CLIP_ONLY=1 requires vtrace or copc-vtrace." >&2; exit 2;
        }
        ALGO=copc-vtrace
    elif [[ $ALGO == copc-vtrace ]]; then
        echo "copc-vtrace requires VTRACE_CLIP_ONLY=1 or unset." >&2; exit 2
    fi
fi
require_env MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR
USE_CRITIC=1
[[ $ALGO != grpo && $ALGO != dapo ]] || USE_CRITIC=0
if (( USE_CRITIC )); then
    require_env CRITIC_LOAD_DIR
    CRITIC_STEP=${CRITIC_STEP:-49}
    [[ $CRITIC_STEP =~ ^[1-9][0-9]*$ ]] || {
        echo "CRITIC_STEP must be positive; explicit step 0 loading is not supported by this launcher." >&2
        exit 2
    }
fi
normalize_paths MODEL_DIR MODEL_TORCH_DIST TRAIN_DATA OUTPUT_DIR CRITIC_LOAD_DIR ASYNC_RESUME
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Choose a fresh OUTPUT_DIR, including for resume; the source checkpoint remains read-only." >&2
    exit 2
fi

DEFAULT_TRAIN_GPUS=2; DEFAULT_ROLLOUT_GPUS=6
DEFAULT_CLIP_HIGH=0.28
DEFAULT_SAVE=10
if [[ $TASK == retool ]]; then
    export MODEL_ARCH=qwen3
    DEFAULT_STEPS=120; DEFAULT_BATCH=64; DEFAULT_SAMPLES=8; DEFAULT_GLOBAL=512
    DEFAULT_STALENESS=8
    case "$ALGO" in
        grpo) DEFAULT_BATCH=128; DEFAULT_GLOBAL=1024 ;;
        areal) DEFAULT_STEPS=60; DEFAULT_BATCH=128; DEFAULT_SAVE=5 ;;
    esac
    DEFAULT_PROMPT=2048; DEFAULT_RESPONSE=16384; DEFAULT_MAX_TOKENS=32768
    WEIGHT_DECAY=0.1; ADAM_BETA2=0.98; LR_WARMUP_ITERS=0
    export MAX_TURNS=${MAX_TURNS:-8} MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-8}
    export RETOOL_REWARD_MODE=math_verify
    TASK_ARGS=(--input-key prompt --label-key label --reward-key score
        --custom-generate-function-path generate_with_retool.generate
        --custom-rm-path generate_with_retool.reward_func
        --sglang-router-disable-circuit-breaker --sequence-parallel)
else
    export MODEL_ARCH=llama
    require_env RETRIEVAL_SERVICE_URL
    DEFAULT_STEPS=450; DEFAULT_BATCH=128; DEFAULT_SAMPLES=4; DEFAULT_GLOBAL=512
    DEFAULT_STALENESS=14
    case "$ALGO" in
        grpo) DEFAULT_TRAIN_GPUS=1; DEFAULT_ROLLOUT_GPUS=7; DEFAULT_CLIP_HIGH=0.2 ;;
        sao) DEFAULT_BATCH=512; DEFAULT_SAMPLES=1 ;;
        areal) DEFAULT_STEPS=225; DEFAULT_BATCH=256; DEFAULT_SAVE=5 ;;
    esac
    DEFAULT_PROMPT=512; DEFAULT_RESPONSE=8192; DEFAULT_MAX_TOKENS=8704
    WEIGHT_DECAY=0.01; ADAM_BETA2=0.999; LR_WARMUP_ITERS=50
    [[ $ALGO != areal ]] || LR_WARMUP_ITERS=25
    export RETRIEVAL_SERVICE_URL SEARCH_PROTOCOL=llama3
    export SEARCH_REWARD_MODE=${SEARCH_REWARD_MODE:-grpo}
    export SEARCH_MAX_TURNS=${SEARCH_MAX_TURNS:-8} SEARCH_TOPK=${SEARCH_TOPK:-3}
    export SEARCH_MAX_DOC_LENGTH=${SEARCH_MAX_DOC_LENGTH:-200}
    export SEARCH_TOOL_TIMEOUT=${SEARCH_TOOL_TIMEOUT:-65} SEARCH_TOOL_MAX_RETRIES=${SEARCH_TOOL_MAX_RETRIES:-3}
    export SEARCH_TOOL_RATE_LIMIT=${SEARCH_TOOL_RATE_LIMIT:-120}
    export SEARCH_RETRIEVAL_FAILURE_MODE=${SEARCH_RETRIEVAL_FAILURE_MODE:-observation}
    TASK_ARGS=(--input-key prompt --label-key reward_model --reward-key score --eval-reward-key em
        --tool-key tools --eval-tool-key tools
        --apply-chat-template-kwargs '{"date_string":"16 Sep 2026"}'
        --custom-generate-function-path generate_with_search.generate
        --custom-rm-path search_reward.reward_func
        --sglang-dtype bfloat16 --sglang-disable-cuda-graph --sglang-watchdog-timeout 1800)
    [[ $ALGO == grpo || $ALGO == dapo ]] || TASK_ARGS+=(--calculate-per-token-loss)
fi
NUM_STEPS=${NUM_STEPS:-$DEFAULT_STEPS}
TRAIN_GPUS=${TRAIN_GPUS:-$DEFAULT_TRAIN_GPUS}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$DEFAULT_ROLLOUT_GPUS}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$DEFAULT_BATCH}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-$DEFAULT_SAMPLES}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$DEFAULT_GLOBAL}
SAVE_INTERVAL=${SAVE_INTERVAL:-$DEFAULT_SAVE}
UPDATE_WEIGHTS_INTERVAL=${UPDATE_WEIGHTS_INTERVAL:-1}
FULLY_ASYNC_MAX_STALENESS=${FULLY_ASYNC_MAX_STALENESS:-$DEFAULT_STALENESS}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-$DEFAULT_PROMPT}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-$DEFAULT_RESPONSE}
NUM_UPDATES=${NUM_STEPS_PER_ROLLOUT:-}
ASYNC_SAVE=${ASYNC_SAVE:-1}
ESTIMATOR=$ALGO
[[ $ALGO != sao && $ALGO != copc-wo-rc && $ALGO != ppo-sync ]] || ESTIMATOR=ppo
[[ $ALGO != copc-vtrace ]] || ESTIMATOR=vtrace
[[ $ALGO != dapo ]] || ESTIMATOR=grpo
EXTRA_ARGS=()
while (( $# )); do
    option=${1%%=*}
    if [[ $ALGO == ppo-sync ]]; then
        case "$option" in
            --colocate|--rollout-function-path|--fully-async-max-staleness|--async-checkpoint*|--async-resume)
                echo "ppo-sync uses decoupled synchronous rollout and does not accept $option." >&2; exit 2 ;;
        esac
    fi
    case "$option" in
        --num-rollout) name=NUM_STEPS ;;
        --rollout-batch-size) name=ROLLOUT_BATCH_SIZE ;;
        --n-samples-per-prompt) name=N_SAMPLES_PER_PROMPT ;;
        --global-batch-size) name=GLOBAL_BATCH_SIZE ;;
        --num-steps-per-rollout) name=NUM_UPDATES ;;
        --save-interval) name=SAVE_INTERVAL ;;
        --update-weights-interval) name=UPDATE_WEIGHTS_INTERVAL ;;
        --fully-async-max-staleness) name=FULLY_ASYNC_MAX_STALENESS ;;
        --advantage-estimator) name=ESTIMATOR_OVERRIDE ;;
        --async-save)
            [[ $1 == --async-save ]] || { echo "Use ASYNC_SAVE=0 or 1." >&2; exit 2; }
            ASYNC_SAVE=1; shift; continue ;;
        --areal-max-train-steps)
            echo "Use NUM_STEPS or --num-rollout; AReaL completes every rollout." >&2; exit 2 ;;
        *) EXTRA_ARGS+=("$1"); shift; continue ;;
    esac
    if [[ $1 == *=* ]]; then
        value=${1#*=}; shift
    else
        (( $# >= 2 )) && [[ $2 != --* ]] || { echo "Missing value for $option." >&2; exit 2; }
        value=$2; shift 2
    fi
    [[ -n $value ]] || { echo "Missing value for $option." >&2; exit 2; }
    printf -v "$name" '%s' "$value"
done
set -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
[[ ${ESTIMATOR_OVERRIDE:-$ESTIMATOR} == "$ESTIMATOR" ]] || {
    echo "Select the algorithm with the launcher argument, not --advantage-estimator." >&2; exit 2;
}
[[ $ASYNC_SAVE == 0 || $ASYNC_SAVE == 1 ]] || { echo "ASYNC_SAVE must be 0 or 1." >&2; exit 2; }
for name in NUM_STEPS TRAIN_GPUS ROLLOUT_GPUS ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE SAVE_INTERVAL UPDATE_WEIGHTS_INTERVAL MAX_PROMPT_LEN MAX_RESPONSE_LEN; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer." >&2; exit 2; }
done
TRAIN_ENTRYPOINT=train_async.py
ROLLOUT_FUNCTION=slime.rollout.fully_async_rollout.generate_rollout_fully_async
STALENESS_ARGS=(--fully-async-max-staleness "$FULLY_ASYNC_MAX_STALENESS")
if [[ $ALGO == ppo-sync ]]; then
    [[ -z ${ASYNC_RESUME:-} ]] || { echo "ASYNC_RESUME requires an async training mode." >&2; exit 2; }
    (( UPDATE_WEIGHTS_INTERVAL == 1 )) || { echo "ppo-sync requires UPDATE_WEIGHTS_INTERVAL=1." >&2; exit 2; }
    TRAIN_ENTRYPOINT=train.py
    ROLLOUT_FUNCTION=slime.rollout.sglang_rollout.generate_rollout
    STALENESS_ARGS=()
else
    [[ $FULLY_ASYNC_MAX_STALENESS =~ ^[1-9][0-9]*$ ]] || {
        echo "FULLY_ASYNC_MAX_STALENESS must be a positive integer." >&2; exit 2;
    }
fi
if [[ $ALGO == dapo ]] && (( N_SAMPLES_PER_PROMPT < 2 )); then
    echo "DAPO dynamic sampling requires at least two responses per prompt." >&2; exit 2
fi
(( (ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) % GLOBAL_BATCH_SIZE == 0 )) || {
    echo "The sampled batch must be divisible by GLOBAL_BATCH_SIZE." >&2; exit 2;
}
(( NUM_STEPS % UPDATE_WEIGHTS_INTERVAL == 0 )) || {
    echo "NUM_STEPS must end on a weight publication boundary." >&2; exit 2;
}
EXPECTED_UPDATES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / GLOBAL_BATCH_SIZE))
NUM_UPDATES=${NUM_UPDATES:-$EXPECTED_UPDATES}
[[ $NUM_UPDATES =~ ^[1-9][0-9]*$ ]] && (( NUM_UPDATES == EXPECTED_UPDATES )) || {
    echo "Updates per rollout must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / GLOBAL_BATCH_SIZE." >&2; exit 2;
}
printf '[config] algorithm=%s rollouts=%s updates_per_rollout=%s total_optimizer_updates=%s\n' \
    "$ALGO" "$NUM_STEPS" "$NUM_UPDATES" "$((NUM_STEPS * NUM_UPDATES))"
printf '[config] save_interval=%s rollouts (%s optimizer updates), weight_sync_interval=%s rollouts, async_save=%s\n' \
    "$SAVE_INTERVAL" "$((SAVE_INTERVAL * NUM_UPDATES))" "$UPDATE_WEIGHTS_INTERVAL" "$ASYNC_SAVE"
if [[ $TASK == search-r1 ]]; then
    CONTEXT_LENGTH=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
    TASK_ARGS+=(--rollout-max-context-len "$CONTEXT_LENGTH" --seq-length "$CONTEXT_LENGTH"
        --sglang-context-length "$CONTEXT_LENGTH" --sglang-max-prefill-tokens "$CONTEXT_LENGTH")
fi
load_model_args
if [[ ${DRY_RUN:-0} != 1 ]]; then
    [[ -f "$MODEL_TORCH_DIST/latest_checkpointed_iteration.txt" && -f "$TRAIN_DATA" ]] || {
        echo "Provide a native SFT checkpoint and an existing training data file." >&2; exit 2;
    }
fi

if [[ $TASK == search-r1 ]]; then check_search_service; fi

ALGO_ARGS=(--advantage-estimator "$ESTIMATOR" --num-critic-only-steps 0)
case "$ALGO" in
    copc|copc-wo-rc|ppo|ppo-sync|sao)
        ALGO_ARGS+=(--gamma 1 --lambd 1 --value-clip 0.2 --normalize-advantages --use-rollout-logprobs)
        [[ $ALGO != sao && $ALGO != copc-wo-rc ]] || ALGO_ARGS+=(--clip-only)
        if [[ $ALGO == copc ]]; then
            ALGO_ARGS+=(--copc-rho-clip "${RHO_CLIP:-0.2}" --copc-rho-clip-high "${RHO_CLIP_HIGH:-0.28}")
        fi ;;
    grpo) ALGO_ARGS+=(--use-rollout-logprobs) ;;
    dapo)
        ALGO_ARGS+=(--use-rollout-logprobs --calculate-per-token-loss
            --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std) ;;
    areal)
        ALGO_ARGS+=(--gamma 1 --lambd 1 --value-clip 0.2 --normalize-advantages
            --areal-is-weight-max "${AREAL_IS_WEIGHT_MAX:-5}") ;;
    vtrace|copc-vtrace)
        ALGO_ARGS+=(--gamma 1 --lambd 1 --normalize-advantages --use-rollout-logprobs --value-clip "${VALUE_CLIP:-0.2}"
            --vtrace-rho-clip "${VTRACE_RHO_CLIP:-1}" --vtrace-c-clip "${VTRACE_C_CLIP:-1}"
            --vtrace-pg-rho-clip "${VTRACE_PG_RHO_CLIP:-1}")
        [[ $ALGO != copc-vtrace ]] || ALGO_ARGS+=(--clip-only) ;;
esac
if [[ $ALGO != vtrace ]]; then
    ALGO_ARGS+=(--eps-clip "${EPS_CLIP:-0.2}" --eps-clip-high "${EPS_CLIP_HIGH:-$DEFAULT_CLIP_HIGH}")
fi
CKPT_ARGS=(--hf-checkpoint "$MODEL_DIR" --load "$MODEL_TORCH_DIST" --ref-load "$MODEL_TORCH_DIST"
    --save "$OUTPUT_DIR/actor_megatron" --save-interval "$SAVE_INTERVAL" --ckpt-format torch_dist)
[[ $ASYNC_SAVE == 0 ]] || CKPT_ARGS+=(--async-save)
if [[ -n ${ASYNC_RESUME:-} ]]; then
    CKPT_ARGS+=(--async-resume "$ASYNC_RESUME")
else
    CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
fi
launch mkdir -p -- "$OUTPUT_DIR"
if (( USE_CRITIC )); then
    ROLE_CONFIG="$OUTPUT_DIR/roles.json"
    launch "$PYTHON" - "$ROLE_CONFIG" "$CRITIC_LOAD_DIR" "$CRITIC_STEP" \
        "$OUTPUT_DIR/critic_megatron" "${CRITIC_LR:-5e-6}" <<'PY_ROLES'
import json
import sys
from pathlib import Path
path, load, step, save, lr = sys.argv[1:]
config = {"megatron": [{"name": "critic", "role": "critic", "overrides": {
    "load": load, "ckpt_step": int(step), "save": save, "lr": float(lr),
    "lr_warmup_iters": 0, "finetune": True, "no_load_optim": True,
    "no_load_rng": True, "use_checkpoint_opt_param_scheduler": False,
}}]}
Path(path).write_text(json.dumps(config, indent=2) + "\n")
PY_ROLES
    CKPT_ARGS+=(--megatron-config-path "$ROLE_CONFIG")
fi
TRACKING_ARGS=()
if [[ -n ${WANDB_PROJECT:-} ]]; then
    TRACKING_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-dir "$OUTPUT_DIR/wandb")
    [[ -z ${WANDB_ENTITY:-} ]] || TRACKING_ARGS+=(--wandb-team "$WANDB_ENTITY")
    [[ -z ${WANDB_RUN_NAME:-} ]] || TRACKING_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
fi
TRAIN_ARGS=(--actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS" --num-gpus-per-node 8
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${TASK_ARGS[@]}" "${ALGO_ARGS[@]}"
    --rollout-function-path "$ROLLOUT_FUNCTION"
    --prompt-data "$TRAIN_DATA" --apply-chat-template --rollout-shuffle
    --seed "${SEED:-1234}" --rollout-seed "${ROLLOUT_SEED:-42}"
    --num-rollout "$NUM_STEPS" --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" --global-batch-size "$GLOBAL_BATCH_SIZE"
    --num-steps-per-rollout "$NUM_UPDATES" --update-weights-interval "$UPDATE_WEIGHTS_INTERVAL"
    ${STALENESS_ARGS[@]+"${STALENESS_ARGS[@]}"}
    --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine 1
    --rollout-max-prompt-len "$MAX_PROMPT_LEN" --rollout-max-response-len "$MAX_RESPONSE_LEN"
    --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --balance-data --log-passrate
    --optimizer adam --lr "${ACTOR_LR:-1e-6}" --lr-decay-style constant --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY" --adam-beta1 0.9 --adam-beta2 "$ADAM_BETA2" --clip-grad 1
    --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-$DEFAULT_MAX_TOKENS}" --offload-train
    --sglang-mem-fraction-static 0.85 --sglang-server-concurrency 128
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash)
if [[ -n ${WANDB_PROJECT:-} ]]; then TRAIN_ARGS+=("${TRACKING_ARGS[@]}"); fi
launch_training "$TRAIN_ENTRYPOINT" "${TRAIN_ARGS[@]}" "$@"
