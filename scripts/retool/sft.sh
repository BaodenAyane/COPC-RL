#!/usr/bin/env bash
# Convert and prepare ReTool conversations, then train a Qwen3-4B SFT model.
# Required: MODEL_DIR (base HF model), TRAIN_DATA (ReTool records), OUTPUT_DIR.
# Outputs: data/train.parquet, hf/, and megatron/. DRY_RUN=1 only prints commands.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
case "${1:-}" in
    --help|-h)
        cat <<'HELP'
Usage: bash scripts/retool/sft.sh
Required environment:
  MODEL_DIR        Base Qwen3-4B Hugging Face checkpoint and tokenizer.
  TRAIN_DATA       Local ReTool JSON/JSONL/Parquet records.
  OUTPUT_DIR       Fresh output directory; produces hf/ and megatron/.
Optional environment:
  SFT_DATA_FORMAT  raw (default) or structured (messages/tools Parquet).
  TRAIN_GPUS      Training GPUs on one node (default 8).
  MEGATRON_ROOT   Megatron-LM checkout if it is not already importable.
  RAY_ADDRESS     Existing Ray cluster; otherwise start a local runtime.
  DRY_RUN=1       Print preparation and training commands without executing them.
HELP
        exit 0 ;;
esac
[[ $# == 0 ]] || { echo "Unknown argument; use --help." >&2; exit 2; }
require_env MODEL_DIR TRAIN_DATA OUTPUT_DIR
normalize_paths MODEL_DIR TRAIN_DATA OUTPUT_DIR
SFT_DATA_FORMAT=${SFT_DATA_FORMAT:-raw}
case "$SFT_DATA_FORMAT" in
    raw|structured) ;;
    *) echo "SFT_DATA_FORMAT must be raw or structured." >&2; exit 2 ;;
esac
[[ -z "${ASYNC_RESUME:-}" ]] || { echo "SFT starts a new run; ASYNC_RESUME is not supported here." >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR already exists; choose a fresh directory." >&2; exit 2; }
TRAIN_GPUS=${TRAIN_GPUS:-8}
load_model_args

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON" - "$MODEL_DIR" <<'PY_MODEL'
import json
import sys
from pathlib import Path
config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
if config.get("model_type") != "qwen3" or config.get("rope_scaling"):
    raise SystemExit("ReTool SFT expects an unscaled Qwen3-4B checkpoint.")
PY_MODEL
fi

PREPARED_DATA="$OUTPUT_DIR/data/train.parquet"
PREPARATION_REPORT="$OUTPUT_DIR/data/preparation-report.json"
launch mkdir -p -- "$OUTPUT_DIR/data"
STRUCTURED_DATA="$TRAIN_DATA"
if [[ "$SFT_DATA_FORMAT" == raw ]]; then
    STRUCTURED_DATA="$OUTPUT_DIR/data/structured.parquet"
    launch "$PYTHON" "$REPO_ROOT/examples/retool/sft_data_processing.py" \
        --input "$TRAIN_DATA" --output "$STRUCTURED_DATA"
fi
launch "$PYTHON" "$REPO_ROOT/examples/retool/prepare_sft_data.py" \
    --input "$STRUCTURED_DATA" --output "$PREPARED_DATA" --report "$PREPARATION_REPORT" \
    --model-dir "$MODEL_DIR" --max-length 16384
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON" - "$PREPARATION_REPORT" <<'PY_DATA'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
if report.get("protocol") != "qwen-tools-v1" or not report.get("data_written"):
    raise SystemExit("The dataset must be prepared with the shared Qwen tool protocol.")
rows = report.get("accepted_rows")
if type(rows) is not int or rows < 16:
    raise SystemExit("SFT requires at least one complete batch of 16 conversations.")
print("[sft] Prepared %d conversations; two epochs, batch size 16." % rows)
PY_DATA
fi

# A large periodic interval leaves the driver's epoch-end and final saves enabled.
# Native --save-hf also preserves the tokenizer, template and generation config.
TRAIN_ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$MODEL_DIR" --load "$MODEL_DIR"
    --finetune --no-load-optim --no-load-rng --start-rollout-id 0
    --save "$OUTPUT_DIR/megatron" --save-hf "$OUTPUT_DIR/hf" --save-interval 2147483647
    --prompt-data "$PREPARED_DATA" --input-key messages --tool-key tools --metadata-key metadata
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --rollout-shuffle --rollout-seed 42 --seed 42
    --num-epoch 2 --rollout-batch-size 16 --n-samples-per-prompt 1
    --global-batch-size 16 --micro-batch-size 1
    --loss-type sft_loss --loss-mask-type qwen3
    --disable-compute-advantages-and-returns --disable-rewards-normalization --debug-train-only
    --optimizer adam --lr 1e-5 --min-lr 0 --weight-decay 0.01
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
    --lr-decay-style cosine --lr-warmup-fraction 0.1
    --seq-length 16384
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 16384 --balance-data
    --attention-dropout 0 --hidden-dropout 0
    --attention-backend flash --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
    --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0
)
launch_training "train.py" "${TRAIN_ARGS[@]}"
