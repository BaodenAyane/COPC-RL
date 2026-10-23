#!/usr/bin/env bash
# Convert the supplied Search SFT checkpoint to Megatron format without training.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Search SFT preparation (no supervised updates):
  MODEL_DIR=/path/to/search-sft OUTPUT_DIR=/path/to/converted bash scripts/search-r1/sft.sh
Requires the prepared Llama-3.2-3B Search SFT checkpoint and its original tool template.
Optional: DRY_RUN=1, CONVERT_GPUS=4, CONVERT_TP=1, CONVERT_PP=4, MEGATRON_ROOT.
Output: OUTPUT_DIR/megatron. The supplied HF model is preserved.
USAGE
    exit 0
fi
[[ $# -eq 0 ]] || { echo "Configure this stage with environment variables; no positional arguments are accepted." >&2; exit 2; }
export MODEL_ARCH=llama
require_env MODEL_DIR OUTPUT_DIR
normalize_paths MODEL_DIR OUTPUT_DIR

CONVERT_GPUS=${CONVERT_GPUS:-4}
CONVERT_TP=${CONVERT_TP:-1}
CONVERT_PP=${CONVERT_PP:-4}
for value in "$CONVERT_GPUS" "$CONVERT_TP" "$CONVERT_PP"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Conversion GPU and parallel sizes must be positive integers." >&2; exit 2; }
done
(( CONVERT_GPUS == CONVERT_TP * CONVERT_PP )) || { echo "CONVERT_GPUS must equal CONVERT_TP * CONVERT_PP." >&2; exit 2; }
load_model_args
printf '%s\n' "Search SFT preparation: validate and convert an existing task-aligned HF checkpoint; no SFT updates."
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON" - "$MODEL_DIR" "$OUTPUT_DIR" <<'PYCODE'
import json
from pathlib import Path
import sys
model, output = (Path(value).expanduser().resolve() for value in sys.argv[1:])
if model == output or model in output.parents or output in model.parents:
    raise SystemExit("OUTPUT_DIR must be separate from the input HF checkpoint")
config = json.loads((model / "config.json").read_text())
expected = {"model_type": "llama", "num_hidden_layers": 28, "hidden_size": 3072,
            "num_attention_heads": 24, "num_key_value_heads": 8, "vocab_size": 128256}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit("This Search stage expects the Llama-3.2-3B Search SFT model: " + key)
tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
template_path = model / "chat_template.jinja"
template = template_path.read_text() if template_path.is_file() else tokenizer_config.get("chat_template", "")
markers = ("<|start_header_id|>", "<|eot_id|>", "ipython", "parameters", "16 Sep 2026")
if not isinstance(template, str) or not all(marker in template for marker in markers):
    raise SystemExit("Preserve the Search SFT tool chat template and its fixed date (16 Sep 2026)")
if not (model / "tokenizer.json").is_file():
    raise SystemExit("The supplied Search SFT model must include tokenizer.json")
if not any(model.glob("*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
    raise SystemExit("The supplied Search SFT model contains no HF weights")
if (output / "megatron").exists():
    raise SystemExit("Refusing to overwrite an existing OUTPUT_DIR/megatron checkpoint")
output.mkdir(parents=True, exist_ok=True)
PYCODE
fi
launch "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$CONVERT_GPUS" \
    "$REPO_ROOT/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "$OUTPUT_DIR/megatron" \
    --tensor-model-parallel-size "$CONVERT_TP" --pipeline-model-parallel-size "$CONVERT_PP" \
    --context-parallel-size 1 --seq-length 131072 --bf16 --use-cpu-initialization
printf '%s\n' "For pretraining, keep MODEL_DIR unchanged and set MODEL_TORCH_DIST=$OUTPUT_DIR/megatron."
