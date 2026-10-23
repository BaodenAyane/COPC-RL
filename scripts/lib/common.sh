#!/usr/bin/env bash
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python3}
if [[ -n ${MEGATRON_ROOT:-} && $MEGATRON_ROOT != /* ]]; then
    MEGATRON_ROOT="$PWD/$MEGATRON_ROOT"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/examples/retool:${REPO_ROOT}/examples/search-r1${MEGATRON_ROOT:+:${MEGATRON_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export RAY_USE_UVLOOP=0

require_env() {
    local name
    for name in "$@"; do
        if [[ -z ${!name:-} ]]; then
            printf 'Set %s before running this script. See --help.\n' "$name" >&2
            return 2
        fi
    done
}

launch() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        "$@"
    fi
}

check_search_service() {
    [[ ${DRY_RUN:-0} != 1 ]] || return 0
    "$PYTHON" - "$REPO_ROOT/examples/search-r1/retrieval/sources.json" <<'PY_SEARCH'
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, build_opener

from generate_with_search import SearchSettings
from retrieval_client import SearchClient

try:
    settings = SearchSettings.from_env()
    client = SearchClient(settings.url, topk=settings.topk, timeout=settings.timeout,
                          max_retries=settings.max_retries, concurrency=settings.concurrency)
    url = urlsplit(settings.url)
    health_url = urlunsplit((url.scheme, url.netloc, "/health", "", ""))
    with build_opener(ProxyHandler({})).open(health_url, timeout=10) as response:
        health = json.load(response)
    expected = dict(status="ready", model="intfloat/e5-base-v2", dimension=768,
                    index_type="hnsw64", corpus_rows=21015324,
                    sources=json.loads(Path(sys.argv[1]).read_text()))
    if not isinstance(health, dict) or any(health.get(key) != value for key, value in expected.items()):
        raise ValueError("Retriever is not ready or does not match the pinned Search-R1 assets")
    asyncio.run(client.retrieve(["Wikipedia"]))
except (OSError, ValueError, RuntimeError) as exc:
    raise SystemExit(f"Search service check failed: {exc}") from None
print("Search service ready; retrieval check passed.")
PY_SEARCH
}

load_model_args() {
    local args_text
    args_text=$("$PYTHON" "$REPO_ROOT/scripts/lib/model_args.py" \
        "$MODEL_DIR" "${MODEL_ARCH:-qwen3}" "${DRY_RUN:-0}") || return
    mapfile -t MODEL_ARGS <<< "$args_text"
}

launch_training() {
    local entrypoint=$1
    shift
    launch "$PYTHON" "$REPO_ROOT/scripts/lib/launch.py" "$entrypoint" "$@"
}

normalize_paths() {
    local name value
    for name in "$@"; do
        value=${!name:-}
        [[ -n "$value" ]] || continue
        [[ "$value" = /* ]] || printf -v "$name" '%s/%s' "$PWD" "$value"
        export "$name"
    done
}
