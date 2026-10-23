"""Start a local Ray runtime or connect to RAY_ADDRESS, then run training."""
import os
import runpy
import sys
from pathlib import Path


def main():
    import ray

    root = Path(__file__).resolve().parents[2]
    entrypoint = (root / sys.argv[1]).resolve()
    if entrypoint not in (root / "train.py", root / "train_async.py"):
        raise SystemExit("Expected train.py or train_async.py")
    # Task settings must reach actors on every Ray node, including tool workers.
    names = {"PYTHONPATH", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
             "CUDA_DEVICE_MAX_CONNECTIONS", "MAX_TURNS", "MAX_TOOL_CALLS",
             "RETRIEVAL_SERVICE_URL"}
    prefixes = ("RETOOL_", "SEARCH_", "NCCL_", "NVTE_", "SGLANG_", "RAY_", "WANDB_")
    env = {key: value for key, value in os.environ.items()
           if (key in names or key.startswith(prefixes)) and key != "RAY_ADDRESS"}
    os.chdir(root)
    sys.path.insert(0, str(root))
    # All nodes must see the checkout, models and data at the same paths.
    ray.init(address=os.environ.get("RAY_ADDRESS", "local"),
             runtime_env={"env_vars": env})
    sys.argv = [str(entrypoint)] + sys.argv[2:]
    try:
        runpy.run_path(str(entrypoint), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
