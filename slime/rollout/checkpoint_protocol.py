"""Checkpointable generation and reward protocols; validation uses only the standard library."""
from __future__ import annotations

import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[2]
_GENERATORS = {
    None: ("single_turn", "slime/rollout/sglang_rollout.py"),
    "slime.rollout.sglang_rollout.generate": ("single_turn", "slime/rollout/sglang_rollout.py"),
    "generate_with_retool.generate": ("retool", "examples/retool/generate_with_retool.py"),
    "examples.retool.generate_with_retool.generate": ("retool", "examples/retool/generate_with_retool.py"),
    "generate_with_search.generate": ("search", "examples/search-r1/generate_with_search.py"),
}
_REWARDS = {
    "generate_with_retool.reward_func": ["examples/retool/generate_with_retool.py", "slime/rollout/rm_hub/math_dapo_utils.py"],
    "examples.retool.generate_with_retool.reward_func": ["examples/retool/generate_with_retool.py", "slime/rollout/rm_hub/math_dapo_utils.py"],
    "search_reward.reward_func": ["examples/search-r1/search_reward.py", "examples/search-r1/reference_reward.py", "examples/search-r1/search_protocol.py"],
}
_BUILTIN_REWARDS = {
    "deepscaler": ["deepscaler.py", "math_utils.py"],
    "dapo": ["math_dapo_utils.py"],
    "math": ["math_utils.py"],
    "f1": ["f1.py"],
    "gpqa": ["gpqa.py"],
}


_DYNAMIC_FILTERS = {
    "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std":
        "slime/rollout/filter_hub/dynamic_sampling_filters.py",
}


def describe_dynamic_filter_protocol(args):
    path = getattr(args, "dynamic_sampling_filter_path", None)
    if path is None:
        return None
    if path not in _DYNAMIC_FILTERS:
        raise ValueError("Async checkpoints require an audited stateless dynamic sampling filter: " + str(path))
    source = _DYNAMIC_FILTERS[path]
    _check_module_resolution(path, source)
    return {"name": path, "stateless": True,
            "source_files": [source, "slime/rollout/filter_hub/base_types.py"]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _check_module_resolution(path, relative_file):
    module = path.rsplit(".", 1)[0]
    locations = None
    parts = module.split(".")
    for index in range(len(parts)):
        spec = importlib.machinery.PathFinder.find_spec(".".join(parts[:index + 1]), locations)
        if spec is None:
            # Fingerprint the source even when it is not yet importable.
            return
        if index < len(parts) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                raise ValueError("Async checkpoint adapter has an invalid module path: " + module)
    if spec.origin and Path(spec.origin).resolve() != (_ROOT / relative_file).resolve():
        raise ValueError("Async checkpoint adapter resolves to an unaudited source: " + module)


def describe_reward_protocol(args):
    """Allow only rewards without unsaved process state."""
    custom = getattr(args, "custom_rm_path", None)
    if custom == "":
        raise ValueError("Async checkpoint custom_rm_path must be None or an audited reward, not an empty string")
    if custom:
        if custom not in _REWARDS:
            raise ValueError("Async checkpoint requires an audited stateless custom reward: " + str(custom))
        sources = _REWARDS[custom]
        _check_module_resolution(custom, sources[0])
        config = {}
        if custom == "search_reward.reward_func":
            mode = os.environ.get("SEARCH_REWARD_MODE", "grpo").strip().lower()
            if mode not in {"grpo", "base"}:
                raise ValueError("Unsupported SEARCH_REWARD_MODE for async checkpoint")
            config = {"mode": mode, "protocol": os.environ.get("SEARCH_PROTOCOL", "qwen")}
        elif "generate_with_retool" in custom:
            config = {"mode": os.environ.get("RETOOL_REWARD_MODE", "math_verify")}
        return {"name": custom, "stateless": True, "config": config, "source_files": list(sources)}
    reward_type = (getattr(args, "rm_type", None) or "").strip()
    base_type = reward_type[len("boxed_"):] if reward_type.startswith("boxed_") else reward_type
    if base_type not in _BUILTIN_REWARDS:
        raise ValueError("Async checkpoint requires an audited built-in reward; rm_type=" + repr(reward_type))
    files = ["slime/rollout/rm_hub/__init__.py"]
    files += ["slime/rollout/rm_hub/" + name for name in _BUILTIN_REWARDS[base_type]]
    return {"name": reward_type, "stateless": True, "config": {}, "source_files": files}


def _search_config():
    protocol = os.environ.get("SEARCH_PROTOCOL", "llama3")
    if protocol not in {"qwen", "llama", "llama3"}:
        raise ValueError("Unsupported SEARCH_PROTOCOL for async checkpoint")
    endpoint = os.environ.get("RETRIEVAL_SERVICE_URL", "")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Async Search requires a concrete credential-free RETRIEVAL_SERVICE_URL")
    config = {"protocol": protocol, "retrieval_endpoint_sha256": _digest(endpoint)}
    integers = {"max_turns": ("SEARCH_MAX_TURNS", "8"), "topk": ("SEARCH_TOPK", "3"),
                "max_doc_length": ("SEARCH_MAX_DOC_LENGTH", "200"),
                "max_retries": ("SEARCH_TOOL_MAX_RETRIES", "3"),
                "concurrency": ("SEARCH_TOOL_RATE_LIMIT", "120")}
    for key, (variable, default) in integers.items():
        try:
            config[key] = int(os.environ.get(variable, default))
        except (TypeError, ValueError):
            raise ValueError("Invalid " + variable) from None
        if config[key] <= 0:
            raise ValueError(variable + " must be positive")
    if config["topk"] > 10 or config["max_retries"] > 10:
        raise ValueError("Search topk and max_retries must be <= 10")
    config["timeout"] = float(os.environ.get("SEARCH_TOOL_TIMEOUT", "65"))
    if not math.isfinite(config["timeout"]) or config["timeout"] <= 0:
        raise ValueError("SEARCH_TOOL_TIMEOUT must be finite and positive")
    config["retrieval_failure_mode"] = os.environ.get(
        "SEARCH_RETRIEVAL_FAILURE_MODE", "observation" if protocol in {"llama", "llama3"} else "abort")
    if config["retrieval_failure_mode"] not in {"abort", "observation"}:
        raise ValueError("SEARCH_RETRIEVAL_FAILURE_MODE must be abort or observation")
    # Pin external corpus/index revisions when supplied; saved observations remain in Sample.
    config["retrieval_revision_sha256"] = _digest(os.environ.get("SEARCH_RETRIEVAL_REVISION", ""))
    return config


def describe_checkpoint_protocol(args):
    generator = getattr(args, "custom_generate_function_path", None)
    if generator not in _GENERATORS:
        raise ValueError("Unsupported async checkpoint generator: " + str(generator))
    for flag in ("partial_rollout", "group_rm", "use_rollout_routing_replay"):
        if getattr(args, flag, False):
            raise ValueError("Async generator checkpoint does not support " + flag)
    if getattr(args, "multimodal_keys", None):
        raise ValueError("Async generator checkpoint currently requires text-only inputs")
    name, generator_file = _GENERATORS[generator]
    if generator:
        _check_module_resolution(generator, generator_file)
    files = ["slime/rollout/checkpoint_protocol.py", "slime/rollout/checkpoint_barrier.py", generator_file]
    config = {}
    if name == "search":
        config = _search_config()
        files += ["examples/search-r1/retrieval_client.py", "examples/search-r1/search_protocol.py"]
    elif name == "retool":
        files += ["examples/retool/tool_sandbox.py", "slime/utils/retool_protocol.py"]
        config = {"runtime_compat": "legacy-tolerant-v1", "tool_environment": "independent_python_subprocess",
                  "max_turns": int(os.environ.get("MAX_TURNS", "16")),
                  "max_tool_calls": int(os.environ.get("MAX_TOOL_CALLS", "16"))}
        if config["max_turns"] <= 0 or config["max_tool_calls"] < 0:
            raise ValueError("ReTool max_turns must be positive and max_tool_calls nonnegative")
    dynamic_filter = describe_dynamic_filter_protocol(args)
    if dynamic_filter is not None:
        files.extend(dynamic_filter.pop("source_files"))
    reward = describe_reward_protocol(args)
    files.extend(reward.pop("source_files"))
    result = {"name": name, "version": 1, "generation_boundary": "completed_request_or_tool_action",
              "sampling_protocol": "request_seed_v1", "source_files": sorted(set(files)),
              "config": config, "reward": reward, "dynamic_sampling_filter": dynamic_filter}
    json.dumps(result, allow_nan=False)
    return result


def request_seed(args, sample, request_index, *, namespace):
    """Stable seeds depend on saved IDs and request cursor, not coroutine order."""
    sample_id, group_id = getattr(sample, "index", None), getattr(sample, "group_index", None)
    for value in (sample_id, group_id, request_index):
        if type(value) is not int or value < 0:
            raise ValueError("Checkpointed generation requires non-negative integer sample/group/request IDs")
    identity = [namespace, int(getattr(args, "rollout_seed", 42)), group_id, sample_id, request_index]
    return int.from_bytes(hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()[:8], "big") % (2**31 - 1)


def validate_weight_version(meta):
    version = meta.get("weight_version")
    valid = type(version) is int and version >= 0
    if isinstance(version, str):
        valid = bool(version) and version.isascii() and version.isdigit()
    if not valid:
        raise ValueError("Checkpointed generation requires a non-negative numeric meta_info.weight_version")


def validate_checkpoint_sample(args, sample):
    """Reject dataset-level protocol overrides before external work starts."""
    if getattr(sample, "multimodal_inputs", None) or getattr(sample, "multimodal_train_inputs", None):
        raise ValueError("Async checkpoints currently support text-only samples")
    generator = getattr(sample, "generate_function_path", None)
    if generator and generator != getattr(args, "custom_generate_function_path", None):
        raise ValueError("Per-sample generator overrides are unsupported by async checkpoints")
    custom = getattr(sample, "custom_rm_path", None)
    if custom and custom != getattr(args, "custom_rm_path", None):
        raise ValueError("Per-sample reward overrides are unsupported by async checkpoints")
    metadata = getattr(sample, "metadata", None) or {}
    if not custom and not getattr(args, "custom_rm_path", None):
        reward_type = metadata.get("rm_type")
        if reward_type and reward_type != getattr(args, "rm_type", None):
            raise ValueError("Per-sample reward-type overrides are unsupported by async checkpoints")
