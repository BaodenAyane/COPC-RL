"""A committed manifest binds native model checkpoints to complete pipeline state."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import time
import uuid

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)
PHASE = "after_weight_sync_before_next_batch"
CONTRACT_KEYS = (
    "advantage_estimator", "use_critic", "train_backend", "num_rollout",
    "rollout_batch_size", "n_samples_per_prompt", "global_batch_size",
    "num_steps_per_rollout", "update_weights_interval", "num_critic_only_steps",
    "fully_async_max_staleness", "seed", "rollout_seed", "rollout_shuffle",
    "rollout_temperature", "rollout_top_p", "rollout_top_k", "rollout_max_prompt_len",
    "rollout_max_response_len", "rollout_max_context_len", "rollout_stop", "rollout_stop_token_ids",
    "rollout_function_path", "custom_generate_function_path", "custom_rm_path", "rm_type",
    "reward_key", "input_key", "label_key", "metadata_key", "apply_chat_template",
    "apply_chat_template_kwargs", "lr", "lr_decay_style", "lr_warmup_iters", "min_lr",
    "optimizer", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "clip_grad", "ckpt_format",
    "gamma", "lambd", "eps_clip", "eps_clip_high", "copc_rho_clip", "copc_rho_clip_high",
    "value_clip", "normalize_advantages", "kl_coef", "kl_loss_coef", "entropy_coef",
    "use_rollout_logprobs", "balance_data", "use_dynamic_batch_size", "max_tokens_per_gpu",
    "actor_num_nodes", "actor_num_gpus_per_node", "rollout_num_gpus", "rollout_num_gpus_per_engine",
    "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
    "sglang_server_concurrency", "sglang_enable_deterministic_inference", "group_rm",
    "bf16", "fp16", "async_save", "async_checkpoint_timeout", "async_checkpoint_interval", "save_interval",
    "sglang_dp_size", "data_source_path", "tool_key", "loss_type", "loss_reduction",
    "lr_decay_iters", "lr_warmup_fraction", "lr_warmup_init", "lr_wsd_decay_iters",
    "start_weight_decay", "end_weight_decay", "weight_decay_incr_style",
    "sequence_parallel", "attention_dropout", "hidden_dropout", "rotary_base",
    "use_distributed_optimizer", "offload_train", "offload_rollout",
    "get_mismatch_metrics", "use_tis", "use_tis_rejection_sampling",
    "custom_advantage_function_path", "rollout_sample_hook_path", "rollout_data_postprocess_path",
    "grpo_std_normalization", "rewards_normalization", "compute_advantages_and_returns",
    "dynamic_sampling_filter_path", "calculate_per_token_loss",
    "vtrace_rho_clip", "vtrace_c_clip", "vtrace_pg_rho_clip", "areal_is_weight_max",
    "clip_only", "dual_clip", "use_rollout_entropy", "reset_optimizer_states",
    "tis_clip", "tis_clip_low", "use_opsm", "opsm_delta", "over_sampling_batch_size",
    "num_epoch", "micro_batch_size", "recompute_granularity", "recompute_method", "recompute_num_layers",
)
SOURCE_FILES = (
    "train_async.py", "slime/utils/async_checkpoint.py", "slime/utils/async_checkpoint_policy.py", "slime/utils/types.py",
    "slime/rollout/checkpoint_barrier.py", "slime/rollout/fully_async_rollout.py",
    "slime/rollout/data_source.py", "slime/rollout/sglang_rollout.py",
    "slime/ray/rollout.py", "slime/ray/placement_group.py", "slime/ray/actor_group.py",
    "slime/backends/megatron_utils/actor.py", "slime/backends/megatron_utils/model.py",
    "slime/backends/megatron_utils/loss.py", "slime/utils/ppo_utils.py",
    "slime/utils/vtrace_utils.py", "slime/utils/retool_protocol.py",
    "slime/utils/areal_utils.py", "slime/rollout/areal_queue.py",
    "slime/backends/megatron_utils/areal_loss.py",
    "examples/retool/generate_with_retool.py", "examples/retool/tool_sandbox.py",
    "slime/utils/data.py", "slime/utils/dp_schedule.py", "slime/utils/arguments.py",
    "slime/backends/megatron_utils/checkpoint.py", "slime/backends/megatron_utils/async_checkpoint.py",
    "slime/backends/megatron_utils/arguments.py",
    "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py",
    "slime/backends/sglang_utils/sglang_engine.py",
    "slime/rollout/rm_hub/math_dapo_utils.py", "slime/rollout/checkpoint_protocol.py",
    "examples/search-r1/generate_with_search.py", "examples/search-r1/retrieval_client.py",
    "examples/search-r1/search_protocol.py", "examples/search-r1/search_reward.py",
    "examples/search-r1/reference_reward.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = None
    try:
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}:
            raise
        logger.warning("Directory fsync is unsupported for %s: %s", path, error)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_mode(args, *, validate_roles=True):
    if not getattr(args, "async_checkpoint", False):
        if getattr(args, "async_resume", None):
            raise ValueError("--async-resume requires --async-checkpoint")
        return
    expected = {
        "train_backend": "megatron",
        "rollout_function_path": "slime.rollout.fully_async_rollout.generate_rollout_fully_async",
    }
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            raise ValueError("Async checkpoints require %s=%r" % (key, value))
    if getattr(args, "ckpt_format", "torch_dist") != "torch_dist":
        raise ValueError("Async checkpoints require the Megatron torch_dist checkpoint format")
    if getattr(args, "save_retain_interval", None) is not None:
        raise ValueError("Complete checkpoints do not support native --save-retain-interval")
    if getattr(args, "advantage_estimator", None) not in {
        "copc", "ppo", "vtrace", "grpo", "gspo", "cispo", "areal",
        "reinforce_plus_plus", "reinforce_plus_plus_baseline",
    }:
        raise ValueError("Async checkpoints require a supported built-in advantage estimator")
    if type(getattr(args, "use_critic", None)) is not bool:
        raise ValueError("Async checkpoints require an explicit actor/critic role configuration")
    sync_interval = getattr(args, "update_weights_interval", None)
    if type(sync_interval) is not int or sync_interval <= 0:
        raise ValueError("Async checkpoints require a positive update_weights_interval")
    steps = getattr(args, "num_steps_per_rollout", None)
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("num_steps_per_rollout must be positive when explicitly set")
    warmup = getattr(args, "num_critic_only_steps", 0)
    if type(warmup) is not int or warmup < 0 or (warmup and not args.use_critic):
        raise ValueError("Critic-only warmup requires a critic and a non-negative step count")
    describe_checkpoint_protocol(args)
    for key in ("colocate", "release_train", "partial_rollout", "debug_train_only", "debug_rollout_only",
                "use_fault_tolerance", "rollout_external", "group_rm", "use_stateless_adam",
                "use_routing_replay", "use_rollout_routing_replay", "reset_optimizer_states"):
        if getattr(args, key, False):
            raise ValueError("Async checkpoints do not support %s" % key)
    for key in ("custom_reward_post_process_path",
                "custom_convert_samples_to_train_data_path", "buffer_filter_path", "rollout_sample_hook_path",
                "rollout_data_postprocess_path", "custom_advantage_function_path",
                "custom_loss_function_path", "custom_tis_function_path", "custom_pg_loss_reducer_function_path",
                "custom_megatron_init_path", "custom_megatron_before_log_prob_hook_path",
                "custom_megatron_before_train_step_hook_path", "custom_update_weight_post_write_path",
                "custom_model_provider_path"):
        if getattr(args, key, None):
            raise ValueError("Hook state is not supported by async checkpoints: " + key)
    for key, default in (("update_weight_transport", "nccl"), ("update_weight_mode", "full"),
                         ("rollout_data_transport", "object-store")):
        if getattr(args, key, default) != default:
            raise ValueError("Async checkpoints require %s=%s" % (key, default))
    if getattr(args, "data_source_path", "slime.rollout.data_source.RolloutDataSourceWithBuffer") != "slime.rollout.data_source.RolloutDataSourceWithBuffer":
        raise ValueError("Async checkpoints require the standard buffered datasource")
    if any(getattr(args, key, False) for key in ("use_kl_loss", "use_opd", "keep_old_actor")) or any(
        getattr(args, key, 0) not in (None, 0) for key in ("kl_coef", "kl_loss_coef", "ref_update_interval")
    ):
        raise ValueError("Async checkpoints do not yet save reference, old-actor, or teacher policy state")
    if getattr(args, "sglang_dp_size", 1) not in (None, 1):
        raise ValueError("Async checkpoints require sglang_dp_size=1")
    if getattr(args, "eval_interval", None) is not None:
        raise ValueError("Async checkpoints require separate evaluation jobs")
    timeout = getattr(args, "async_checkpoint_timeout", 300.0)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("async_checkpoint_timeout must be finite and positive")
    interval = getattr(args, "async_checkpoint_interval", None) or getattr(args, "save_interval", None)
    if not isinstance(interval, int) or interval <= 0:
        raise ValueError("A positive async checkpoint/save interval is required")
    if not getattr(args, "save", None):
        raise ValueError("--save is required for async checkpoints")
    if not getattr(args, "rollout_global_dataset", True):
        raise ValueError("Async checkpoint requires a global dataset")
    if type(getattr(args, "num_rollout", None)) is not int or args.num_rollout <= 0:
        raise ValueError("Async checkpoints require an explicit positive --num-rollout")
    if args.num_rollout % sync_interval:
        raise ValueError("The final rollout must coincide with a natural weight-sync boundary")
    if not getattr(args, "async_resume", None) and getattr(args, "start_rollout_id", None) not in (None, 0):
        raise ValueError("A new async checkpoint run must start at rollout 0")
    if validate_roles and getattr(args, "megatron_config_path", None):
        # Validate explicit role overrides before allocating GPUs.
        import yaml
        roles = yaml.safe_load(Path(args.megatron_config_path).read_text()) or {}
        entries = roles.get("megatron")
        if not isinstance(entries, list):
            raise ValueError("Megatron configuration requires a role list")
        seen = set()
        role_saves = {"actor": args.save, "critic": args.save}
        for entry in entries:
            role = entry.get("role")
            if role not in ("actor", "critic") or role in seen:
                raise ValueError("Async checkpoint requires unique actor/critic role entries")
            if role == "critic" and not args.use_critic:
                raise ValueError("A critic role override cannot be used in actor-only training")
            seen.add(role)
            effective = copy.copy(args)
            for key, value in (entry.get("overrides") or entry.get("args") or {}).items():
                if key in ("num_nodes", "num_gpus_per_node"):
                    continue
                original = getattr(effective, key, None)
                if original is not None and isinstance(value, str) and isinstance(original, (int, float)):
                    try:
                        value = type(original)(value)
                    except (TypeError, ValueError):
                        pass
                setattr(effective, key, value)
            if not effective.async_checkpoint:
                raise ValueError("Role overrides cannot disable async checkpointing")
            if bool(getattr(effective, "async_save", False)) != bool(getattr(args, "async_save", False)):
                raise ValueError("Role overrides cannot change the pipeline async_save setting")
            for key in ("use_critic", "num_rollout", "update_weights_interval", "num_critic_only_steps"):
                if getattr(effective, key, None) != getattr(args, key, None):
                    raise ValueError("Role override changes a pipeline coordination setting: " + key)
            validate_mode(effective, validate_roles=False)
            role_saves[role] = effective.save
        if args.use_critic:
            _validate_distinct_role_saves(role_saves)


def _validate_distinct_role_saves(role_saves):
    if len({str(Path(value).resolve()) for value in role_saves.values()}) != len(role_saves):
        raise ValueError("Actor and critic checkpoint save directories must be distinct")


def describe_checkpoint_protocol(args):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "rollout/checkpoint_protocol.py"
    spec = importlib.util.spec_from_file_location("_slime_async_checkpoint_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.describe_checkpoint_protocol(args)


def manifest_roles(manifest):
    return tuple(manifest.get("roles", ("actor", "critic")))


def native_checkpoint_iteration(manifest):
    return manifest.get("checkpoint_iteration", manifest["iteration"])


def runtime_contract(args):
    """Fingerprint semantic inputs, excluding output paths and fresh/load switches."""
    config = {key: getattr(args, key, None) for key in CONTRACT_KEYS}
    config["checkpoint_protocol"] = describe_checkpoint_protocol(args)
    config["tool_environment"] = {key: os.environ.get(key) for key in (
        "MAX_TURNS", "MAX_TOOL_CALLS", "TOOL_TIMEOUT", "PYTHON_EXEC_TIMEOUT", "RETOOL_REWARD_MODE",
    )}
    role_path = getattr(args, "megatron_config_path", None)
    if role_path:
        import yaml
        roles = yaml.safe_load(Path(role_path).read_text())
        ignored = {"load", "save", "ckpt_step", "finetune", "no_load_optim", "no_load_rng",
                   "use_checkpoint_opt_param_scheduler", "override_opt_param_scheduler", "no_save_optim", "no_save_rng"}
        config["roles"] = [
            {"role": role["role"], "overrides": {
                key: val for key, val in (role.get("overrides") or role.get("args") or {}).items() if key not in ignored
            }} for role in roles.get("megatron", [])
        ]
    root = Path(__file__).resolve().parents[2]
    protocol_sources = config["checkpoint_protocol"].get("source_files", ())
    source_files = tuple(dict.fromkeys(SOURCE_FILES + tuple(protocol_sources)))
    for relative in source_files:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Checkpoint protocol source paths must be repository-relative")
    inputs = {relative: sha256_file(root / relative) for relative in source_files}
    from importlib import metadata
    packages = {}
    for package in ("torch", "ray", "sglang", "transformers", "megatron-core", "math-verify"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    config["runtime_packages"] = packages
    data = getattr(args, "prompt_data", None)
    if data:
        inputs["dataset"] = sha256_file(data)
    model = getattr(args, "hf_checkpoint", None)
    if model:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
            path = Path(model) / name
            if path.is_file():
                inputs["tokenizer/" + name] = sha256_file(path)
    return {"config": config, "input_sha256": inputs}


def checkpoint_root(args):
    return Path(getattr(args, "async_checkpoint_dir", None) or (Path(args.save).parent / "resume"))


def _model_inventory(directory):
    directory = Path(directory).resolve()
    for name in (".metadata", "metadata.json", "common.pt"):
        if not (directory / name).is_file():
            raise ValueError("Incomplete model checkpoint: " + str(directory / name))
    shards = list(directory.rglob("*.distcp"))
    if not shards:
        raise ValueError("No distributed model shards in " + str(directory))
    files = {str(p.relative_to(directory)): p.stat().st_size for p in directory.rglob("*") if p.is_file()}
    if any(size <= 0 for size in files.values()):
        raise ValueError("Empty checkpoint component in " + str(directory))
    return {"directory": str(directory), "files": files}


def _validate_inventory(inventory):
    root = Path(inventory["directory"])
    for name, size in inventory["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid checkpoint component path")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("Missing or changed model checkpoint component: " + str(path))
    _model_inventory(root)


def begin_snapshot(args, iteration):
    root = checkpoint_root(args)
    root.mkdir(parents=True, exist_ok=True)
    final = root / ("step_%07d" % iteration)
    if final.exists():
        raise ValueError("Refusing to overwrite a committed async checkpoint: " + str(final))
    staging = root / (".step_%07d.incomplete-%s" % (iteration, uuid.uuid4().hex))
    staging.mkdir()
    return staging


def commit_snapshot(args, iteration, staging, actor_state, critic_state, driver_state, *, checkpoint_iteration=None):
    staging = Path(staging)
    payloads = {}
    for name in ("rollout.pt", "driver.pt"):
        path = staging / name
        if not path.is_file():
            raise ValueError("Missing async checkpoint payload: " + str(path))
        payloads[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if checkpoint_iteration is None:
        checkpoint_iteration = iteration
    if type(checkpoint_iteration) is not int or checkpoint_iteration < 0:
        raise ValueError("Invalid native checkpoint iteration")
    if bool(getattr(args, "use_critic", critic_state is not None)) != (critic_state is not None):
        raise ValueError("The saved native roles do not match use_critic")
    states = {"actor": actor_state}
    if critic_state is not None:
        states["critic"] = critic_state
    _validate_distinct_role_saves({role: state["save"] for role, state in states.items()})
    inventories = {role: _model_inventory(Path(state["save"]) / ("iter_%07d" % checkpoint_iteration))
                   for role, state in states.items()}
    protocol = args._async_runtime_contract.get("config", {}).get("checkpoint_protocol")
    if protocol is None:
        protocol = describe_checkpoint_protocol(args)
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": PHASE, "iteration": iteration,
        "checkpoint_iteration": checkpoint_iteration, "roles": list(states),
        "next_rollout_id": iteration + 1, "weight_version": actor_state["weight_version"],
        **inventories, "payloads": payloads,
        "runtime_contract": args._async_runtime_contract,
        "next_batch_rollout_id": driver_state["next_batch_rollout_id"],
        "generation_boundary": protocol["generation_boundary"],
        "sampling_protocol": protocol["sampling_protocol"],
        "generator_protocol": protocol,
    }
    atomic_json(staging / "manifest.json", manifest)
    destination = staging.parent / ("step_%07d" % iteration)
    if destination.exists():
        raise ValueError("Committed async checkpoint already exists: " + str(destination))
    os.rename(str(staging), str(destination))
    _fsync_directory(destination.parent)
    atomic_json(destination.parent / "latest.json", {"manifest": destination.name + "/manifest.json"})
    return str(destination / "manifest.json")


def _read_manifest(path, *, follow_pointer=True):
    if path.parent.name.startswith(".step_") and ".incomplete-" in path.parent.name:
        raise ValueError("An incomplete async checkpoint has not been committed")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Invalid async checkpoint manifest")
    if "manifest" in data and "schema_version" not in data:
        if not follow_pointer or not isinstance(data["manifest"], str):
            raise ValueError("Invalid async checkpoint pointer")
        relative = Path(data["manifest"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid async checkpoint pointer")
        return _read_manifest(path.parent / relative, follow_pointer=False)
    if (type(data.get("schema_version")) is not int
            or data["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS or data.get("phase") != PHASE):
        raise ValueError("Unsupported async checkpoint schema or phase")
    required = {"iteration", "next_rollout_id", "weight_version", "next_batch_rollout_id",
                "actor", "payloads", "runtime_contract"}
    if data.get("schema_version") == 1:
        required.add("critic")
    else:
        required.update(("roles", "checkpoint_iteration", "generator_protocol"))
    if not required.issubset(data):
        raise ValueError("Incomplete async checkpoint manifest")
    if (not isinstance(data["runtime_contract"], dict)
            or not {"config", "input_sha256"}.issubset(data["runtime_contract"])
            or not isinstance(data["runtime_contract"]["config"], dict)
            or not isinstance(data["runtime_contract"]["input_sha256"], dict)):
        raise ValueError("Invalid saved async checkpoint contract")
    if type(data["iteration"]) is not int or data["iteration"] < 0:
        raise ValueError("Invalid async checkpoint iteration")
    if re.fullmatch(r"step_\d{7,}", path.parent.name) and path.parent.name != "step_%07d" % data["iteration"]:
        raise ValueError("Async checkpoint directory and manifest iteration mismatch")
    if type(data["next_rollout_id"]) is not int or data["next_rollout_id"] != data["iteration"] + 1:
        raise ValueError("Inconsistent async checkpoint step counters")
    if type(data["weight_version"]) is not int or data["weight_version"] < 1:
        raise ValueError("Invalid saved policy version")
    if data.get("next_batch_rollout_id") not in (None, data["next_rollout_id"]):
        raise ValueError("Prefetched batch does not match the next training step")
    if "roles" in data and not isinstance(data["roles"], list):
        raise ValueError("Invalid saved native role declaration")
    roles = manifest_roles(data)
    if roles not in (("actor",), ("actor", "critic")):
        raise ValueError("Invalid saved native roles")
    if data.get("schema_version") == 1 and roles != ("actor", "critic"):
        raise ValueError("Legacy async checkpoints must contain actor and critic")
    if any((role in data) != (role in roles) for role in ("actor", "critic")):
        raise ValueError("Saved native role inventories do not match the role declaration")
    saved_use_critic = data["runtime_contract"]["config"].get("use_critic")
    if isinstance(saved_use_critic, bool) and saved_use_critic != ("critic" in roles):
        raise ValueError("Saved native roles do not match the runtime contract")
    native_iteration = native_checkpoint_iteration(data)
    if type(native_iteration) is not int or native_iteration < 0:
        raise ValueError("Invalid native checkpoint iteration")
    if data.get("schema_version") == 1 and native_iteration != data["iteration"]:
        raise ValueError("Legacy native and rollout iterations must match")
    for role in roles:
        if Path(data[role]["directory"]).name != "iter_%07d" % native_iteration:
            raise ValueError("Actor/critic iteration mismatch")
        _validate_inventory(data[role])
    _validate_distinct_role_saves({role: str(Path(data[role]["directory"]).parent) for role in roles})
    if data.get("schema_version") != 1:
        protocol = data["generator_protocol"]
        if (not isinstance(protocol, dict)
                or not {"name", "version", "generation_boundary", "sampling_protocol"}.issubset(protocol)):
            raise ValueError("Invalid saved generator checkpoint protocol")
        saved_protocol = data["runtime_contract"]["config"].get("checkpoint_protocol")
        if saved_protocol is not None and saved_protocol != protocol:
            raise ValueError("Saved generator protocol differs from the runtime contract")
        for key in ("generation_boundary", "sampling_protocol"):
            if data.get(key) != protocol[key]:
                raise ValueError("Inconsistent saved generator protocol metadata")
    for name in ("rollout.pt", "driver.pt"):
        meta = data["payloads"][name]
        payload = path.parent / name
        if payload.stat().st_size != meta["bytes"] or sha256_file(payload) != meta["sha256"]:
            raise ValueError("Corrupt async checkpoint payload: " + str(payload))
    return path, data


def resolve_manifest(path):
    """Honor pinned commits; otherwise find the newest valid commit, even if latest.json is stale."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if (path / "manifest.json").is_file() or re.fullmatch(r"step_\d{7,}", path.name):
            return _read_manifest(path / "manifest.json")
        root = path
    elif path.name == "latest.json":
        root = path.parent
    else:
        return _read_manifest(path)

    if not root.is_dir():
        raise FileNotFoundError("No committed async checkpoint in " + str(root))
    candidates = sorted(
        (entry for entry in root.iterdir() if re.fullmatch(r"step_\d{7,}", entry.name)),
        key=lambda entry: int(entry.name[5:]), reverse=True,
    )
    if not candidates:
        if (root / "latest.json").exists():
            raise ValueError("Async checkpoint pointer exists without any published step in " + str(root))
        raise FileNotFoundError("No committed async checkpoint in " + str(root))
    failures = []
    for candidate in candidates:
        try:
            return _read_manifest(candidate / "manifest.json", follow_pointer=False)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failures.append(candidate.name)
            logger.warning("Skipping invalid committed async checkpoint %s: %s", candidate, error)
    raise ValueError("No valid committed async checkpoint in %s (invalid: %s)" % (root, ", ".join(failures)))


def prepare_resume(args):
    validate_mode(args)
    if not getattr(args, "async_checkpoint", False):
        return None
    args._async_runtime_contract = runtime_contract(args)
    if not getattr(args, "async_resume", None):
        # SFT/critic initialization iterations are not pipeline training steps.
        args.start_rollout_id = 0
        return None
    path, manifest = resolve_manifest(args.async_resume)
    expected_roles = ("actor", "critic") if args.use_critic else ("actor",)
    if manifest_roles(manifest) != expected_roles:
        raise ValueError("Resume native roles do not match the configured actor/critic roles")
    if manifest["runtime_contract"] != args._async_runtime_contract:
        before, after = manifest["runtime_contract"], args._async_runtime_contract
        changed = [key for key in ("config", "input_sha256") if before.get(key) != after.get(key)]
        raise ValueError("Resume contract changed (%s); refusing to mix experiments" % ", ".join(changed))
    if manifest["next_rollout_id"] > args.num_rollout:
        raise ValueError("This async checkpoint exceeds the requested training budget")
    completed = manifest["next_rollout_id"] == args.num_rollout
    if completed and manifest["next_batch_rollout_id"] is not None:
        raise ValueError("A completed run must not have a prefetched next batch saved")
    if not completed and manifest["next_batch_rollout_id"] != manifest["next_rollout_id"]:
        raise ValueError("An unfinished run must have its prefetched next batch saved")
    args._async_resume_manifest = manifest
    args._async_resume_directory = str(path.parent)
    args.start_rollout_id = manifest["next_rollout_id"]
    if completed:
        logger.info("Async checkpoint already completed the training budget: updates=%s checkpoint=%s",
                    args.num_rollout, path)
        return manifest
    args.update_weight_start_version = manifest["weight_version"]
    apply_role_resume(args, "actor")
    return manifest


def apply_role_resume(args, role):
    manifest = getattr(args, "_async_resume_manifest", None)
    if manifest is None:
        return
    if role not in manifest_roles(manifest):
        raise ValueError("The async checkpoint has no native state for role: " + str(role))
    args.load = str(Path(manifest[role]["directory"]).parent)
    args.ckpt_step = native_checkpoint_iteration(manifest)
    args.finetune = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.use_checkpoint_opt_param_scheduler = True
    args.override_opt_param_scheduler = False
    args.update_weight_start_version = manifest["weight_version"]


def driver_rng_state():
    state = {"python": random.getstate()}
    import numpy as np
    state["numpy"] = np.random.get_state()
    return state


def restore_driver_rng(state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _log_checkpoint_metrics(args, rollout_id, metrics, *, payload_dir):
    if not (getattr(args, "use_wandb", False) or getattr(args, "use_tensorboard", False)):
        return
    try:
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step
        values = dict(metrics)
        for name in ("rollout", "driver"):
            values["perf/checkpoint_" + name + "_bytes"] = (Path(payload_dir) / (name + ".pt")).stat().st_size
        values["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, values, step_key="rollout/step")
    except Exception as error:
        logger.warning("Checkpoint telemetry unavailable at rollout %s (%s); save state is unchanged",
                       rollout_id, type(error).__name__)


class PendingPipelineCheckpoint:

    def __init__(self, args, rollout_id, checkpoint_iteration, staging, actor_model,
                 critic_model, actor_state, critic_state, driver_state, started_at):
        self.args = copy.copy(args)
        self.args._async_runtime_contract = copy.deepcopy(args._async_runtime_contract)
        self.rollout_id = rollout_id
        self.checkpoint_iteration = checkpoint_iteration
        self.staging = Path(staging)
        self.actor_state = copy.deepcopy(actor_state)
        self.critic_state = copy.deepcopy(critic_state)
        self.driver_state = copy.deepcopy(driver_state)
        self.started_at = started_at
        self._models = [("actor", actor_model)]
        if critic_model is not None:
            self._models.append(("critic", critic_model))
        self._completed_roles = set()
        self._manifest_path = None
        self._failure = None

    def finalize(self, blocking=False):
        if self._failure is not None:
            raise RuntimeError("The pending async checkpoint previously failed") from self._failure
        if self._manifest_path is not None:
            return self._manifest_path
        try:
            for role, model in self._models:
                if role in self._completed_roles:
                    continue
                complete = model.finalize_checkpoint_save(self.checkpoint_iteration, blocking=blocking)
                if type(complete) is not bool:
                    raise RuntimeError("Native checkpoint finalization did not return a completion acknowledgement")
                if complete:
                    self._completed_roles.add(role)
                elif blocking:
                    raise RuntimeError("A blocking native checkpoint finalization did not finish")
            if len(self._completed_roles) != len(self._models):
                return None
            self._manifest_path = commit_snapshot(
                self.args, self.rollout_id, self.staging, self.actor_state, self.critic_state,
                self.driver_state, checkpoint_iteration=self.checkpoint_iteration,
            )
            commit_latency = time.monotonic() - self.started_at
            logger.info("Committed background checkpoint: iteration=%s weight_version=%s "
                        "commit_latency_seconds=%.3f (includes finalization polling) manifest=%s",
                        self.rollout_id, self.actor_state["weight_version"],
                        commit_latency, self._manifest_path)
            _log_checkpoint_metrics(self.args, self.rollout_id,
                                    {"perf/checkpoint_commit_latency": commit_latency},
                                    payload_dir=Path(self._manifest_path).parent)
            return self._manifest_path
        except BaseException as exc:
            self._failure = exc
            raise


def save_pipeline_checkpoint(args, rollout_id, actor_model, critic_model, rollout_manager, has_next_batch,
                             *, checkpoint_iteration=None):
    """Release the barrier after pipeline persistence and native staging; commit after all writers finish."""
    import ray
    import torch
    if bool(getattr(args, "use_critic", critic_model is not None)) != (critic_model is not None):
        raise ValueError("Live native roles do not match use_critic")
    if checkpoint_iteration is None:
        checkpoint_iteration = rollout_id
    background = bool(getattr(args, "async_save", False))
    started_at = time.monotonic()
    staging = begin_snapshot(args, rollout_id)
    next_id = rollout_id + 1 if has_next_batch else None
    save_succeeded = False
    commit_latency = None
    payload_dir = staging
    try:
        frozen = ray.get(rollout_manager.freeze_checkpoint.remote(str(staging / "rollout.pt"), next_id))
        actor_state = copy.deepcopy(actor_model.get_checkpoint_state())
        critic_state = copy.deepcopy(critic_model.get_checkpoint_state()) if critic_model is not None else None
        if critic_state is not None:
            _validate_distinct_role_saves({"actor": actor_state["save"], "critic": critic_state["save"]})
        if actor_state["weight_version"] != frozen["weight_version"]:
            raise ValueError("Actor and frozen rollout policy versions do not match")
        save_kwargs = {"force_sync": not background, "full_state": True}
        if background:
            save_kwargs["defer_finalize"] = True
        actor_model.save_model(checkpoint_iteration, **save_kwargs)
        if critic_model is not None:
            critic_model.save_model(checkpoint_iteration, **save_kwargs)
        driver = {"rng": driver_rng_state(), "next_batch_rollout_id": next_id,
                  "checkpoint_iteration": checkpoint_iteration}
        with (staging / "driver.pt").open("wb") as stream:
            torch.save(driver, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if background:
            pending = PendingPipelineCheckpoint(
                args, rollout_id, checkpoint_iteration, staging, actor_model, critic_model,
                actor_state, critic_state, driver, started_at,
            )
            logger.info("Staged background checkpoint: iteration=%s weight_version=%s "
                        "staging_seconds=%.3f; native persistence continues in background",
                        rollout_id, actor_state["weight_version"], time.monotonic() - started_at)
            save_succeeded = True
            return pending
        manifest_path = commit_snapshot(args, rollout_id, staging, actor_state, critic_state, driver,
                                        checkpoint_iteration=checkpoint_iteration)
        commit_latency = time.monotonic() - started_at
        payload_dir = Path(manifest_path).parent
        logger.info("Committed async checkpoint: iteration=%s weight_version=%s manifest=%s",
                    rollout_id, actor_state["weight_version"], manifest_path)
        save_succeeded = True
        return manifest_path
    finally:
        # A delayed finalizer must never release a newer checkpoint's barrier.
        ray.get(rollout_manager.release_checkpoint.remote())
        if save_succeeded:
            pause_time = time.monotonic() - started_at
            logger.info("Released checkpoint barrier: iteration=%s pause_seconds=%.3f", rollout_id, pause_time)
            metrics = {"perf/checkpoint_pause_time": pause_time}
            if commit_latency is not None:
                metrics["perf/checkpoint_commit_latency"] = commit_latency
            _log_checkpoint_metrics(args, rollout_id, metrics, payload_dir=payload_dir)


def restore_pipeline_checkpoint(args, actor_model, rollout_manager):
    import ray
    import torch
    manifest = args._async_resume_manifest
    directory = Path(args._async_resume_directory)
    # Install the saved policy; loading it is not a new optimizer/policy update.
    actor_model.update_weights(restore_version=manifest["weight_version"])
    batch = ray.get(rollout_manager.restore_checkpoint.remote(
        str(directory / "rollout.pt"), manifest["next_rollout_id"], manifest["weight_version"],
    ))
    driver = torch.load(directory / "driver.pt", map_location="cpu", weights_only=False)
    if driver["next_batch_rollout_id"] != manifest["next_rollout_id"]:
        raise ValueError("Driver and rollout checkpoint disagree on the next batch")
    if driver.get("checkpoint_iteration", native_checkpoint_iteration(manifest)) != native_checkpoint_iteration(manifest):
        raise ValueError("Driver and manifest disagree on the native checkpoint iteration")
    restore_driver_rng(driver["rng"])
    ray.get(rollout_manager.release_checkpoint.remote())
    logger.info("Restored async pipeline: next_rollout_id=%s weight_version=%s checkpoint=%s",
                manifest["next_rollout_id"], manifest["weight_version"], directory)
    return batch


def validate_sample_ledger(worker_state, next_batch, consumed_indices):
    """Every dispatched sample must have exactly one owner at the saved cut."""
    def sample_id(sample):
        value = sample["index"] if isinstance(sample, dict) else sample.index
        if type(value) is not int or value < 0:
            raise ValueError("Async sample IDs must be non-negative integers")
        return value

    consumed = list(consumed_indices)
    if any(type(index) is not int or index < 0 for index in consumed):
        raise ValueError("Invalid consumed sample ID")
    filtered = list(worker_state.get("filtered_indices", []))
    if any(type(index) is not int or index < 0 for index in filtered):
        raise ValueError("Invalid filtered sample ID")
    partitions = {"consumed": consumed, "filtered": filtered}
    partitions["prefetched"] = [] if next_batch is None else [sample_id(s) for s in next_batch["samples"]]
    partitions["ready"] = [sample_id(s) for _, group in worker_state["ready"] for s in group]
    partitions["pending"] = [sample_id(s) for _, group in worker_state["pending_completions"] for s in group]
    partitions["active"] = [sample_id(s) for item in worker_state["active"] for s in item["samples"]]
    partitions["retry"] = [sample_id(s) for group in worker_state["data_source"].get("buffer", []) for s in group]
    seen = set()
    for label, indices in partitions.items():
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError("Duplicate sample ownership in " + label)
        seen.update(indices)
    issued = worker_state["data_source"]["sample_index"]
    if len(seen) != issued or (seen and (min(seen) != 0 or max(seen) != issued - 1)):
        raise ValueError("Dispatched samples are missing from the committed pipeline ledger")
    return {label: len(indices) for label, indices in partitions.items()}
