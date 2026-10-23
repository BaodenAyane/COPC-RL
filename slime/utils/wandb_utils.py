# Modified for COPC.
import json
import logging
import os
import re
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _attach_existing_run_enabled() -> bool:
    return os.environ.get("WANDB_ATTACH_EXISTING_RUN") == "1"


def _standalone_eval_run_enabled() -> bool:
    enabled = os.environ.get("EVAL_STANDALONE_RUN") == "1"
    if enabled and _attach_existing_run_enabled():
        raise ValueError("EVAL_STANDALONE_RUN=1 conflicts with WANDB_ATTACH_EXISTING_RUN=1")
    return enabled


def _define_eval_metrics():
    prefix = os.environ.get("EVAL_METRIC_PREFIX", "eval")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", prefix):
        raise ValueError("EVAL_METRIC_PREFIX must be a simple metric namespace")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    if prefix != "eval":
        wandb.define_metric(f"{prefix}/*", step_metric="eval/step")


def _init_wandb_existing_run(args):
    """Append evaluation history without taking ownership of the training run."""
    if not getattr(args, "use_wandb", False):
        raise ValueError("WANDB_ATTACH_EXISTING_RUN=1 requires --use-wandb")
    required = ("wandb_run_id", "wandb_team", "wandb_project")
    if any(not getattr(args, key, None) for key in required):
        raise ValueError(
            "WANDB_ATTACH_EXISTING_RUN=1 requires explicit --wandb-run-id, "
            "--wandb-team, and --wandb-project"
        )
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", "online")
    if mode not in ("online", "shared"):
        raise ValueError("Attaching to an existing W&B run requires online mode")

    if args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)
    init_kwargs = {
        "id": args.wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "must",
        "settings": wandb.Settings(
            mode="shared", x_primary=False, x_update_finish_state=False, init_timeout=300
        ),
    }
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
    # Training retains ownership of this run's metadata and finish state.
    wandb.init(**init_kwargs)
    _define_eval_metrics()


def _init_wandb_standalone_eval(args, *, primary):
    """Initialize optional W&B logging for a standalone checkpoint evaluation."""
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", "online")
    if os.environ.get("EVAL_LOG_TO_WANDB") == "0" or mode == "disabled":
        args.use_wandb = False
        return
    if not getattr(args, "use_wandb", False):
        raise ValueError("EVAL_STANDALONE_RUN=1 requires --use-wandb")
    required = ("wandb_run_id", "wandb_run_name", "wandb_team", "wandb_project")
    if any(not getattr(args, key, None) for key in required):
        raise ValueError(
            "EVAL_STANDALONE_RUN=1 requires explicit --wandb-run-id, --wandb-run-name, "
            "--wandb-team, and --wandb-project"
        )
    source_run_id = os.environ.get("EVAL_SOURCE_RUN_ID") or None
    if source_run_id == args.wandb_run_id:
        raise ValueError("The standalone evaluation run ID must differ from the source training run ID")
    source_checkpoint_dir = os.environ.get("EVAL_SOURCE_CKPT_DIR")
    protocol_path = os.environ.get("EVAL_PROTOCOL_PATH")
    if not source_checkpoint_dir or not protocol_path:
        raise ValueError("Standalone evaluation requires EVAL_SOURCE_CKPT_DIR and EVAL_PROTOCOL_PATH")
    if mode not in ("online", "shared"):
        raise ValueError("Standalone evaluation W&B runs require online mode")

    init_kwargs = {
        "id": args.wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "allow" if primary else "must",
        "settings": wandb.Settings(
            mode="shared", x_primary=primary, x_update_finish_state=primary, init_timeout=300
        ),
    }
    if primary:
        with open(protocol_path, encoding="utf-8") as stream:
            protocol = json.load(stream)
        if not isinstance(protocol, dict):
            raise ValueError("EVAL_PROTOCOL_PATH must contain a JSON object")
        # This is an evaluation run: do not copy training arguments or credentials.
        init_kwargs.update(
            name=args.wandb_run_name,
            job_type="eval",
            config={
                "eval_protocol": protocol,
                "eval_source": {"run_id": source_run_id, "checkpoint_dir": source_checkpoint_dir},
            },
        )
        if getattr(args, "wandb_group", None):
            init_kwargs["group"] = args.wandb_group
    if getattr(args, "wandb_dir", None):
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
    if getattr(args, "wandb_key", None) is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)
    run = wandb.init(**init_kwargs)
    if run.id != args.wandb_run_id:
        raise ValueError("W&B did not initialize the requested standalone evaluation run ID")
    _define_eval_metrics()


def init_wandb_primary(args):
    if _standalone_eval_run_enabled():
        _init_wandb_standalone_eval(args, primary=True)
        return
    if _attach_existing_run_enabled():
        _init_wandb_existing_run(args)
        return

    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    if getattr(args, "wandb_run_name", None) is not None:
        group = args.wandb_group
        run_name = args.wandb_run_name
    elif args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if getattr(args, "wandb_run_id", None):
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True, init_timeout=300)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    if _standalone_eval_run_enabled():
        _init_wandb_standalone_eval(args, primary=False)
        return
    if _attach_existing_run_enabled():
        _init_wandb_existing_run(args)
        return

    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
            init_timeout=300,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    _define_eval_metrics()
    wandb.define_metric("perf/*", step_metric="rollout/step")
