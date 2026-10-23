
FULLY_ASYNC_ROLLOUT = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"


def resolve_checkpoint_mode(args):
    """Require pipeline snapshots for fully asynchronous runs with persistence configured."""
    fully_async = getattr(args, "rollout_function_path", None) == FULLY_ASYNC_ROLLOUT
    configured = any(getattr(args, key, None) is not None for key in (
        "save", "save_interval", "async_checkpoint_dir", "async_checkpoint_interval", "async_resume",
    ))
    requested = getattr(args, "async_checkpoint", None)
    if fully_async and configured and requested is False:
        raise ValueError(
            "Fully-async training saves require complete pipeline checkpoints; "
            "remove the async_checkpoint=False override. Run without --save and "
            "checkpoint options to disable persistence."
        )
    if requested is None:
        args.async_checkpoint = bool(fully_async and configured)
    return bool(args.async_checkpoint)
