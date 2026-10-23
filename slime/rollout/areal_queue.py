"""AReaL's queue capacity in prompt groups, accounting for PPO minibatches."""

from __future__ import annotations


def get_areal_queue_capacity(args) -> tuple[int, str]:
    """Convert max(S - 1, 1) optimizer minibatches to queued prompt groups; S < 0 is unbounded."""
    prompts = args.rollout_batch_size
    samples = args.n_samples_per_prompt
    minibatches = args.num_steps_per_rollout
    global_batch_size = args.global_batch_size
    for name, value in (("rollout-batch-size", prompts), ("n-samples-per-prompt", samples)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"AReaL requires a positive integer --{name}.")
    responses = prompts * samples
    if minibatches is None:
        if not isinstance(global_batch_size, int) or global_batch_size <= 0:
            raise ValueError("AReaL requires positive --num-steps-per-rollout or --global-batch-size.")
        if responses % global_batch_size:
            raise ValueError("AReaL rollout responses must be divisible by --global-batch-size.")
        minibatches = responses // global_batch_size
    if not isinstance(minibatches, int) or minibatches <= 0:
        raise ValueError("AReaL requires a positive integer --num-steps-per-rollout.")
    if responses % minibatches:
        raise ValueError("AReaL rollout responses must be divisible by --num-steps-per-rollout.")
    train_batch_size = responses // minibatches
    if global_batch_size is not None and global_batch_size != train_batch_size:
        raise ValueError(
            f"AReaL --global-batch-size must equal rollout responses / minibatches "
            f"({responses} / {minibatches} = {train_batch_size})."
        )

    staleness = args.fully_async_max_staleness
    if not isinstance(staleness, int):
        raise ValueError("AReaL requires an integer --fully-async-max-staleness.")
    if staleness < 0:
        return 0, "unbounded (areal)"

    # Round to whole groups; zero capacity would make the queue unbounded.
    queued_optimizer_steps = max(staleness - 1, 1)
    groups = max(queued_optimizer_steps * prompts // minibatches, 1)
    description = (
        f"{groups} groups / {groups * samples} responses (areal: "
        f"max(S={staleness} - 1, 1)={queued_optimizer_steps} optimizer-minibatch equivalents "
        f"* {prompts} prompt groups // {minibatches} PPO minibatches per rollout; "
        f"{train_batch_size} responses per optimizer step)"
    )
    return groups, description
