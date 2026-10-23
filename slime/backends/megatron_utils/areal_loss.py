"""Megatron integration for AReaL."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch

from slime.utils.areal_utils import compute_areal_policy_loss
from slime.utils.misc import load_function
from slime.utils.ppo_utils import compute_approx_kl
from slime.utils.types import RolloutBatch

from .loss import get_log_probs_and_entropy, get_rollout_top_p_logprob_kwargs


def areal_policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute AReaL losses using cached proximal and behavior log-probabilities."""
    if not batch.get("log_probs"):
        raise ValueError("AReaL requires cached pre-update actor log_probs in the training batch.")
    if not batch.get("rollout_log_probs"):
        raise ValueError("AReaL requires rollout_log_probs from the behavior policy.")
    sample_count = len(batch["response_lengths"])
    for key in ("log_probs", "rollout_log_probs", "advantages"):
        if len(batch[key]) != sample_count:
            raise ValueError(f"AReaL requires one {key} tensor per response.")

    _, outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    if len(outputs["log_probs"]) != sample_count:
        raise ValueError("AReaL requires one current log-prob tensor per response.")
    for index, current in enumerate(outputs["log_probs"]):
        proximal = batch["log_probs"][index]
        behavior = batch["rollout_log_probs"][index]
        advantage = batch["advantages"][index]
        if not (current.shape == proximal.shape == behavior.shape == advantage.shape):
            raise ValueError("AReaL requires response-aligned current, proximal, behavior log-probs and advantages.")

    current = torch.cat(outputs["log_probs"], dim=0)
    proximal = torch.cat(batch["log_probs"], dim=0).detach()
    behavior = torch.cat(batch["rollout_log_probs"], dim=0).detach()
    advantages = torch.cat(batch["advantages"], dim=0)
    token_loss, token_metrics = compute_areal_policy_loss(
        current,
        proximal,
        behavior,
        advantages,
        args.eps_clip,
        args.eps_clip_high,
        max_is_weight=args.areal_is_weight_max,
    )

    # Keep the original tool masks and rollout denominators after rejection.
    reducer = sum_of_sample_mean
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        reducer = load_function(args.custom_pg_loss_reducer_function_path)(
            batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], args.calculate_per_token_loss
        )
    pg_loss = reducer(token_loss)
    entropy_loss = sum_of_sample_mean(torch.cat(outputs["entropy"], dim=0))
    loss = pg_loss - args.entropy_coef * entropy_loss

    kl_loss = None
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = torch.exp(current - proximal) if args.use_unbiased_kl else None
        kl_loss = sum_of_sample_mean(
            compute_approx_kl(
                current, ref_log_probs, kl_loss_type=args.kl_loss_type, importance_ratio=importance_ratio
            )
        )
        loss = loss + args.kl_loss_coef * kl_loss

    if current.numel() == 0:
        loss = loss + 0 * logits.sum()
    metrics = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "ppo_kl": sum_of_sample_mean(proximal - current).detach(),
        "train_rollout_logprob_abs_diff": sum_of_sample_mean((proximal - behavior).abs()).detach(),
    }
    metrics.update({key: sum_of_sample_mean(value).detach() for key, value in token_metrics.items()})
    if kl_loss is not None:
        metrics["kl_loss"] = kl_loss.detach()
    return loss, metrics
