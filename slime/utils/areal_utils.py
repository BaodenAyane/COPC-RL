"""AReaL configuration and policy objective."""

from __future__ import annotations

import math

import torch


def validate_areal_config(args) -> None:
    if args.advantage_estimator != "areal":
        return
    incompatible_flags = (
        ("use_rollout_logprobs", "--use-rollout-logprobs"),
        ("keep_old_actor", "--keep-old-actor"),
        ("clip_only", "--clip-only"),
        ("use_tis", "--use-tis"),
        ("get_mismatch_metrics", "--get-mismatch-metrics"),
        ("use_opsm", "--use-opsm"),
        ("custom_tis_function_path", "--custom-tis-function-path"),
        ("custom_advantage_function_path", "--custom-advantage-function-path"),
    )
    for attribute, flag in incompatible_flags:
        if getattr(args, attribute, None):
            raise ValueError(
                f"AReaL does not support {flag}: it uses cached pre-update log-probs, "
                "ordinary GAE, and its own decoupled PPO loss and behavior-weight filtering."
            )
    if getattr(args, "loss_type", "policy_loss") != "policy_loss":
        raise ValueError("--advantage-estimator areal requires --loss-type policy_loss.")
    maximum = args.areal_is_weight_max
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("--areal-is-weight-max must be finite and positive.")
    high = args.eps_clip if args.eps_clip_high is None else args.eps_clip_high
    if not math.isfinite(args.eps_clip) or not 0 <= args.eps_clip < 1:
        raise ValueError("AReaL requires finite --eps-clip in [0, 1).")
    if not math.isfinite(high) or high < 0:
        raise ValueError("AReaL requires finite, non-negative --eps-clip-high.")


def compute_areal_policy_loss(
    log_probs: torch.Tensor,
    proximal_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float | None = None,
    max_is_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use fixed proximal and behavior snapshots.
    Rejected tokens retain their place in the original response denominator."""
    if any(tensor.shape != log_probs.shape for tensor in (proximal_log_probs, rollout_log_probs, advantages)):
        raise ValueError("AReaL log-probs and advantages must have identical token shapes.")
    if not math.isfinite(max_is_weight) or max_is_weight <= 0:
        raise ValueError("AReaL max_is_weight must be finite and positive.")

    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    current = log_probs.to(dtype)
    proximal = proximal_log_probs.detach().to(dtype)
    behavior = rollout_log_probs.detach().to(dtype)
    advantage = advantages.detach().to(dtype)
    log_weight = proximal - behavior
    keep = torch.isfinite(log_weight) & (log_weight <= math.log(max_is_weight))
    log_ratio = current - proximal
    # Mask rejected tokens before exponentiation to prevent NaN gradients.
    ratio = torch.exp(torch.where(keep, log_ratio, torch.zeros_like(log_ratio)))
    high = eps_clip if eps_clip_high is None else eps_clip_high
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + high)

    weight = torch.exp(torch.where(keep, log_weight, torch.zeros_like(log_weight)))
    weight = torch.where(keep, weight, torch.zeros_like(weight))
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    loss = -torch.where(keep, weight * surrogate, torch.zeros_like(surrogate))

    reported_ratio = torch.exp(log_ratio.detach())
    metrics = {
        "pg_clipfrac": ((reported_ratio < 1.0 - eps_clip) | (reported_ratio > 1.0 + high)).float(),
        "areal_proximal_ratio": reported_ratio,
        "areal_behavior_ratio": torch.exp(log_weight).detach(),
        "areal_reject_fraction": (~keep).float().detach(),
    }
    return loss, metrics
