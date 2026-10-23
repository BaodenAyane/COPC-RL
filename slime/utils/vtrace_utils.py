"""V-trace targets and actor advantages."""

from __future__ import annotations

import math

import torch


def _validate_parameters(gamma, lambd, rho_clip, c_clip, pg_rho_clip):
    for name, value in (("gamma", gamma), ("lambd", lambd)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"V-trace {name} must be finite and in [0, 1].")
    for name, value in (("rho-clip", rho_clip), ("c-clip", c_clip), ("pg-rho-clip", pg_rho_clip)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--vtrace-{name} must be finite and positive.")
    if c_clip > min(1.0, rho_clip):
        raise ValueError("--vtrace-c-clip must be <= 1 and <= --vtrace-rho-clip.")


def validate_vtrace_config(args):
    if args.advantage_estimator != "vtrace":
        return
    _validate_parameters(args.gamma, args.lambd, args.vtrace_rho_clip, args.vtrace_c_clip, args.vtrace_pg_rho_clip)
    if getattr(args, "clip_only", False) and not getattr(args, "use_rollout_logprobs", False):
        raise ValueError("V-trace --clip-only requires --use-rollout-logprobs for the actor learner/behavior ratio.")
    for name in (
        "keep_old_actor", "use_tis", "get_mismatch_metrics", "use_opsm",
        "custom_tis_function_path", "custom_advantage_function_path", "use_opd",
    ):
        if getattr(args, name, None):
            raise ValueError(f"V-trace does not support --{name.replace('_', '-')}; it supplies its own off-policy correction.")
    if getattr(args, "loss_type", "policy_loss") != "policy_loss":
        raise ValueError("--advantage-estimator vtrace requires --loss-type policy_loss.")
    if not getattr(args, "compute_advantages_and_returns", True):
        raise ValueError("V-trace requires built-in advantage/return computation.")


@torch.no_grad()
def vtrace_from_tensors(
    rewards: torch.Tensor,
    values: torch.Tensor,
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    lambd: float = 1.0,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
    pg_rho_clip: float = 1.0,
    apply_pg_rho: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute terminal-response V-trace (Espeholt et al., 2018).
    Inputs are aligned 1D tensors; masked tokens are skipped and terminal bootstrap is zero.
    Disable apply_pg_rho when the actor supplies its own importance ratio."""
    _validate_parameters(gamma, lambd, rho_clip, c_clip, pg_rho_clip)
    tensors = (rewards, values, target_log_probs, behavior_log_probs)
    if any(x.ndim != 1 or x.shape != values.shape for x in tensors):
        raise ValueError("V-trace rewards, values and log-probs must have identical 1D shapes.")
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in tensors) else torch.float32
    device = values.device
    mask = torch.ones_like(values, dtype=torch.bool) if loss_mask is None else loss_mask.to(device=device)
    if mask.shape != values.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("V-trace loss_mask must be binary and match values.")
    indices = mask.nonzero(as_tuple=True)[0]
    advantages = torch.zeros(values.shape, dtype=dtype, device=device)
    returns = torch.zeros_like(advantages)
    metric_names = (
        "vtrace_rhos", "vtrace_clipped_rhos", "vtrace_cs", "vtrace_pg_rhos",
        "vtrace_rho_clipfrac", "vtrace_c_clipfrac", "vtrace_pg_rho_clipfrac",
    )
    metrics = {name: torch.zeros_like(advantages) for name in metric_names}
    if indices.numel() == 0:
        return advantages, returns, metrics

    reward, value, target, behavior = [x.to(device=device, dtype=dtype)[indices] for x in tensors]
    if not all(torch.isfinite(x).all() for x in (reward, value, target, behavior)):
        raise ValueError("V-trace requires finite rewards, values and log-probs on valid actions.")
    log_rhos = target - behavior
    # Clip before exponentiation to avoid overflow.
    rho = log_rhos.clamp(max=math.log(rho_clip)).exp()
    c = log_rhos.clamp(max=math.log(c_clip)).exp() * lambd
    if apply_pg_rho:
        pg_rho = log_rhos.clamp(max=math.log(pg_rho_clip)).exp()
        pg_clipfrac = (log_rhos > math.log(pg_rho_clip)).to(dtype)
    else:
        # Construct raw advantages directly; division by tiny rho can lose the signal.
        pg_rho = torch.ones_like(log_rhos)
        pg_clipfrac = torch.zeros_like(log_rhos)
    next_value = torch.cat((value[1:], value.new_zeros(1)))
    delta = rho * (reward + gamma * next_value - value)

    correction = delta
    coefficient = gamma * c
    stride = 1
    while stride < value.numel():
        correction = torch.cat((
            correction[:-stride] + coefficient[:-stride] * correction[stride:], correction[-stride:],
        ))
        coefficient = torch.cat((coefficient[:-stride] * coefficient[stride:], coefficient[-stride:]))
        stride *= 2
    vs = value + correction
    next_vs = torch.cat((vs[1:], vs.new_zeros(1)))
    advantages[indices] = pg_rho * (reward + gamma * next_vs - value)
    returns[indices] = vs
    # Only diagnostics cap raw rho at exp(20); algorithm weights use the configured caps.
    for name, tensor in zip(metric_names, (
        log_rhos.clamp(max=20).exp(), rho, c, pg_rho,
        (log_rhos > math.log(rho_clip)).to(dtype),
        (log_rhos > math.log(c_clip)).to(dtype),
        pg_clipfrac,
    ), strict=True):
        metrics[name][indices] = tensor
    return advantages, returns, metrics


@torch.no_grad()
def get_vtrace_advantages_and_returns_batch(
    total_lengths, response_lengths, values_list, target_log_probs_list,
    behavior_log_probs_list, scalar_rewards, kl, loss_masks, *,
    gamma, lambd, kl_coef, rho_clip, c_clip, pg_rho_clip, apply_pg_rho=True,
):
    """Compute V-trace on full responses, then return CP-local slices."""
    from megatron.core import mpu

    size = len(response_lengths)
    if any(len(xs) != size for xs in (
        total_lengths, values_list, target_log_probs_list, behavior_log_probs_list, scalar_rewards, kl, loss_masks,
    )):
        raise ValueError("V-trace batch fields must contain the same number of responses.")
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
    advantages_list, returns_list, metric_lists = [], [], {}
    for i in range(size):
        total_len, resp_len = total_lengths[i], response_lengths[i]
        full = [values_list[i], target_log_probs_list[i], behavior_log_probs_list[i], kl[i]]
        if cp_size > 1:
            full = [all_gather_with_cp(x, total_len, resp_len) for x in full]
        values, target, behavior, full_kl = full
        mask = loss_masks[i].to(device=values.device)
        if mask.shape != (resp_len,) or any(x.shape != (resp_len,) for x in full):
            raise ValueError("V-trace gathered tensors and full loss masks must match response_lengths.")
        dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        # Avoid 0 * NaN in the unused KL field.
        token_rewards = -kl_coef * full_kl.to(dtype) if kl_coef else torch.zeros_like(values, dtype=dtype)
        valid = mask.nonzero(as_tuple=True)[0]
        if valid.numel():
            token_rewards[valid[-1]] += scalar_rewards[i]
        advantages, returns, metrics = vtrace_from_tensors(
            token_rewards, values, target, behavior, mask, gamma=gamma, lambd=lambd,
            rho_clip=rho_clip, c_clip=c_clip, pg_rho_clip=pg_rho_clip,
            apply_pg_rho=apply_pg_rho,
        )
        if cp_size > 1:
            advantages = slice_log_prob_with_cp(advantages, total_len, resp_len)
            returns = slice_log_prob_with_cp(returns, total_len, resp_len)
            metrics = {k: slice_log_prob_with_cp(v, total_len, resp_len) for k, v in metrics.items()}
        advantages_list.append(advantages)
        returns_list.append(returns)
        for key, value in metrics.items():
            metric_lists.setdefault(key, []).append(value)
    return advantages_list, returns_list, metric_lists
