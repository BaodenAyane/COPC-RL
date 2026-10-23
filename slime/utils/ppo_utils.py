# Modified for COPC.
# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate KL divergence; see http://joschu.net/blog/kl-approx.html."""
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the OPSM token mask and masked-sequence statistic."""
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Broadcast full-response GSPO KL values to CP-local token shapes."""
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
    use_min: bool = False,
):
    ratio = (-ppo_kl).exp()
    advantages_actor = advantages.detach()
    clipped_ratio = ratio.clamp(1 - eps_clip, 1 + eps_clip_high)
    pg_losses = -clipped_ratio * advantages_actor
    clipfrac = ((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip_high)).float()

    # COPC uses clamp-only; use_min selects PPO. Clamp retains gradients at the boundaries.
    if use_min:
        surr1 = ratio * advantages_actor
        surr2 = clipped_ratio * advantages_actor
        pg_losses = -torch.min(surr1, surr2)

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses_c = -eps_clip_c * advantages_actor
        pg_losses = torch.where(advantages_actor < 0, torch.min(pg_losses_c, pg_losses), pg_losses)

    return pg_losses, clipfrac


@torch.compile(dynamic=True)
def compute_cispo_loss(
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
):
    """Compute CISPO with detached clipped ratios (https://arxiv.org/abs/2506.13585).
    Clipping margins are offsets from 1; eps_clip >= 1 disables the lower bound."""
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac = (ratio_truncated != ratio).float()
    return pg_losses, clipfrac


def _maybe_all_reduce(tensor: torch.Tensor, op: dist.ReduceOp, process_group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=process_group)


def _get_vocab_parallel_rank_size(process_group) -> tuple[int, int]:
    if process_group is not None and hasattr(process_group, "rank") and hasattr(process_group, "size"):
        return process_group.rank(), process_group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)
    return 0, 1


class _VocabParallelLogProbEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        log_prob_keep_mask: torch.Tensor | None,
        process_group,
        with_entropy: bool,
        with_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with_entropy_grad = with_entropy and with_entropy_grad
        vocab_parallel_logits = vocab_parallel_logits.float()
        seq_len, vocab_parallel_size = vocab_parallel_logits.shape
        rank, _world_size = _get_vocab_parallel_rank_size(process_group)
        vocab_start_index = rank * vocab_parallel_size
        vocab_end_index = vocab_start_index + vocab_parallel_size

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target_1d = (target - vocab_start_index).clone()
        masked_target_1d[target_mask] = 0
        arange_1d = torch.arange(seq_len, device=vocab_parallel_logits.device)

        def vocab_parallel_softmax(
            logits: torch.Tensor,
            inplace: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_max = logits.max(dim=-1, keepdim=True).values
            _maybe_all_reduce(logits_max, dist.ReduceOp.MAX, process_group)
            # In-place softmax requires an owned scratch buffer.
            normalized_logits = logits.sub_(logits_max) if inplace else logits - logits_max
            # Copy target logits before in-place exponentiation overwrites them.
            predicted_logits = normalized_logits.view(-1, vocab_parallel_size)[arange_1d, masked_target_1d]
            exp_logits = normalized_logits.exp_()
            sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
            _maybe_all_reduce(sum_exp_logits, dist.ReduceOp.SUM, process_group)
            softmax = exp_logits.div_(sum_exp_logits)
            return predicted_logits, sum_exp_logits, softmax, logits_max

        entropy = vocab_parallel_logits.new_zeros((0,))
        entropy_softmax = vocab_parallel_logits.new_empty((0,))
        sum_softmax_times_logits = vocab_parallel_logits.new_empty((0,))

        def sum_softmax_logits(softmax: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            if softmax.is_cuda:
                return torch.einsum("ij,ij->i", softmax, logits).unsqueeze(-1)
            return (softmax * logits).sum(dim=-1, keepdim=True)

        if log_prob_keep_mask is None:
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, log_prob_logits_max = vocab_parallel_softmax(
                vocab_parallel_logits
            )
            if with_entropy:
                entropy_softmax = log_prob_softmax
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = log_prob_logits_max + log_prob_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)
        else:
            if with_entropy:
                _entropy_predicted_logits, entropy_sum_exp_logits, entropy_softmax, entropy_logits_max = (
                    vocab_parallel_softmax(vocab_parallel_logits)
                )
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = entropy_logits_max + entropy_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)

            local_target_rows = torch.nonzero(~target_mask, as_tuple=False).squeeze(-1)
            log_prob_logits = vocab_parallel_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
            if local_target_rows.numel() > 0:
                log_prob_logits[local_target_rows, masked_target_1d[local_target_rows]] = vocab_parallel_logits[
                    local_target_rows, masked_target_1d[local_target_rows]
                ]
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, _log_prob_logits_max = vocab_parallel_softmax(
                log_prob_logits, inplace=True
            )

        predicted_logits = predicted_logits.masked_fill_(target_mask, 0.0).unsqueeze(-1)
        _maybe_all_reduce(predicted_logits, dist.ReduceOp.SUM, process_group)
        log_prob = predicted_logits - log_prob_sum_exp_logits.log()

        if not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)

        ctx.with_entropy_grad = with_entropy_grad
        # Skip entropy-backward buffers when entropy is only a metric.
        saved_entropy_softmax = entropy_softmax if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        saved_sum_softmax_times_logits = (
            sum_softmax_times_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        )
        saved_logits = vocab_parallel_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        ctx.save_for_backward(
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            saved_entropy_softmax,
            saved_sum_softmax_times_logits,
            saved_logits,
        )
        return log_prob, entropy

    @staticmethod
    def backward(
        ctx, grad_log_prob: torch.Tensor | None, grad_entropy: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            entropy_softmax,
            sum_softmax_times_logits,
            vocab_parallel_logits,
        ) = ctx.saved_tensors

        if grad_log_prob is None:
            raise RuntimeError(
                "_VocabParallelLogProbEntropy expected a materialized grad_log_prob. "
                "Do not call ctx.set_materialize_grads(False)."
            )

        grad_entropy_input = None
        if ctx.with_entropy_grad and grad_entropy is not None and grad_entropy.numel() > 0:
            # Compute entropy gradients before mutating the aliased softmax buffer.
            grad_entropy_input = sum_softmax_times_logits - vocab_parallel_logits
            grad_entropy_input.mul_(entropy_softmax)
            grad_entropy_input.mul_(grad_entropy.reshape(-1, 1))

        vocab_parallel_size = log_prob_softmax.size(-1)
        grad_input = log_prob_softmax.neg_()
        grad_2d = grad_input.view(-1, vocab_parallel_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        target_update = (~target_mask).to(dtype=grad_2d.dtype)
        grad_2d[arange_1d, masked_target_1d] += target_update
        grad_input.mul_(grad_log_prob.reshape(-1, 1))

        if grad_entropy_input is not None:
            grad_input.add_(grad_entropy_input)

        return grad_input, None, None, None, None, None


def _calculate_log_probs_and_entropy_chunk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    *,
    with_entropy: bool,
    with_entropy_grad: bool = True,
    log_prob_keep_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    log_prob, entropy = _VocabParallelLogProbEntropy.apply(
        logits,
        tokens,
        log_prob_keep_mask,
        tp_group,
        with_entropy,
        with_entropy_grad,
    )
    if not with_entropy:
        entropy = None
    return log_prob, entropy


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """Compute CP-local REINFORCE++ returns (https://arxiv.org/pdf/2501.03262).
    Loss masks cover full responses; rewards are added to the last valid action."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """Broadcast group-centered rewards and subtract token-level KL penalties."""
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
):
    """Compute batched GAE, gathering full responses before slicing CP-local results."""

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def copc_trace_scan(c: torch.Tensor, w: float, chunk_size: int = 128) -> torch.Tensor:
    """Scan [B, L] COPC residuals with a constant recurrence coefficient."""
    B, L = c.shape
    if L == 0:
        return c.clone()
    if w == 1.0:
        return torch.cumsum(c, dim=1)

    pad = (-L) % chunk_size
    if pad:
        c = F.pad(c, (0, pad))
    n = c.shape[1] // chunk_size
    chunks = c.view(B, n, chunk_size)

    idx = torch.arange(chunk_size, device=c.device, dtype=c.dtype)
    w_t = torch.full((), w, device=c.device, dtype=c.dtype)
    E = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp_min(0)
    M = torch.tril(w_t**E)
    local = torch.einsum("ij,bnj->bni", M, chunks)

    w_pow = w_t ** torch.arange(1, chunk_size + 1, device=c.device, dtype=c.dtype)
    out = torch.empty_like(local)
    s_in = torch.zeros(B, device=c.device, dtype=c.dtype)
    for k in range(n):
        out[:, k, :] = local[:, k, :] + w_pow * s_in.unsqueeze(-1)
        s_in = out[:, k, -1]
    return out.view(B, -1)[:, :L]


def copc_from_tensors(
    rewards: torch.Tensor,
    values: torch.Tensor,
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    gamma: float,
    lambd: float,
    rho_clip: float,
    rho_clip_high: float,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute detached COPC advantages, targets, and diagnostics for one response.
    Inputs are aligned 1D tensors with frozen learner and behavior log-probabilities.
    The bootstrap advantage assumes lambda=1."""

    assert rewards.ndim == values.ndim == target_log_probs.ndim == behavior_log_probs.ndim == 1
    assert rewards.shape == values.shape == target_log_probs.shape == behavior_log_probs.shape

    rho_min = 1.0 - rho_clip
    rho_max = 1.0 + rho_clip_high

    dtype = values.dtype
    device = values.device
    rewards = rewards.to(device=device, dtype=dtype)
    target_log_probs = target_log_probs.to(device=device)
    behavior_log_probs = behavior_log_probs.to(device=device)

    if loss_mask is None:
        mask = torch.ones_like(values, dtype=dtype, device=device)
    else:
        mask = loss_mask.to(device=device, dtype=dtype)
        assert mask.shape == values.shape

    log_rhos = (target_log_probs.float() - behavior_log_probs.float()).clamp(min=-20.0, max=20.0)
    rhos = torch.exp(log_rhos).to(dtype=dtype)
    rhos_return = rhos.detach()
    in_band = (~((rhos_return < rho_min) | (rhos_return > rho_max))).to(dtype=dtype) * mask
    # Tool observations are environment outputs: use ratio 1, not their model likelihood.
    tool_rhos = torch.ones_like(rhos_return)
    clipped_rhos = torch.where(
        mask.bool(), torch.clamp(rhos_return, min=rho_min, max=rho_max), tool_rhos
    )

    masked_rewards = rewards * mask
    next_values = torch.cat([values[1:], torch.zeros(1, dtype=dtype, device=device)])
    deltas = clipped_rhos * (masked_rewards + gamma * next_values - values)

    # Detached wide-band reference for diagnostics only.
    REF_RHO_MIN, REF_RHO_MAX = 0.2, 5.0
    ref_clipped_rhos = torch.where(
        mask.bool(), torch.clamp(rhos_return, min=REF_RHO_MIN, max=REF_RHO_MAX), tool_rhos
    )
    ref_deltas = ref_clipped_rhos * (masked_rewards + gamma * next_values - values)

    # Accumulate both traces in float32 for long responses.
    w = gamma * lambd
    c = deltas + values - w * next_values
    c_ref = ref_deltas + values - w * next_values
    c_both = torch.stack([c, c_ref]).to(torch.float32).flip(1)
    S = copc_trace_scan(c_both, w)
    v = S[0].flip(0).to(dtype=dtype)
    v_ref = S[1].flip(0).to(dtype=dtype)

    copc_returns = v.detach()

    next_v = torch.cat([v[1:], torch.zeros(1, dtype=dtype, device=device)])
    advantages = (masked_rewards + gamma * next_v - values).detach() * mask

    next_v_ref = torch.cat([v_ref[1:], torch.zeros(1, dtype=dtype, device=device)])
    ref_advantages = (masked_rewards + gamma * next_v_ref - values).detach() * mask

    metrics = {
        "copc_rhos": rhos.detach(),
        "copc_clipped_rhos": clipped_rhos.detach(),
        "copc_rho_clipfrac": (((rhos < rho_min) | (rhos > rho_max)).to(dtype=dtype) * mask).detach(),
        "copc_rho_weight": (in_band).detach(),
    }

    w = clipped_rhos * mask
    n_tokens = mask.sum().clamp(min=1)
    sum_w = w.sum()
    sum_w2 = (w * w).sum()
    ess = (sum_w * sum_w) / (n_tokens * sum_w2.clamp(min=1e-8))
    metrics["copc_ess"] = ess.detach().unsqueeze(0)

    # Clipped-ratio diagnostics are not the actor's masked weights.
    mask_f = mask.to(dtype=torch.float32)
    n_tok = mask_f.sum().clamp(min=1)
    rho_f = rhos.detach().to(dtype=torch.float32)
    omega_A = rho_f.clamp(1.0 - eps_clip, 1.0 + eps_clip_high)
    adv_f = advantages.detach().to(dtype=torch.float32)
    ref_f = ref_advantages.detach().to(dtype=torch.float32)
    da = ((omega_A - rho_f).abs() * mask_f).sum() / n_tok
    dv = ((ref_f - adv_f).abs() * mask_f).sum() / n_tok
    q = ((omega_A * adv_f) ** 2 * mask_f).sum() / n_tok
    coupling = omega_A * adv_f
    mean_q = (coupling * mask_f).sum() / n_tok
    var_q = (q - mean_q * mean_q).clamp(min=0.0)
    interaction = (da * dv).clamp(min=0.0).sqrt()
    metrics["copc_da"] = da.detach().unsqueeze(0)
    metrics["copc_dv"] = dv.detach().unsqueeze(0)
    metrics["copc_q"] = q.detach().unsqueeze(0)
    metrics["copc_mean_q"] = mean_q.detach().unsqueeze(0)
    metrics["copc_var_q"] = var_q.detach().unsqueeze(0)
    metrics["copc_interaction"] = interaction.detach().unsqueeze(0)
    pg_clipfrac_tok = ((rho_f < 1.0 - eps_clip) | (rho_f > 1.0 + eps_clip_high)).to(dtype=torch.float32)
    metrics["copc_pg_clipfrac"] = ((pg_clipfrac_tok * mask_f).sum() / n_tok).detach().unsqueeze(0)
    metrics["copc_pg_clipped_rhos"] = ((omega_A * mask_f).sum() / n_tok).detach().unsqueeze(0)

    return advantages, copc_returns, metrics


def get_copc_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    target_log_probs_list,
    behavior_log_probs_list,
    scalar_rewards,
    kl,
    loss_masks,
    gamma,
    lambd,
    kl_coef,
    rho_clip,
    rho_clip_high,
    eps_clip,
    eps_clip_high,
):
    """Compute COPC targets with context-parallel gather and slice support."""

    from megatron.core import mpu

    with torch.no_grad():
        assert len(response_lengths) == len(values_list)
        assert len(values_list) == len(target_log_probs_list)
        assert len(values_list) == len(behavior_log_probs_list)
        assert len(values_list) == len(kl)
        assert len(values_list) == len(loss_masks)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype
        scalar_rewards = torch.as_tensor(scalar_rewards, dtype=dtype, device=device)

        advantages_list = []
        returns_list = []
        metric_lists: dict[str, list[torch.Tensor]] = {}

        for i, (total_len, resp_len, values, target_log_probs, behavior_log_probs, kl_tensor, loss_mask) in enumerate(
            zip(
                total_lengths,
                response_lengths,
                values_list,
                target_log_probs_list,
                behavior_log_probs_list,
                kl,
                loss_masks,
                strict=False,
            )
        ):
            if cp_size > 1:
                from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp

                full_values = all_gather_with_cp(values, total_len, resp_len)
                full_target_log_probs = all_gather_with_cp(target_log_probs, total_len, resp_len)
                full_behavior_log_probs = all_gather_with_cp(behavior_log_probs, total_len, resp_len)
                full_kl = all_gather_with_cp(kl_tensor, total_len, resp_len)
            else:
                full_values = values
                full_target_log_probs = target_log_probs
                full_behavior_log_probs = behavior_log_probs
                full_kl = kl_tensor

            full_mask = loss_mask.to(device=device, dtype=dtype)
            assert full_mask.numel() == resp_len

            token_rewards = -kl_coef * full_kl.to(device=device, dtype=dtype)
            token_rewards = token_rewards * full_mask

            valid_indices = full_mask.nonzero(as_tuple=True)[0]
            if valid_indices.numel() > 0:
                token_rewards[valid_indices[-1]] += scalar_rewards[i]

            advantages, returns, metrics = copc_from_tensors(
                rewards=token_rewards,
                values=full_values,
                target_log_probs=full_target_log_probs,
                behavior_log_probs=full_behavior_log_probs,
                loss_mask=full_mask,
                gamma=gamma,
                lambd=lambd,
                rho_clip=rho_clip,
                rho_clip_high=rho_clip_high,
                eps_clip=eps_clip,
                eps_clip_high=eps_clip_high,
            )

            if cp_size > 1:
                advantages = slice_log_prob_with_cp(advantages, total_len, resp_len)
                returns = slice_log_prob_with_cp(returns, total_len, resp_len)
                metrics = {
                    key: slice_log_prob_with_cp(value, total_len, resp_len)
                    if value.numel() == resp_len
                    else value
                    for key, value in metrics.items()
                }

            advantages_list.append(advantages)
            returns_list.append(returns)
            for key, value in metrics.items():
                metric_lists.setdefault(key, []).append(value)

    return advantages_list, returns_list, metric_lists




def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    response_lengths: list[int] | None = None,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """Compute [B, T] GAE using a FlashLinearAttention-style chunked scan with zero terminal bootstrap."""

    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])

    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]

    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    log_prob_keep_mask=None,
    with_entropy_grad: bool = True,
):
    logits = logits.contiguous()
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            mask_chunks = (
                log_prob_keep_mask.chunk(num_chunks, dim=0) if log_prob_keep_mask is not None else [None] * num_chunks
            )

            log_probs = []
            entropy_chunks = []
            for tokens_chunk, logits_chunk, mask_chunk in zip(tokens_chunks, logits_chunks, mask_chunks, strict=True):
                log_prob, entropy_chunk = _calculate_log_probs_and_entropy_chunk(
                    logits_chunk,
                    tokens_chunk,
                    tp_group,
                    with_entropy=with_entropy,
                    with_entropy_grad=with_entropy_grad,
                    log_prob_keep_mask=mask_chunk,
                )
                log_probs.append(log_prob)
                if entropy_chunk is not None:
                    entropy_chunks.append(entropy_chunk)
            log_prob = torch.cat(log_probs, dim=0)
            if entropy_chunks:
                entropy = torch.cat(entropy_chunks, dim=0)
        else:
            log_prob, entropy = _calculate_log_probs_and_entropy_chunk(
                logits,
                tokens,
                tp_group,
                with_entropy=with_entropy,
                with_entropy_grad=with_entropy_grad,
                log_prob_keep_mask=log_prob_keep_mask,
            )
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy
