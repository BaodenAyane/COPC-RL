# Modified for COPC.
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_cispo_loss,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_copc_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)

ROLLOUT_TOP_P_TOKEN_KEYS = (
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
)


def get_rollout_top_p_logprob_kwargs(args: Namespace, batch: dict[str, Any]) -> dict[str, Any]:
    if args.rollout_top_p == 1.0:
        return {}

    top_p_token_ids = batch.get("rollout_top_p_token_ids")
    top_p_token_offsets = batch.get("rollout_top_p_token_offsets")
    if top_p_token_ids is None or top_p_token_offsets is None:
        raise ValueError("rollout_top_p != 1.0 requires rollout_top_p_token_ids and rollout_top_p_token_offsets.")
    return {
        "top_p_token_ids": top_p_token_ids,
        "top_p_token_offsets": top_p_token_offsets,
    }


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    apply_temperature: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned logit and next-token pairs for each local CP slice."""
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"
    assert logits.size(0) == 1, f"{logits.shape}"
    logits = logits.squeeze(0)

    if apply_temperature and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # All-gather CP owns contiguous global positions; other CP paths use zigzag chunks.
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    """Convert contiguous all-gather CP responses to the zigzag layout without breaking autograd."""
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        if all(v is None for v in values):
            continue

        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=ref_value.requires_grad,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        new_values = []
        for full_resp, total_length, response_length in zip(
            all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False
        ):
            new_values.append(slice_log_prob_with_cp(full_resp, total_length, response_length))

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> torch.Tensor:
    """Build next-token targets for packed and padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for tokens, total_length, response_length in zip(
            unconcat_tokens, total_lengths, response_lengths, strict=False
        ):
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    offset = 0
    for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
        full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
        offset += total_length

    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _fill_topp_mask_rows(
    keep: torch.Tensor,
    ids: list[int],
    offsets: list[int],
    response_start: int,
    local_start: int,
    length: int,
    vocab_start: int,
    vocab_end: int,
) -> None:
    end = min(response_start + length, max(len(offsets) - 1, 0))
    for response_idx in range(response_start, end):
        local_ids = [
            token_id - vocab_start
            for token_id in ids[offsets[response_idx] : offsets[response_idx + 1]]
            if vocab_start <= token_id < vocab_end
        ]
        row = local_start + response_idx - response_start
        keep[row].fill_(False)
        if local_ids:
            keep[row, torch.tensor(local_ids, device=keep.device, dtype=torch.long)] = True


def _build_topp_keep_mask(
    T: int,
    vocab_local: int,
    device: torch.device,
    top_p_token_ids: list[list[int]],
    top_p_token_offsets: list[list[int]],
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> torch.Tensor:
    """Build a [T, vocab_local] nucleus mask using each response token's recorded top-p support."""
    cp_size = mpu.get_context_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_start = tp_rank * vocab_local
    vocab_end = vocab_start + vocab_local

    top_p_token_ids = [t.tolist() if torch.is_tensor(t) else list(t) for t in top_p_token_ids]
    top_p_token_offsets = [t.tolist() if torch.is_tensor(t) else list(t) for t in top_p_token_offsets]

    keep = torch.ones((T, vocab_local), dtype=torch.bool, device=device)

    if cp_size > 1 and not allgather_cp:
        local_base = 0
        for ids, offsets, total_length, response_length in zip(
            top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
        ):
            prompt_length = total_length - response_length
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            for half, base in ((0, local_base), (1, local_base + chunk_size_cp)):
                local_start = base + logits_offset[half][0] - chunks_offset[half][0]
                length = logits_offset[half][1] - logits_offset[half][0]
                response_start = tokens_offset[half][0] - prompt_length
                _fill_topp_mask_rows(keep, ids, offsets, response_start, local_start, length, vocab_start, vocab_end)
            local_base += 2 * chunk_size_cp
        return keep

    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        seq_start = 0
        for ids, offsets, total_length, response_length in zip(
            top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
        ):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1
            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e > s:
                _fill_topp_mask_rows(
                    keep,
                    ids,
                    offsets,
                    s - logit_global_start,
                    s - chunk_start,
                    e - s,
                    vocab_start,
                    vocab_end,
                )
            seq_start += total_length
        return keep

    offset = 0
    for ids, offsets, total_length, response_length in zip(
        top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
    ):
        end = offset + total_length
        start = end - response_length
        _fill_topp_mask_rows(keep, ids, offsets, 0, start - 1, response_length, vocab_start, vocab_end)
        offset += total_length

    return keep


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice response log-probabilities and entropy from full-length tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        pos = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(log_prob_full[:0])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[:0])
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            end = offset + total_length
            start = end - response_length
            log_probs_list.append(log_prob_full[start - 1 : end - 1])
            if entropy_full is not None:
                entropy_list.append(entropy_full[start - 1 : end - 1])
            offset += total_length

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    top_p_token_ids: list[list[int]] | None = None,
    top_p_token_offsets: list[list[int]] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Top-p masking applies only to log-probabilities; outputs use the zigzag CP layout."""
    assert non_loss_data
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"
    assert logits.size(0) == 1, f"{logits.shape}"
    logits = logits.squeeze(0)

    # Match the rollout distribution's temperature.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size
    # Keep entropy metrics without backward buffers when its loss coefficient is zero.
    with_entropy_grad = with_entropy and getattr(args, "entropy_coef", 0.0) != 0

    full_tokens = _build_shifted_tokens(T, device, unconcat_tokens, total_lengths, response_lengths, args.allgather_cp)

    top_p_keep_mask = None
    if top_p_token_ids is not None and top_p_token_offsets is not None:
        top_p_keep_mask = _build_topp_keep_mask(
            T,
            logits.size(-1),
            device,
            top_p_token_ids,
            top_p_token_offsets,
            total_lengths,
            response_lengths,
            args.allgather_cp,
        )

    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy_grad,
        chunk_size=chunk_size,
        log_prob_keep_mask=top_p_keep_mask,
    )
    log_prob_full = log_prob_full.squeeze(-1)

    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )

    return torch.empty((0,), device=device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Extract response-aligned value predictions from local logits."""
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    # Reference: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """Subtract the distillation KL penalty from advantages in-place."""

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Optional whitening changes advantages only; custom estimators must populate both outputs."""
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    if args.advantage_estimator == "vtrace":
        # V-trace targets use learner log-probabilities even when the actor uses rollout log-probs.
        log_probs = rollout_data.get("log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    if not mpu.is_pipeline_last_stage():
        return

    if args.advantage_estimator == "vtrace":
        for key in ("values", "log_probs", "rollout_log_probs"):
            if rollout_data.get(key) is None:
                raise ValueError(f"V-trace requires rollout_data['{key}'].")
        if args.kl_coef != 0 and ref_log_probs is None:
            raise ValueError("V-trace with reward KL requires rollout_data['ref_log_probs'].")

    if args.kl_coef == 0 or not log_probs:
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo", "cispo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # Replacing an advantage must not replace its corresponding return.
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "areal":
        if values is None:
            raise ValueError("AReaL requires critic values in rollout_data['values'].")
        token_rewards = []
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(rewards, kl, strict=False):
            token_reward = -args.kl_coef * k
            if cp_rank == 0 and token_reward.numel() > 0:
                token_reward[-1] += reward
            token_rewards.append(token_reward)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, token_rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "vtrace":
        from slime.utils.vtrace_utils import get_vtrace_advantages_and_returns_batch

        advantages, returns, vtrace_metrics = get_vtrace_advantages_and_returns_batch(
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            values_list=values,
            target_log_probs_list=rollout_data["log_probs"],
            behavior_log_probs_list=rollout_log_probs,
            scalar_rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            gamma=args.gamma,
            lambd=args.lambd,
            kl_coef=args.kl_coef,
            rho_clip=args.vtrace_rho_clip,
            c_clip=args.vtrace_c_clip,
            pg_rho_clip=args.vtrace_pg_rho_clip,
            apply_pg_rho=not getattr(args, "clip_only", False),
        )
        rollout_data.update(vtrace_metrics)

    elif args.advantage_estimator == "copc":
        if values is None:
            raise ValueError("COPC requires critic values in rollout_data['values'].")
        if rollout_data.get("log_probs") is None:
            raise ValueError("COPC requires train log_probs in rollout_data['log_probs'].")
        if rollout_log_probs is None:
            raise ValueError("COPC requires rollout log_probs in rollout_data['rollout_log_probs'].")

        advantages, returns, copc_metrics = get_copc_advantages_and_returns_batch(
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            values_list=values,
            # Return correction always compares the frozen learner against the behavior policy.
            target_log_probs_list=rollout_data["log_probs"],
            behavior_log_probs_list=rollout_log_probs,
            scalar_rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            gamma=args.gamma,
            lambd=args.lambd,
            kl_coef=args.kl_coef,
            rho_clip=args.copc_rho_clip,
            rho_clip_high=args.copc_rho_clip_high,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
        )
        rollout_data.update(copc_metrics)

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    if args.advantage_estimator == "vtrace" and not getattr(args, "clip_only", False):
        # The default V-trace advantage already contains pg_rho; clip-only supplies its own ratio.
        pg_loss = -log_probs * advantages.detach()
        pg_clipfrac = torch.zeros_like(pg_loss)
    elif args.advantage_estimator == "cispo":
        pg_loss, pg_clipfrac = compute_cispo_loss(ppo_kl, log_probs, advantages, args.eps_clip, args.eps_clip_high)
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(
            ppo_kl, advantages, args.eps_clip, args.eps_clip_high,
            use_min=(
                args.advantage_estimator == "grpo"
                or (args.advantage_estimator == "ppo" and not getattr(args, "clip_only", False))
            ),
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    if args.get_mismatch_metrics or args.use_tis:
        # Mismatch statistics retain the original mask, including rejected tokens.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # Rejection changes the numerator, not the precomputed rollout denominator.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        log_probs_to_compare = log_probs if args.use_rollout_logprobs else old_log_probs
        train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs_to_compare - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    # COPC clip diagnostics are logged on the rollout channel.
    if args.advantage_estimator != "copc":
        reported_loss["pg_clipfrac"] = pg_clipfrac.clone().detach()

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped critic regression and explained-variance moments on trainable tokens."""
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0).detach()

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    ev_moments = values.new_zeros(8, dtype=torch.float64)
    metric_masks = batch["loss_masks"]
    if getattr(args, "advantage_estimator", None) == "vtrace" and mpu.get_context_parallel_world_size() > 1:
        # Explained-variance masks must match the local CP value slices.
        metric_masks = [
            slice_log_prob_with_cp(mask, total_length, response_length)
            for mask, total_length, response_length in zip(
                metric_masks, batch["total_lengths"], batch["response_lengths"], strict=True
            )
        ]
    cat_loss_masks = torch.cat(metric_masks, dim=0).to(dtype=torch.float64)
    mask = cat_loss_masks
    target = returns.detach().to(dtype=torch.float64)
    pred = values.detach().to(dtype=torch.float64)
    error = target - pred
    ev_moments += torch.stack(
        [
            mask.sum(),
            (target * mask).sum(),
            (target.square() * mask).sum(),
            (error * mask).sum(),
            (error.square() * mask).sum(),
            (pred * mask).sum(),
            (pred.square() * mask).sum(),
            (pred * target * mask).sum(),
        ]
    )

    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
        "_ev_count": ev_moments[0],
        "_ev_target_sum": ev_moments[1],
        "_ev_target_sq_sum": ev_moments[2],
        "_ev_error_sum": ev_moments[3],
        "_ev_error_sq_sum": ev_moments[4],
        "_ev_pred_sum": ev_moments[5],
        "_ev_pred_sq_sum": ev_moments[6],
        "_ev_cross_sum": ev_moments[7],
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked next-token negative log-likelihood for supervised fine-tuning."""
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Apply Megatron accumulation and parallelism scaling.
    step_global_batch_size is the actual sample count across DP ranks."""
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
    )

    match args.loss_type:
        case "policy_loss":
            if args.advantage_estimator == "areal":
                from .areal_loss import areal_policy_loss_function

                func = areal_policy_loss_function
            else:
                func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Empty CP ranks must traverse backward collectives to avoid deadlocking other ranks.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Cancel Megatron's CP gradient multiplier.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            # Token counts are reduced across micro-batches; per-rollout mode uses the step batch size.
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
