# Modified for COPC.
import logging
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.types import RolloutBatch

from ...utils import logging_utils
from .cp_utils import (
    gather_and_reduce_log_dict,
    get_sum_of_sample_mean,
    rollout_log_metric_contribution,
    slice_with_cp,
)

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """Build a packed micro-batch in the selected CP layout, retaining full tokens as unconcat_tokens."""

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens = batch["tokens"]
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # CP log-probability extraction needs the full token sequences.
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if allgather_cp:
        cu_seqlens_list: list[int] = [0]
        for t in tokens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

        tokens = torch.cat(tokens, dim=0)

        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
        tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]

        cu_seqlens = [0]
        for t in tokens:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))

        tokens = torch.cat(tokens)

        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)

        # Packed THD metadata uses unpadded sequence lengths.
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    tokens = tokens.unsqueeze(0)

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        # Shift masks with next-token targets.
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(loss_mask, 0)
        loss_masks.append(loss_mask)

    if allgather_cp:
        # Keep the mask layout identical to the token layout.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    else:
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
) -> dict[str, float] | None:
    """Reduce per-rank metrics on the DP source rank and log them to W&B or TensorBoard."""
    reduced = gather_and_reduce_log_dict(
        log_dict,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
    )
    if reduced is None:
        return None
    reduced_log_dict = {f"{metric_name}/{k}": v for k, v in reduced.items()}
    logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")
    step = compute_rollout_step(args, rollout_id)
    reduced_log_dict["rollout/step"] = step
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
    return reduced_log_dict


class DataIterator:
    """Iterate rollout data by scheduled micro-batch indices."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return requested fields for the next micro-batch, using None for absent fields."""
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one scheduled data iterator per VPP stage."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Summarize rollout fields on the last PP stage and TP rank 0."""
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        # Use the same rollout denominators as training.
        rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
        dp_world = mpu.get_data_parallel_world_size(with_context_parallel=False)
        num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_routed_experts",
                "global_batch_sizes",
                "num_microbatches",
                "micro_batch_indices",
                "source_names",
                "weight_versions",
            ]:
                continue
            # Weight uneven DP partitions by their sample counts.
            if isinstance(val, (list, tuple)):
                count = len(val)
                if isinstance(val[0], torch.Tensor):
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                        "vtrace_rhos",
                        "vtrace_clipped_rhos",
                        "vtrace_cs",
                        "vtrace_pg_rhos",
                        "vtrace_rho_clipfrac",
                        "vtrace_c_clipfrac",
                        "vtrace_pg_rho_clipfrac",
                    ]:
                        tensor = torch.cat(val).clone().detach()
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            rollout_mask_sums,
                        )
                        sum_value, count = rollout_log_metric_contribution(
                            sum_of_sample_mean(tensor).item(),
                            cp_size=cp_size,
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (sum_value, count)
                        continue
                    tensor = torch.cat(val).clone().detach()
                    per_rank_sum = tensor.mean() * cp_size * count
                    sum_value = per_rank_sum.item()
                else:
                    sum_value = sum(val)
                log_dict[key] = (sum_value, count)
            elif isinstance(val, torch.Tensor):
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        emit_explained_var = "returns" in rollout_data and "values" in rollout_data
        if emit_explained_var:
            sum_of_token = get_sum_of_sample_mean(
                total_lengths,
                response_lengths,
                loss_masks,
                rollout_mask_sums,
                calculate_per_token_loss=True,
            )
            returns_t = torch.cat(rollout_data["returns"]).clone().detach().double()
            values_t = torch.cat(rollout_data["values"]).clone().detach().double()
            diff_t = returns_t - values_t
            token_count = sum_of_token(torch.ones_like(returns_t)).item()
            moment_dict = {
                name: (sum_of_token(tensor).item(), token_count)
                for name, tensor in (
                    ("r", returns_t),
                    ("r2", returns_t * returns_t),
                    ("d", diff_t),
                    ("d2", diff_t * diff_t),
                )
            }

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)

        if emit_explained_var:
            reduced_moments = gather_and_reduce_log_dict(
                moment_dict,
                dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
                dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
                dp_group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
            )
            if reduced_moments is not None:
                e_r, e_r2 = reduced_moments["r"], reduced_moments["r2"]
                e_d, e_d2 = reduced_moments["d"], reduced_moments["d2"]
                var_r = e_r2 - e_r * e_r
                var_d = e_d2 - e_d * e_d
                explained_var = 1.0 - var_d / (var_r + 1e-5)
                payload = {
                    "rollout/vf_explained_var": explained_var,
                    "rollout/step": compute_rollout_step(args, rollout_id),
                }
                logger.info(f"rollout-ev {rollout_id}: {payload}")
                logging_utils.log(args, payload, step_key="rollout/step")
                if reduced_log_dict is not None:
                    reduced_log_dict["rollout/vf_explained_var"] = explained_var
        if args.ci_test and reduced_log_dict is not None:
            # Routing replay differs between actor and reference, so skip their zero-KL check.
            if (
                rollout_id == 0
                and not getattr(args, "ci_disable_kl_checker", False)
                and not getattr(args, "use_rollout_routing_replay", False)
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -1 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 1

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward >= 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # Normalize correct responses independently; correct_entropy uses sampled negative log-probs.
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, sample_denoms=None
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Log response lengths and interaction counts on the last PP stage and TP rank 0."""
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:
                    device = val[0].device

                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute pass@k from raw rewards grouped by prompt."""
    if (
        args.dynamic_sampling_filter_path is not None
        and args.rollout_function_path == "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
        and not args.load_debug_rollout_data
    ):
        # DAPO passrate is already logged before candidate filtering.
        return
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace, extra_metrics: dict | None = None) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
        extra_metrics=extra_metrics,
    )


def tensors_to_cpu(tensor_list):
    """Detach tensors and copy them to CPU for Ray transfer; preserve None."""
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Copy tensors to the selected CUDA device as float32; preserve None."""
    if tensor_list is None:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
