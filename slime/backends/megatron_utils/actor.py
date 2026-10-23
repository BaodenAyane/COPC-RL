# Modified for COPC.
import logging
import os
from argparse import Namespace
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_torch_dist,
    register_default_process_group,
    reload_process_groups,
)
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import prepare_routed_experts_for_routing_replay, slice_log_prob_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_disk import UpdateWeightFromDisk
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _build_checkpointing_context(args):
    """Initialize native checkpoint context when supported; async loading requires its manager keys."""
    context = {}
    try:
        from megatron.training.checkpointing import setup_checkpointing_context

        context = setup_checkpointing_context(context, args)
    except ImportError:
        pass
    return context


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)
        self._is_offloaded = False
        self._keep_actor_resident = (
            role == "actor"
            and getattr(args, "_keep_actor_resident", False)
            and args.offload_train
            and not args.colocate
        )
        # Destroying and recreating WORLD invalidates raw dist.group.WORLD references cached by external code.
        # Set SLIME_DESTROY_WORLD_PROCESS_GROUP=0 when such references may outlive a train sleep/wake cycle.
        if os.getenv("SLIME_DESTROY_WORLD_PROCESS_GROUP", "1").lower() not in {"0", "false", "no"}:
            register_default_process_group(timeout=timedelta(minutes=args.distributed_timeout_minutes))
        else:
            logger.info("Default WORLD process-group destruction is disabled")

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        # Serialize config/tokenizer loading to avoid concurrent writes.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        if getattr(args, "async_resume", None):
            if any(getattr(args, name, False) for name in ("finetune", "no_load_optim", "no_load_rng")):
                raise ValueError("Full async resume must restore optimizer, scheduler, RNG, and iteration")
            if type(getattr(args, "ckpt_step", None)) is not int or args.ckpt_step < 0:
                raise ValueError("Full async resume requires an explicit non-negative checkpoint iteration")

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )
        self._loaded_checkpoint_iteration = loaded_rollout_id
        self._weights_published = False
        if getattr(args, "async_resume", None) and loaded_rollout_id != args.ckpt_step:
            raise RuntimeError(
                f"{role} loaded iteration {loaded_rollout_id}, expected {args.ckpt_step} for full async resume"
            )

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=True,
            ),
            single_tag=None,
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        if self.args.keep_old_actor:
            self.load_other_checkpoint("old_actor", args.load)
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Preserve model-native vocabulary padding from the HF config.
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        update_weight_mode = self.args.update_weight_mode
        update_weight_transport = self.args.update_weight_transport

        if update_weight_mode == "delta":
            # Disk delta sync updates a host-local checkpoint before engines reload it.
            assert not self.args.colocate, "--update-weight-mode=delta is not supported with --colocate"
            assert (
                update_weight_transport == "disk"
            ), "--update-weight-mode=delta requires --update-weight-transport=disk"
            from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

            update_weight_cls = UpdateWeightFromDiskDelta
        elif update_weight_transport == "disk":
            update_weight_cls = UpdateWeightFromDisk
        elif self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            assert update_weight_mode == "full"
            assert (
                update_weight_transport == "nccl"
            ), f"unsupported weight sync mode/transport: {update_weight_mode!r}/{update_weight_transport!r}"
            update_weight_cls = UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
        self.weight_updater.weight_version = getattr(self.args, "update_weight_start_version", 0)

        clear_memory()

        if self.args.offload_train:
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train
        if self._is_offloaded:
            return

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()
        self._is_offloaded = True

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        if not self._is_offloaded:
            clear_memory()
            if self.role == "actor" and self._active_model_tag != "actor":
                self._switch_model("actor")
            return
        print_memory("before wake_up model")

        torch_memory_saver.resume()
        self._is_offloaded = False

        clear_memory()
        reload_process_groups()

        if mpu.get_pipeline_model_parallel_world_size() > 2:
            # Initialize WORLD on all ranks before pipeline P2P first uses the recreated group.
            dist.barrier(device_ids=[torch.cuda.current_device()])
        if self.role == "actor":
            self._switch_model("actor")
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        device = torch.cuda.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            rollout_data["rollout_mask_sums"] = rollout_data["rollout_mask_sums"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        if "multimodal_train_inputs" in rollout_data:
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                        for key, value in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(log_prob, total_length, response_length).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = prepare_routed_experts_for_routing_replay(
                batch["rollout_routed_experts"],
                batch["tokens"],
                num_experts=self.args.num_experts,
                data_pad_size_multiplier=self.args.data_pad_size_multiplier,
                sequence_parallel=self.args.sequence_parallel,
                allgather_cp=self.args.allgather_cp,
            )

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                use_rollout_top_p_replay=True,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data, external_data=external_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            del rollout_data
            # In async COPC, the next actor forward releases the GPUs to the critic.
            if not self._keep_actor_resident:
                self.sleep()

        return result

    def forward_log_probs(self, rollout_data_ref: Box):
        """Return CPU policy log-probs for critic targets without updating weights."""
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]

        if self.args.advantage_estimator == "vtrace" and self.args.kl_coef != 0:
            # The critic needs the same reference KL as the actor's targets.
            self._switch_model("ref")
            rollout_data.update(self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_"))

        self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
        rollout_data.update(
            self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="",
            )
        )

        result = None
        if mpu.is_pipeline_last_stage():
            from slime.backends.megatron_utils.data import tensors_to_cpu

            result = {"log_probs": tensors_to_cpu(rollout_data["log_probs"])}
            if self.args.advantage_estimator == "vtrace" and self.args.kl_coef != 0:
                result["ref_log_probs"] = tensors_to_cpu(rollout_data["ref_log_probs"])

        if self.args.offload_train:
            del rollout_data
            self.sleep()

        return result

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if external_data is not None and mpu.is_pipeline_last_stage():
            log_probs = external_data.get("log_probs")
            if log_probs is not None:
                from slime.backends.megatron_utils.data import tensors_to_gpu

                rollout_data["log_probs"] = tensors_to_gpu(log_probs)
            if self.args.advantage_estimator == "vtrace" and self.args.kl_coef != 0:
                ref_log_probs = external_data.get("ref_log_probs")
                if ref_log_probs is None:
                    raise ValueError("V-trace with reward KL requires pre-forward ref_log_probs.")
                from slime.backends.megatron_utils.data import tensors_to_gpu

                rollout_data["ref_log_probs"] = tensors_to_gpu(ref_log_probs)

        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        # Critic-only training also needs value and explained-variance metrics.
        log_rollout_data(
            rollout_id,
            self.args,
            rollout_data,
        )

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            result = {"values": tensors_to_cpu(rollout_data["values"])}
            if "log_probs" in rollout_data:
                result["log_probs"] = tensors_to_cpu(rollout_data["log_probs"])
            logger.info(f"COPC train_critic: returning keys={list(result.keys())}, values_len={len(result.get('values', []))}")
            return result
        logger.warning(f"COPC train_critic: no values to return! is_last_stage={mpu.is_pipeline_last_stage()}, values_in_data={'values' in rollout_data}")
        return {}

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                log_probs_pre_computed = (
                    external_data is not None
                    and mpu.is_pipeline_last_stage()
                    and external_data.get("log_probs") is not None
                )
                if log_probs_pre_computed:
                    from slime.backends.megatron_utils.data import tensors_to_gpu

                    rollout_data["log_probs"] = tensors_to_gpu(external_data["log_probs"])
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and (not self.args.use_routing_replay or self.args.use_rollout_routing_replay)
                    and self.args.advantage_estimator != "gspo"
                )
                if (
                    not log_probs_pre_computed
                    and (
                        not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                    )
                    and not can_reuse_log_probs_in_loss
                ):
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        logger.info(f"COPC train_actor: external_data type={type(external_data)}, keys={list(external_data.keys()) if isinstance(external_data, dict) else 'N/A'}")
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                        else:
                            logger.warning(f"COPC: values not in external_data! external_data={external_data}")
                            self._switch_model("critic")
                            rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))
                            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

        # Compute advantages over the full rollout before minibatching.
                with timer("copc_advantages"):
                    compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            with timer("rollout_log"):
                log_rollout_data(
                    rollout_id,
                    self.args,
                    rollout_data,
                )

            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        self.weights_backuper.backup("actor")

        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        # Scale publication lag by optimizer steps per rollout; exclude within-rollout updates.
        _extra = self.weight_updater.pop_metrics()
        _wvs = rollout_data.get("weight_versions")
        if _wvs:
            _data_wv = max(int(v) for v in _wvs)
            _cur_wv = self.weight_updater.weight_version
            _extra["perf/staleness"] = (_cur_wv - _data_wv) * len(num_microbatches)
            _extra["perf/data_weight_version"] = _data_wv
            _extra["perf/current_weight_version"] = _cur_wv
        log_perf_data(rollout_id, self.args, extra_metrics=_extra)

    def get_checkpoint_state(self) -> dict:
        """Report loaded native state without advancing any RNG or weight version."""
        if self.args.debug_rollout_only:
            raise RuntimeError("No native trainer state is available in rollout-only mode")
        return {
            "role": self.role,
            "rank": self._rank,
            "loaded_iteration": self._loaded_checkpoint_iteration,
            "weight_version": self.weight_updater.weight_version if self.role == "actor" else None,
            "load": str(self.args.load) if self.args.load is not None else None,
            "save": str(self.args.save) if self.args.save is not None else None,
            "ckpt_step": getattr(self.args, "ckpt_step", None),
            "finetune": bool(self.args.finetune),
            "no_load_optim": bool(self.args.no_load_optim),
            "no_load_rng": bool(self.args.no_load_rng),
        }

    @timer
    def save_model(
        self, rollout_id: int, force_sync: bool = False, full_state: bool = False,
        defer_finalize: bool = False,
    ) -> dict | None:
        if self.args.debug_rollout_only:
            if full_state:
                raise RuntimeError("Cannot save full trainer state in rollout-only mode")
            return
        if full_state and (self.optimizer is None or self.opt_param_scheduler is None):
            raise RuntimeError("A full checkpoint requires a native optimizer and scheduler")
        if getattr(self, "_pending_checkpoint_iteration", None) is not None:
            raise RuntimeError("Finalize the pending native checkpoint before starting another save")
        if defer_finalize and (not full_state or force_sync or not self.args.async_save):
            raise ValueError("Deferred checkpoint save requires full_state, async_save, and force_sync=False")

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            self.wake_up()

        saved_flags = []
        try:
            if self.args.async_save:
                from megatron.training.async_utils import maybe_finalize_async_save

                maybe_finalize_async_save(blocking=True)

            if full_state:
                from megatron.training.global_vars import get_args

                # Native saving reads global args; keep wrapper and global namespaces aligned.
                global_args = get_args()
                targets = [self.args] if global_args is self.args else [self.args, global_args]
                for target in targets:
                    for name in ("no_save_optim", "no_save_rng"):
                        saved_flags.append((target, name, getattr(target, name)))
                        setattr(target, name, False)

            if defer_finalize:
                from slime.backends.megatron_utils.async_checkpoint import isolated_native_async_save

                self._pending_checkpoint_iteration = rollout_id
                self._checkpoint_save_error = None
                with isolated_native_async_save(global_args):
                    save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
            else:
                save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

            if (force_sync or full_state) and self.args.async_save and not defer_finalize:
                maybe_finalize_async_save(blocking=True)

            if self.args.save_hf is not None and self.role == "actor":
                save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
        except BaseException as error:
            if defer_finalize:
                self._checkpoint_save_error = str(error)
            raise
        finally:
            for target, name, value in reversed(saved_flags):
                setattr(target, name, value)
            if self.args.offload_train:
                self.sleep()

        if full_state:
            result = {"role": self.role, "iteration": rollout_id, "full_state": True,
                      "completed": not defer_finalize}
            if defer_finalize:
                result["staged"] = True
            return result

    def finalize_checkpoint_save(self, rollout_id: int, blocking: bool = False) -> bool:
        """Finalize on all ranks, restoring only communicators for offloaded workers."""
        pending = getattr(self, "_pending_checkpoint_iteration", None)
        if pending is None:
            if getattr(self, "_completed_checkpoint_iteration", None) == rollout_id:
                return True
            raise RuntimeError("No pending native checkpoint for the requested iteration")
        if pending != rollout_id:
            raise RuntimeError("Native checkpoint finalization iteration does not match the pending save")
        if getattr(self, "_checkpoint_save_error", None) is not None:
            raise RuntimeError("Native checkpoint save previously failed: " + self._checkpoint_save_error)

        from megatron.training.async_utils import is_empty_async_queue, maybe_finalize_async_save

        offloaded = self.args.offload_train and self._is_offloaded
        try:
            with torch_memory_saver.disable() if offloaded else nullcontext():
                try:
                    if offloaded:
                        reload_process_groups()
                    maybe_finalize_async_save(blocking=blocking)
                    complete = is_empty_async_queue()
                    if complete:
                        maybe_finalize_async_save(blocking=True, terminate=True)
                finally:
                    if offloaded:
                        destroy_process_groups()
            if complete:
                self._completed_checkpoint_iteration = rollout_id
                self._pending_checkpoint_iteration = None
            return complete
        except BaseException as error:
            # Retain finalizer failures after Megatron removes the request from its queue.
            self._checkpoint_save_error = str(error)
            raise

    @timer
    def update_weights(self, restore_version: int | None = None) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            if restore_version is not None:
                raise RuntimeError("Cannot restore a published version in debug mode")
            return
        if restore_version is not None:
            if type(restore_version) is not int or restore_version < 0:
                raise ValueError("restore_version must be a non-negative integer")
            if (
                self.role != "actor" or self.args.colocate
                or getattr(self.args, "release_train", False)
                or self.args.update_weight_mode != "full"
                or self.args.update_weight_transport != "nccl"
            ):
                raise ValueError("Restoring a published version requires a resident full/NCCL actor group")
            if getattr(self, "_weights_published", False):
                raise RuntimeError("A checkpoint version may only be republished before the first weight update")
            if self.weight_updater.weight_version != restore_version:
                raise RuntimeError("The initialized actor version must match the checkpoint's published version")

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

        # Reload live CUDA weights for distributed/disk sync; colocated sync uses the CPU copy.
        reconnect_rollout_engines = self.args.offload_train and not self.args.colocate

        if restore_version is not None and not rollout_engines:
            raise RuntimeError("Cannot restore the checkpoint's published version without rollout engines")
        if not rollout_engines and not reconnect_rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines are running; skip weight update.")
            return

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            if restore_version is not None:
                # Offset the first NCCL publication to preserve the restored version.
                self.weight_updater.weight_version = restore_version - 1
            self.weight_updater.update_weights()
            if restore_version is not None and self.weight_updater.weight_version != restore_version:
                raise RuntimeError("Native weight publication did not preserve the checkpoint version")
            self._weights_published = True
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            if not self._keep_actor_resident:
                self.sleep()
            elif hasattr(self.weight_updater, "disconnect_rollout_engines"):
                # Free broadcast buffers; keep the model and training groups resident.
                self.weight_updater.disconnect_rollout_engines()
        elif self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            _build_checkpointing_context(self.args),
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag
