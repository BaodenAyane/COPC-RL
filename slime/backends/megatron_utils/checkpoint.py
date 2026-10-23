# Modified for COPC.
# PyTorch-derived portions: see the license header in slime/utils/distributed_utils.py.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Megatron-LM portions (BSD-3-Clause):
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
import re
from pathlib import Path

from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

try:
    # Skip expensive overlap validation; shard metadata must already be valid.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """Construct from trusted metadata without cross-rank validation."""
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    explicit_step = getattr(args, "ckpt_step", None)
    pin_iteration = bool(getattr(args, "async_resume", None)) and explicit_step is not None
    if pin_iteration:
        if type(explicit_step) is not int or explicit_step < 0:
            raise ValueError("Full async resume requires a non-negative integer checkpoint step")
        tracker = (Path(load_path) / "latest_checkpointed_iteration.txt").resolve()
        if not tracker.is_file():
            raise ValueError("Full async resume requires a native checkpoint tracker: " + str(tracker))

    if _is_megatron_checkpoint(load_path):
        # Full resume must request optimizer/RNG shards even when later saves omit them.
        save_flags = (
            {name: getattr(args, name) for name in ("no_save_optim", "no_save_rng")}
            if getattr(args, "async_resume", None)
            else {}
        )
        native_checkpointing = None
        original_read_metadata = None
        try:
            for name in save_flags:
                setattr(args, name, False)
            if pin_iteration:
                from megatron.training import checkpointing as native_checkpointing

                original_read_metadata = native_checkpointing.read_metadata

                def read_pinned_metadata(filename, *positional, **kwargs):
                    # Pin iteration zero locally without rewriting the shared checkpoint tracker.
                    if Path(filename).resolve() == tracker:
                        return explicit_step, False
                    return original_read_metadata(filename, *positional, **kwargs)

                native_checkpointing.read_metadata = read_pinned_metadata
            return _load_checkpoint_megatron(
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
        finally:
            if original_read_metadata is not None:
                native_checkpointing.read_metadata = original_read_metadata
            for name, value in save_flags.items():
                setattr(args, name, value)
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    # Converted torch_dist checkpoints require the native Megatron loader.
    return (p / "latest_checkpointed_iteration.txt").is_file() or bool(re.fullmatch(r"iter_\d{7}", p.name))


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")
    from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights

    load_hf_weights(args, ddp_model, load_path)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # Hugging Face weights initialize a fresh training run at iteration zero.
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
