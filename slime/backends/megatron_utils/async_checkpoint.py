
from contextlib import contextmanager
from io import BytesIO


def isolate_write_buckets(write_buckets):
    """Copy writer payloads while training is quiescent."""
    result = []
    for filename, storage_key, (byte_data, tensor_data) in write_buckets:
        isolated_bytes = []
        for item, value in byte_data:
            if not isinstance(value, BytesIO):
                raise RuntimeError("Deferred checkpoint writer requires serialized BytesIO payloads")
            isolated_bytes.append((item, BytesIO(value.getvalue())))
        isolated_tensors = [
            (item, tensor.detach().to(device="cpu", copy=True, non_blocking=False))
            for item, tensor in tensor_data
        ]
        result.append((filename, storage_key, (isolated_bytes, isolated_tensors)))
    return result


@contextmanager
def isolated_native_async_save(args):
    if not getattr(args, "async_save", False) or getattr(args, "ckpt_format", None) != "torch_dist":
        raise RuntimeError("Deferred full checkpoints require --async-save with --ckpt-format torch_dist")
    try:
        from megatron.core.dist_checkpointing.strategies import torch as native_strategy
        from megatron.training.async_utils import is_empty_async_queue, maybe_finalize_async_save
    except ImportError as error:
        raise RuntimeError("Megatron lacks the native asynchronous checkpoint capability") from error

    writer_cls = getattr(native_strategy, "FileSystemWriterAsync", None)
    strategy_cls = getattr(native_strategy, "TorchDistSaveShardedStrategy", None)
    prepare = getattr(writer_cls, "prepare_write_data", None)
    callbacks = getattr(strategy_cls, "_get_save_and_finalize_callbacks", None)
    if not all(callable(function) for function in
               (prepare, callbacks, is_empty_async_queue, maybe_finalize_async_save)):
        raise RuntimeError("Unsupported Megatron asynchronous writer/callback API")

    prepared = []
    checked = []

    def prepare_isolated(writer, plan, planner):
        prepare(writer, plan, planner)
        if not isinstance(getattr(writer, "write_buckets", None), list):
            raise RuntimeError("Native checkpoint writer did not expose its staged write buckets")
        writer.write_buckets = isolate_write_buckets(writer.write_buckets)
        prepared.append(writer)

    def checked_callbacks(strategy, writer, save_state_dict_ret):
        # Offload recreates WORLD; finalizers must resolve its current group.
        if (not isinstance(save_state_dict_ret, (tuple, list)) or len(save_state_dict_ret) != 3
                or not hasattr(save_state_dict_ret[2], "group")
                or save_state_dict_ret[2].group is not None):
            raise RuntimeError("Deferred checkpoint finalization requires a dynamic WORLD process group")
        if not any(writer is item for item in prepared):
            raise RuntimeError("Native checkpoint writer bypassed immutable CPU snapshot staging")
        request = callbacks(strategy, writer, save_state_dict_ret)
        if not callable(getattr(request, "add_finalize_fn", None)):
            raise RuntimeError("Unsupported Megatron asynchronous request API")
        checked.append(writer)
        return request

    original_prepare = writer_cls.__dict__.get("prepare_write_data")
    original_callbacks = strategy_cls.__dict__.get("_get_save_and_finalize_callbacks")
    writer_cls.prepare_write_data = prepare_isolated
    strategy_cls._get_save_and_finalize_callbacks = checked_callbacks
    try:
        yield
        if not checked:
            raise RuntimeError("Native save did not use the supported asynchronous torch_dist writer")
    finally:
        if original_prepare is None:
            del writer_cls.prepare_write_data
        else:
            writer_cls.prepare_write_data = original_prepare
        if original_callbacks is None:
            del strategy_cls._get_save_and_finalize_callbacks
        else:
            strategy_cls._get_save_and_finalize_callbacks = original_callbacks
