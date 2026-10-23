# Modified for COPC.
"""Asynchronous rollout with a completion queue and checkpoint barriers."""

from __future__ import annotations

import asyncio
import atexit
import copy
import random
import sys
import uuid
import logging
import queue
import threading
import time
from collections import defaultdict, deque

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.checkpoint_barrier import CheckpointBarrier
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.metric_utils import compute_pass_rate
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
    "freeze_for_checkpoint",
    "restore_from_checkpoint",
    "resume_after_checkpoint",
]

logger = logging.getLogger("slime.rollout.fully_async")


_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if getattr(args, "_fully_async_shutdown_requested", False):
            raise RuntimeError("Fully-async generation was stopped for task cleanup")
        if _global_worker is not None and getattr(_global_worker, "_fatal_checkpoint_error", None) is not None:
            raise RuntimeError("Refusing to recreate a failed async checkpoint worker") from _global_worker._fatal_checkpoint_error
        if _global_worker is None or (
            _global_worker.worker_thread is not None and not _global_worker.worker_thread.is_alive()
        ):
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker(timeout: float = 5, require_stopped: bool = False) -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop(timeout=timeout, require_stopped=require_stopped)
            if not (_global_worker.worker_thread and _global_worker.worker_thread.is_alive()):
                _global_worker = None


atexit.register(_stop_global_worker)


def freeze_for_checkpoint(args, data_source, timeout=None):
    """Freeze admissions and external actions, returning an isolated full state."""
    if not getattr(args, "async_checkpoint", False):
        raise RuntimeError("Full async snapshots require --async-checkpoint")
    worker = _get_global_worker(args, data_source)
    return worker.freeze_for_checkpoint(
        timeout if timeout is not None else getattr(args, "async_checkpoint_timeout", 300.0)
    )


def restore_from_checkpoint(args, data_source, state):
    """Restore a trusted, already decoded snapshot without starting generation."""
    global _global_worker
    if not getattr(args, "async_checkpoint", False):
        raise RuntimeError("Async restore requires --async-checkpoint")
    with _worker_lock:
        if _global_worker is not None:
            raise RuntimeError("Cannot restore over an existing rollout worker")
        worker = AsyncRolloutWorker(
            args, data_source, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
        )
        worker.restore_checkpoint_state(state)
        _global_worker = worker


def resume_after_checkpoint():
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.resume_after_checkpoint()


class AsyncRolloutWorker:

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Capacity excludes in-flight and prefetched work; AReaL uses optimizer-step units.
        max_staleness = args.fully_async_max_staleness
        if getattr(args, "advantage_estimator", None) == "areal":
            from slime.rollout.areal_queue import get_areal_queue_capacity

            queue_maxsize, queue_desc = get_areal_queue_capacity(args)
        elif max_staleness < 0:
            queue_maxsize = 0
            queue_desc = "unbounded"
        else:
            # A zero maxsize means unbounded, so retain at least one batch.
            queue_batches = max(max_staleness - 1, 1)
            queue_maxsize = queue_batches * args.rollout_batch_size
            queue_desc = f"{queue_maxsize} (=max(staleness {max_staleness} - 1, 1) * rollout_batch_size {args.rollout_batch_size})"
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=queue_maxsize)
        logger.info(
            "fully-async worker: output_queue maxsize=%s; full-queue behavior=blocking event loop", queue_desc
        )
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self.checkpoint_enabled = bool(getattr(args, "async_checkpoint", False))
        if self.checkpoint_enabled:
            if getattr(args, "group_rm", False):
                raise ValueError("Async checkpoints currently require per-sample rewards")
            self._checkpoint_requested = threading.Event()
            self._checkpoint_frozen = threading.Event()
            self._checkpoint_release_pending = threading.Event()
            self._checkpoint_guard = threading.Lock()
            self._barrier = CheckpointBarrier(self._checkpoint_requested, self._checkpoint_release_pending)
            self._active_groups = {}
            self._pending_completions = deque()
            self._restored_groups = []
            self._gid_counter = 0
            # Terminal DAPO discards still own issued IDs in the snapshot ledger.
            self._filtered_indices = []
            self._snapshot = None
            self._checkpoint_generation = 0
            self._checkpoint_error = None
            self._fatal_checkpoint_error = None
            self._restored_rng = None


    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self, timeout: float = 5, require_stopped: bool = False) -> None:
        """Stop admissions and wait for in-flight cancellation and cleanup."""
        self.running = False
        if self.checkpoint_enabled and self._checkpoint_requested.is_set():
            self._checkpoint_error = RuntimeError("Async worker stopped during checkpoint")
            self._checkpoint_frozen.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        if require_stopped and self.worker_thread and self.worker_thread.is_alive():
            raise TimeoutError(f"Fully-async rollout cleanup did not finish within {timeout}s")

    def get_completed_groups(self, max_items: int | None = None) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            if max_items is not None and len(completed) >= max_items:
                break
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def record_filtered_group(self, group):
        """Record terminal discards without retaining or requeueing trajectories."""
        if self.checkpoint_enabled:
            # The driver waits for batch collection before freezing the worker.
            self._filtered_indices.extend(sample.index for sample in group)

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def freeze_for_checkpoint(self, timeout=300.0):
        if not self.checkpoint_enabled:
            raise RuntimeError("Worker was not constructed with async checkpoints")
        if timeout <= 0:
            raise ValueError("Checkpoint timeout must be positive")
        if self._fatal_checkpoint_error is not None or not self.running:
            raise RuntimeError("Cannot checkpoint a failed/stopped rollout worker") from self._fatal_checkpoint_error
        with self._checkpoint_guard:
            if self._checkpoint_requested.is_set():
                raise RuntimeError("Rollout checkpoint is already in progress")
            self._snapshot = None
            self._checkpoint_error = None
            self._checkpoint_frozen.clear()
            self._checkpoint_generation += 1
            self._checkpoint_requested.set()
            if not self._checkpoint_frozen.wait(timeout):
                self._checkpoint_release_pending.set()
                self._checkpoint_requested.clear()
                raise TimeoutError("Rollout requests did not reach safe points before checkpoint timeout")
            if self._checkpoint_error is not None:
                error = self._checkpoint_error
                self._checkpoint_release_pending.set()
                self._checkpoint_requested.clear()
                raise RuntimeError("Cannot create a consistent rollout checkpoint") from error
            return self._snapshot

    def resume_after_checkpoint(self):
        if not self.checkpoint_enabled or not self._checkpoint_requested.is_set():
            return
        self._snapshot = None
        self._checkpoint_frozen.clear()
        self._checkpoint_release_pending.set()
        self._checkpoint_requested.clear()
        self.start()

    @staticmethod
    def _capture_rng():
        state = {"python": random.getstate()}
        numpy = sys.modules.get("numpy")
        torch = sys.modules.get("torch")
        if numpy is not None:
            state["numpy"] = numpy.random.get_state()
        if torch is not None and hasattr(torch, "get_rng_state"):
            state["torch"] = torch.get_rng_state()
        return state

    @staticmethod
    def _restore_rng(state):
        random.setstate(state["python"])
        if "numpy" in state:
            import numpy
            numpy.random.set_state(state["numpy"])
        if "torch" in state:
            import torch
            torch.set_rng_state(state["torch"])

    @staticmethod
    def _validate_sample_partition(state):
        """Each issued sample belongs to exactly one persisted pipeline stage."""
        groups = [("ready", group) for _, group in state["ready"]]
        groups += [("pending", group) for _, group in state["pending_completions"]]
        groups += [("active", item["samples"]) for item in state["active"]]
        groups += [("retry", group) for group in state["data_source"].get("buffer", [])]
        seen = {}
        next_index = state["data_source"].get("sample_index")

        def register(index, stage):
            if type(index) is not int or index < 0:
                raise ValueError("Persisted rollout sample must have a nonnegative integer ID")
            if next_index is not None and index >= next_index:
                raise ValueError("Persisted sample ID exceeds the datasource cursor")
            if index in seen:
                raise ValueError("Sample ID %s appears in both %s and %s" % (index, seen[index], stage))
            seen[index] = stage

        for stage, group in groups:
            for sample in group:
                register(getattr(sample, "index", None), stage)
        for index in state.get("filtered_indices", []):
            register(index, "filtered")

    def _capture_checkpoint_state(self):
        if self._fatal_checkpoint_error is not None:
            raise RuntimeError("An in-flight group failed; refusing to checkpoint lost work") from self._fatal_checkpoint_error
        with self.output_queue.mutex:
            ready = list(self.output_queue.queue)
        active = [
            {"gid": gid, "samples": group,
             "completed": [self._barrier.is_done(sample) for sample in group]}
            for gid, group in self._active_groups.items()
        ]
        # Include restored groups still waiting behind saved completions.
        active.extend(self._restored_groups)
        snapshot = {
            "schema_version": 1,
            "queue_capacity": self.output_queue.maxsize,
            "concurrency": self.concurrency,
            "gid_counter": self._gid_counter,
            "filtered_indices": list(self._filtered_indices),
            "ready": ready,
            "pending_completions": list(self._pending_completions),
            "active": active,
            "data_source": self.data_buffer.checkpoint_state_dict(),
            "sampling_params": self.state.sampling_params,
            "rng": self._capture_rng(),
        }
        self._validate_sample_partition(snapshot)
        return copy.deepcopy(snapshot)

    def restore_checkpoint_state(self, state):
        if not self.checkpoint_enabled or self.worker_thread is not None:
            raise RuntimeError("Restore requires a fresh checkpoint-enabled worker")
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported async rollout checkpoint schema")
        if state["queue_capacity"] != self.output_queue.maxsize or state["concurrency"] != self.concurrency:
            raise ValueError("Rollout capacity/concurrency changed since checkpoint")
        if state["sampling_params"] != self.state.sampling_params:
            raise ValueError("Rollout sampling configuration changed since checkpoint")
        ready = copy.deepcopy(state["ready"])
        if self.output_queue.maxsize and len(ready) > self.output_queue.maxsize:
            raise ValueError("Checkpoint ready queue exceeds configured capacity")
        gids = [gid for gid, _ in ready]
        gids.extend(gid for gid, _ in state["pending_completions"])
        gids.extend(item["gid"] for item in state["active"])
        if len(set(gids)) != len(gids) or any(gid < 0 or gid >= state["gid_counter"] for gid in gids):
            raise ValueError("Invalid or duplicate group IDs in async checkpoint")
        for item in state["active"]:
            if len(item["samples"]) != len(item["completed"]):
                raise ValueError("Active group completion flags do not match its samples")
        self._validate_sample_partition(state)
        self.data_buffer.load_checkpoint_state_dict(state["data_source"])
        for item in ready:
            self.output_queue.put_nowait(item)
        self._pending_completions.extend(copy.deepcopy(state["pending_completions"]))
        self._restored_groups = copy.deepcopy(state["active"])
        self._gid_counter = state["gid_counter"]
        self._filtered_indices = list(state.get("filtered_indices", []))
        self._restored_rng = copy.deepcopy(state["rng"])
        # Restore RNG after initialization, just before the worker starts.
        self._checkpoint_requested.set()
        self._checkpoint_frozen.set()

    async def _generate_checkpoint_group(self, group):
        from slime.rollout.sglang_rollout import generate_and_rm

        async def one(sample, index):
            if self._barrier.is_done(sample):
                return sample
            sampling = self.state.sampling_params.copy()
            if getattr(self.args, "sglang_enable_deterministic_inference", False):
                sampling["sampling_seed"] = self.state.group_sampling_seeds[index]
            result = await generate_and_rm(self.args, sample, sampling, evaluation=False)
            if result is not sample:
                raise RuntimeError("Async checkpoint requires one in-place Sample per generator call")
            self._barrier.mark_done(sample)
            return result

        with self._barrier.bind():
            tasks = [asyncio.create_task(one(sample, index)) for index, sample in enumerate(group)]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    def _spawn_checkpoint_group(self, gid, group, completed=None):
        completed = completed or [False] * len(group)
        from slime.rollout.checkpoint_protocol import validate_checkpoint_sample

        for sample in group:
            validate_checkpoint_sample(self.args, sample)
        for sample, done in zip(group, completed):
            if sample.session_id is None:
                sample.session_id = str(uuid.uuid4())
            self._barrier.register(sample, done=done)
        self._active_groups[gid] = group
        task = asyncio.create_task(self._generate_checkpoint_group(group))
        task.add_done_callback(self._make_done_cb(gid))
        return task

    def _publish_checkpoint_result(self, item, *, front=False):
        # During a checkpoint, persist completions outside the blocked output queue.
        while self.running:
            if self._checkpoint_requested.is_set() or (not front and self._checkpoint_release_pending.is_set()):
                if front:
                    self._pending_completions.appendleft(item)
                else:
                    self._pending_completions.append(item)
                return
            try:
                self.output_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    async def _checkpoint_loop_tick(self, active_tasks):
        if self._checkpoint_requested.is_set():
            # Drain completed group callbacks before snapshotting to preserve retry order.
            finishing = any(
                all(self._barrier.is_done(sample) for sample in group)
                for group in self._active_groups.values()
            )
            if not self._checkpoint_frozen.is_set() and self._barrier.is_quiescent() and not finishing:
                generation = self._checkpoint_generation
                try:
                    snapshot = self._capture_checkpoint_state()
                    error = None
                except Exception as caught:
                    snapshot, error = None, caught
                if self._checkpoint_requested.is_set() and generation == self._checkpoint_generation:
                    self._snapshot, self._checkpoint_error = snapshot, error
                    self._checkpoint_frozen.set()
            await asyncio.sleep(0.01)
            return True
        # Drain saved completions before resuming parked or new work.
        while self._pending_completions and self.running:
            item = self._pending_completions.popleft()
            self._publish_checkpoint_result(item, front=True)
            if self._checkpoint_requested.is_set():
                return True
        for item in self._restored_groups:
            active_tasks.add(self._spawn_checkpoint_group(item["gid"], item["samples"], item["completed"]))
        self._restored_groups.clear()
        self._checkpoint_release_pending.clear()
        return False


    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        if self.checkpoint_enabled and self._restored_rng is not None:
            self._restore_rng(self._restored_rng)
            self._restored_rng = None
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        try:
            while self.running:
                try:
                    # Completion callbacks publish before this loop reaps tasks.
                    done = {t for t in active_tasks if t.done()}
                    for task in done:
                        if task.cancelled():
                            continue
                        try:
                            task.result()
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                    if self.checkpoint_enabled:
                        if self._fatal_checkpoint_error is not None:
                            raise RuntimeError("Async rollout lost an in-flight group") from self._fatal_checkpoint_error
                        if await self._checkpoint_loop_tick(active_tasks):
                            continue

                    while len(active_tasks) < max_concurrent and self.running:
                        if self.checkpoint_enabled and self._checkpoint_requested.is_set():
                            break
                        groups = self.data_buffer.get_samples(1)
                        if not groups:
                            break
                        for group in groups:
                            if self.checkpoint_enabled:
                                task = self._spawn_checkpoint_group(self._gid_counter, group)
                                self._gid_counter += 1
                                active_tasks.add(task)
                                continue
                            task = asyncio.create_task(
                                generate_and_rm_group(
                                    self.args,
                                    group,
                                    sampling_params=self.state.sampling_params.copy(),
                                    evaluation=False,
                                )
                            )
                            task.add_done_callback(self._make_done_cb(gid_counter))
                            gid_counter += 1
                            active_tasks.add(task)

                    await asyncio.sleep(1)
                except Exception as e:  # noqa: BLE001
                    logger.exception("fully-async loop iteration error: %s", e)
                    if self.checkpoint_enabled:
                        self._fatal_checkpoint_error = e
                        self.running = False
                        if self._checkpoint_requested.is_set():
                            self._checkpoint_error = e
                            self._checkpoint_frozen.set()
                        break
                    await asyncio.sleep(1)
        finally:
            # Cancel on shutdown without waiting for the trainer to drain results.
            self.running = False
            for task in active_tasks:
                task.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            if done_task.cancelled():
                return
            try:
                result = done_task.result()
            except Exception as error:  # noqa: BLE001
                if self.checkpoint_enabled:
                    self._fatal_checkpoint_error = error
                    group = self._active_groups.pop(gid, [])
                    for sample in group:
                        self._barrier.unregister(sample)
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            if not self.running:
                return
            if self.checkpoint_enabled:
                group = self._active_groups.pop(gid)
                for sample in group:
                    self._barrier.unregister(sample)
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            if self.checkpoint_enabled:
                self._publish_checkpoint_result((gid, result))
                return
            while self.running:
                try:
                    # The timeout permits shutdown while the queue is full.
                    self.output_queue.put((gid, result), timeout=0.1)
                    return
                except queue.Full:
                    continue

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )
    drop_reasons = defaultdict(int)
    # Count rejected DAPO candidates in passrate; retain rewards only.
    passrate_rewards = [] if dynamic_filter is not None and getattr(args, "log_passrate", False) else None

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        if not worker.running:
            raise RuntimeError("Fully-async worker stopped before batch collection completed")
        if worker.checkpoint_enabled and worker._fatal_checkpoint_error is not None:
            raise RuntimeError("Fully async rollout failed; refusing to skip unfinished samples") from worker._fatal_checkpoint_error
        drained = 0
        for gid, group in worker.get_completed_groups(max_items=target - len(collected)):
            if passrate_rewards is not None:
                passrate_rewards.extend(sample.get_reward_value(args) for sample in group)
            # Never requeue filtered completions: unchanged rewards would cause an infinite loop.
            if dynamic_filter is not None:
                f_out = call_dynamic_filter(dynamic_filter, args, group)
                if not f_out.keep:
                    worker.record_filtered_group(group)
                    drop_reasons[f_out.reason or "unspecified"] += 1
                    continue
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, dropped=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                sum(drop_reasons.values()),
                now - started,
            )
            last_log = now

    if not worker.running:
        raise RuntimeError("Fully-async worker stopped before batch collection completed")

    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    dropped_total = sum(drop_reasons.values())
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d, dropped=%d%s",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
        dropped_total,
        f" drop_reasons={dict(drop_reasons)}" if drop_reasons else "",
    )
    metrics = None
    if passrate_rewards is not None:
        metrics = {
            f"passrate/{key}": value
            for key, value in compute_pass_rate(passrate_rewards, args.n_samples_per_prompt).items()
        }
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
