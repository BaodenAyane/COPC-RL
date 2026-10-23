"""Park requests between external actions without cancelling in-flight work."""
from __future__ import annotations

import asyncio
import contextvars
import threading
from contextlib import contextmanager

_controller = contextvars.ContextVar("rollout_checkpoint_controller", default=None)


class CheckpointBarrier:
    def __init__(self, requested: threading.Event, release_pending=None):
        self.requested = requested
        self.release_pending = release_pending
        self.samples = {}

    def register(self, sample, *, done=False):
        key = id(sample)
        if key in self.samples:
            raise RuntimeError("Sample is registered in multiple in-flight groups")
        # Idle generators must reach a safe point before external I/O.
        self.samples[key] = "done" if done else "idle"

    def unregister(self, sample):
        self.samples.pop(id(sample), None)

    def mark_done(self, sample):
        self.samples[id(sample)] = "done"

    def is_done(self, sample):
        return self.samples.get(id(sample)) == "done"

    def is_quiescent(self):
        return all(state in {"idle", "parked", "done"} for state in self.samples.values())

    @contextmanager
    def bind(self):
        token = _controller.set(self)
        try:
            yield
        finally:
            _controller.reset(token)

    async def safe_point(self, sample):
        key = id(sample)
        if key not in self.samples:
            raise RuntimeError("Unregistered sample reached the async checkpoint barrier")
        while self.requested.is_set() or (self.release_pending is not None and self.release_pending.is_set()):
            self.samples[key] = "parked"
            await asyncio.sleep(0.01)
        self.samples[key] = "busy"


async def checkpoint_safe_point(sample):
    controller = _controller.get()
    if controller is not None:
        await controller.safe_point(sample)
