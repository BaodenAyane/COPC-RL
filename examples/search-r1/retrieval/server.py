"""Search-R1 retrieval HTTP service."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import fcntl
import ipaddress
import json
import logging
import os
from pathlib import Path
import socket
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictBool, StrictInt, field_validator
import uvicorn

from core import BatchEngine, BusyError, E5Faiss

class HTTPConcurrencyMetrics(logging.Filter):
    """Expose the actual Uvicorn guard counters, separate from engine overloads."""

    def __init__(self, server):
        super().__init__()
        self.server = server
        self.rejections = 0

    def snapshot(self):
        state = self.server.server_state
        return dict(http_connections=len(state.connections), http_tasks=len(state.tasks),
                    http_concurrency_limit=self.server.config.limit_concurrency,
                    http_concurrency_rejections=self.rejections)

    def filter(self, record):
        if record.getMessage() == "Exceeded concurrency limit.":
            self.rejections += 1
            stats = self.snapshot()
            record.msg = ("Exceeded concurrency limit. connections=%d tasks=%d limit=%d "
                          "rejections=%d")
            record.args = (stats["http_connections"], stats["http_tasks"],
                           stats["http_concurrency_limit"], self.rejections)
        return True


class QueryRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=32)
    topk: StrictInt = Field(default=3, ge=1, le=10)
    return_scores: StrictBool = False

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries):
        if any(not q.strip() or len(q) > 2048 for q in queries):
            raise ValueError("Queries must be nonblank and at most 2048 characters")
        return [q.strip() for q in queries]

def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)

def discover_host():
    for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        host = result[4][0]
        addr = ipaddress.ip_address(host)
        if addr.is_private and not addr.is_loopback:
            return host
    raise RuntimeError("Set SEARCH_ADVERTISE_HOST to an address reachable from RL workers")

def create_app(engine, manifest=None, advertise_host=None, port=8000, request_timeout=60):
    instance = uuid.uuid4().hex
    lock = None
    heartbeat = None
    started = time.time()
    manifest = Path(manifest) if manifest else None

    def publish(status):
        if manifest:
            atomic_json(manifest, dict(schema_version=1, instance_id=instance, status=status,
                        updated_at=time.time(), started_at=started, pid=os.getpid(),
                        retrieval_service_url=f"http://{advertise_host}:{port}/retrieve",
                        health_url=f"http://{advertise_host}:{port}/health",
                        **(engine.backend.info if engine.backend else {})))

    async def beat():
        while True:
            publish("ready")
            await asyncio.sleep(10)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal lock, heartbeat
        if manifest:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            lock = manifest.with_suffix(".lock").open("a")
            # Prevent two jobs from replacing each other's endpoint discovery file.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish("starting")
        try:
            await engine.start()
            publish("ready")
            heartbeat = asyncio.create_task(beat())
            yield
        finally:
            if heartbeat:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
            publish("stopped")
            await engine.close()
            if lock:
                lock.close()

    app = FastAPI(title="Offline Wikipedia E5 retrieval", lifespan=lifespan)
    app.state.http_metrics = lambda: {}

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        # Bound chunked and ordinary JSON bodies before FastAPI parses them.
        from starlette.responses import JSONResponse
        if request.url.path == "/retrieve":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 256 * 1024:
                    return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            request._body = bytes(body)
        return await call_next(request)

    @app.get("/health")
    async def health():
        if not engine.ready:
            raise HTTPException(503, "Retriever not ready")
        return dict(status="ready", instance_id=instance, uptime_seconds=time.time()-started,
                    **engine.backend.info)

    @app.get("/metrics")
    async def metrics():
        return dict(**engine.stats, queue_queries=engine.queue.qsize(), cache_queries=len(engine.cache),
                    ready=engine.ready, uptime_seconds=time.time()-started, **app.state.http_metrics())

    @app.post("/retrieve")
    async def retrieve(request: QueryRequest):
        try:
            results = await engine.submit(request.queries, request.topk, timeout=request_timeout)
        except BusyError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "1"})
        except asyncio.TimeoutError:
            raise HTTPException(504, "Retrieval queue deadline exceeded")
        except Exception:
            logging.exception("Request failed")
            raise HTTPException(500, "Retrieval backend failure")
        if not request.return_scores:
            results = [[hit["document"] for hit in row] for row in results]
        return {"result": results}

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index-fp32", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--cache-size", type=int, default=50000)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--advertise-host")
    args = parser.parse_args()
    listener = None
    port = args.port
    if port == 0:
        # Bind before publishing to prevent a free-port race.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, 0))
        listener.listen(2048)
        port = listener.getsockname()[1]
    advertise = args.advertise_host
    if args.manifest and not advertise:
        advertise = discover_host()
    engine = BatchEngine(lambda: E5Faiss(args.assets, args.device, not args.index_fp32, args.cpu_threads,
                                         args.hnsw_ef_search),
                         max_batch=args.batch_size, max_wait_ms=args.batch_wait_ms,
                         queue_size=args.queue_size, cache_size=args.cache_size)
    app = create_app(engine, args.manifest, advertise, port, args.request_timeout)
    # Multiple workers would replicate the entire index/model and defeat batching.
    config = uvicorn.Config(app, host=args.host, port=port, workers=1, access_log=False,
                            limit_concurrency=256, timeout_keep_alive=30,
                            timeout_graceful_shutdown=30)
    server = uvicorn.Server(config)
    http_metrics = HTTPConcurrencyMetrics(server)
    app.state.http_metrics = http_metrics.snapshot
    error_logger = logging.getLogger("uvicorn.error")
    error_logger.addFilter(http_metrics)
    try:
        server.run(sockets=[listener] if listener else None)
    finally:
        error_logger.removeFilter(http_metrics)

if __name__ == "__main__":
    main()
