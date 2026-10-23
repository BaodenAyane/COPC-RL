"""Single-owner E5/FAISS backend and bounded asynchronous micro-batching."""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import time

class BusyError(Exception):
    pass

class Corpus:
    def __init__(self, root, expected_rows):
        import numpy as np
        self.offsets = np.memmap(Path(root) / "corpus.offsets.u64", mode="r", dtype="<u8")
        self.path = Path(root) / "wiki-18.jsonl"
        if len(self.offsets) != expected_rows + 1 or int(self.offsets[-1]) != self.path.stat().st_size:
            raise ValueError("Corpus offsets do not match corpus/index")
        self.fd = os.open(self.path, os.O_RDONLY)
        self.cache = OrderedDict()

    def get(self, index):
        if index < 0 or index >= len(self.offsets) - 1:
            raise ValueError("FAISS returned an invalid document row")
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start, end = int(self.offsets[index]), int(self.offsets[index+1])
        doc = json.loads(os.pread(self.fd, end-start, start))
        if not isinstance(doc.get("contents"), str):
            raise ValueError("Corpus document has no text contents")
        self.cache[index] = doc
        if len(self.cache) > 10000:
            self.cache.popitem(last=False)
        return doc

    def close(self):
        os.close(self.fd)
        self.offsets._mmap.close()

class E5Faiss:
    def __init__(self, root, device="cpu", index_fp16=True, cpu_threads=8, hnsw_ef_search=128):
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.root = Path(root)
        self.assets = json.loads((self.root / "assets.ready.json").read_text())
        info = self.assets["index"]
        index_file = info.get("file")
        if info.get("index_type") != "hnsw64" or not index_file:
            raise ValueError("CPU retrieval requires a verified HNSW64 index manifest")
        if (self.root / index_file).stat().st_size != info["size"]:
            raise ValueError("Index size differs from verified manifest")
        # Bound encoder threads to leave CPU capacity for FAISS.
        encoder_cpu_threads = min(cpu_threads, 16)
        torch.set_num_threads(encoder_cpu_threads)
        faiss.omp_set_num_threads(cpu_threads)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("GPU service requires visible CUDA GPU and GPU-enabled FAISS; no silent CPU fallback")
            torch.cuda.set_device(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            required = info["ntotal"] * info["dimension"] * (2 if index_fp16 else 4) + 6 * 1024**3
            if free < required:
                raise RuntimeError(f"Insufficient free GPU memory: need approximately {required / 1024**3:.1f} GiB")
        print(f"[retrieval startup] E5 model load started: device={self.device}", flush=True)
        model_load_started = time.monotonic()
        self.model = AutoModel.from_pretrained(self.root / "e5-base-v2", local_files_only=True,
                                                trust_remote_code=False, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "e5-base-v2",
                                                       local_files_only=True, trust_remote_code=False)
        self.model.eval().to(self.device)
        if self.device.type == "cuda":
            self.model.half()  # Same query encoder precision as Search-R1's retrieval launcher.
        print(f"[retrieval startup] E5 model load completed in {time.monotonic() - model_load_started:.2f}s", flush=True)
        print(f"[retrieval startup] FAISS read_index started: size={info['size'] / 1024**3:.2f} GiB", flush=True)
        index_load_started = time.monotonic()
        cpu_index = faiss.read_index(str(self.root / index_file))
        print(f"[retrieval startup] FAISS read_index completed in {time.monotonic() - index_load_started:.2f}s", flush=True)
        if cpu_index.d != 768 or cpu_index.ntotal != self.assets["corpus"]["rows"] or cpu_index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("E5 index/corpus dimension, row count or metric mismatch")
        if not hasattr(cpu_index, "hnsw"):
            raise ValueError("Configured CPU index is not a FAISS HNSW index")
        cpu_index.hnsw.efSearch = hnsw_ef_search
        self.corpus = Corpus(self.root, cpu_index.ntotal)
        if self.device.type == "cuda":
            self.resources = faiss.StandardGpuResources()
            self.resources.setTempMemory(512 * 1024**2)
            options = faiss.GpuClonerOptions()
            options.useFloat16 = index_fp16
            self.index = faiss.index_cpu_to_gpu(self.resources, self.device.index or 0, cpu_index, options)
            del cpu_index
            gc.collect()
        else:
            self.index = cpu_index
        self.info = dict(model="intfloat/e5-base-v2", dimension=768, corpus_rows=self.index.ntotal,
                         index_sha256=info["sha256"], index_type=info["index_type"],
                         hnsw_ef_search=hnsw_ef_search, device=str(self.device), index_fp16=index_fp16 and self.device.type == "cuda",
                         sources=self.assets["sources"], faiss_version=faiss.__version__,
                         torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                         faiss_cpu_threads=cpu_threads, encoder_cpu_threads=encoder_cpu_threads)

    def search(self, queries, topk):
        import numpy as np
        import torch
        inputs = self.tokenizer(["query: " + query for query in queries], max_length=256,
                                padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][...,None].bool(), 0.0)
            vectors = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[...,None]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            vectors = np.ascontiguousarray(vectors.float().cpu().numpy())
        scores, indices = self.index.search(vectors, topk)
        result = []
        for row, values in zip(indices, scores):
            result.append([{"document": self.corpus.get(int(i)), "score": float(score)}
                           for i, score in zip(row, values) if int(i) >= 0 and math.isfinite(float(score))])
        if any(len(row) != topk for row in result):
            raise RuntimeError("Index did not return the requested number of finite results")
        return result

    def close(self):
        self.corpus.close()

class BatchEngine:
    def __init__(self, factory, max_batch=128, max_wait_ms=10, queue_size=4096,
                 cache_size=50000, max_topk=10):
        if min(max_batch, queue_size, max_topk) <= 0 or max_wait_ms < 0 or cache_size < 0:
            raise ValueError("Invalid batch, queue or cache limits")
        self.factory = factory
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.max_topk = max_topk
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-owner")
        self.backend = None
        self.ready = False
        self.task = None
        self.inflight = []
        self.stats = dict(queries=0, cache_hits=0, batches=0, backend_queries=0,
                          overloads=0, failures=0, backend_seconds=0.0)

    async def start(self):
        loop = asyncio.get_running_loop()
        self.backend = await loop.run_in_executor(self.executor, self.factory)
        print("[retrieval startup] Warmup retrieval query started: queries=1 topk=3", flush=True)
        warmup_started = time.monotonic()
        await loop.run_in_executor(self.executor, self.backend.search, ["Who wrote Pride and Prejudice?"], 3)
        print(f"[retrieval startup] Warmup retrieval query completed in {time.monotonic() - warmup_started:.2f}s", flush=True)
        self.ready = True
        self.task = asyncio.create_task(self._run())

    async def submit(self, queries, topk, timeout=120):
        if not self.ready:
            raise BusyError("Retriever not ready")
        if topk < 1 or topk > self.max_topk:
            raise ValueError("Invalid topk")
        misses = sum(query not in self.cache for query in queries)
        if self.queue.qsize() + misses > self.queue.maxsize:
            self.stats["overloads"] += 1
            raise BusyError("Retriever queue full; retry later")
        loop = asyncio.get_running_loop()
        futures = []
        for query in queries:
            self.stats["queries"] += 1
            future = loop.create_future()
            if query in self.cache:
                self.stats["cache_hits"] += 1
                self.cache.move_to_end(query)
                future.set_result(self.cache[query])
            else:
                self.queue.put_nowait((query, future))
            futures.append(future)
        try:
            rows = await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
            return [row[:topk] for row in rows]
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            self.inflight = batch
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
            active = [(q, f) for q, f in batch if not f.done()]
            missing = list(dict.fromkeys(q for q, f in active if q not in self.cache))
            try:
                values = {q: self.cache[q] for q, f in active if q in self.cache}
                if missing:
                    start = time.monotonic()
                    rows = await loop.run_in_executor(self.executor, self.backend.search, missing, self.max_topk)
                    if len(rows) != len(missing):
                        raise RuntimeError("Backend returned the wrong number of query results")
                    self.stats["backend_seconds"] += time.monotonic() - start
                    self.stats["batches"] += 1
                    self.stats["backend_queries"] += len(missing)
                    values.update(zip(missing, rows))
                    for query, row in zip(missing, rows):
                        if self.cache_size:
                            self.cache[query] = row
                            if len(self.cache) > self.cache_size:
                                self.cache.popitem(last=False)
                for query, future in active:
                    if not future.done():
                        future.set_result(values[query])
            except asyncio.CancelledError:
                for query, future in active:
                    if not future.done():
                        future.set_exception(BusyError("Retriever shutting down"))
                raise
            except Exception:
                self.stats["failures"] += 1
                for query, future in active:
                    if not future.done():
                        future.set_exception(RuntimeError("Retrieval backend failed; inspect service log"))
                import logging
                logging.exception("Retrieval batch failed")
            finally:
                for item in batch:
                    self.queue.task_done()
                self.inflight = []

    async def close(self):
        self.ready = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        # Cancellation may happen while collecting a micro-batch, before its try/finally.
        for _, future in self.inflight:
            if not future.done():
                future.set_exception(BusyError("Retriever shutting down"))
            self.queue.task_done()
        self.inflight = []
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if not future.done():
                future.set_exception(BusyError("Retriever shutting down"))
            self.queue.task_done()
        if self.backend:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
        self.executor.shutdown(wait=True)

