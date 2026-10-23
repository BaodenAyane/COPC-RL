"""Bounded asynchronous client for the Search-R1 retrieval service."""

from __future__ import annotations

import asyncio
import math
import threading
import weakref
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

aiohttp = None


def _http_module():
    global aiohttp
    if aiohttp is None:
        try:
            import aiohttp as http
        except ImportError as exc:
            raise RetrievalError("SearchClient requires aiohttp in the rollout Python environment") from exc
        aiohttp = http
    return aiohttp


class RetrievalError(RuntimeError):
    """A retrieval infrastructure/protocol failure, never an empty search result."""

    def __init__(self, message: str, *, attempts: int = 0, status_code: int | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class _AttemptFailure(Exception):
    def __init__(self, message, *, retryable, status_code=None, retry_after=None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


class _LoopState:
    def __init__(self, concurrency):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = None
        self.users = 0


class SearchClient:
    """Retrieve with per-loop connection pools; max_retries includes the first attempt.

    Timeouts start after admission; cancellation propagates and the last caller closes the pool."""

    _RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, url, topk=3, timeout=65, max_retries=3, concurrency=120):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/retrieve"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Search URL must be an HTTP(S) /retrieve endpoint without credentials or query parameters")
        for name, value, lower, upper in (
            ("topk", topk, 1, 10),
            ("max_retries", max_retries, 1, None),
            ("concurrency", concurrency, 1, None),
        ):
            if type(value) is not int or value < lower or (upper is not None and value > upper):
                raise ValueError(f"Invalid {name}: {value!r}")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency = concurrency
        self._states = weakref.WeakKeyDictionary()
        self._states_lock = threading.Lock()

    def _state(self):
        loop = asyncio.get_running_loop()
        with self._states_lock:
            state = self._states.get(loop)
            if state is None:
                state = _LoopState(self.concurrency)
                self._states[loop] = state
            return state

    def _semaphore(self):
        return self._state().semaphore

    @asynccontextmanager
    async def _session(self, http):
        state = self._state()
        if state.session is None:
            state.session = http.ClientSession(
                connector=http.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency),
                timeout=http.ClientTimeout(total=self.timeout), trust_env=False,
            )
        session = state.session
        state.users += 1
        try:
            yield session
        finally:
            state.users -= 1
            if state.users == 0:
                # Detach before awaiting close: a new caller may enter meanwhile.
                state.session = None
                await session.close()

    @staticmethod
    def _retry_after(value):
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        return min(5.0, max(0.0, seconds)) if math.isfinite(seconds) else None

    def _documents(self, payload, query_count):
        rows = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != query_count:
            raise _AttemptFailure("Retrieval response has the wrong number of query rows", retryable=False)
        for row in rows:
            if not isinstance(row, list) or len(row) != self.topk:
                raise _AttemptFailure("Retrieval response did not return the requested topk", retryable=False)
            if any(not isinstance(doc, dict) or not isinstance(doc.get("contents"), str) for doc in row):
                raise _AttemptFailure("Retrieval documents must directly contain string contents", retryable=False)
        return rows

    async def _request(self, queries, session):
        async with session.post(
            self.url,
            json={"queries": queries, "topk": self.topk, "return_scores": False},
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                # Drain errors so retries can reuse the connection.
                await response.read()
                raise _AttemptFailure(
                    f"Retrieval service returned HTTP {response.status}",
                    retryable=response.status in self._RETRYABLE_HTTP,
                    status_code=response.status,
                    retry_after=self._retry_after(response.headers.get("Retry-After")),
                )
            try:
                payload = await response.json(content_type=None)
            except ValueError as exc:
                raise _AttemptFailure("Retrieval response is not valid JSON", retryable=False) from exc
            return self._documents(payload, len(queries))

    async def retrieve(self, queries: list[str]) -> list[list[dict]]:
        if not isinstance(queries, list) or not 1 <= len(queries) <= 32:
            raise ValueError("queries must be a list containing 1 to 32 strings")
        if any(not isinstance(query, str) or not query.strip() or len(query) > 2048 for query in queries):
            raise ValueError("Each query must be nonblank and at most 2048 characters")
        queries = [query.strip() for query in queries]
        http = _http_module()
        semaphore = self._semaphore()
        async with self._session(http) as session:
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with semaphore:
                        return await asyncio.wait_for(self._request(queries, session), timeout=self.timeout)
                except _AttemptFailure as exc:
                    failure = exc
                except (http.ClientError, asyncio.TimeoutError) as exc:
                    failure = _AttemptFailure(f"Retrieval transport failed ({type(exc).__name__})", retryable=True)
                    failure.__cause__ = exc
                if not failure.retryable or attempt == self.max_retries:
                    raise RetrievalError(
                        f"{failure}; attempts={attempt}/{self.max_retries}",
                        attempts=attempt,
                        status_code=failure.status_code,
                    ) from failure
                delay = failure.retry_after
                if delay is None:
                    delay = min(5.0, 0.5 * 2 ** min(attempt - 1, 4))
                await asyncio.sleep(delay)
        raise AssertionError("Unreachable: positive attempt limit is validated in the constructor")
