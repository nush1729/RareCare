"""Session store, rate limiting and security headers.

Both stores are in-process: correct for a single-worker Hugging Face Space. A
multi-worker deployment must move them to Redis (same interface).
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict, deque


class TTLSessionStore[T]:
    def __init__(self, ttl_seconds: int, max_items: int):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, value: T) -> str:
        sid = secrets.token_urlsafe(24)
        with self._lock:
            self._evict()
            self._data[sid] = (time.monotonic(), value)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
        return sid

    def get(self, sid: str) -> T | None:
        with self._lock:
            self._evict()
            item = self._data.get(sid)
            if item is None:
                return None
            self._data[sid] = (time.monotonic(), item[1])
            self._data.move_to_end(sid)
            return item[1]

    def delete(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)

    def __len__(self) -> int:
        return len(self._data)

    def _evict(self) -> None:
        cutoff = time.monotonic() - self.ttl
        while self._data:
            _sid, (ts, _) = next(iter(self._data.items()))
            if ts >= cutoff:
                break
            self._data.popitem(last=False)


class RateLimiter:
    """Sliding-window limiter keyed by client id."""

    def __init__(self, per_minute: int, max_keys: int = 50_000):
        self.per_minute = per_minute
        self.max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = deque()
                self._hits[key] = q
                if len(self._hits) > self.max_keys:
                    self._hits.popitem(last=False)
            while q and q[0] <= now - 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; "
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}
