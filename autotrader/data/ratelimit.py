import threading
import time


class TokenBucket:
    """Thread-safe token bucket. `acquire()` blocks until a token is available."""

    def __init__(self, rate_per_sec: float, capacity: float | None = None):
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_sec)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)
