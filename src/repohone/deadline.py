"""One process-local deadline for a host hook.

Every blocking primitive consults this budget.  The reserve belongs to the
caller so capture can stop attempting the primary write while there is still
time to append a refusal before the host terminates the hook.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_CURRENT: ContextVar[Optional[float]] = ContextVar("repohone_deadline", default=None)


@contextmanager
def budget(seconds: float, reserve: float = 0.5) -> Iterator[None]:
    usable = max(0.0, float(seconds) - max(0.0, float(reserve)))
    token = _CURRENT.set(time.monotonic() + usable)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> Optional[float]:
    return _CURRENT.get()


def until(maximum_wait: float) -> float:
    local = time.monotonic() + max(0.0, float(maximum_wait))
    inherited = current()
    return min(local, inherited) if inherited is not None else local


def remaining(maximum_wait: float) -> float:
    return max(0.0, until(maximum_wait) - time.monotonic())


def expired() -> bool:
    inherited = current()
    return inherited is not None and time.monotonic() >= inherited
