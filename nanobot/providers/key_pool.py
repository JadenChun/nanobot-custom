"""Request-local key ordering and provider-wide pacing for keyed transports."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class KeyPoolCoordinator:
    """Own ordering/cooldowns without mutating any transport's credential."""

    def __init__(self, slots: int, rate_limit: int = 0, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if slots < 1:
            raise ValueError('Key pool requires at least one credential slot')
        self._cooldowns = [0.0] * slots
        self._preferred: int | None = 0
        self._rotation = 0
        self._rate_limit = rate_limit
        self._clock = clock
        self._sleep = sleep
        self._admission_lock = asyncio.Lock()
        self._admitted: deque[float] = deque()

    def request_order(self) -> tuple[int, ...]:
        now = self._clock()
        ready = [slot for slot, expiry in enumerate(self._cooldowns) if expiry <= now]
        cooling = [slot for slot, expiry in enumerate(self._cooldowns) if expiry > now]
        result: list[int] = []
        preferred = self._preferred
        if preferred is not None and preferred in ready:
            result.append(preferred)
            ready.remove(preferred)
        if len(ready) > 1:
            offset = self._rotation % len(ready)
            self._rotation = (self._rotation + 1) % len(self._cooldowns)
            ready = ready[offset:] + ready[:offset]
        cooling.sort(key=lambda slot: (self._cooldowns[slot], slot))
        return tuple([*result, *ready, *cooling])

    def succeeded(self, slot: int) -> None:
        self._preferred = slot
        self._cooldowns[slot] = 0.0

    def failed(self, slot: int, cooldown_seconds: float) -> None:
        self._cooldowns[slot] = max(self._cooldowns[slot], self._clock() + cooldown_seconds)

    async def admit(self) -> None:
        """Reserve one provider-wide request slot; recheck after every wake."""
        if self._rate_limit <= 0:
            return
        while True:
            async with self._admission_lock:
                now = self._clock()
                while self._admitted and self._admitted[0] <= now - 60.0:
                    self._admitted.popleft()
                if len(self._admitted) < self._rate_limit:
                    self._admitted.append(now)
                    return
                delay = self._admitted[0] + 61.0 - now
            # No lock is held across waiting or a provider request. Cancelled
            # waiters have not reserved a slot and cannot consume capacity.
            await self._sleep(max(0.0, delay))
