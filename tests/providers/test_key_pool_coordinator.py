import asyncio

from nanobot.providers.key_pool import KeyPoolCoordinator


def test_order_snapshot_prefers_winner_and_keeps_cooling_candidates():
    pool = KeyPoolCoordinator(3, clock=lambda: 100.0)
    snapshot = pool.request_order()
    pool.succeeded(2)
    assert pool.request_order()[0] == 2
    pool.failed(2, 60)
    pool.failed(0, 15)
    assert pool.request_order() == (1, 0, 2)
    pool.failed(1, 1800)
    assert pool.request_order() == (0, 2, 1)
    assert snapshot == (0, 1, 2)
    pool.succeeded(1)
    assert pool.request_order()[0] == 1


async def test_simultaneous_pacing_waiters_recheck_capacity_and_cancellation():
    now = 0.0
    sleepers = []
    admitted = []
    async def sleep(delay):
        future = asyncio.get_running_loop().create_future()
        sleepers.append(future)
        await future
    pool = KeyPoolCoordinator(3, rate_limit=1, clock=lambda: now, sleep=sleep)
    await pool.admit()
    async def request():
        await pool.admit()
        admitted.append(now)
    tasks = [asyncio.create_task(request()) for _ in range(4)]
    await asyncio.sleep(0)
    assert len(sleepers) == 4
    tasks[-1].cancel()
    await asyncio.gather(tasks[-1], return_exceptions=True)
    for instant in (61.0, 122.0, 183.0):
        now = instant
        pending, sleepers[:] = list(sleepers), []
        for waiter in pending:
            if not waiter.done():
                waiter.set_result(None)
        await asyncio.sleep(0)
        assert admitted == [value for value in (61.0, 122.0, 183.0) if value <= instant]
    await asyncio.gather(*tasks[:-1])


def test_initial_preference_and_concurrent_failures_preserve_long_cooldown():
    now = 100.0
    pool = KeyPoolCoordinator(3, clock=lambda: now)
    assert all(pool.request_order()[0] == 0 for _ in range(5))
    pool.failed(0, 1800)
    now = 110.0
    pool.failed(0, 15)
    pool.failed(1, 100)
    pool.failed(2, 200)
    assert pool.request_order() == (1, 2, 0)
    now = 500.0
    assert pool.request_order()[-1] == 0
