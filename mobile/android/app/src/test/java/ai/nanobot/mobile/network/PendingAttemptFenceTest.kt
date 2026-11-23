package ai.nanobot.mobile.network

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class PendingAttemptFenceTest {
    @Test fun cancel_is_durable_while_executor_is_blocked_and_late_writer_cannot_revive_it() {
        var disk = PendingAttemptFence.State(0, false)
        val owner = PendingAttemptFence({ disk }, { disk = it })
        val generation = owner.begin()
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val executor = Executors.newSingleThreadExecutor()
        val late = executor.submit<Boolean> {
            entered.countDown(); release.await()
            try { owner.withCurrent(generation) { fail("late custody write") }; false }
            catch (_: IllegalStateException) { true }
        }
        assertTrue(entered.await(1, TimeUnit.SECONDS))
        owner.cancel()
        val restarted = PendingAttemptFence({ disk }, { disk = it })
        assertFalse(restarted.isCurrent(generation))
        release.countDown()
        assertTrue(late.get(1, TimeUnit.SECONDS))
        executor.shutdownNow()
    }
    @Test fun failed_cancel_is_reported_and_fences_memory() {
        var disk = PendingAttemptFence.State(0, false)
        var failWrite = false
        val owner = PendingAttemptFence({ disk }, { if (failWrite) throw java.io.IOException("disk full") else disk = it })
        val generation = owner.begin()
        failWrite = true
        assertThrows(java.io.IOException::class.java) { owner.cancel() }
        assertFalse(owner.isCurrent(generation))
        assertThrows(IllegalStateException::class.java) { owner.withCurrent(generation) { fail("write after failed fence") } }
    }
    @Test fun superseded_attempt_and_restarted_old_generation_are_rejected() {
        var disk = PendingAttemptFence.State(0, false)
        val owner = PendingAttemptFence({ disk }, { disk = it })
        val first = owner.begin()
        val second = owner.begin()
        val restarted = PendingAttemptFence({ disk }, { disk = it })
        assertFalse(restarted.isCurrent(first))
        assertTrue(restarted.isCurrent(second))
    }
}
