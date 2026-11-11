package ai.nanobot.mobile.network

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class NativeRequestCorrelatorTest {
    @Test fun reserves_before_poller_can_observe_immediate_completion() {
        val owner = NativeRequestCorrelator<String>()
        val poller = Executors.newSingleThreadExecutor()
        val entered = CountDownLatch(1)
        val admission = owner.launch(true) {
            poller.submit { entered.countDown(); assertFalse(owner.receive(1, "credential")) }
            assertTrue(entered.await(1, TimeUnit.SECONDS))
            1L
        }
        assertEquals("credential", admission.future.get(1, TimeUnit.SECONDS))
        poller.shutdownNow()
    }
    @Test fun timeout_retains_internal_classification_until_late_completion_and_refuses_at_capacity() {
        val owner = NativeRequestCorrelator<String>()
        for (id in 1L..64L) {
            owner.launch(true) { id }
            owner.abandon(id)
        }
        assertThrows(IllegalStateException::class.java) { owner.launch(true) { 65L } }
        assertFalse(owner.receive(1L, "late credential"))
        assertFalse(owner.receive(999L, "unmatched credential"))
        assertEquals(65L, owner.launch(false) { 65L }.id)
        assertTrue(owner.receive(65L, "public response"))
        owner.close()
        owner.close()
        assertFalse(owner.receive(2L, "disposed credential"))
    }
    @Test fun concurrent_requests_stay_bound_to_their_own_completion() {
        val owner = NativeRequestCorrelator<String>()
        val first = owner.launch(true) { 1L }
        val second = owner.launch(true) { 2L }
        assertFalse(owner.receive(2L, "second"))
        assertFalse(owner.receive(1L, "first"))
        assertEquals("first", first.future.get())
        assertEquals("second", second.future.get())
    }
}
