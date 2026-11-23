package ai.nanobot.mobile.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class NativeRequestLimitsTest {
    @Test
    fun accepts_a_bounded_request_below_the_pending_limit() {
        assertNull(NativeRequestLimits.refusal(1024, NativeRequestLimits.MAX_PENDING_REQUESTS - 1))
    }

    @Test
    fun rejects_an_oversize_request_before_jni() {
        assertEquals(
            "native request exceeds the 2 MiB limit",
            NativeRequestLimits.refusal(NativeRequestLimits.MAX_REQUEST_BYTES + 1, 0),
        )
    }

    @Test
    fun rejects_a_full_pending_set() {
        assertEquals(
            "native request queue is full",
            NativeRequestLimits.refusal(1024, NativeRequestLimits.MAX_PENDING_REQUESTS),
        )
    }
}
