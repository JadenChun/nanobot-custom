package ai.nanobot.mobile.network

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeNetworkInstrumentedTest {
    @Test
    fun abiVersionIsOne() {
        assertEquals(1, NativeNetwork.abiVersion())
    }

    @Test
    fun invalidConfigFailsClosed() {
        assertEquals(0L, NativeNetwork.start("{}"))
        assertTrue(NativeNetwork.lastError().contains("origin"))
    }

    @Test
    fun unknownHandlesFailClosed() {
        assertNull(NativeNetwork.status(999))
        assertNull(NativeNetwork.loginUrl(999))
        assertNull(NativeNetwork.pollEvent(999))
        assertEquals(-4, NativeNetwork.stop(999))
        assertEquals(-4, NativeNetwork.cancel(999, 1))
        assertEquals(-4, NativeNetwork.wsClose(999, 1))
    }
}
