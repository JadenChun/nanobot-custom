package ai.nanobot.mobile.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeMediaInterceptorLogTest {
    @Test
    fun logLineCarriesOnlyStatusCacheAndCounts() {
        val line = NativeMediaInterceptor.mediaLogLine(
            206,
            true,
            mapOf(
                "Content-Type" to "image/png",
                "Content-Length" to "4096",
                "Content-Range" to "bytes 0-4095/9000",
            ),
        )
        assertEquals("open status=206 cached=true bytes=4096 range=0-4095/9000", line)
    }

    @Test
    fun logLineNeverContainsTheSignedMediaUrl() {
        // The signature is the authorization material for /api/media/<sig>/<payload>.
        val line = NativeMediaInterceptor.mediaLogLine(
            200,
            false,
            mapOf(
                "Content-Length" to "-1",
                "Content-Range" to "bytes 0-3/4",
            ),
        )
        assertFalse(line.contains("/api/media/"))
        assertFalse(line.contains("/mobile/v1/media/"))
        assertFalse(line.contains("sig="))
        assertFalse(line.contains("Bearer"))
        assertFalse(line.contains("token"))
        assertFalse(line.contains("http"))
        assertTrue(line.startsWith("open status="))
    }

    @Test
    fun nonNumericHeaderValuesAreDropped() {
        val line = NativeMediaInterceptor.mediaLogLine(
            200,
            false,
            mapOf(
                "Content-Length" to "https://example.test/api/media/signature/payload",
                "Content-Range" to "bytes https://example.test/secret",
            ),
        )
        assertEquals("open status=200 cached=false bytes=-1 range=-", line)
        assertFalse(line.contains("example.test"))
        assertFalse(line.contains("secret"))
    }
}
