package ai.nanobot.mobile.media

import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.ArrayDeque
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeMediaFetcherInstrumentedTest {
    private class FakeTransport(var response: String?) : NativeMediaTransport {
        var submitted: Map<String, String> = emptyMap()
        var canceled = 0L
        override fun submit(operation: String, method: String, path: String, headers: Map<String, String>): Long {
            submitted = headers
            return 1L
        }
        override fun pollEvent(): String? = response.also { response = null }
        override fun cancel(requestId: Long) { canceled = requestId }
    }

    private fun response(body: ByteArray, mediaType: String = "image/png"): String = JSONObject()
        .put("type", "http_response")
        .put("request_id", 1)
        .put("status", 206)
        .put("headers", JSONObject().put("content-type", mediaType))
        .put("body_base64", Base64.encodeToString(body, Base64.NO_WRAP))
        .toString()

    @Test
    fun fetchesBoundedMediaAndPreservesRange() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = MediaHandleStore(context, maxBytes = 1024)
        val transport = FakeTransport(response(byteArrayOf(1, 2, 3)))
        val fetcher = NativeMediaFetcher(transport, store, maxBytes = 1024)
        val handle = fetcher.fetch("sig", "payload", "bytes=0-2")
        assertEquals(3, handle.bytes.toInt())
        assertEquals("bytes=0-2", transport.submitted["Range"])
        assertEquals("image/png", handle.mediaType)
        store.purge()
    }

    @Test
    fun rejectsOversizeAndTimesOutWithCancel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = MediaHandleStore(context, maxBytes = 1024)
        val oversized = NativeMediaFetcher(FakeTransport(response(ByteArray(1025))), store, maxBytes = 1024)
        assertThrows(MediaFetchError::class.java) { oversized.fetch("sig", "payload") }
        val empty = FakeTransport(null)
        val timeout = NativeMediaFetcher(empty, store, maxBytes = 1024, timeoutMs = 50)
        assertThrows(MediaFetchError::class.java) { timeout.fetch("sig", "payload") }
        assertEquals(1L, empty.canceled)
    }
}
