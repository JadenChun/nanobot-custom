package ai.nanobot.mobile.media

import android.util.Base64
import org.json.JSONObject

interface NativeMediaTransport {
    fun submit(operation: String, method: String, path: String, headers: Map<String, String>): Long
    fun pollEvent(): String?
    fun cancel(requestId: Long)
}

class MediaFetchError(message: String) : Exception(message)

class NativeMediaFetcher(
    private val transport: NativeMediaTransport,
    private val store: MediaHandleStore,
    private val maxBytes: Int = 8 * 1024 * 1024,
    private val timeoutMs: Long = 30_000,
) {
    fun fetch(signature: String, payload: String, range: String? = null): MediaHandle {
        if (!SIGNATURE.matches(signature) || !PAYLOAD.matches(payload)) {
            throw MediaFetchError("signed media segments are invalid")
        }
        val headers = if (range.isNullOrBlank()) emptyMap() else mapOf("Range" to range)
        val requestId = transport.submit(
            "mobile_media",
            "GET",
            "/mobile/v1/media/$signature/$payload",
            headers,
        )
        if (requestId == 0L) throw MediaFetchError("media request was not admitted")
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val raw = transport.pollEvent() ?: continue
            val event = try { JSONObject(raw) } catch (exception: Exception) { continue }
            if (event.optLong("request_id") != requestId) continue
            if (event.optString("type") == "http_error") {
                throw MediaFetchError("media fetch failed: ${event.optString("code")}")
            }
            if (event.optString("type") != "http_response") continue
            if (event.optInt("status") !in 200..299) {
                throw MediaFetchError("media status ${event.optInt("status")}")
            }
            val body = try {
                Base64.decode(event.optString("body_base64"), Base64.NO_WRAP)
            } catch (exception: Exception) {
                throw MediaFetchError("media body is not base64")
            }
            if (body.size > maxBytes) throw MediaFetchError("media exceeds the preview limit")
            val contentType = event.optJSONObject("headers")?.optString("content-type").orEmpty()
            if (contentType.isBlank()) throw MediaFetchError("media content type is missing")
            val extension = contentType.substringAfter('/', "bin").substringBefore(';')
                .replace(Regex("[^A-Za-z0-9]"), "").take(8).ifEmpty { "bin" }
            return store.put(body, contentType, extension)
        }
        transport.cancel(requestId)
        throw MediaFetchError("media fetch timed out")
    }

    companion object {
        private val SIGNATURE = Regex("^[A-Za-z0-9_-]+$")
        private val PAYLOAD = Regex("^[A-Za-z0-9_-]+$")
    }
}
