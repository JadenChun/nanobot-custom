package ai.kohaku.mobile.network

import ai.kohaku.nativenetwork.NativeNetwork
import android.util.Log
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.FilterInputStream
import java.io.InputStream

private val contentRangePattern = Regex("""bytes\s+(\d+)-(\d+)/(\d+|\*)""")

/**
 * Streams signed media through the native transport.
 *
 * The WebView asks for `/api/media/<sig>/<payload>` on its own origin, so the
 * bytes never cross the JS bridge as base64, no extra listener is exposed, and
 * there is no mixed-content or CSP change. Ranges are forwarded, which is what
 * lets video seek and images stream progressively.
 */
object NativeMediaInterceptor {
    private const val PREFIX = "/api/media/"

    @Volatile
    var transportHandle: Long = 0L

    fun intercept(request: WebResourceRequest): WebResourceResponse? {
        val path = request.url?.encodedPath ?: return null
        if (!path.startsWith(PREFIX)) return null
        val handle = transportHandle
        if (handle == 0L) return null
        val mobilePath = "/mobile/v1/media/" + path.removePrefix(PREFIX)
        val range = request.requestHeaders?.entries
            ?.firstOrNull { it.key.equals("Range", ignoreCase = true) }
            ?.value ?: ""
        val streamId = try {
            NativeNetwork.mediaOpen(handle, mobilePath, range)
        } catch (_: Throwable) {
            0L
        }
        if (streamId == 0L) return failure(503, "Service Unavailable")
        val info = try {
            NativeNetwork.mediaInfo(handle, streamId)?.let { JSONObject(String(it)) }
        } catch (_: Throwable) {
            null
        }
        if (info == null) {
            NativeNetwork.mediaClose(handle, streamId)
            return failure(503, "Service Unavailable")
        }
        val status = info.optInt("status", 200)
        if (status < 200 || status >= 300) {
            NativeNetwork.mediaClose(handle, streamId)
            return failure(status, reasonPhrase(status))
        }
        val headers = mutableMapOf<String, String>()
        info.optJSONObject("headers")?.let { headerJson ->
            for (key in headerJson.keys()) {
                val value = headerJson.optString(key)
                if (value.isNotEmpty()) headers[key] = value
            }
        }
        Log.i(
            "NanobotMedia",
            mediaLogLine(status, info.optBoolean("cached"), headers),
        )
        val mime = headerValue(headers, "Content-Type") ?: "application/octet-stream"
        return WebResourceResponse(
            mime,
            null,
            status,
            reasonPhrase(status),
            headers,
            mediaBodyForWebView(MediaStreamInput(handle, streamId), headerValue(headers, "Content-Range")),
        )
    }

    /**
     * Signed media URLs are bearer capabilities: the signature is the
     * authorization material. A log line may carry numeric response facts only,
     * so no URL, signature or credential can be replayed from logcat.
     */
    internal fun mediaLogLine(
        status: Int,
        cached: Boolean,
        headers: Map<String, String>,
    ): String {
        val bytes = headerValue(headers, "Content-Length")?.trim()?.toLongOrNull() ?: -1L
        val range = headerValue(headers, "Content-Range")?.let(::contentRangeCounts) ?: "-"
        return "open status=$status cached=$cached bytes=$bytes range=$range"
    }

    private fun headerValue(headers: Map<String, String>, name: String): String? =
        headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value

    private fun contentRangeCounts(value: String): String? {
        val match = contentRangePattern.find(value.trim()) ?: return null
        val (start, end, total) = match.destructured
        return "$start-$end/$total"
    }

    private fun failure(status: Int, reason: String): WebResourceResponse =        WebResourceResponse(
            "text/plain",
            "utf-8",
            status,
            reason,
            emptyMap(),
            ByteArrayInputStream(ByteArray(0)),
        )

    private fun reasonPhrase(status: Int): String = when (status) {
        200 -> "OK"
        206 -> "Partial Content"
        304 -> "Not Modified"
        404 -> "Not Found"
        416 -> "Range Not Satisfiable"
        else -> "Media"
    }
}

/**
 * The WebView seeks on the stream before it reads a range response. Report the
 * file size and account for the prefix that the gateway already skipped.
 */
internal fun mediaBodyForWebView(body: InputStream, contentRange: String?): InputStream {
    val match = contentRange?.trim()?.let(contentRangePattern::matchEntire)
        ?: return body
    val start = match.groupValues[1].toLongOrNull() ?: return body
    val size = match.groupValues[3].toLongOrNull() ?: return body
    return object : FilterInputStream(body) {
        private var prefix = start

        override fun available(): Int = if (size <= Int.MAX_VALUE) size.toInt() else 0

        override fun skip(n: Long): Long {
            if (n <= 0) return 0
            val skipped = minOf(n, prefix)
            prefix -= skipped
            return if (n == skipped) skipped else skipped + super.skip(n - skipped)
        }
    }
}

private class MediaStreamInput(
    private val handle: Long,
    private val streamId: Long,
) : InputStream() {
    private val buffer = ByteArray(64 * 1024)
    private var pending = 0
    private var pendingOffset = 0
    private var closed = false

    override fun read(): Int {
        if (read(buffer, 0, 1) <= 0) return -1
        return buffer[0].toInt() and 0xff
    }

    override fun read(b: ByteArray, off: Int, len: Int): Int {
        if (closed) return -1
        if (len == 0) return 0
        if (pendingOffset >= pending) {
            val read = try {
                NativeNetwork.mediaRead(handle, streamId, buffer)
            } catch (_: Throwable) {
                -1
            }
            if (read <= 0) return -1
            pending = read
            pendingOffset = 0
        }
        val count = minOf(len, pending - pendingOffset)
        System.arraycopy(buffer, pendingOffset, b, off, count)
        pendingOffset += count
        return count
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            NativeNetwork.mediaClose(handle, streamId)
        } catch (_: Throwable) {
            // Closing best-effort: the core also drops streams with the transport.
        }
    }
}
