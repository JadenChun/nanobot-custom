package ai.nanobot.mobile.network

import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.InputStream

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
        if (streamId == 0L) return failure(404, "Not Found")
        val info = try {
            NativeNetwork.mediaInfo(handle, streamId)?.let { JSONObject(String(it)) }
        } catch (_: Throwable) {
            null
        }
        if (info == null) {
            NativeNetwork.mediaClose(handle, streamId)
            return failure(404, "Not Found")
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
        val mime = headers["Content-Type"] ?: "application/octet-stream"
        return WebResourceResponse(
            mime,
            null,
            status,
            reasonPhrase(status),
            headers,
            MediaStreamInput(handle, streamId),
        )
    }

    private fun failure(status: Int, reason: String): WebResourceResponse =
        WebResourceResponse(
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
