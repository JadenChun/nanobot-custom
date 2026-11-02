package ai.nanobot.mobile.media

import android.content.Context
import java.io.File
import java.security.SecureRandom

class MediaHandleError(message: String) : Exception(message)

data class MediaHandle(val id: String, val file: File, val mediaType: String, val bytes: Long)

/**
 * Bounded app-private media handles for native previews. Files are never
 * exposed through a general URL bridge and are purged on logout/switch.
 */
class MediaHandleStore(context: Context, private val maxBytes: Long = 64L * 1024 * 1024) {
    private val root = File(context.cacheDir, "nanobot-media")
    private val handles = mutableMapOf<String, MediaHandle>()

    @Synchronized
    fun put(data: ByteArray, mediaType: String, extension: String = "bin"): MediaHandle {
        if (data.size > maxBytes) throw MediaHandleError("media exceeds the handle limit")
        if (totalBytes() + data.size > maxBytes) purge()
        val safeExtension = extension.replace(Regex("[^A-Za-z0-9]"), "").take(8).ifEmpty { "bin" }
        val id = ByteArray(16).also { SecureRandom().nextBytes(it) }.joinToString("") { "%02x".format(it) }
        if (!root.exists() && !root.mkdirs()) throw MediaHandleError("media directory unavailable")
        root.setReadable(false, false)
        root.setWritable(false, false)
        root.setReadable(true, true)
        root.setWritable(true, true)
        val file = File(root, "$id.$safeExtension")
        file.writeBytes(data)
        file.setReadable(false, false)
        file.setReadable(true, true)
        val handle = MediaHandle(id, file, mediaType, data.size.toLong())
        handles[id] = handle
        return handle
    }

    @Synchronized
    fun get(id: String): MediaHandle? = handles[id]?.takeIf { it.file.isFile }

    @Synchronized
    fun purge() {
        handles.clear()
        root.deleteRecursively()
    }

    @Synchronized
    fun totalBytes(): Long = handles.values.sumOf { it.bytes } + (root.listFiles()?.sumOf { it.length() } ?: 0L)
}
