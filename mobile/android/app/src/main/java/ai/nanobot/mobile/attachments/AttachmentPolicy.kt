package ai.nanobot.mobile.attachments

import java.io.File
import java.io.InputStream

data class AttachmentLimits(
    val maxBytes: Int = 256 * 1024,
    val allowedMime: Set<String> = setOf(
        "image/jpeg", "image/png", "image/webp", "image/gif",
        "application/pdf", "text/plain", "text/markdown",
    ),
)

class AttachmentRejectedError(message: String) : Exception(message)

object AttachmentPolicy {
    fun validate(name: String, mediaType: String, size: Long, limits: AttachmentLimits): String {
        if (name.isBlank() || name.length > 128) throw AttachmentRejectedError("attachment name is invalid")
        if (mediaType !in limits.allowedMime) throw AttachmentRejectedError("attachment MIME is not allowed")
        if (size <= 0 || size > limits.maxBytes) throw AttachmentRejectedError("attachment size is outside the limit")
        return name
    }

    fun copyBounded(
        input: InputStream,
        output: File,
        limits: AttachmentLimits,
        mediaType: String,
        name: String,
    ): Long {
        var total = 0L
        output.parentFile?.mkdirs()
        output.outputStream().use { sink ->
            val buffer = ByteArray(8192)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                total += read
                if (total > limits.maxBytes) {
                    output.delete()
                    throw AttachmentRejectedError("attachment exceeds the limit")
                }
                sink.write(buffer, 0, read)
            }
        }
        validate(name, mediaType, total, limits)
        return total
    }
}
