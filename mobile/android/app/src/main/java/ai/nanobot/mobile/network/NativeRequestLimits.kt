package ai.nanobot.mobile.network

/** Admission bounds for requests crossing the Android native bridge. */
internal object NativeRequestLimits {
    const val MAX_REQUEST_BYTES = 2 * 1024 * 1024
    const val MAX_PENDING_REQUESTS = 64

    /** Return a stable error message when the request cannot be admitted. */
    fun refusal(encodedBytes: Int, pendingRequests: Int): String? {
        if (encodedBytes <= 0) return "native request is empty"
        if (encodedBytes > MAX_REQUEST_BYTES) {
            return "native request exceeds the ${MAX_REQUEST_BYTES / (1024 * 1024)} MiB limit"
        }
        if (pendingRequests >= MAX_PENDING_REQUESTS) {
            return "native request queue is full"
        }
        return null
    }
}
