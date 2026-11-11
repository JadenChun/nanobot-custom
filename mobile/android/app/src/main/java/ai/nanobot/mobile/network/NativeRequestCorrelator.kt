package ai.nanobot.mobile.network

import java.util.concurrent.CompletableFuture

/** Serializes launch/ID publication with poll delivery. No credential body goes
 * to the WebView unless this owner admitted that ID as an external request.
 */
internal class NativeRequestCorrelator<T> {
    data class Admission<T>(val id: Long, val future: CompletableFuture<T>)
    private data class Entry<T>(val internal: Boolean, var future: CompletableFuture<T>?)
    private val entries = mutableMapOf<Long, Entry<T>>()
    private var closed = false

    @Synchronized fun pendingCount(): Int = entries.size

    @Synchronized fun launch(internal: Boolean, start: () -> Long): Admission<T> {
        check(!closed) { "native transport stopped" }
        check(entries.size < NativeRequestLimits.MAX_PENDING_REQUESTS) { "native request queue is full" }
        val future = CompletableFuture<T>()
        // Poll delivery takes this monitor, so it cannot observe completion
        // until both the returned ID and its native classification are stored.
        val id = start()
        check(id > 0 && id !in entries) { "native request was not admitted" }
        entries[id] = Entry(internal, future)
        return Admission(id, future)
    }

    /** true only for an admitted external terminal response. Unknown is private. */
    @Synchronized fun receive(id: Long, result: T): Boolean {
        val entry = entries.remove(id) ?: return false
        entry.future?.complete(result)
        return !entry.internal && !closed
    }

    @Synchronized fun abandon(id: Long) {
        // Keep classification and count against admission until the actual
        // terminal completion arrives. Timeouts never turn credentials public.
        entries[id]?.future = null
    }

    @Synchronized fun close() {
        if (closed) return
        closed = true
        for (entry in entries.values) entry.future?.completeExceptionally(IllegalStateException("native transport stopped"))
        entries.clear()
    }
}
