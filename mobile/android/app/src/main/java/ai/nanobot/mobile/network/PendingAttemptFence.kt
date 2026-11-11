package ai.nanobot.mobile.network

/** Durable native owner; the same monitor guards comparison and custody writes. */
internal class PendingAttemptFence(read: () -> State, private val persist: (State) -> Unit) {
    data class State(val generation: Long, val active: Boolean)
    private var state = read()
    private var persistenceFailed = false

    @Synchronized fun begin(): Long = advance(true)
    @Synchronized fun cancel(): Long = advance(false)
    @Synchronized fun generation(): Long = state.generation
    @Synchronized fun isCurrent(generation: Long): Boolean =
        !persistenceFailed && state.active && state.generation == generation

    @Synchronized fun <T> withCurrent(generation: Long, requireActive: Boolean = true, write: () -> T): T {
        check(!persistenceFailed && state.generation == generation && (!requireActive || state.active)) {
            "pairing attempt was cancelled or superseded"
        }
        return write()
    }

    private fun advance(active: Boolean): Long {
        check(state.generation < Long.MAX_VALUE) { "pairing generation exhausted" }
        // Fence in memory even if persistence fails; the caller must receive the
        // error and cannot claim successful durable cancellation.
        state = State(state.generation + 1, active)
        persistenceFailed = true
        persist(state)
        persistenceFailed = false
        return state.generation
    }
}
