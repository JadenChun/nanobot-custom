package ai.nanobot.mobile.network

import java.util.concurrent.CompletableFuture
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Why a native launch did not produce a request ID. */
internal enum class AdmissionFailureKind { REFUSED, NOT_ADMITTED, CANCELLED }

/** One admitted launch failed. [reason] is the broker's own text, never a
 * correlation placeholder. A refusal keeps the reason exactly; an exhausted
 * admission prefixes the reason so callers and logs can tell the two apart.
 */
internal class NativeAdmissionFailure(
    val reason: String,
    val kind: AdmissionFailureKind,
    val admissionExhausted: Boolean = false,
) : IllegalStateException(
    if (kind == AdmissionFailureKind.NOT_ADMITTED && admissionExhausted) {
        "native request was not admitted: $reason"
    } else {
        reason
    }
)

/** Classifies the broker's stored diagnostic without inventing a code path.
 * A missing/replaced transport, a policy refusal, a bad request or an oversize
 * payload is permanent. A busy transport or a full reservation slot is not.
 */
internal object NativeAdmissionPolicy {
    fun retryable(reason: String): Boolean =
        reason.startsWith("bounded queue is full") ||
            reason.startsWith("transport is not ready") ||
            reason.startsWith("operation already in progress")

    fun kindFor(reason: String): AdmissionFailureKind =
        if (retryable(reason)) AdmissionFailureKind.NOT_ADMITTED else AdmissionFailureKind.REFUSED
}

/** Fixed bounds for one launch. Tests supply the same values through the
 * underlying case specification, so production and cases share one shape.
 */
internal data class NativeRetryPolicy(
    val maxAttempts: Int = 8,
    val baseDelayMs: Long = 50,
    val maxDelayMs: Long = 1_000,
    val deadlineMs: Long = 10_000,
) {
    init {
        require(maxAttempts >= 1)
        require(baseDelayMs > 0)
        require(maxDelayMs >= baseDelayMs)
        require(deadlineMs > 0)
    }

    /** Delay after [attemptCount] attempts, capped before any wait. */
    fun delayFor(attemptCount: Int): Long {
        var delay = baseDelayMs
        repeat((attemptCount - 1).coerceAtLeast(0)) {
            delay = if (delay > maxDelayMs / 2) maxDelayMs else delay * 2
        }
        return delay.coerceAtMost(maxDelayMs)
    }
}

internal fun interface Cancellable {
    fun cancel()
}

/** One pending backoff. The count is observable so a cancelled launch can
 * prove that it left no timer behind.
 */
internal interface RetryTimer {
    fun schedule(delayMs: Long, task: () -> Unit): Cancellable
    fun pendingCount(): Int
}

internal class ScheduledRetryTimer(
    private val scheduler: ScheduledExecutorService,
) : RetryTimer {
    private val pending = AtomicInteger()

    override fun schedule(delayMs: Long, task: () -> Unit): Cancellable {
        val done = AtomicBoolean(false)
        pending.incrementAndGet()
        val future: ScheduledFuture<*> = scheduler.schedule(
            {
                if (done.compareAndSet(false, true)) {
                    pending.decrementAndGet()
                    task()
                }
            },
            delayMs.coerceAtLeast(0),
            TimeUnit.MILLISECONDS,
        )
        return Cancellable {
            if (done.compareAndSet(false, true)) {
                future.cancel(false)
                pending.decrementAndGet()
            }
        }
    }

    override fun pendingCount(): Int = pending.get()
}

/** Bounded admission retry for startup requests.
 *
 * A read retry re-sends the same encoded request, so it has no effect to
 * duplicate. A mutation retry re-sends the same bytes too, including its
 * request identity, so the server's replay protection returns the stored
 * result instead of applying the effect twice.
 *
 * Only [AdmissionFailureKind.NOT_ADMITTED] failures are retried, and only
 * inside the attempt and deadline bounds. A refusal fails at once. [close]
 * cancels every pending backoff and settles every waiting launch, so a
 * stopped transport leaves no timer or handle.
 */
internal class NativeRequestAdmission(
    private val policy: NativeRetryPolicy = NativeRetryPolicy(),
    private val timer: RetryTimer,
    private val clock: () -> Long = { System.nanoTime() / 1_000_000 },
) {
    private val lock = Any()
    private var closed = false
    private val active = mutableSetOf<CompletableFuture<*>>()
    private val scheduled = mutableSetOf<Cancellable>()

    fun <T> run(attempt: () -> T): CompletableFuture<T> {
        val result = CompletableFuture<T>()
        val deadline = clock() + policy.deadlineMs
        synchronized(lock) {
            if (closed) {
                result.completeExceptionally(closedFailure())
                return result
            }
            active.add(result)
        }
        attemptOnce(result, attempt, deadline, 0)
        return result
    }

    fun close() {
        val cancels: List<Cancellable>
        val waiting: List<CompletableFuture<*>>
        synchronized(lock) {
            if (closed) return
            closed = true
            cancels = scheduled.toList()
            scheduled.clear()
            waiting = active.toList()
            active.clear()
        }
        for (cancel in cancels) cancel.cancel()
        for (future in waiting) future.completeExceptionally(closedFailure())
    }

    fun pendingCount(): Int = timer.pendingCount()

    private fun <T> attemptOnce(
        result: CompletableFuture<T>,
        attempt: () -> T,
        deadline: Long,
        attempts: Int,
    ) {
        synchronized(lock) {
            if (closed) {
                active.remove(result)
                result.completeExceptionally(closedFailure())
                return
            }
        }
        try {
            val value = attempt()
            synchronized(lock) {
                active.remove(result)
                if (closed) result.completeExceptionally(closedFailure()) else result.complete(value)
            }
        } catch (failure: NativeAdmissionFailure) {
            val made = attempts + 1
            val remaining = deadline - clock()
            if (failure.kind != AdmissionFailureKind.NOT_ADMITTED ||
                made >= policy.maxAttempts ||
                remaining <= 0
            ) {
                val terminal = if (failure.kind == AdmissionFailureKind.NOT_ADMITTED) {
                    NativeAdmissionFailure(failure.reason, failure.kind, admissionExhausted = true)
                } else {
                    failure
                }
                synchronized(lock) { active.remove(result) }
                result.completeExceptionally(terminal)
                return
            }
            val delay = policy.delayFor(made)
            if (delay >= remaining) {
                // The next backoff would consume the deadline. Do not start an
                // attempt that the deadline has already ended.
                synchronized(lock) { active.remove(result) }
                result.completeExceptionally(
                    NativeAdmissionFailure(failure.reason, failure.kind, admissionExhausted = true)
                )
                return
            }
            synchronized(lock) {
                if (closed) {
                    active.remove(result)
                    result.completeExceptionally(closedFailure())
                    return
                }
                lateinit var handle: Cancellable
                handle = timer.schedule(delay) {
                    synchronized(lock) { scheduled.remove(handle) }
                    attemptOnce(result, attempt, deadline, made)
                }
                scheduled.add(handle)
            }
        } catch (failure: Throwable) {
            synchronized(lock) { active.remove(result) }
            result.completeExceptionally(failure)
        }
    }

    private fun closedFailure(): NativeAdmissionFailure =
        NativeAdmissionFailure("native transport stopped", AdmissionFailureKind.CANCELLED)
}

/** Serializes launch/ID publication with poll delivery. No credential body goes
 * to the WebView unless this owner admitted that ID as an external request.
 */
internal class NativeRequestCorrelator<T> {
    data class Admission<T>(val id: Long, val future: CompletableFuture<T>)
    private data class Entry<T>(val internal: Boolean, var future: CompletableFuture<T>?)
    private val entries = mutableMapOf<Long, Entry<T>>()
    private var closed = false

    @Synchronized fun pendingCount(): Int = entries.size

    @Synchronized fun launch(
        internal: Boolean,
        lastError: () -> String = { "native request was not admitted" },
        start: () -> Long,
    ): Admission<T> {
        check(!closed) { "native transport stopped" }
        check(entries.size < NativeRequestLimits.MAX_PENDING_REQUESTS) { "native request queue is full" }
        val future = CompletableFuture<T>()
        // Poll delivery takes this monitor, so it cannot observe completion
        // until both the returned ID and its native classification are stored.
        val id = start()
        if (id <= 0L) {
            // The native call recorded the broker's own message on this thread.
            // Capture it here, before any later call on this thread can
            // overwrite it, and before the correlator discards the launch.
            val reason = lastError().ifBlank { "native request was not admitted" }
            throw NativeAdmissionFailure(reason, NativeAdmissionPolicy.kindFor(reason))
        }
        check(id !in entries) { "native request returned a duplicate id" }
        entries[id] = Entry(internal, future)
        return Admission(id, future)
    }

    /** true only for an admitted external terminal response. Unknown is private. */
    @Synchronized fun receive(id: Long, result: T): Boolean {
        val entry = entries.remove(id) ?: return false
        entry.future?.complete(result)
        return !entry.internal && !closed
    }

    /** Forward only external terminal events to the WebView event sink. */
    @Synchronized fun dispatch(id: Long, result: T, deliver: (T) -> Unit): Boolean {
        val external = receive(id, result)
        if (external) deliver(result)
        return external
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
