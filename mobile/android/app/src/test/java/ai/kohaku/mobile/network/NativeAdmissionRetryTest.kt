package ai.kohaku.mobile.network

import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test

/**
 * U1-U4 cases from the requirements-only admission specification. The broker
 * is a scripted fake: it returns typed admission outcomes, counts attempts and
 * applies a mutation at most once per identity. The clock and scheduler are
 * injected, so attempt counts, backoff and deadlines are deterministic.
 */
class NativeAdmissionRetryTest {
    private class ManualTimer : RetryTimer {
        private data class Task(val due: Long, val delay: Long, val action: () -> Unit)

        private val tasks = mutableListOf<Task>()
        val delays = mutableListOf<Long>()
        var now = 0L

        override fun schedule(delayMs: Long, task: () -> Unit): Cancellable {
            val entry = Task(now + delayMs, delayMs, task)
            tasks.add(entry)
            delays.add(delayMs)
            return Cancellable { tasks.remove(entry) }
        }

        override fun pendingCount(): Int = tasks.size

        fun advance(ms: Long) {
            val target = now + ms
            while (true) {
                val next = tasks.filter { it.due <= target }.minByOrNull { it.due } ?: break
                tasks.remove(next)
                now = next.due
                next.action()
            }
            now = target
        }
    }

    private fun policy(
        maxAttempts: Int = 5,
        baseDelayMs: Long = 50,
        maxDelayMs: Long = 1_000,
        deadlineMs: Long = 10_000,
    ) = NativeRetryPolicy(maxAttempts, baseDelayMs, maxDelayMs, deadlineMs)

    private fun engine(timer: ManualTimer, policy: NativeRetryPolicy) =
        NativeRequestAdmission(policy, timer) { timer.now }

    private fun failureOf(future: CompletableFuture<*>): NativeAdmissionFailure {
        val event = assertThrows(ExecutionException::class.java) {
            future.get(1, TimeUnit.SECONDS)
        }
        return event.cause as NativeAdmissionFailure
    }

    /** C1 - startup reads are admitted, not lost. */
    @Test
    fun c1_startup_reads_are_admitted_not_lost() {
        val timer = ManualTimer()
        val admission = engine(timer, policy(maxAttempts = 5))
        val attempts = AtomicInteger()
        val perRequest = IntArray(10)
        val results = (0 until 10).map { index ->
            admission.run {
                attempts.incrementAndGet()
                val made = perRequest[index]++
                if (made < 3) {
                    throw NativeAdmissionFailure(
                        "transport_starting",
                        AdmissionFailureKind.NOT_ADMITTED,
                    )
                }
                index.toLong() + 1
            }
        }

        var guard = 0
        while (timer.pendingCount() > 0 && guard++ < 100) timer.advance(1_000)

        assertEquals(40, attempts.get())
        assertEquals((1L..10L).toList(), results.map { it.get(1, TimeUnit.SECONDS) })
        assertEquals(0, admission.pendingCount())
        assertTrue(timer.delays.isNotEmpty())
        assertTrue(timer.delays.all { it <= 1_000 })
    }

    /** C2 - a real refusal fails at once and is not retried. */
    @Test
    fun c2_a_real_refusal_fails_at_once_and_is_not_retried() {
        for (reason in listOf("bad_origin", "revoked access", "invalid scope")) {
            val timer = ManualTimer()
            val admission = engine(timer, policy(maxAttempts = 5))
            assertFalse(NativeAdmissionPolicy.retryable(reason))
            var attempts = 0
            val future = admission.run {
                attempts += 1
                throw NativeAdmissionFailure(reason, NativeAdmissionPolicy.kindFor(reason))
            }

            val failure = failureOf(future)
            assertEquals(reason, failure.reason)
            assertEquals(AdmissionFailureKind.REFUSED, failure.kind)
            assertEquals(1, attempts)
            assertEquals(0, admission.pendingCount())
            timer.advance(5_000)
            assertEquals(1, attempts)
        }
    }

    /** C3 - the retry bound is enforced and reported as not-admitted. */
    @Test
    fun c3_retry_bound_is_enforced_and_reported_as_not_admitted() {
        val limit = policy(maxAttempts = 4, baseDelayMs = 20, maxDelayMs = 100, deadlineMs = 400)
        val timer = ManualTimer()
        val admission = engine(timer, limit)
        var attempts = 0
        val future = admission.run {
            attempts += 1
            throw NativeAdmissionFailure(
                "transport_starting",
                AdmissionFailureKind.NOT_ADMITTED,
            )
        }
        timer.advance(5_000)

        val failure = failureOf(future)
        assertEquals(4, attempts)
        assertEquals("transport_starting", failure.reason)
        assertEquals(AdmissionFailureKind.NOT_ADMITTED, failure.kind)
        assertTrue(failure.admissionExhausted)
        assertEquals(0, admission.pendingCount())
        assertTrue(timer.delays.sum() <= limit.deadlineMs + limit.maxDelayMs)
    }

    /** C4 - a refusal keeps the broker reason; it is not masked or cleared. */
    @Test
    fun c4_refusal_keeps_the_broker_reason() {
        val timer = ManualTimer()
        val admission = engine(timer, policy())
        val future = admission.run {
            throw NativeAdmissionFailure(
                "access_revoked",
                NativeAdmissionPolicy.kindFor("access_revoked"),
            )
        }
        val failure = failureOf(future)
        val firstRead = failure.reason
        val secondRead = failure.reason
        assertEquals("access_revoked", firstRead)
        assertEquals("access_revoked", secondRead)
        assertFalse(firstRead.contains("correlation"))
        assertFalse(firstRead.contains("not admitted"))
        assertEquals(0, admission.pendingCount())
    }

    /** C5 - a mutation retried after not-admitted has one effect. */
    @Test
    fun c5_mutation_retried_after_not_admitted_has_one_effect() {
        val timer = ManualTimer()
        val admission = engine(timer, policy(maxAttempts = 3))
        var effects = 0
        var attempts = 0
        val identities = mutableListOf<String>()
        val handled = mutableMapOf<String, Long>()
        val identity = "id-1"
        val future = admission.run {
            attempts += 1
            identities.add(identity)
            val stored = handled[identity]
            if (stored != null) {
                stored
            } else {
                effects += 1
                handled[identity] = 41L
                throw NativeAdmissionFailure(
                    "transport_busy",
                    AdmissionFailureKind.NOT_ADMITTED,
                )
            }
        }
        timer.advance(1_000)

        assertEquals(1, effects)
        assertEquals(2, attempts)
        assertEquals(41L, future.get(1, TimeUnit.SECONDS))
        assertEquals(listOf("id-1", "id-1"), identities)
        assertEquals(0, admission.pendingCount())
    }

    /** C6 - one read versus one mutation: retries differ only by replay. */
    @Test
    fun c6_read_and_mutation_retries_keep_separate_identities() {
        val timer = ManualTimer()
        val admission = engine(timer, policy(maxAttempts = 3))
        var readExecutions = 0
        val readValues = mutableListOf<Long>()
        var mutationAttempts = 0
        var mutationEffects = 0
        var storedMutation: Long? = null
        val readIdentities = mutableListOf<String>()
        val mutationIdentities = mutableListOf<String>()

        val read = admission.run {
            readExecutions += 1
            readIdentities.add("read-1")
            readValues.add(7L)
            if (readExecutions == 1) {
                throw NativeAdmissionFailure("transport_busy", AdmissionFailureKind.NOT_ADMITTED)
            }
            7L
        }
        val mutation = admission.run {
            mutationAttempts += 1
            mutationIdentities.add("mutation-1")
            val stored = storedMutation
            if (stored != null) {
                stored
            } else {
                mutationEffects += 1
                storedMutation = 9L
                throw NativeAdmissionFailure("transport_busy", AdmissionFailureKind.NOT_ADMITTED)
            }
        }
        timer.advance(1_000)

        assertEquals(2, readExecutions)
        assertEquals(listOf(7L, 7L), readValues)
        assertEquals(7L, read.get(1, TimeUnit.SECONDS))
        assertEquals(2, mutationAttempts)
        assertEquals(1, mutationEffects)
        assertEquals(9L, mutation.get(1, TimeUnit.SECONDS))
        assertTrue(readIdentities.intersect(mutationIdentities.toSet()).isEmpty())
        assertEquals(0, admission.pendingCount())
    }

    /** C7 - close during backoff stops the retry and leaves no handle. */
    @Test
    fun c7_close_during_backoff_stops_the_retry() {
        val timer = ManualTimer()
        val admission = engine(timer, policy())
        var attempts = 0
        var effects = 0
        val future = admission.run {
            attempts += 1
            effects += 1
            throw NativeAdmissionFailure("transport_starting", AdmissionFailureKind.NOT_ADMITTED)
        }
        assertEquals(1, admission.pendingCount())

        admission.close()

        assertTrue(future.isCompletedExceptionally)
        assertEquals(AdmissionFailureKind.CANCELLED, failureOf(future).kind)
        assertEquals(0, admission.pendingCount())
        timer.advance(5_000)
        assertEquals(1, attempts)
        assertEquals(1, effects)
    }

    /** C8 - a deadline during backoff stops the retry with nothing pending. */
    @Test
    fun c8_deadline_during_backoff_stops_the_retry() {
        val limit = policy(maxAttempts = 10, baseDelayMs = 100, maxDelayMs = 1_000, deadlineMs = 30)
        val timer = ManualTimer()
        val admission = engine(timer, limit)
        var attempts = 0
        val future = admission.run {
            attempts += 1
            throw NativeAdmissionFailure("transport_starting", AdmissionFailureKind.NOT_ADMITTED)
        }
        timer.advance(30)

        val failure = failureOf(future)
        assertEquals(1, attempts)
        assertEquals("transport_starting", failure.reason)
        assertEquals(AdmissionFailureKind.NOT_ADMITTED, failure.kind)
        assertTrue(failure.admissionExhausted)
        assertEquals(0, admission.pendingCount())
        timer.advance(10_000)
        assertEquals(1, attempts)
    }
}
