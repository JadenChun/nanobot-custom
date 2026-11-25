package ai.kohaku.mobile.network

import ai.kohaku.mobile.links.PairingChallenge
import ai.kohaku.mobile.network.pairing.CollectedPairing
import ai.kohaku.mobile.network.pairing.NativeHttpException
import ai.kohaku.mobile.network.pairing.PairingAttempt
import ai.kohaku.mobile.network.pairing.PairingConnection
import ai.kohaku.mobile.network.pairing.PairingCoordinator
import ai.kohaku.mobile.network.pairing.PairingCredentials
import ai.kohaku.mobile.network.pairing.PairingCustody
import ai.kohaku.mobile.network.pairing.PairingMethod
import ai.kohaku.mobile.network.pairing.PairingPollResult
import ai.kohaku.mobile.network.pairing.PairingProgress
import ai.kohaku.mobile.network.pairing.PairingRequest
import ai.kohaku.mobile.network.pairing.PairingScheduler
import ai.kohaku.mobile.network.pairing.PairingSnapshot
import ai.kohaku.mobile.network.pairing.PairingStage
import ai.kohaku.mobile.network.pairing.PairingStage.*
import ai.kohaku.mobile.network.pairing.PairingTarget
import ai.kohaku.mobile.network.pairing.PairingTransport
import ai.kohaku.mobile.network.pairing.StoredPairing
import ai.kohaku.mobile.network.pairing.TailscaleState
import java.util.PriorityQueue
import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class PairingCoordinatorTest {
    @Test
    fun qrPairingCompletesWithoutBridgePolling() = with(Fixture()) {
        assertEquals(IDLE, coordinator.state().stage)
        events.clear()
        assertEquals(STARTING_TRANSPORT, scan().stage)
        timer.advance(0)
        assertEquals(WAITING_APPROVAL, coordinator.state().stage)
        assertEquals("01234567", coordinator.state().comparisonCode)
        assertEquals(1_764_000_000L, coordinator.state().expiresAt)
        timer.advance(1_999)
        assertEquals(WAITING_APPROVAL, coordinator.state().stage)
        timer.advance(1)
        assertEquals(listOf(STARTING_TRANSPORT, COLLECTING, WAITING_APPROVAL, PROVING, PAIRED), stages())
        assertTrue(custody.load().target!!.paired)
    }

    @Test
    fun readyEventResumesPersistedQrImmediately() = with(Fixture(TailscaleState.NEEDS_LOGIN)) {
        val attempt = scan()
        timer.advance(0)
        assertEquals(TAILSCALE_LOGIN_REQUIRED, coordinator.state().stage)
        assertEquals(TailscaleState.NEEDS_LOGIN, coordinator.state().tailscaleState)
        assertTrue(coordinator.state().tailscaleHasLoginUrl)

        coordinator = newCoordinator()
        assertEquals(attempt.attemptId, coordinator.state().attemptId)
        timer.advance(0)
        events.clear()
        setReady()
        assertEquals(COLLECTING, coordinator.state().stage)
        assertFalse(coordinator.state().tailscaleHasLoginUrl)
        timer.advance(2_000)
        assertEquals(listOf(COLLECTING, WAITING_APPROVAL, PROVING, PAIRED), stages())
    }

    @Test
    fun resumeCheckReadsTailscaleStatusWithoutAnEvent() = with(Fixture(TailscaleState.NEEDS_LOGIN)) {
        scan()
        timer.advance(0)
        transport.connection = PairingConnection(1, TailscaleState.READY)
        assertEquals(COLLECTING, coordinator.checkResume().stage)
    }

    @Test
    fun expiredQrAfterLoginRequiresAFreshAttempt() = with(Fixture(TailscaleState.NEEDS_LOGIN)) {
        val expiredAttempt = scan()
        timer.advance(0)
        transport.collectResult = { throw NativeHttpException(410, "pairing_expired", "/mobile/v1/pair/collect") }
        setReady()
        timer.advance(0)
        val failed = coordinator.state()
        assertEquals(FAILED, failed.stage)
        assertEquals("rescan_qr", failed.error!!.remediation)
        assertFalse(failed.canRetry)

        transport.collectResult = ::collected
        events.clear()
        val freshAttempt = scan(id = SECOND_ID)
        assertNotEquals(expiredAttempt.attemptId, freshAttempt.attemptId)
        UUID.fromString(freshAttempt.attemptId)
        assertNull(freshAttempt.pairingId)
        timer.advance(0)
        assertEquals(listOf(STARTING_TRANSPORT, COLLECTING, WAITING_APPROVAL), stages())
        assertEquals(SECOND_ID, coordinator.state().pairingId)
        assertEquals(TailscaleState.READY, coordinator.state().tailscaleState)
        timer.advance(2_000)
        assertEquals(PAIRED, coordinator.state().stage)
    }

    @Test
    fun switchingDesktopKeepsInstallationAndTailscaleIdentity() = with(Fixture()) {
        val first = scan()
        timer.advance(0)
        val originalIdentity = transport.networkIdentity
        val second = scan("https://second.ts.net", SECOND_ID)
        timer.advance(0)

        assertEquals("https://second.ts.net", coordinator.state().origin)
        assertEquals(first.installationId, second.installationId)
        assertEquals(TailscaleState.READY, coordinator.state().tailscaleState)
        assertSame(originalIdentity, transport.networkIdentity)
        assertNotEquals(first.attemptId, second.attemptId)
    }

    @Test
    fun reportsInvalidCodeDenialAndMissingServeRoute() {
        data class Case(val status: Int, val code: String, val guidance: String, val retryable: Boolean)
        val cases = listOf(
            Case(403, "pairing_code_invalid", "edit_code", false),
            Case(403, "pairing_denied", "rescan_qr", false),
            Case(404, "not_found", "check_desktop_serve", true),
        )
        for (case in cases) with(Fixture()) {
            val manual = case.code == "pairing_code_invalid"
            val endpoint = if (manual) "code" else "collect"
            transport.collectResult = { throw NativeHttpException(case.status, case.code, "/mobile/v1/pair/$endpoint") }
            if (manual) coordinator.startWithCode(ORIGIN, "01234567") else scan()
            timer.advance(0)
            val state = coordinator.state()
            assertEquals(FAILED, state.stage)
            assertEquals(case.guidance, state.error!!.remediation)
            assertEquals(case.retryable, state.canRetry)
            if (case.status == 404) {
                assertTrue(state.error.message.contains("gateway"))
                assertTrue(state.error.message.contains("Tailscale Serve"))
                assertTrue(state.error.message.contains("/mobile/v1"))
            }
        }
    }

    @Test
    fun cancelClearsLocalAttemptWhenDesktopReturns4xx() = with(Fixture()) {
        scan()
        timer.advance(0)
        transport.cancelResult = {
            throw NativeHttpException(404, "pairing_not_found", "/mobile/v1/pair/cancel")
        }
        val identity = transport.networkIdentity
        assertEquals(CANCELLING, coordinator.cancel().stage)
        timer.advance(2_000)
        assertEquals(listOf(FIRST_ID), transport.cancelled)
        assertEquals(IDLE, coordinator.state().stage)
        assertEquals(ORIGIN, coordinator.state().origin)
        assertEquals(TailscaleState.READY, coordinator.state().tailscaleState)
        assertNull(custody.load().attempt)
        assertSame(identity, transport.networkIdentity)
    }

    @Test
    fun cancelBeforeCollectorSecretIsAvailable() = with(Fixture(TailscaleState.NEEDS_LOGIN)) {
        coordinator.startWithCode(ORIGIN, "12345678")
        timer.advance(0)
        coordinator.cancel()
        timer.advance(0)
        assertEquals(IDLE, coordinator.state().stage)
        assertEquals(TailscaleState.NEEDS_LOGIN, coordinator.state().tailscaleState)
        assertTrue(transport.cancelled.isEmpty())
        assertNull(custody.load().attempt)
    }

    @Test
    fun oldCollectionCannotReplaceNewAttempt() = with(Fixture()) {
        var replacement: PairingSnapshot? = null
        transport.collectResult = { attempt ->
            if (attempt.request.challenge!!.pairingId == FIRST_ID) replacement = scan(id = SECOND_ID)
            collected(attempt)
        }
        scan()
        timer.advance(0)
        assertEquals(replacement!!.attemptId, coordinator.state().attemptId)
        assertEquals(SECOND_ID, coordinator.state().pairingId)
        assertEquals(WAITING_APPROVAL, coordinator.state().stage)
        assertTrue(events.none { it.pairingId == FIRST_ID })
    }

    @Test
    fun approvalInFlightCannotCompleteAfterCancellation() = with(Fixture()) {
        transport.pollResult = {
            PairingPollResult.Approved {
                assertEquals(CANCELLING, coordinator.cancel().stage)
                credentials()
            }
        }
        scan()
        timer.advance(2_000)
        assertEquals(IDLE, coordinator.state().stage)
        assertFalse(custody.load().target!!.paired)
        assertTrue(PAIRED !in stages())
    }

    @Test
    fun expiredApprovalPollCanResumeAfterRenewal() = with(Fixture()) {
        scan()
        transport.pollResult = { throw expiredPoll() }
        timer.advance(2_000)
        assertEquals(EXPIRED_RENEWABLE, coordinator.state().stage)

        transport.pollResult = { attempt -> PairingPollResult.Waiting(attempt.progress!!.copy(expiresAt = 1_764_000_300L)) }
        timer.advance(4_999)
        assertEquals(EXPIRED_RENEWABLE, coordinator.state().stage)
        timer.advance(1)
        assertEquals(WAITING_APPROVAL, coordinator.state().stage)
        assertEquals(1_764_000_300L, coordinator.state().expiresAt)
    }

    @Test
    fun expiredApprovalPollingStopsAfterSixtyChecks() = with(Fixture()) {
        scan()
        transport.pollResult = { throw expiredPoll() }
        timer.advance(2_000)
        timer.advance(59 * 5_000L)
        assertEquals(EXPIRED_RENEWABLE, coordinator.state().stage)
        timer.advance(5_000)
        assertEquals(FAILED, coordinator.state().stage)
        assertEquals("rescan_qr", coordinator.state().error!!.remediation)
        timer.advance(5_000)
        assertEquals(61, transport.pollTimes.size)
        assertTrue(transport.pollTimes.zipWithNext().all { (first, second) -> second - first == 5_000L })
    }

    private class Fixture(state: TailscaleState = TailscaleState.READY) {
        val timer = ManualScheduler()
        val custody = TestCustody()
        val transport = TestTransport(state, timer)
        val events = mutableListOf<PairingSnapshot>()
        var coordinator = newCoordinator()

        fun newCoordinator() = PairingCoordinator(transport, custody, timer) { events.add(it) }
        fun stages(): List<PairingStage> = events.map { it.stage }.filterIndexed { index, stage ->
            index == 0 || events[index - 1].stage != stage
        }
        fun scan(origin: String = ORIGIN, id: String = FIRST_ID): PairingSnapshot =
            coordinator.startWithLink("kohaku://pair?origin=$origin&challenge=$id.$NONCE.$SECRET", PairingMethod.QR)

        fun setReady() {
            transport.connection = PairingConnection(1, TailscaleState.READY)
            coordinator.onTransportChanged(transport.connection)
        }
    }

    private class TestCustody : PairingCustody {
        private var stored = StoredPairing()
        private var generation = 0L

        override fun load() = stored

        override fun begin(request: PairingRequest, attemptId: String): PairingAttempt {
            val target = PairingTarget(request.origin, INSTALLATION, false)
            val attempt = PairingAttempt(attemptId, ++generation, target, request)
            stored = StoredPairing(target, attempt)
            return attempt
        }

        override fun saveCollected(attempt: PairingAttempt, collected: CollectedPairing): PairingAttempt {
            check(attempt.generation == generation)
            val updated = attempt.copy(
                request = attempt.request.copy(challenge = collected.challenge), progress = collected.progress,
            )
            stored = stored.copy(attempt = updated)
            return updated
        }

        override fun complete(attempt: PairingAttempt, handle: Long, credentials: PairingCredentials) {
            check(attempt.generation == generation)
            stored = StoredPairing(attempt.target.copy(paired = true))
        }

        override fun fence(): Long {
            stored = stored.copy(cancelling = true)
            return ++generation
        }

        override fun resetUnpairedAttempt() {
            ++generation
            stored = StoredPairing(stored.target)
        }
    }

    private class TestTransport(state: TailscaleState, val timer: ManualScheduler) : PairingTransport {
        val networkIdentity = Any()
        var connection = PairingConnection(1, state, state == TailscaleState.NEEDS_LOGIN)
        var collectResult: (PairingAttempt) -> CollectedPairing = ::collected
        var pollResult: (PairingAttempt) -> PairingPollResult = { PairingPollResult.Approved(::credentials) }
        var cancelResult: () -> Unit = {}
        val pollTimes = mutableListOf<Long>()
        val cancelled = mutableListOf<String>()

        override fun start(attempt: PairingAttempt) = connection
        override fun current() = connection
        override fun collect(attempt: PairingAttempt, handle: Long) = collectResult(attempt)
        override fun poll(attempt: PairingAttempt, handle: Long): PairingPollResult {
            pollTimes.add(timer.now)
            return pollResult(attempt)
        }

        override fun cancel(attempt: PairingAttempt, generation: Long) {
            cancelled.add(attempt.request.challenge!!.pairingId)
            cancelResult()
        }
    }

    private class ManualScheduler : PairingScheduler {
        private data class Task(val at: Long, val order: Long, val run: () -> Unit)
        private val pending = PriorityQueue(compareBy<Task> { it.at }.thenBy { it.order })
        private var order = 0L
        var now = 0L
            private set

        override fun schedule(delayMillis: Long, action: () -> Unit) {
            pending.add(Task(now + delayMillis, order++, action))
        }

        fun advance(millis: Long) {
            val until = now + millis
            while (pending.peek()?.let { it.at <= until } == true) {
                val task = pending.remove()
                now = task.at
                task.run()
            }
            now = until
        }
    }

    private companion object {
        const val ORIGIN = "https://first.ts.net"
        const val INSTALLATION = "00000000-0000-4000-8000-000000000001"
        const val FIRST_ID = "00000000-0000-4000-8000-000000000002"
        const val SECOND_ID = "00000000-0000-4000-8000-000000000003"
        const val NONCE = "fixture-nonce-000"
        const val SECRET = "fixture-secret-000"

        fun collected(attempt: PairingAttempt): CollectedPairing {
            val challenge = attempt.request.challenge ?: PairingChallenge(FIRST_ID, NONCE, SECRET)
            return CollectedPairing(challenge, PairingProgress(challenge.pairingId, "01234567", 1_764_000_000))
        }

        fun credentials() = PairingCredentials(INSTALLATION, "test-access", "test-refresh", 600, 0)
        fun expiredPoll() = NativeHttpException(410, "pairing_expired", "/mobile/v1/pair/poll")
    }
}
