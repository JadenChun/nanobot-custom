package ai.kohaku.mobile.network.pairing

import ai.kohaku.mobile.links.PairingChallenge
import ai.kohaku.mobile.links.PairingChallengeCodec
import ai.kohaku.mobile.links.PairingLinkParser
import java.net.URI
import java.util.UUID

internal data class PairingRequest(
    val method: PairingMethod,
    val origin: String,
    val label: String,
    val challenge: PairingChallenge? = null,
    val code: String? = null,
    val installationId: String? = null,
)

internal data class PairingTarget(val origin: String, val installationId: String, val paired: Boolean)

internal data class PairingProgress(
    val pairingId: String,
    val comparisonCode: String? = null,
    val expiresAt: Long? = null,
)

internal data class PairingAttempt(
    val id: String,
    val generation: Long,
    val target: PairingTarget,
    val request: PairingRequest,
    val progress: PairingProgress? = null,
)

internal data class StoredPairing(
    val target: PairingTarget? = null,
    val attempt: PairingAttempt? = null,
    val cancelling: Boolean = false,
)

internal data class PairingConnection(
    val handle: Long = 0,
    val state: TailscaleState = TailscaleState.IDLE,
    val hasLoginUrl: Boolean = false,
)

internal data class PairingCredentials(
    val installationId: String,
    val accessToken: String,
    val refreshToken: String,
    val expiresIn: Int,
    val refreshExpiresIn: Int,
)

internal data class CollectedPairing(val challenge: PairingChallenge, val progress: PairingProgress)

internal sealed interface PairingPollResult {
    data class Waiting(val progress: PairingProgress) : PairingPollResult
    class Approved(val prove: () -> PairingCredentials) : PairingPollResult
}

internal interface PairingTransport {
    fun start(attempt: PairingAttempt): PairingConnection
    fun current(): PairingConnection
    fun collect(attempt: PairingAttempt, handle: Long): CollectedPairing
    fun poll(attempt: PairingAttempt, handle: Long): PairingPollResult
    fun cancel(attempt: PairingAttempt, generation: Long)
}

internal interface PairingCustody {
    fun load(): StoredPairing
    fun begin(request: PairingRequest, attemptId: String): PairingAttempt
    fun saveCollected(attempt: PairingAttempt, collected: CollectedPairing): PairingAttempt
    fun complete(attempt: PairingAttempt, handle: Long, credentials: PairingCredentials)
    fun fence(): Long
    fun resetUnpairedAttempt()
}

internal fun interface PairingScheduler {
    fun schedule(delayMillis: Long, action: () -> Unit)
}

/**
 * The coordinator serializes state changes and custody writes. Network calls run
 * outside that monitor. Each completion must still own the same run before it
 * can advance the state or write to custody.
 */
internal class PairingCoordinator(
    private val transport: PairingTransport,
    private val custody: PairingCustody,
    private val scheduler: PairingScheduler,
    private val onSnapshot: (PairingSnapshot) -> Unit,
) {
    private class Run(var attempt: PairingAttempt) {
        var handle: Long = 0
        var expiredChecks: Int = 0
        var cancelling: Boolean = false
    }

    private var hydrated = false
    private var run: Run? = null
    private var snapshot = PairingSnapshot()

    @Synchronized
    fun state(): PairingSnapshot {
        if (!hydrated) {
            hydrated = true
            try {
                val stored = custody.load()
                setTarget(stored.target, transport.current())
                stored.attempt?.let {
                    val restored = Run(it)
                    run = restored
                    if (stored.cancelling) cancel() else startTransport(restored)
                }
            } catch (failure: Exception) {
                fail(PairingErrorMapper.map(failure, snapshot.tailscaleState))
            }
        }
        return snapshot
    }

    @Synchronized
    fun startWithLink(
        raw: String,
        method: PairingMethod = PairingMethod.LINK,
        label: String = "",
        restore: Boolean = false,
    ): PairingSnapshot {
        val request = try {
            val link = PairingLinkParser.parse(raw)
            PairingRequest(method, link.origin, label, PairingChallengeCodec.decode(link.challenge))
        } catch (_: Exception) {
            return invalid(method)
        }
        val restoring = restore && !hydrated
        state()
        val active = run?.attempt?.request
        if (restoring && active?.origin == request.origin && active.challenge == request.challenge) return snapshot
        // The link describes the offer. The phone retains its own installation identity.
        return start(request)
    }

    @Synchronized
    fun startWithCode(origin: String, code: String, label: String = "", installationId: String? = null): PairingSnapshot {
        val normalized = httpsOrigin(origin)
        if (normalized == null || !code.matches(Regex("[0-9]{8}"))) return invalid(PairingMethod.CODE)
        return start(PairingRequest(PairingMethod.CODE, normalized, label, code = code, installationId = installationId))
    }

    @Synchronized
    fun startWithChallenge(origin: String, installationId: String, raw: String, label: String): PairingSnapshot {
        val normalized = httpsOrigin(origin) ?: return invalid(PairingMethod.LINK)
        val challenge = try {
            PairingChallengeCodec.decode(raw)
        } catch (_: Exception) {
            return invalid(PairingMethod.LINK)
        }
        return start(PairingRequest(PairingMethod.LINK, normalized, label, challenge, installationId = installationId))
    }

    private fun start(request: PairingRequest): PairingSnapshot {
        state()
        if (snapshot.stage == PairingStage.PAIRED) return snapshot
        try {
            val attempt = custody.begin(request, UUID.randomUUID().toString())
            val next = Run(attempt)
            run = next
            startTransport(next)
        } catch (failure: Exception) {
            run = null
            fail(PairingErrorMapper.map(failure, snapshot.tailscaleState))
        }
        return snapshot
    }

    @Synchronized
    fun retry(): PairingSnapshot {
        state()
        val current = run
        if (current == null) {
            if (snapshot.stage == PairingStage.FAILED && snapshot.canRetry) {
                hydrated = false
                return state()
            }
            return snapshot
        }
        when {
            current.cancelling && snapshot.stage == PairingStage.FAILED && snapshot.canRetry -> cancel()
            snapshot.stage == PairingStage.TAILSCALE_LOGIN_REQUIRED -> checkResume()
            snapshot.stage == PairingStage.FAILED && snapshot.canRetry -> {
                val next = Run(current.attempt)
                run = next
                startTransport(next)
            }
        }
        return snapshot
    }

    @Synchronized
    fun checkResume(): PairingSnapshot {
        state()
        onTransportChanged(transport.current())
        return snapshot
    }

    @Synchronized
    fun onTransportChanged(connection: PairingConnection) {
        if (!hydrated) return
        val current = run
        if (current != null && current.handle != 0L && current.handle != connection.handle) return
        if (current != null && current.handle != 0L &&
            !current.cancelling &&
            snapshot.stage in setOf(PairingStage.STARTING_TRANSPORT, PairingStage.TAILSCALE_LOGIN_REQUIRED)
        ) {
            applyConnection(current, connection)
        } else {
            publish(snapshot.withConnection(connection))
            if (current?.cancelling == true && snapshot.stage == PairingStage.FAILED &&
                snapshot.error?.remediation == "open_tailscale_login" && connection.state == TailscaleState.READY
            ) cancel()
        }
    }

    @Synchronized
    fun onTransportReplaced(previousHandle: Long) {
        val current = run
        if (current != null && current.handle == previousHandle && !current.cancelling) {
            val next = Run(current.attempt)
            run = next
            startTransport(next)
        }
    }

    @Synchronized
    fun cancel(): PairingSnapshot {
        state()
        if (snapshot.stage == PairingStage.PAIRED || snapshot.stage == PairingStage.CANCELLING) return snapshot
        val current = run ?: return clearAttempt()
        val generation = try {
            custody.fence()
        } catch (_: Exception) {
            fail(PairingErrorMapper.custody())
            return snapshot
        }
        current.cancelling = true
        transition(current, PairingStage.CANCELLING)
        scheduler.schedule(0) {
            if (!owns(current, PairingStage.CANCELLING)) return@schedule
            val failure = try {
                if (current.attempt.request.challenge != null) transport.cancel(current.attempt, generation)
                null
            } catch (failure: Exception) {
                failure
            }
            synchronized(this) {
                if (!owns(current, PairingStage.CANCELLING)) return@synchronized
                if (failure == null || failure is NativeHttpException && failure.status in 400..499) {
                    clearAttempt()
                } else {
                    val connection = transport.current()
                    current.handle = connection.handle
                    val error = if (connection.state == TailscaleState.NEEDS_LOGIN) {
                        PairingErrorMapper.tailscale(needsLogin = true)
                    } else PairingErrorMapper.map(failure, connection.state)
                    transition(current, PairingStage.FAILED, error, connection)
                }
            }
        }
        return snapshot
    }

    @Synchronized
    fun resetToIdle(): PairingSnapshot {
        state()
        if (snapshot.stage !in setOf(PairingStage.IDLE, PairingStage.FAILED, PairingStage.EXPIRED_RENEWABLE)) {
            return snapshot
        }
        return clearAttempt()
    }

    private fun clearAttempt(): PairingSnapshot {
        try {
            custody.resetUnpairedAttempt()
            run = null
            setTarget(custody.load().target, transport.current())
        } catch (_: Exception) {
            run = null
            fail(PairingErrorMapper.custody())
        }
        return snapshot
    }

    @Synchronized
    fun invalidate() {
        run = null
        hydrated = false
        publish(PairingSnapshot())
    }

    private fun startTransport(current: Run) {
        transition(current, PairingStage.STARTING_TRANSPORT, connection = PairingConnection(state = TailscaleState.STARTING))
        scheduler.schedule(0) {
            synchronized(this) {
                if (!owns(current, PairingStage.STARTING_TRANSPORT)) return@synchronized
                try {
                    val connection = transport.start(current.attempt)
                    current.handle = connection.handle
                    applyConnection(current, connection)
                } catch (_: Exception) {
                    fail(PairingErrorMapper.tailscale())
                }
            }
        }
        scheduler.schedule(90_000) {
            synchronized(this) {
                if (owns(current, PairingStage.STARTING_TRANSPORT)) fail(PairingErrorMapper.tailscale())
            }
        }
    }

    private fun applyConnection(current: Run, connection: PairingConnection) {
        when (connection.state) {
            TailscaleState.READY -> {
                if (current.attempt.progress == null) {
                    transition(current, PairingStage.COLLECTING, connection = connection)
                    scheduler.schedule(0) { collect(current) }
                } else {
                    transition(current, PairingStage.WAITING_APPROVAL, connection = connection)
                    schedulePoll(current, 2_000)
                }
            }
            TailscaleState.NEEDS_LOGIN -> transition(
                current, PairingStage.TAILSCALE_LOGIN_REQUIRED, PairingErrorMapper.tailscale(needsLogin = true), connection,
            )
            TailscaleState.ERROR, TailscaleState.IDLE -> transition(
                current, PairingStage.FAILED, PairingErrorMapper.tailscale(), connection,
            )
            TailscaleState.STARTING -> publish(snapshot.withConnection(connection))
        }
    }

    private fun collect(current: Run) {
        if (!owns(current, PairingStage.COLLECTING)) return
        try {
            val collected = transport.collect(current.attempt, current.handle)
            synchronized(this) {
                if (!owns(current, PairingStage.COLLECTING)) return
                current.attempt = custody.saveCollected(current.attempt, collected)
                transition(current, PairingStage.WAITING_APPROVAL)
                schedulePoll(current, 2_000)
            }
        } catch (failure: Exception) {
            failIfCurrent(current, PairingStage.COLLECTING, failure)
        }
    }

    private fun schedulePoll(current: Run, delayMillis: Long) {
        val stage = snapshot.stage
        scheduler.schedule(delayMillis) { poll(current, stage) }
    }

    private fun poll(current: Run, stage: PairingStage) {
        if (!owns(current, stage)) return
        try {
            when (val result = transport.poll(current.attempt, current.handle)) {
                is PairingPollResult.Waiting -> synchronized(this) {
                    if (!owns(current, stage)) return
                    current.expiredChecks = 0
                    current.attempt = current.attempt.copy(progress = result.progress)
                    transition(current, PairingStage.WAITING_APPROVAL)
                    schedulePoll(current, 2_000)
                }
                is PairingPollResult.Approved -> {
                    synchronized(this) {
                        if (!owns(current, stage)) return
                        transition(current, PairingStage.PROVING)
                    }
                    prove(current, result)
                }
            }
        } catch (failure: Exception) {
            synchronized(this) {
                if (!owns(current, stage)) return
                if (failure is NativeHttpException && failure.status == 410 &&
                    failure.serverCode == "pairing_expired" && current.expiredChecks < 60
                ) {
                    current.expiredChecks++
                    val error = PairingErrorMapper.map(failure, snapshot.tailscaleState).copy(
                        message = "Renew the pairing code on your desktop to continue.",
                    )
                    transition(current, PairingStage.EXPIRED_RENEWABLE, error)
                    schedulePoll(current, 5_000)
                } else {
                    failIfCurrent(current, stage, failure)
                }
            }
        }
    }

    private fun prove(current: Run, approval: PairingPollResult.Approved) {
        if (!owns(current, PairingStage.PROVING)) return
        try {
            val credentials = approval.prove()
            synchronized(this) {
                if (!owns(current, PairingStage.PROVING)) return
                custody.complete(current.attempt, current.handle, credentials)
                transition(current, PairingStage.PAIRED)
                run = null
            }
        } catch (failure: Exception) {
            failIfCurrent(current, PairingStage.PROVING, failure)
        }
    }

    @Synchronized
    private fun owns(current: Run, stage: PairingStage): Boolean = current === run && snapshot.stage == stage

    @Synchronized
    private fun failIfCurrent(current: Run, stage: PairingStage, failure: Exception) {
        if (!owns(current, stage)) return
        val connection = transport.current()
        if (connection.handle == current.handle && connection.state == TailscaleState.NEEDS_LOGIN) {
            transition(current, PairingStage.TAILSCALE_LOGIN_REQUIRED, PairingErrorMapper.tailscale(needsLogin = true), connection)
            return
        }
        fail(PairingErrorMapper.map(failure, snapshot.tailscaleState))
    }

    private fun invalid(method: PairingMethod): PairingSnapshot {
        state()
        val error = PairingErrorMapper.invalid(method)
        if (run != null || snapshot.stage == PairingStage.PAIRED) return snapshot.copy(error = error)
        publish(snapshot.copy(stage = PairingStage.FAILED, method = method, error = error, canRetry = false))
        return snapshot
    }

    private fun fail(error: PairingError) {
        publish(snapshot.copy(stage = PairingStage.FAILED, canCancel = run != null, canRetry = error.retryable, error = error))
    }

    private fun setTarget(target: PairingTarget?, connection: PairingConnection) {
        publish(PairingSnapshot(
            stage = if (target?.paired == true) PairingStage.PAIRED else PairingStage.IDLE,
            origin = target?.origin,
            installationId = target?.installationId,
        ).withConnection(connection))
    }

    private fun transition(
        current: Run,
        stage: PairingStage,
        error: PairingError? = null,
        connection: PairingConnection? = null,
    ) {
        val attempt = current.attempt
        val next = snapshot.copy(
            stage = stage,
            attemptId = attempt.id,
            method = attempt.request.method,
            origin = attempt.target.origin,
            installationId = attempt.target.installationId,
            pairingId = attempt.progress?.pairingId,
            comparisonCode = attempt.progress?.comparisonCode,
            expiresAt = attempt.progress?.expiresAt,
            canCancel = stage !in setOf(PairingStage.CANCELLING, PairingStage.PAIRED),
            canRetry = error?.retryable == true,
            error = error,
        )
        publish(if (connection == null) next else next.withConnection(connection))
    }

    private fun publish(next: PairingSnapshot) {
        if (next == snapshot) return
        snapshot = next
        onSnapshot(next)
    }

    private fun PairingSnapshot.withConnection(connection: PairingConnection) =
        copy(tailscaleState = connection.state, tailscaleHasLoginUrl = connection.hasLoginUrl)

    private fun httpsOrigin(value: String): String? {
        val origin = value.trim().trimEnd('/')
        val uri = runCatching { URI(origin) }.getOrNull() ?: return null
        if (!uri.scheme.equals("https", ignoreCase = true) || uri.host.isNullOrBlank() || uri.userInfo != null ||
            !uri.rawPath.isNullOrEmpty() || uri.rawQuery != null || uri.rawFragment != null
        ) return null
        return "https://${uri.rawAuthority}"
    }
}
