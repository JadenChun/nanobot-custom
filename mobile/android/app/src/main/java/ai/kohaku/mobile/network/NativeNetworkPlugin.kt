package ai.kohaku.mobile.network

import ai.kohaku.nativenetwork.NativeNetwork
import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Base64
import android.util.Log
import com.getcapacitor.BridgeActivity
import com.getcapacitor.JSObject
import com.getcapacitor.PermissionState
import com.getcapacitor.Plugin
import androidx.browser.customtabs.CustomTabsIntent
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.ActivityCallback
import com.getcapacitor.annotation.CapacitorPlugin
import com.getcapacitor.annotation.Permission
import com.getcapacitor.annotation.PermissionCallback
import java.net.URI
import java.nio.charset.StandardCharsets
import java.text.Normalizer
import ai.kohaku.mobile.attachments.AttachmentLimits
import ai.kohaku.mobile.attachments.AttachmentPolicy
import ai.kohaku.mobile.MainActivity
import ai.kohaku.mobile.RunKeepAliveRegistry
import ai.kohaku.mobile.RunKeepAliveService
import ai.kohaku.mobile.links.PairingChallenge
import ai.kohaku.mobile.links.PairingChallengeCodec
import ai.kohaku.mobile.links.PairingLinkParser
import ai.kohaku.mobile.media.MediaHandleStore
import ai.kohaku.mobile.network.pairing.CollectedPairing
import ai.kohaku.mobile.network.pairing.NativeHttpException
import ai.kohaku.mobile.network.pairing.NativeTransportException
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
import ai.kohaku.mobile.network.pairing.PairingTarget
import ai.kohaku.mobile.network.pairing.PairingTransport
import ai.kohaku.mobile.network.pairing.StoredPairing
import ai.kohaku.mobile.network.pairing.TailscaleState
import java.util.Locale
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import androidx.activity.result.ActivityResult
import androidx.core.content.FileProvider
import com.google.zxing.integration.android.IntentIntegrator
import java.io.ByteArrayOutputStream

@CapacitorPlugin(
    name = "NativeNetwork",
    permissions = [Permission(alias = "camera", strings = [Manifest.permission.CAMERA])],
)
class NativeNetworkPlugin : Plugin() {
    private companion object {
        private const val NOTIFICATION_GRACE_MS = 2_000L
        private const val PAIRING_NOTIFICATION_CHANNEL = "kohaku_pairing_status"
        private const val PAIRING_NOTIFICATION_ID = 4102
        private const val TAG = "NanobotNetwork"
        private const val HOSTNAME_PREFIX = "kohaku-"
        private const val MAX_DNS_LABEL_LENGTH = 63
        private val combiningMarks = Regex("\\p{M}+")
        private val hostnameSeparators = Regex("[^a-z0-9]+")
    }

    @Volatile private var transportHandle: Long = 0L
    @Volatile private var stateRecoveryAttempted = false
    private val pollExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val retryExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val retryTimer = ScheduledRetryTimer(retryExecutor)
    private val pairingExecutor = Executors.newSingleThreadExecutor()
    private val operationExecutor = Executors.newCachedThreadPool()
    @Volatile private var requests = NativeRequestCorrelator<JSONObject>()
    @Volatile private var admission = NativeRequestAdmission(timer = retryTimer)
    private data class PendingWsFrame(val kind: Int, val data: ByteArrayOutputStream)
    private val pendingWsFrames = ConcurrentHashMap<Long, PendingWsFrame>()
    @Volatile private var loginTabOpen = false
    @Volatile private var isActivityResumed = false
    @Volatile private var destroyed = false
    private var transportTarget: Pair<String, String>? = null
    private var legacyPendingLink: PairingSnapshot? = null
    private val pairingWaiters = mutableListOf<(PairingSnapshot) -> Boolean>()
    private data class RefreshAttempt(
        val origin: String,
        val installationId: String,
        val generation: Long,
        val handle: Long,
    ) {
        val result = CompletableFuture<Unit>()
    }
    private var refreshInFlight: RefreshAttempt? = null
    private var pollTask: ScheduledFuture<*>? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val custody by lazy { CustodyStore(context) }
    private val mediaHandles by lazy { MediaHandleStore(context) }
    private val attachmentLimits = AttachmentLimits()
    private var pendingCameraFile: java.io.File? = null
    private var networkCallback: ConnectivityManager.NetworkCallback? = null

    private val stopIdleKeepAlive = Runnable {
        if (RunKeepAliveRegistry.isEmpty()) RunKeepAliveService.stop(context.applicationContext)
    }

    override fun load() {
        super.load()
        // Capacitor keeps the default tone when createChannel omits sound.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val silent = NotificationChannel(
                "kohaku_task_alerts_silent", "Task updates (silent)", NotificationManager.IMPORTANCE_DEFAULT
            )
            silent.setSound(null, null)
            silent.enableVibration(false)
            context.getSystemService(NotificationManager::class.java).createNotificationChannel(silent)
        }
        registerNetworkCallback()
    }

    override fun handleOnResume() {
        super.handleOnResume()
        isActivityResumed = true
        loginTabOpen = false
        context.getSystemService(NotificationManager::class.java).cancel(PAIRING_NOTIFICATION_ID)
        dispatchPairing {
            readPairingState()
            pairingCoordinator.checkResume()
        }
        notifyNativeNetworkChanged()
    }

    override fun handleOnPause() {
        isActivityResumed = false
        super.handleOnPause()
    }

    private fun registerNetworkCallback() {
        try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return
            val callback = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    notifyNativeNetworkChanged()
                }

                override fun onLost(network: Network) {
                    notifyNativeNetworkChanged()
                }

                override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
                    notifyNativeNetworkChanged()
                }
            }
            cm.registerDefaultNetworkCallback(callback)
            networkCallback = callback
        } catch (exception: Exception) {
            Log.w(TAG, "could not register default network callback", exception)
        }
    }

    private fun unregisterNetworkCallback() {
        val callback = networkCallback ?: return
        networkCallback = null
        try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            cm?.unregisterNetworkCallback(callback)
        } catch (_: Exception) {
            // ignore unregister errors on teardown
        }
    }

    private fun notifyNativeNetworkChanged() {
        val current = transportHandle
        if (current == 0L) return
        operationExecutor.execute {
            try {
                if (transportHandle == current) {
                    NativeNetwork.networkChanged(current)
                }
            } catch (_: Throwable) {
                // Best-effort flush on OS network handoff.
            }
        }
    }

    @PluginMethod
    fun abiVersion(call: PluginCall) {
        call.resolve(JSObject().put("version", NativeNetwork.abiVersion()))
    }

    /**
     * Keep the process alive while a phone-started run is active. Android
     * requires the foreground service to be started while the app is visible;
     * a background start is rejected and treated as best-effort.
     */
    @PluginMethod
    fun startRunKeepAlive(call: PluginCall) {
        val chatId = call.getString("chatId")?.trim()
        if (chatId.isNullOrBlank()) {
            call.reject("chatId is required")
            return
        }
        mainHandler.removeCallbacks(stopIdleKeepAlive)
        RunKeepAliveRegistry.add(chatId)
        try {
            RunKeepAliveService.start(context.applicationContext)
            call.resolve(JSObject().put("running", true))
        } catch (exception: Exception) {
            RunKeepAliveRegistry.remove(chatId)
            call.reject(exception.message ?: "foreground service start failed")
        }
    }

    @PluginMethod
    fun stopRunKeepAlive(call: PluginCall) {
        val chatId = call.getString("chatId")?.trim()
        if (chatId.isNullOrBlank()) {
            RunKeepAliveRegistry.clear()
        } else {
            RunKeepAliveRegistry.remove(chatId)
        }
        if (RunKeepAliveRegistry.isEmpty()) {
            // Let the completion notification cross the bridge before the
            // foreground service is removed.
            mainHandler.removeCallbacks(stopIdleKeepAlive)
            mainHandler.postDelayed(stopIdleKeepAlive, NOTIFICATION_GRACE_MS)
        }
        call.resolve()
    }

    @PluginMethod
    @Synchronized
    fun refresh(call: PluginCall) {
        val generation = custody.attemptGeneration()
        val record = try {
            custody.load()
        } catch (exception: Exception) {
            call.reject(exception.message ?: "custody is unavailable")
            return
        }
        if (record == null || record.revocationPending ||
            (record.accessToken.isBlank() && record.refreshToken.isBlank())
        ) {
            call.reject("No paired device.")
            return
        }
        val candidate = RefreshAttempt(record.origin, record.installationId, generation, transportHandle)
        val existing = refreshInFlight
        val attempt = if (existing != null && existing == candidate) {
            existing
        } else {
            refreshInFlight = candidate
            operationExecutor.execute {
                try {
                    performRefresh(record, candidate)
                    candidate.result.complete(Unit)
                } catch (exception: Exception) {
                    candidate.result.completeExceptionally(settleRefreshFailure(candidate, exception))
                } finally {
                    synchronized(this) {
                        if (refreshInFlight === candidate) refreshInFlight = null
                    }
                }
            }
            candidate
        }
        attempt.result.whenComplete { _, error ->
            mainHandler.post {
                if (error == null) {
                    call.resolve(JSObject().put("refreshed", true))
                } else {
                    call.reject(error.message ?: "Access renewal failed.")
                }
            }
        }
    }

    private fun settleRefreshFailure(attempt: RefreshAttempt, failure: Exception): Exception {
        val status = httpStatus(failure)
        if (!RefreshAuthority.requiresRepair(status, failure.message.orEmpty())) return failure
        val clearedGeneration = try {
            synchronized(this) {
                custody.withAuthority(attempt.generation) {
                    val current = custody.load()
                    if (current?.installationId != attempt.installationId || current.origin != attempt.origin ||
                        attempt.handle == 0L || transportHandle != attempt.handle
                    ) return@withAuthority null
                    custody.cancelPairing()
                    RunKeepAliveRegistry.clear()
                    RunKeepAliveService.stop(context.applicationContext)
                    NativeNetwork.mediaPurge(attempt.handle)
                    stopInternal()
                    custody.logout(serverReachable = true)
                    custody.purgeMedia()
                    custody.attemptGeneration()
                }
            }
        } catch (_: Exception) {
            null
        } ?: return failure
        dispatchPairing {
            if (custody.attemptGeneration() == clearedGeneration) pairingCoordinator.invalidate()
        }
        mainHandler.post {
            synchronized(this) {
                try {
                    custody.withAuthority(clearedGeneration) {
                        if (transportHandle == 0L) {
                            (activity as? BridgeActivity)?.getBridge()?.getWebView()?.reload()
                        }
                    }
                } catch (_: IllegalStateException) {
                    // A newer pairing already owns the app.
                }
            }
        }
        val message = if (status == 426) "pairing upgrade required" else "pairing was revoked"
        return IllegalStateException(message, failure)
    }

    private fun performRefresh(record: CustodyRecord, attempt: RefreshAttempt) {
        custody.requireAuthority(attempt.generation)
        val useDeviceProof = record.refreshToken.isBlank()
        val refreshed = if (useDeviceProof) {
            recoverAccess(record, attempt.generation, attempt.handle)
        } else {
            refreshBearer(record, attempt)
        }
        synchronized(this) {
            check(attempt.handle == transportHandle) { "Native transport changed." }
            custody.withAuthority(attempt.generation) {
                val saved = custody.saveRefreshedTokens(
                    attempt.origin,
                    attempt.installationId,
                    refreshed.getString("access_token"),
                    if (useDeviceProof) refreshed.optString("refresh_token") else refreshed.getString("refresh_token"),
                    refreshed.optInt("expires_in", 600),
                    refreshed.optInt("refresh_expires_in", if (useDeviceProof) 0 else 30 * 24 * 60 * 60),
                    expectedGeneration = attempt.generation,
                )
                NativeNetwork.mediaSetToken(attempt.handle, saved.accessToken)
            }
        }
    }

    private fun refreshBearer(record: CustodyRecord, attempt: RefreshAttempt): JSONObject {
        return try {
            requestRefresh(record, attempt.handle)
        } catch (first: Exception) {
            if (!isUnauthorized(first)) throw first
            custody.requireAuthority(attempt.generation)
            val latest = custody.load()
            if (latest == null || latest.revocationPending || latest.installationId != attempt.installationId ||
                latest.origin != attempt.origin || latest.refreshToken.isBlank() || latest.refreshToken == record.refreshToken
            ) throw first
            requestRefresh(latest, attempt.handle)
        }
    }

    @PluginMethod
    fun start(call: PluginCall) {
        val origin = call.getString("origin")
        val installationId = call.getString("installationId")
        if (origin.isNullOrBlank() || installationId.isNullOrBlank()) {
            call.reject("origin and installationId are required")
            return
        }
        try {
            synchronized(this) {
                val record = requireTransportTarget(origin, installationId)
                check(record.accessToken.isNotBlank()) { "No paired device." }
                val current = transportHandle
                val state = if (current == 0L) "" else
                    NativeNetwork.status(current)?.let { JSONObject(String(it, Charsets.UTF_8)).optString("state") }
                if (state !in setOf("starting", "needs_login", "ready")) startTransport(record)
            }
        } catch (exception: Exception) {
            call.reject(exception.message ?: "native custody rejected the installation")
            return
        }
        call.resolve(JSObject().put("state", "starting"))
    }

    private fun defaultDeviceLabel(): String {
        val configured = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
            try {
                Settings.Global.getString(context.contentResolver, Settings.Global.DEVICE_NAME)
            } catch (_: Exception) {
                null
            }
        } else {
            null
        }
        val model = listOf(Build.MANUFACTURER, Build.MODEL)
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .trim()
        return (configured?.trim()?.takeIf { it.isNotEmpty() } ?: model.ifEmpty { "Android device" })
            .take(64)
    }

    private fun tailscaleHostname(): String {
        val maxDeviceLength = MAX_DNS_LABEL_LENGTH - HOSTNAME_PREFIX.length
        val device = sequenceOf(defaultDeviceLabel(), Build.MODEL, Build.DEVICE)
            .map { label ->
                Normalizer.normalize(label, Normalizer.Form.NFD)
                    .replace(combiningMarks, "")
                    .lowercase(Locale.ROOT)
                    .replace(hostnameSeparators, "-")
                    .trim('-')
                    .take(maxDeviceLength)
                    .trimEnd('-')
            }
            .firstOrNull { it.isNotEmpty() } ?: "android"
        return HOSTNAME_PREFIX + device
    }

    @PluginMethod
    fun pairingIdentity(call: PluginCall) {
        try {
            val record = custody.load()
            val installation = record?.installationId ?: custody.installationIdentity()
            val result = JSObject().put("installationId", installation)
            if (record != null) {
                result.put("origin", record.origin)
                result.put("hasAccess", record.accessToken.isNotBlank())
            }
            call.resolve(result)
        } catch (exception: Exception) {
            call.reject(exception.message ?: "pairing identity is unavailable")
        }
    }

    private val pairingCoordinatorDelegate = lazy {
        PairingCoordinator(
            pairingTransport, pairingCustody,
            PairingScheduler { delay, action ->
                retryExecutor.schedule({ if (!destroyed) operationExecutor.execute(action) }, delay, TimeUnit.MILLISECONDS)
            },
        ) { snapshot ->
            mainHandler.post {
                if (!destroyed) notifyListeners("pairingStateChanged", snapshot.toJSObject())
            }
            dispatchPairing(::settlePairingWaiters)
        }
    }
    private val pairingCoordinator: PairingCoordinator by pairingCoordinatorDelegate

    private fun settlePairingWaiters() {
        if (pairingWaiters.isEmpty()) return
        val current = pairingCoordinator.state()
        pairingWaiters.removeAll { it(current) }
    }

    private fun dispatchPairing(action: () -> Unit) {
        if (destroyed) return
        try {
            pairingExecutor.execute {
                if (!destroyed) action()
            }
        } catch (failure: RejectedExecutionException) {
            if (!destroyed) throw failure
        }
    }

    private fun pairingCall(call: PluginCall, action: () -> Unit) = dispatchPairing {
        try {
            action()
        } catch (_: Exception) {
            call.reject("Pairing is unavailable.")
        }
    }

    private fun readPairingState(): PairingSnapshot {
        val raw = MainActivity.pendingPairingLink ?: return pairingCoordinator.state()
        MainActivity.pendingPairingLink = null
        return startPairingLink(raw, PairingMethod.LINK, restore = true)
    }

    private fun startPairingLink(raw: String, method: PairingMethod, restore: Boolean = false): PairingSnapshot =
        pairingCoordinator.startWithLink(raw, method, defaultDeviceLabel(), restore).also { legacyPendingLink = it }

    @PluginMethod
    fun pairingState(call: PluginCall) = pairingCall(call) { call.resolve(readPairingState().toJSObject()) }

    @PluginMethod
    fun pairWithLink(call: PluginCall) = pairingCall(call) {
        val snapshot = pairingCoordinator.startWithLink(call.getString("rawLink").orEmpty(), label = defaultDeviceLabel())
        call.resolve(snapshot.toJSObject())
    }

    @PluginMethod
    fun pairWithCode(call: PluginCall) = pairingCall(call) {
        val snapshot = pairingCoordinator.startWithCode(
            call.getString("origin").orEmpty(), call.getString("code").orEmpty(), defaultDeviceLabel(),
        )
        call.resolve(snapshot.toJSObject())
    }

    @PluginMethod
    fun pairRetry(call: PluginCall) = pairingCall(call) {
        readPairingState()
        call.resolve(pairingCoordinator.retry().toJSObject())
    }

    @PluginMethod
    fun pairResetToIdle(call: PluginCall) = pairingCall(call) {
        legacyPendingLink = null
        call.resolve(pairingCoordinator.resetToIdle().toJSObject())
    }

    @PluginMethod
    fun pairCheckResume(call: PluginCall) = pairingCall(call) {
        readPairingState()
        call.resolve(pairingCoordinator.checkResume().toJSObject())
    }

    @PluginMethod
    fun pairStart(call: PluginCall) = pairingCall(call) {
        val origin = call.getString("origin").orEmpty()
        val installationId = call.getString("installationId").orEmpty()
        if (origin.isBlank() || installationId.isBlank()) {
            call.reject("origin and installationId are required")
            return@pairingCall
        }
        val raw = call.getString("challenge")
        val snapshot = if (raw.isNullOrBlank()) {
            val current = readPairingState()
            if (current.origin != origin.trimEnd('/') || current.installationId != installationId ||
                current.attemptId == null || current.method == PairingMethod.CODE
            ) {
                call.reject("no pending pairing challenge")
                return@pairingCall
            }
            pairingCoordinator.retry()
        } else {
            pairingCoordinator.startWithChallenge(origin, installationId, raw, pairingLabel(call))
        }
        completeLegacyStart(call, snapshot)
    }

    @PluginMethod
    fun pairStartCode(call: PluginCall) = pairingCall(call) {
        val origin = call.getString("origin").orEmpty()
        val installationId = call.getString("installationId").orEmpty()
        val code = call.getString("code").orEmpty()
        if (origin.isBlank() || installationId.isBlank() || code.isBlank()) {
            call.reject("origin, installationId and code are required")
            return@pairingCall
        }
        completeLegacyStart(call, pairingCoordinator.startWithCode(origin, code, pairingLabel(call), installationId))
    }

    private fun pairingLabel(call: PluginCall): String {
        val label = call.getString("label")?.trim().orEmpty()
        return if (label.isEmpty() || label.equals("android", true) || label.equals("phone", true)) {
            defaultDeviceLabel()
        } else label.take(64)
    }

    private fun completeLegacyStart(call: PluginCall, initial: PairingSnapshot) {
        awaitPairing(initial) { snapshot ->
            when {
                snapshot.attemptId != initial.attemptId || snapshot.stage == PairingStage.CANCELLING ->
                    call.reject("pairing attempt was cancelled or superseded")
                snapshot.error != null -> call.reject(snapshot.legacyError())
                snapshot.pairingId != null -> call.resolve(legacyPairingProgress(snapshot))
                snapshot.stage == PairingStage.PAIRED -> call.reject("No pending pairing.")
                else -> return@awaitPairing false
            }
            true
        }
    }

    private fun awaitPairing(initial: PairingSnapshot, accept: (PairingSnapshot) -> Boolean) {
        if (!accept(initial)) pairingWaiters.add(accept)
    }

    private fun PairingSnapshot.legacyError(): String = when (stage) {
        PairingStage.TAILSCALE_LOGIN_REQUIRED -> "tailnet login required"
        else -> error?.let {
            if (it.httpStatus != null) {
                "HTTP ${it.httpStatus}" + (it.serverCode?.let { code -> " $code" } ?: "")
            } else it.technicalDetail.ifBlank { it.message }
        } ?: "Pairing is unavailable."
    }

    private fun legacyPairingProgress(snapshot: PairingSnapshot): JSObject = JSObject()
        .put("pairing_id", snapshot.pairingId)
        .put("origin", snapshot.origin)
        .put("installationId", snapshot.installationId)
        .put("status", if (snapshot.stage == PairingStage.PAIRED) "approved" else "pending_approval")
        .put("comparison_code", snapshot.comparisonCode)
        .apply { snapshot.expiresAt?.let { put("expires_at", it) } }

    @PluginMethod
    fun connectTailscale(call: PluginCall) {
        val origin = call.getString("origin")
        val installationId = call.getString("installationId")
        if (origin.isNullOrBlank() || installationId.isNullOrBlank()) {
            call.reject("origin and installationId are required")
            return
        }
        operationExecutor.execute {
            try {
                val current = synchronized(this) {
                    val record = requireTransportTarget(origin, installationId)
                    if (!matchesTransport(record)) startTransport(record)
                    transportHandle
                }
                val deadline = System.currentTimeMillis() + 5_000
                var status = JSONObject()
                while (System.currentTimeMillis() < deadline) {
                    val bytes = NativeNetwork.status(current) ?: break
                    status = JSONObject(String(bytes, StandardCharsets.UTF_8))
                    if (status.optString("state") != "starting") break
                    Thread.sleep(250)
                }
                val result = JSObject()
                    .put("state", status.optString("state", "starting"))
                    .put("message", status.optString("message"))
                mainHandler.post { call.resolve(result) }
            } catch (exception: Exception) {
                mainHandler.post { call.reject(exception.message ?: "tailscale connection failed") }
            }
        }
    }

    @PluginMethod
    fun pairCancel(call: PluginCall) = pairingCall(call) {
        legacyPendingLink = null
        val initial = pairingCoordinator.cancel()
        awaitPairing(initial) { snapshot ->
            if (snapshot.stage == PairingStage.CANCELLING) return@awaitPairing false
            val result = snapshot.toJSObject()
            if (snapshot.stage == PairingStage.IDLE) result.put("status", "cancelled")
            call.resolve(result)
            true
        }
    }

    @PluginMethod
    fun pendingPairing(call: PluginCall) = pairingCall(call) {
        val snapshot = readPairingState()
        call.resolve(JSObject()
            .put("pending", snapshot.canCancel && snapshot.pairingId != null)
            .put("cancellationPending", snapshot.stage == PairingStage.CANCELLING)
            .put("origin", snapshot.origin)
            .put("pairingId", snapshot.pairingId)
            .put("attemptId", snapshot.attemptId))
    }

    @PluginMethod
    fun pairPoll(call: PluginCall) = pairingCall(call) {
        val pairingId = call.getString("pairingId")
        if (pairingId.isNullOrBlank()) {
            call.reject("pairingId is required")
            return@pairingCall
        }
        val snapshot = pairingCoordinator.state()
        when {
            snapshot.pairingId != pairingId -> call.reject("pairing challenge is not active")
            snapshot.error != null -> call.reject(snapshot.legacyError())
            else -> call.resolve(legacyPairingProgress(snapshot))
        }
    }

    @PluginMethod
    fun pairAccept(call: PluginCall) = pairingCall(call) {
        val snapshot = pairingCoordinator.state()
        if (snapshot.stage == PairingStage.PAIRED && snapshot.pairingId == call.getString("pairingId")) {
            call.resolve()
        } else {
            call.reject("pairing approval is not current")
        }
    }

    private fun pairingHeaders(challenge: PairingChallenge): Map<String, String> = mapOf(
        "X-Nanobot-Pairing-Id" to challenge.pairingId,
        "X-Nanobot-Pairing-Nonce" to challenge.nonce,
        "X-Nanobot-Pairing-Secret" to challenge.collectorSecret,
    )

    private fun CustodyRecord.pairingTarget() = PairingTarget(
        origin, installationId, accessToken.isNotBlank() || refreshToken.isNotBlank(),
    )

    private val pairingCustody = object : PairingCustody {
        override fun load(): StoredPairing {
            val record = custody.load() ?: return StoredPairing()
            val cancellation = custody.loadPairingCancellation()
            val pending = cancellation ?: custody.loadPendingPairing()
            val target = record.pairingTarget()
            if (pending == null || !record.isUnpaired) return StoredPairing(target)
            val method = when (pending.method) {
                "qr" -> PairingMethod.QR
                "code" -> PairingMethod.CODE
                "link" -> PairingMethod.LINK
                else -> if (pending.code.isNotBlank()) PairingMethod.CODE else PairingMethod.LINK
            }
            val challenge = pending.challenge.takeIf(String::isNotBlank)?.let {
                if (cancellation != null) runCatching { PairingChallengeCodec.decode(it) }.getOrNull()
                else PairingChallengeCodec.decode(it)
            }
            val request = PairingRequest(
                method, pending.origin, pending.label.ifBlank(::defaultDeviceLabel), challenge,
                pending.code.takeIf(String::isNotBlank),
            )
            val attemptId = pending.attemptId.ifBlank {
                UUID.nameUUIDFromBytes("${record.installationId}:${pending.generation}".toByteArray()).toString()
            }
            val progress = pending.pairingId.takeIf(String::isNotBlank)?.let {
                PairingProgress(it, pending.comparisonCode.takeIf(String::isNotBlank), pending.expiresAt.takeIf { it > 0 })
            }
            return StoredPairing(target, PairingAttempt(attemptId, pending.generation, target, request, progress), cancellation != null)
        }

        override fun begin(request: PairingRequest, attemptId: String): PairingAttempt {
            val installationId = request.installationId ?: custody.installationIdentity()
            val generation = custody.beginPairing(
                request.origin, installationId, request.challenge?.let(PairingChallengeCodec::encode).orEmpty(),
                request.code.orEmpty(), request.label, attemptId, request.method.name.lowercase(Locale.ROOT),
            )
            return PairingAttempt(attemptId, generation, PairingTarget(request.origin, installationId, false), request)
        }

        override fun saveCollected(attempt: PairingAttempt, collected: CollectedPairing): PairingAttempt {
            check(!destroyed)
            val progress = collected.progress
            custody.savePendingPairing(
                attempt.target.origin, attempt.target.installationId, progress.pairingId,
                PairingChallengeCodec.encode(collected.challenge), attempt.generation,
                progress.comparisonCode.orEmpty(), progress.expiresAt ?: 0,
            )
            return attempt.copy(request = attempt.request.copy(challenge = collected.challenge, code = null), progress = progress)
        }

        override fun complete(attempt: PairingAttempt, handle: Long, credentials: PairingCredentials) {
            synchronized(this@NativeNetworkPlugin) {
                check(!destroyed)
                if (transportHandle != handle) throw NativeTransportException("network")
                val saved = custody.savePairingTokens(
                    attempt.generation, checkNotNull(attempt.progress).pairingId,
                    JSONObject()
                        .put("installation_id", credentials.installationId)
                        .put("access_token", credentials.accessToken)
                        .put("refresh_token", credentials.refreshToken)
                        .put("expires_in", credentials.expiresIn)
                        .put("refresh_expires_in", credentials.refreshExpiresIn),
                )
                NativeNetwork.mediaSetToken(handle, saved.accessToken)
            }
        }

        override fun fence(): Long {
            custody.cancelPairing()
            return custody.attemptGeneration()
        }
        override fun resetUnpairedAttempt() = custody.resetUnpairedAttempt()
    }

    private val pairingTransport = object : PairingTransport {
        override fun start(attempt: PairingAttempt): PairingConnection = synchronized(this@NativeNetworkPlugin) {
            check(!destroyed)
            custody.withPairing(attempt.generation) {
                val record = requireTransportTarget(attempt.target.origin, attempt.target.installationId)
                val connection = current()
                if (!matchesTransport(record) ||
                    connection.state !in setOf(TailscaleState.STARTING, TailscaleState.NEEDS_LOGIN, TailscaleState.READY)
                ) {
                    startTransport(record)
                    current()
                } else connection
            }
        }

        override fun current(): PairingConnection = synchronized(this@NativeNetworkPlugin) {
            val handle = transportHandle
            if (handle == 0L) return@synchronized PairingConnection()
            val status = NativeNetwork.status(handle)?.let { JSONObject(String(it, Charsets.UTF_8)) }
            PairingConnection(
                handle,
                TailscaleState.fromWire(status?.optString("state").orEmpty()),
                isApprovedLoginURL(status?.optString("login_url").orEmpty()) || approvedLoginUrl(handle) != null,
            )
        }

        override fun collect(attempt: PairingAttempt, handle: Long): CollectedPairing {
            custody.requirePairing(attempt.generation)
            val request = attempt.request
            val code = request.code
            val suffix = if (code == null) "collect" else "code"
            val headers = if (code == null) pairingHeaders(checkNotNull(request.challenge)) else
                mapOf("X-Nanobot-Pairing-Code" to code)
            val response = jsonRequest(
                "mobile_pair_$suffix", "/mobile/v1/pair/$suffix", body = null, expectedHandle = handle,
                authorityGeneration = attempt.generation,
                headers = headers + mapOf(
                    "X-Nanobot-Installation" to attempt.target.installationId,
                    "X-Nanobot-Label" to request.label,
                    "X-Nanobot-Device-Key" to DeviceProofCanonicalizer.encode(custody.deviceProofPublicKey()),
                ),
            )
            val challenge = request.challenge ?: PairingChallengeCodec.decode(
                response.optString("challenge").ifBlank {
                    PairingChallengeCodec.encode(PairingChallenge(
                        response.getString("pairing_id"), response.getString("nonce"), response.getString("collector_secret"),
                    ))
                },
            )
            val progress = pairingProgress(response, challenge.pairingId)
            return CollectedPairing(challenge, progress)
        }

        override fun poll(attempt: PairingAttempt, handle: Long): PairingPollResult {
            custody.requirePairing(attempt.generation)
            val challenge = checkNotNull(attempt.request.challenge)
            val response = try {
                jsonRequest(
                    "mobile_pair_poll", "/mobile/v1/pair/poll", body = null,
                    headers = pairingHeaders(challenge), expectedHandle = handle,
                    authorityGeneration = attempt.generation,
                )
            } catch (failure: NativeHttpException) {
                if (failure.serverCode !in setOf("pairing_not_found", "pairing_consumed")) throw failure
                return PairingPollResult.Approved {
                    custody.requirePairing(attempt.generation)
                    val record = requireTransportTarget(attempt.target.origin, attempt.target.installationId)
                    credentials(recoverAccess(record, attempt.generation, handle, challenge.pairingId))
                }
            }
            return if (response.optString("status") == "approved") {
                val proof = response.optJSONObject("proof_challenge")
                PairingPollResult.Approved {
                    credentials(if (proof != null && !response.has("access_token")) {
                        provePairing(challenge.pairingId, proof, attempt.generation, handle)
                    } else response)
                }
            } else {
                PairingPollResult.Waiting(pairingProgress(response, challenge.pairingId))
            }
        }

        override fun cancel(attempt: PairingAttempt, generation: Long) {
            val challenge = attempt.request.challenge ?: return
            val handle = synchronized(this@NativeNetworkPlugin) {
                custody.withAuthority(generation) {
                    val record = requireTransportTarget(attempt.target.origin, attempt.target.installationId)
                    check(record.isUnpaired && !destroyed)
                    if (!matchesTransport(record)) startTransport(record)
                    transportHandle
                }
            }
            jsonRequest(
                "mobile_pair_cancel", "/mobile/v1/pair/cancel", body = null, expectedHandle = handle, timeoutSeconds = 5,
                authorityGeneration = generation,
                headers = pairingHeaders(challenge) + ("X-Nanobot-Installation" to attempt.target.installationId),
            )
        }
    }

    private fun pairingProgress(response: JSONObject, expectedId: String): PairingProgress {
        val pairingId = response.optString("pairing_id", expectedId)
        check(pairingId == expectedId) { "Pairing ID changed." }
        return PairingProgress(
            pairingId,
            (response.opt("comparison_code") as? String)?.takeIf { it.matches(Regex("[0-9]{8}")) },
            response.optLong("expires_at").takeIf { it > 0 },
        )
    }

    private fun credentials(response: JSONObject) = PairingCredentials(
        response.getString("installation_id"), response.getString("access_token"), response.optString("refresh_token"),
        response.optInt("expires_in", 600), response.optInt("refresh_expires_in", 0),
    )

    private fun recoverAccess(
        record: CustodyRecord,
        generation: Long,
        handle: Long,
        pairingId: String? = null,
    ): JSONObject {
        val installationId = record.installationId
        val headers = mutableMapOf(
            "X-Nanobot-Installation" to installationId,
            "X-Nanobot-Device-Key" to DeviceProofCanonicalizer.encode(custody.deviceProofPublicKey()),
        )
        if (pairingId != null) headers["X-Nanobot-Pairing-Id"] = pairingId
        val begun = jsonRequest(
            "mobile_pair_recover",
            "/mobile/v1/pair/recover",
            body = null,
            expectedHandle = handle,
            headers = headers,
            authorityGeneration = generation,
        )
        val claims = begun.getJSONObject("proof_challenge")
        val canonical = DeviceProofCanonicalizer.canonicalBytes(
            DeviceProofCanonicalizer.fromJson(claims)
        )
        validateProofBinding(claims, record, generation)
        val signature = custody.signDeviceProof(canonical)
        custody.requireAuthority(generation)
        return jsonRequest(
            "mobile_pair_recover_complete",
            "/mobile/v1/pair/recover/complete",
            body = null,
            expectedHandle = handle,
            authorityGeneration = generation,
            headers = mapOf(
                "X-Nanobot-Installation" to installationId,
                "X-Nanobot-Device-Signature" to DeviceProofCanonicalizer.encode(signature),
                "X-Nanobot-Proof-Claims" to DeviceProofCanonicalizer.encode(canonical),
            ),
        )
    }

    @PluginMethod
    fun pendingPairingLink(call: PluginCall) = pairingCall(call) { resolvePendingLink(call, required = false) }

    @PluginMethod
    fun consumePairingLink(call: PluginCall) = pairingCall(call) { resolvePendingLink(call, required = true) }

    private fun resolvePendingLink(call: PluginCall, required: Boolean) {
        readPairingState()
        val link = legacyPendingLink
        legacyPendingLink = null
        when {
            link == null && required -> call.reject("no pending pairing link")
            link == null -> call.resolve(JSObject())
            link.error != null -> call.reject(link.legacyError())
            else -> call.resolve(JSObject()
                .put("origin", link.origin)
                .put("installationId", link.installationId)
                .put("attemptId", link.attemptId))
        }
    }

    @PluginMethod
    fun target(call: PluginCall) {
        val record = try {
            custody.load()
        } catch (exception: Exception) {
            call.reject(exception.message ?: "custody is unavailable")
            return
        }
        if (record == null) {
            call.reject("no paired target")
            return
        }
        call.resolve(
            JSObject()
                .put("origin", record.origin)
                .put("installationId", record.installationId)
                .put("hasAccess", record.accessToken.isNotBlank())
                .put("hasRefresh", record.refreshToken.isNotBlank())
        )
    }

    @PluginMethod
    fun remoteStatus(call: PluginCall) {
        ownerRequest(call, "remote_status", "/remote/v1/status", null)
    }

    @PluginMethod
    fun remoteDevices(call: PluginCall) {
        ownerRequest(call, "remote_devices", "/remote/v1/devices", null)
    }

    @PluginMethod
    fun remoteRevoke(call: PluginCall) {
        val installationId = call.getString("installationId")
        if (
            installationId.isNullOrBlank() ||
            !Regex("^[A-Za-z0-9._-]{1,128}$").matches(installationId)
        ) {
            call.reject("invalid installation id")
            return
        }
        ownerRequest(
            call,
            "remote_device_revoke",
            "/remote/v1/devices/$installationId/revoke",
            JSONObject(),
        )
    }

    private fun ownerRequest(
        call: PluginCall,
        operation: String,
        path: String,
        body: JSONObject?,
    ) {
        operationExecutor.execute {
            try {
                val record = custody.load()
                val accessToken = record?.accessToken.orEmpty()
                val headers = if (accessToken.isNotBlank()) {
                    mapOf("Authorization" to "Bearer $accessToken")
                } else {
                    emptyMap()
                }
                val result = jsonRequest(operation, path, body = body, headers = headers)
                mainHandler.post { call.resolve(JSObject(result.toString())) }
            } catch (exception: Exception) {
                mainHandler.post {
                    call.reject(exception.message ?: "owner request failed")
                }
            }
        }
    }

    @PluginMethod
    fun status(call: PluginCall) {
        val current = transportHandle
        if (current == 0L) {
            call.reject("transport is not started")
            return
        }
        val bytes = NativeNetwork.status(current)
        if (bytes == null) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve(JSObject(String(bytes, StandardCharsets.UTF_8)))
    }

    @PluginMethod
    fun http(call: PluginCall) {
        val current = transportHandle
        if (current == 0L) {
            call.reject("transport is not started")
            return
        }
        val operation = call.getString("operation")
        val method = call.getString("method")
        val path = call.getString("path")
        if (operation.isNullOrBlank() || method.isNullOrBlank() || path.isNullOrBlank()) {
            call.reject("operation, method and path are required")
            return
        }
        val request = JSONObject()
            .put("operation", operation)
            .put("method", method)
            .put("path", path)
        val headers = call.getObject("headers")
        val copied = JSONObject()
        if (headers != null) {
            for (key in headers.keys()) {
                copied.put(key, headers.getString(key))
            }
        }
        // Inject the custody access token natively; JS never receives it.
        val record = try { custody.load() } catch (_: Exception) { null }
        if (record != null && record.accessToken.isNotBlank() && !copied.has("Authorization")) {
            copied.put("Authorization", "Bearer ${record.accessToken}")
        }
        if (copied.length() > 0) {
            request.put("headers", copied)
        }
        call.getString("bodyBase64")?.let { request.put("body_base64", it) }
        call.getInt("timeoutMs")?.let { request.put("timeout_ms", it) }
        // Build once: every admission retry must carry the same bytes, so a
        // mutation keeps its request identity for the server's replay guard.
        val encodedRequest = request.toString()
        val refusal = NativeRequestLimits.refusal(
            encodedRequest.toByteArray(Charsets.UTF_8).size,
            requests.pendingCount(),
        )
        if (refusal != null) {
            call.reject(refusal)
            return
        }
        val requestOwner = requests
        val retry = admission
        operationExecutor.execute {
            try {
                val admitted = retry.run {
                    requestOwner.launch(
                        false,
                        lastError = { NativeNetwork.lastError() },
                        start = { NativeNetwork.http(current, encodedRequest) },
                    )
                }.get()
                mainHandler.post {
                    call.resolve(JSObject().put("requestId", admitted.id))
                }
            } catch (exception: Exception) {
                val cause = (exception as? java.util.concurrent.ExecutionException)?.cause ?: exception
                mainHandler.post {
                    call.reject(cause.message ?: "native request was not admitted")
                }
            }
        }
    }

    @PluginMethod
    fun cancel(call: PluginCall) {
        val current = transportHandle
        val requestId = callLong(call, "requestId")
        if (current == 0L || requestId == null) {
            call.reject("transport and requestId are required")
            return
        }
        val status = NativeNetwork.cancel(current, requestId)
        requests.abandon(requestId)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve()
    }

    @PluginMethod
    fun wsOpen(call: PluginCall) {
        val current = transportHandle
        if (current == 0L) {
            call.reject("transport is not started")
            return
        }
        val operation = call.getString("operation") ?: "fixture_ws"
        val path = call.getString("path")
        if (path.isNullOrBlank()) {
            call.reject("path is required")
            return
        }
        val request = JSONObject().put("operation", operation).put("path", path)
        val headers = call.getObject("headers")
        val copied = JSONObject()
        if (headers != null) {
            for (key in headers.keys()) {
                copied.put(key, headers.getString(key))
            }
        }
        val record = try { custody.load() } catch (_: Exception) { null }
        if (record != null && record.accessToken.isNotBlank() && !copied.has("Authorization")) {
            copied.put("Authorization", "Bearer ${record.accessToken}")
        }
        if (copied.length() > 0) {
            request.put("headers", copied)
        }
        val wsId = NativeNetwork.wsOpen(current, request.toString())
        if (wsId == 0L) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve(JSObject().put("wsId", wsId))
    }

    @PluginMethod
    fun wsSend(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        val dataBase64 = call.getString("dataBase64")
        val kindName = call.getString("kind") ?: "text"
        if (current == 0L || wsId == null || dataBase64 == null) {
            call.reject("transport, wsId and dataBase64 are required")
            return
        }
        val kind = if (kindName == "binary") 1 else 0
        val data = Base64.decode(dataBase64, Base64.NO_WRAP)
        val status = NativeNetwork.wsSend(current, wsId, kind, data)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve()
    }

    @PluginMethod
    fun wsSendBegin(call: PluginCall) {
        val wsId = callLong(call, "wsId")
        val kindName = call.getString("kind") ?: "text"
        if (wsId == null) {
            call.reject("wsId is required")
            return
        }
        pendingWsFrames[wsId] = PendingWsFrame(
            if (kindName == "binary") 1 else 0,
            ByteArrayOutputStream(),
        )
        call.resolve()
    }

    @PluginMethod
    fun wsSendChunk(call: PluginCall) {
        val wsId = callLong(call, "wsId")
        val dataBase64 = call.getString("dataBase64")
        if (wsId == null || dataBase64 == null) {
            call.reject("wsId and dataBase64 are required")
            return
        }
        val frame = pendingWsFrames[wsId]
        if (frame == null) {
            call.reject("no pending websocket frame")
            return
        }
        val chunk = try {
            Base64.decode(dataBase64, Base64.NO_WRAP)
        } catch (exception: Exception) {
            pendingWsFrames.remove(wsId)
            call.reject("websocket chunk is not base64")
            return
        }
        synchronized(frame.data) {
            frame.data.write(chunk)
        }
        call.resolve()
    }

    @PluginMethod
    fun wsSendEnd(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        val discard = call.getBoolean("discard") ?: false
        if (wsId == null) {
            call.reject("wsId is required")
            return
        }
        val frame = pendingWsFrames.remove(wsId)
        if (frame == null) {
            call.reject("no pending websocket frame")
            return
        }
        if (discard) {
            call.resolve()
            return
        }
        if (current == 0L) {
            call.reject("transport is not started")
            return
        }
        val data = synchronized(frame.data) { frame.data.toByteArray() }
        val status = NativeNetwork.wsSend(current, wsId, frame.kind, data)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve()
    }

    @PluginMethod
    fun wsClose(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        if (current == 0L || wsId == null) {
            call.reject("transport and wsId are required")
            return
        }
        pendingWsFrames.remove(wsId)
        val status = NativeNetwork.wsClose(current, wsId)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve()
    }

    @PluginMethod
    fun wsPing(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        if (current == 0L || wsId == null) {
            call.reject("transport and wsId are required")
            return
        }
        val status = NativeNetwork.wsPing(current, wsId)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve(JSObject().put("ok", true))
    }

    @PluginMethod
    fun wsRefreshAuth(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        if (current == 0L || wsId == null) {
            call.reject("transport and wsId are required")
            return
        }
        val token = try {
            custody.load()?.accessToken
        } catch (_: Exception) {
            null
        }
        if (token.isNullOrBlank()) {
            call.reject("no current access token")
            return
        }
        val frame = JSONObject()
            .put("type", "auth_refresh")
            .put("access_token", token)
            .toString()
            .toByteArray(StandardCharsets.UTF_8)
        val status = NativeNetwork.wsSend(current, wsId, 0, frame)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve(JSObject().put("ok", true))
    }

    @PluginMethod
    fun openLogin(call: PluginCall) {
        mainHandler.post {
            val handle = transportHandle
            val loginUrl = if (handle == 0L) null else approvedLoginUrl(handle)
            if (loginUrl == null) {
                call.reject("no approved login URL is available")
                return@post
            }
            val act = activity
            val uri = Uri.parse(loginUrl)
            loginTabOpen = true
            try {
                try {
                    val customTabs = CustomTabsIntent.Builder().setShowTitle(true).build()
                    if (act != null) {
                        customTabs.launchUrl(act, uri)
                    } else {
                        customTabs.intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        customTabs.launchUrl(context, uri)
                    }
                } catch (_: Exception) {
                    val intent = Intent(Intent.ACTION_VIEW, uri)
                    if (act != null) act.startActivity(intent)
                    else context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                }
                call.resolve()
            } catch (_: Exception) {
                loginTabOpen = false
                call.reject("Tailscale sign-in could not open.")
            }
        }
    }

    private fun approvedLoginUrl(handle: Long): String? =
        NativeNetwork.loginUrl(handle)?.let {
            JSONObject(String(it, Charsets.UTF_8)).optString("login_url").takeIf(::isApprovedLoginURL)
        }

    private fun returnFromTailscaleLogin() {
        if (!loginTabOpen || isActivityResumed) return
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        try {
            (activity ?: context).startActivity(intent)
        } catch (_: Exception) {
            // Android can block an activity launch while the browser is visible.
        }
        mainHandler.postDelayed({
            if (!isActivityResumed && loginTabOpen) notifyTailscaleConnected(intent)
        }, 800L)
    }

    private fun notifyTailscaleConnected(intent: Intent) {
        val manager = context.getSystemService(NotificationManager::class.java)
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(PAIRING_NOTIFICATION_CHANNEL, "Pairing status", NotificationManager.IMPORTANCE_HIGH)
            )
            Notification.Builder(context, PAIRING_NOTIFICATION_CHANNEL)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(context).setPriority(Notification.PRIORITY_HIGH).setDefaults(Notification.DEFAULT_ALL)
        }
        val openApp = PendingIntent.getActivity(
            context, PAIRING_NOTIFICATION_ID, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        try {
            manager.notify(PAIRING_NOTIFICATION_ID, builder
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle("Tailscale connected")
                .setContentText("Tap to finish pairing Kohaku")
                .setContentIntent(openApp)
                .setAutoCancel(true)
                .build())
        } catch (_: SecurityException) {
            // The user can still return to Kohaku from the browser.
        }
    }

    @PluginMethod
    fun stop(call: PluginCall) {
        synchronized(this) { stopInternal() }
        call.resolve()
    }

    @PluginMethod
    fun scanPairingQr(call: PluginCall) {
        if (getPermissionState("camera") != PermissionState.GRANTED) {
            requestPermissionForAlias("camera", call, "cameraPermission")
            return
        }
        launchQrScanner(call)
    }

    @PermissionCallback
    fun cameraPermission(call: PluginCall) {
        if (getPermissionState("camera") == PermissionState.GRANTED) {
            launchQrScanner(call)
        } else {
            call.reject("camera permission is required to scan the pairing QR code")
        }
    }

    private fun launchQrScanner(call: PluginCall) {
        val integrator = IntentIntegrator(activity)
        integrator.setDesiredBarcodeFormats(IntentIntegrator.QR_CODE)
        integrator.setPrompt("Scan the Kohaku pairing QR code")
        integrator.setBeepEnabled(false)
        integrator.setOrientationLocked(true)
        startActivityForResult(call, integrator.createScanIntent(), "qrScanned")
    }

    @ActivityCallback
    fun qrScanned(call: PluginCall?, result: ActivityResult) {
        if (call == null) return
        val parsed = IntentIntegrator.parseActivityResult(result.resultCode, result.data)
        val raw = parsed?.contents
        if (raw.isNullOrBlank()) {
            call.reject("QR scan cancelled")
            return
        }
        pairingCall(call) {
            val snapshot = startPairingLink(raw, PairingMethod.QR)
            call.resolve(snapshot.toJSObject().put("scanned", true))
        }
    }

    @PluginMethod
    fun pickCamera(call: PluginCall) {
        if (getPermissionState("camera") != PermissionState.GRANTED) {
            requestPermissionForAlias("camera", call, "cameraPickPermission")
            return
        }
        launchCameraPick(call)
    }

    @PermissionCallback
    fun cameraPickPermission(call: PluginCall) {
        if (getPermissionState("camera") == PermissionState.GRANTED) {
            launchCameraPick(call)
        } else {
            call.reject("camera permission is required to take a photo")
        }
    }

    private fun launchCameraPick(call: PluginCall) {
        val directory = context.cacheDir.resolve("nanobot-media")
        if (!directory.exists() && !directory.mkdirs()) {
            call.reject("camera directory could not be created")
            return
        }
        val file = directory.resolve("camera-" + System.currentTimeMillis() + ".jpg")
        val uri = FileProvider.getUriForFile(context, context.packageName + ".fileprovider", file)
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            putExtra(MediaStore.EXTRA_OUTPUT, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        }
        if (intent.resolveActivity(context.packageManager) == null) {
            call.reject("no camera application is available")
            return
        }
        pendingCameraFile = file
        try {
            startActivityForResult(call, intent, "cameraPicked")
        } catch (exception: SecurityException) {
            pendingCameraFile = null
            file.delete()
            call.reject("camera permission is required to take a photo")
        }
    }

    @ActivityCallback
    fun cameraPicked(call: PluginCall?, result: ActivityResult) {
        val file = pendingCameraFile
        pendingCameraFile = null
        if (call == null) {
            file?.delete()
            return
        }
        if (result.resultCode != Activity.RESULT_OK || file == null || !file.exists()) {
            file?.delete()
            call.reject("camera capture cancelled")
            return
        }
        try {
            val bytes = boundedCameraBytes(file)
            val name = "camera-" + System.currentTimeMillis() + ".jpg"
            AttachmentPolicy.validate(name, "image/jpeg", bytes.size.toLong(), attachmentLimits)
            call.resolve(
                JSObject()
                    .put("name", name)
                    .put("mediaType", "image/jpeg")
                    .put("size", bytes.size)
                    .put("dataBase64", Base64.encodeToString(bytes, Base64.NO_WRAP))
            )
        } catch (exception: Exception) {
            call.reject(exception.message ?: "camera capture failed")
        } finally {
            file.delete()
        }
    }

    private fun boundedCameraBytes(file: java.io.File): ByteArray {
        val original = file.readBytes()
        if (original.size <= attachmentLimits.maxBytes) return original
        val bitmap = BitmapFactory.decodeByteArray(original, 0, original.size)
            ?: throw IllegalArgumentException("camera image cannot be decoded")
        var maxDimension = 1600
        var quality = 82
        while (true) {
            val scaled = scaleDown(bitmap, maxDimension)
            val output = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, quality, output)
            if (scaled !== bitmap) scaled.recycle()
            val bytes = output.toByteArray()
            if (bytes.size <= attachmentLimits.maxBytes) {
                bitmap.recycle()
                return bytes
            }
            if (maxDimension <= 640 && quality <= 55) {
                bitmap.recycle()
                throw IllegalArgumentException("camera image exceeds the attachment limit")
            }
            maxDimension = maxOf(640, maxDimension - 320)
            quality = maxOf(55, quality - 8)
        }
    }

    private fun scaleDown(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val largest = maxOf(bitmap.width, bitmap.height)
        if (largest <= maxDimension) return bitmap
        val ratio = maxDimension.toFloat() / largest.toFloat()
        return Bitmap.createScaledBitmap(
            bitmap,
            maxOf(1, (bitmap.width * ratio).toInt()),
            maxOf(1, (bitmap.height * ratio).toInt()),
            true,
        )
    }

    @PluginMethod
    fun pickAttachment(call: PluginCall) {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }
        startActivityForResult(call, intent, "attachmentPicked")
    }

    @ActivityCallback
    fun attachmentPicked(call: PluginCall?, result: ActivityResult) {
        if (call == null) return
        if (result.resultCode != Activity.RESULT_OK) {
            call.reject("attachment selection cancelled")
            return
        }
        val uri = result.data?.data
        if (uri == null) {
            call.reject("attachment URI is missing")
            return
        }
        try {
            val resolver = context.contentResolver
            val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            } ?: "attachment"
            val mediaType = resolver.getType(uri) ?: "application/octet-stream"
            val bytes = resolver.openInputStream(uri)?.use { it.readNBytes(attachmentLimits.maxBytes + 1) }
                ?: throw IllegalArgumentException("attachment cannot be opened")
            if (bytes.size > attachmentLimits.maxBytes) {
                call.reject("attachment exceeds the limit")
                return
            }
            AttachmentPolicy.validate(name, mediaType, bytes.size.toLong(), attachmentLimits)
            val handle = mediaHandles.put(bytes, mediaType, "bin")
            call.resolve(
                JSObject()
                    .put("handleId", handle.id)
                    .put("name", name)
                    .put("mediaType", mediaType)
                    .put("size", bytes.size)
            )
        } catch (exception: Exception) {
            call.reject(exception.message ?: "attachment selection failed")
        }
    }

    @PluginMethod
    fun custodyStatus(call: PluginCall) {
        val result = JSObject()
        custody.status().forEach { (key, value) -> result.put(key, value) }
        call.resolve(result)
    }

    @PluginMethod
    fun logout(call: PluginCall) = pairingCall(call) {
        RunKeepAliveRegistry.clear()
        RunKeepAliveService.stop(context.applicationContext)
        legacyPendingLink = null
        val pending = synchronized(this) {
            val record = runCatching { custody.load() }.getOrNull()
            if (record?.isUnpaired == true) {
                custody.resetUnpairedAttempt()
                false
            } else {
                custody.cancelPairing()
                val current = transportHandle
                if (current != 0L) NativeNetwork.mediaPurge(current)
                stopInternal()
                custody.logout(call.getBoolean("serverReachable") ?: false)
            }.also { custody.purgeMedia() }
        }
        pairingCoordinator.invalidate()
        call.resolve(JSObject().put("serverRevocationPending", pending))
    }

    override fun handleOnDestroy() {
        destroyed = true
        if (pairingCoordinatorDelegate.isInitialized()) pairingCoordinator.invalidate()
        unregisterNetworkCallback()
        RunKeepAliveRegistry.clear()
        RunKeepAliveService.stop(context.applicationContext)
        synchronized(this) { stopInternal() }
        pollExecutor.shutdownNow()
        retryExecutor.shutdownNow()
        pairingExecutor.shutdownNow()
        operationExecutor.shutdownNow()
    }

    private fun startPoller() {
        pollTask = pollExecutor.scheduleWithFixedDelay(
            {
                try {
                    // Drain up to 64 queued events per tick so LLM token bursts
                    // never back up behind a 50 events/sec cap and trigger Go's
                    // event-queue-full WebSocket disconnect.
                    for (drained in 0 until 64) {
                        val current = transportHandle
                        if (current == 0L) {
                            return@scheduleWithFixedDelay
                        }
                        val bytes = NativeNetwork.pollEvent(current) ?: return@scheduleWithFixedDelay
                        val event = JSONObject(String(bytes, StandardCharsets.UTF_8))
                        if (current != transportHandle) return@scheduleWithFixedDelay
                        logTransportEvent(event)
                        if (event.optString("type") == "node_state") handlePairingNodeEvent(current, event)
                        val requestId = event.optLong("request_id", 0L)
                        val publish: (JSONObject) -> Unit = { admitted ->
                            val jsEvent = JSObject(admitted, admitted.keys().asSequence().toList().toTypedArray())
                            mainHandler.post {
                                if (current == transportHandle) notifyListeners("networkEvent", jsEvent)
                            }
                        }
                        if (event.optString("type") in setOf("http_response", "http_error")) {
                            val requestOwner = requests
                            requestOwner.dispatch(requestId, event, publish)
                        } else {
                            publish(event)
                        }
                    }
                } catch (t: Throwable) {
                    // Never allow an uncaught exception to permanently cancel
                    // the ScheduledExecutorService poller task.
                    Log.e(TAG, "native event poller error", t)
                }
            },
            0,
            20,
            TimeUnit.MILLISECONDS
        )
    }

    private fun logTransportEvent(event: JSONObject) {
        when (event.optString("type")) {
            "ws_open" ->
                Log.i(TAG, "ws_open ws=${event.optLong("ws_id")}")
            "ws_closed" ->
                Log.i(
                    TAG,
                    "ws_closed ws=${event.optLong("ws_id")} code=${event.optInt("code")} " +
                        "reason=${event.optString("reason")}",
                )
            "node_state" ->
                Log.i(TAG, "node_state state=${event.optString("state")} message=${event.optString("message")}")
            "http_error" ->
                Log.w(
                    TAG,
                    "http_error code=${event.optString("code")} status=${event.optInt("status")} " +
                        "message=${event.optString("message")}",
                )
            "http_response" ->
                Log.i(
                    TAG,
                    "http_response request=${event.optLong("request_id")} status=${event.optInt("status")}",
                )
            "ws_error" ->
                Log.w(
                    TAG,
                    "ws_error code=${event.optString("code")} message=${event.optString("message")}",
                )
            else -> Unit
        }
    }

    private fun handlePairingNodeEvent(handle: Long, event: JSONObject) {
        val state = TailscaleState.fromWire(event.optString("state"))
        dispatchPairing {
            if (handle == transportHandle) {
                val connection = pairingTransport.current()
                if (connection.handle == handle) pairingCoordinator.onTransportChanged(connection)
            }
        }
        when (state) {
            TailscaleState.READY -> {
                stateRecoveryAttempted = false
                mainHandler.post {
                    if (handle == transportHandle) returnFromTailscaleLogin()
                }
            }
            TailscaleState.ERROR -> {
                val message = event.optString("message")
                if (message.contains("cannot be decrypted") || message.contains("ReadState")) {
                    recoverCorruptNetworkState(handle)
                }
            }
            else -> Unit
        }
    }

    private fun requestRefresh(record: CustodyRecord, handle: Long): JSONObject = jsonRequest(
        "remote_refresh",
        "/remote/v1/refresh",
        body = JSONObject()
            .put("installation_id", record.installationId)
            .put("refresh_token", record.refreshToken),
        expectedHandle = handle,
    )

    private fun isUnauthorized(exception: Exception): Boolean = httpStatus(exception) == 401

    private fun httpStatus(exception: Exception): Int = (exception as? NativeHttpException)?.status ?: 0

    private fun requireTransportTarget(origin: String, installationId: String): CustodyRecord {
        val record = custody.load() ?: throw CustodyMismatchException("Custody target is missing.")
        check(record.origin == origin && record.installationId == installationId) { "Custody target changed." }
        return record
    }

    private fun matchesTransport(record: CustodyRecord): Boolean =
        transportHandle != 0L && transportTarget == (record.origin to record.installationId)

    private fun startTransport(record: CustodyRecord) {
        stopInternal()
        requests = NativeRequestCorrelator()
        admission = NativeRequestAdmission(timer = retryTimer)
        val stateRoot = context.filesDir.resolve("network-state")
        val safeInstallation = record.installationId.replace(Regex("[^A-Za-z0-9._-]"), "_").take(64)
        val stateDir = stateRoot.resolve(safeInstallation)
        if (!stateDir.exists() && !stateDir.mkdirs()) {
            throw IllegalStateException("native state directory could not be created")
        }
        stateDir.setReadable(false, false)
        stateDir.setWritable(false, false)
        stateDir.setReadable(true, true)
        stateDir.setWritable(true, true)
        val config = JSONObject()
            .put("origin", record.origin)
            .put("installation_id", record.installationId)
            .put("hostname", tailscaleHostname())
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("max_ws_frame_bytes", 37_748_736)
            .put("max_response_bytes", 8 * 1024 * 1024)
        val newHandle = NativeNetwork.start(config.toString())
        if (newHandle == 0L) {
            throw IllegalStateException(NativeNetwork.lastError())
        }
        transportTarget = record.origin to record.installationId
        transportHandle = newHandle
        NativeNetwork.mediaSetToken(newHandle, record.accessToken)
        NativeMediaInterceptor.transportHandle = newHandle
        startPoller()
    }

    private fun provePairing(pairingId: String, claims: JSONObject, generation: Long, handle: Long): JSONObject {
        custody.requirePairing(generation)
        val record = custody.load() ?: throw IllegalStateException("custody target is missing")
        validateProofBinding(claims, record, generation)
        val canonical = DeviceProofCanonicalizer.canonicalBytes(
            DeviceProofCanonicalizer.fromJson(claims)
        )
        val signature = custody.signDeviceProof(canonical)
        custody.requirePairing(generation)
        return jsonRequest(
            "mobile_pair_prove",
            "/mobile/v1/pair/prove",
            body = null,
            expectedHandle = handle,
            authorityGeneration = generation,
            headers = mapOf(
                "X-Nanobot-Pairing-Id" to pairingId,
                "X-Nanobot-Device-Signature" to DeviceProofCanonicalizer.encode(signature),
                "X-Nanobot-Proof-Claims" to DeviceProofCanonicalizer.encode(canonical),
            ),
        )
    }

    private fun validateProofBinding(claims: JSONObject, record: CustodyRecord, generation: Long) {
        custody.requireAuthority(generation)
        val fingerprint = java.security.MessageDigest.getInstance("SHA-256").digest(custody.deviceProofPublicKey())
            .joinToString("") { "%02x".format(it) }
        require(claims.getString("origin") == record.origin && claims.getString("installation_id") == record.installationId
            && claims.getString("device_fingerprint") == fingerprint) { "device proof binding changed" }
        require(claims.getLong("expires_at") > System.currentTimeMillis() / 1000L) { "device proof expired" }
    }

    private fun jsonRequest(
        operation: String,
        path: String,
        body: JSONObject?,
        headers: Map<String, String> = emptyMap(),
        timeoutSeconds: Long = 30,
        expectedHandle: Long? = null,
        authorityGeneration: Long? = null,
    ): JSONObject {
        val (current, requestOwner, retry) = synchronized(this) {
            Triple(transportHandle, requests, admission)
        }
        if (current == 0L || expectedHandle != null && current != expectedHandle) {
            throw NativeTransportException("network")
        }
        val request = JSONObject()
            .put("operation", operation)
            .put("method", if (body == null) "GET" else "POST")
            .put("path", path)
            .put("native", true)
        if (headers.isNotEmpty()) {
            val encodedHeaders = JSONObject()
            headers.forEach { (key, value) -> encodedHeaders.put(key, value) }
            request.put("headers", encodedHeaders)
        }
        if (body != null) {
            request.put(
                "body_base64",
                Base64.encodeToString(body.toString().toByteArray(Charsets.UTF_8), Base64.NO_WRAP)
            )
        }
        // The same encoded bytes carry the same request identity on retry.
        val encodedRequest = request.toString()
        val launch = {
            requestOwner.launch(
                true,
                lastError = { NativeNetwork.lastError() },
                start = { NativeNetwork.http(current, encodedRequest) },
            )
        }
        val admission = try {
            retry.run {
                if (authorityGeneration == null) launch()
                else custody.withAuthority(authorityGeneration, launch)
            }.get()
        } catch (exception: Exception) {
            val cause = (exception as? java.util.concurrent.ExecutionException)?.cause ?: exception
            if (authorityGeneration != null && cause is NativeAdmissionFailure) {
                throw NativeTransportException("network")
            }
            throw cause as? Exception ?: exception
        }
        val requestId = admission.id
        val future = admission.future
        try {
            val event = future.get(timeoutSeconds, TimeUnit.SECONDS)
            if (event.optString("type") == "http_error") {
                throw NativeTransportException(event.optString("code", "network"))
            }
            val status = event.optInt("status")
            if (status < 200 || status >= 300) {
                val code = try {
                    val body = String(Base64.decode(event.optString("body_base64"), Base64.DEFAULT), Charsets.UTF_8)
                    JSONObject(body).opt("error") as? String
                } catch (_: Exception) {
                    null
                }
                throw NativeHttpException(status, code, path)
            }
            val payload = Base64.decode(event.getString("body_base64"), Base64.DEFAULT)
            return JSONObject(String(payload, Charsets.UTF_8))
        } finally {
            requestOwner.abandon(requestId)
            NativeNetwork.cancel(current, requestId)
        }
    }

    private fun recoverCorruptNetworkState(handle: Long) {
        dispatchPairing {
            try {
                val recovered = synchronized(this) {
                    if (handle != transportHandle || stateRecoveryAttempted) return@synchronized false
                    val record = custody.load() ?: return@synchronized false
                    if (!matchesTransport(record)) return@synchronized false
                    stateRecoveryAttempted = true
                    stopInternal()
                    CustodyStore.purgeNetworkState(context, record.installationId)
                    startTransport(record)
                    true
                }
                if (recovered) pairingCoordinator.onTransportReplaced(handle)
            } catch (_: Exception) {
                // Pairing can still retry transport startup.
            }
        }
    }

    private fun stopInternal() {
        loginTabOpen = false
        admission.close()
        requests.close()
        pollTask?.cancel(false)
        pollTask = null
        pendingWsFrames.clear()
        val current = transportHandle
        transportHandle = 0L
        transportTarget = null
        NativeMediaInterceptor.transportHandle = 0L
        if (current != 0L) {
            NativeNetwork.stop(current)
        }
    }

    private fun callLong(call: PluginCall, key: String): Long? {
        return call.getInt(key)?.toLong()
            ?: call.getLong(key)
            ?: call.getDouble(key)?.toLong()
    }

    private fun isApprovedLoginURL(raw: String): Boolean {
        return try {
            val uri = URI(raw)
            uri.scheme == "https" &&
                uri.userInfo == null &&
                uri.fragment == null &&
                uri.host in setOf("login.tailscale.com", "controlplane.tailscale.com") &&
                !uri.path.isNullOrBlank()
        } catch (_: Exception) {
            false
        }
    }
}
