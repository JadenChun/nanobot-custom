package ai.nanobot.mobile.network

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import ai.nanobot.mobile.attachments.AttachmentLimits
import ai.nanobot.mobile.attachments.AttachmentPolicy
import ai.nanobot.mobile.MainActivity
import ai.nanobot.mobile.RunKeepAliveRegistry
import ai.nanobot.mobile.RunKeepAliveService
import ai.nanobot.mobile.links.PairingChallenge
import ai.nanobot.mobile.links.PairingChallengeCodec
import ai.nanobot.mobile.links.PairingLinkParser
import ai.nanobot.mobile.media.MediaHandleStore
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
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
        private const val TAG = "NanobotNetwork"
    }

    @Volatile private var transportHandle: Long = 0L
    private val pollExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val operationExecutor = Executors.newCachedThreadPool()
    private val pendingNative = ConcurrentHashMap<Long, CompletableFuture<JSONObject>>()
    private data class PendingWsFrame(val kind: Int, val data: ByteArrayOutputStream)
    private val pendingWsFrames = ConcurrentHashMap<Long, PendingWsFrame>()
    @Volatile private var activePairing: PairingChallenge? = null
    private val refreshLock = Any()
    private var refreshInFlight: CompletableFuture<CustodyRecord>? = null
    private var pollTask: ScheduledFuture<*>? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val custody by lazy { CustodyStore(context) }
    private val mediaHandles by lazy { MediaHandleStore(context) }
    private val attachmentLimits = AttachmentLimits()
    private var pendingCameraFile: java.io.File? = null

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
            val appContext = context.applicationContext
            mainHandler.postDelayed({
                if (RunKeepAliveRegistry.isEmpty()) {
                    RunKeepAliveService.stop(appContext)
                }
            }, 300)
        }
        call.resolve()
    }

    @PluginMethod
    fun refresh(call: PluginCall) {
        val record = try {
            custody.load()
        } catch (exception: Exception) {
            call.reject(exception.message ?: "custody is unavailable")
            return
        }
        if (record == null || record.refreshToken.isBlank()) {
            call.reject("no refresh credential")
            return
        }
        val shared: CompletableFuture<CustodyRecord>
        synchronized(refreshLock) {
            val existing = refreshInFlight
            if (existing != null) {
                shared = existing
            } else {
                shared = CompletableFuture()
                refreshInFlight = shared
                operationExecutor.execute {
                    try {
                        shared.complete(performRefresh())
                    } catch (exception: Exception) {
                        shared.completeExceptionally(exception)
                    } finally {
                        synchronized(refreshLock) {
                            if (refreshInFlight === shared) refreshInFlight = null
                        }
                    }
                }
            }
        }
        shared.whenComplete { _updated, error ->
            if (error == null) {
                mainHandler.post { call.resolve(JSObject().put("refreshed", true)) }
                return@whenComplete
            }
            val failure = error as? Exception ?: Exception(error)
            val message = failure.message ?: "token refresh failed"
            if (RefreshAuthority.requiresRepair(httpStatus(failure), message)) {
                // A confirmed owner revocation and a v1 upgrade requirement both
                // end local bearer authority. Transient failures keep custody so
                // a raced invalid_refresh cannot orphan encrypted tsnet state.
                try {
                    custody.logout(serverReachable = true)
                    custody.purgeMedia()
                    val current = transportHandle
                    if (current != 0L) {
                        NativeNetwork.mediaPurge(current)
                    }
                } catch (_: Exception) {
                    // Custody may already be gone; the reload still resets UI.
                }
                mainHandler.post {
                    call.reject(
                        if (message.contains("upgrade_required")) {
                            "pairing upgrade required"
                        } else {
                            "pairing was revoked"
                        },
                    )
                    (activity as? BridgeActivity)?.getBridge()?.getWebView()?.reload()
                }
            } else {
                mainHandler.post { call.reject(message) }
            }
        }
    }

    private fun performRefresh(): CustodyRecord {
        var active = custody.load() ?: throw CustodyException("no refresh credential")
        if (active.refreshToken.isBlank() || active.revocationPending) {
            throw CustodyException("no refresh credential")
        }
        val refreshed = try {
            requestRefresh(active)
        } catch (first: Exception) {
            if (!isUnauthorized(first)) throw first
            // Another device may have rotated the token. Retry once with the
            // latest live record before treating the installation as revoked.
            val latest = custody.load()
            if (latest == null || latest.refreshToken == active.refreshToken) throw first
            active = latest
            requestRefresh(active)
        }
        val saved = custody.saveRefreshedTokens(
            active.origin,
            active.installationId,
            refreshed.getString("access_token"),
            refreshed.getString("refresh_token"),
            refreshed.optInt("expires_in", 600),
            refreshed.optInt("refresh_expires_in", 30 * 24 * 60 * 60),
        )
        val current = transportHandle
        if (current != 0L) {
            NativeNetwork.mediaSetToken(current, saved.accessToken)
        }
        return saved
    }

    @PluginMethod
    fun start(call: PluginCall) {
        val origin = call.getString("origin")
        val installationId = call.getString("installationId")
        val hostname = call.getString("hostname") ?: "nanobot-android"
        if (origin.isNullOrBlank() || installationId.isNullOrBlank()) {
            call.reject("origin and installationId are required")
            return
        }
        try {
            startTransport(origin, installationId, hostname)
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

    @PluginMethod
    fun pairStart(call: PluginCall) {
        val origin = call.getString("origin")
        val installationId = call.getString("installationId")
        val rawChallenge = call.getString("challenge")
        val requestedLabel = call.getString("label")?.trim().orEmpty()
        val label = if (
            requestedLabel.isEmpty() ||
            requestedLabel.equals("android", ignoreCase = true) ||
            requestedLabel.equals("phone", ignoreCase = true)
        ) {
            defaultDeviceLabel()
        } else {
            requestedLabel.take(64)
        }
        if (origin.isNullOrBlank() || installationId.isNullOrBlank()) {
            call.reject("origin and installationId are required")
            return
        }
        val challenge = try {
            if (!rawChallenge.isNullOrBlank()) {
                PairingChallengeCodec.decode(rawChallenge).also { activePairing = it }
            } else {
                activePairing ?: throw IllegalStateException("no pending pairing challenge")
            }
        } catch (exception: Exception) {
            call.reject(exception.message ?: "pairing challenge is invalid")
            return
        }
        operationExecutor.execute {
            try {
                val existing = custody.load()
                if (
                    transportHandle == 0L ||
                    existing == null ||
                    existing.origin != origin ||
                    existing.installationId != installationId
                ) {
                    startTransport(origin, installationId, "nanobot-" + installationId.takeLast(6))
                }
                waitUntilReady()
                val collected = jsonRequest(
                    "mobile_pair_collect",
                    "/mobile/v1/pair/collect",
                    body = null,
                    headers = mapOf(
                        "X-Nanobot-Pairing-Id" to challenge.pairingId,
                        "X-Nanobot-Pairing-Nonce" to challenge.nonce,
                        "X-Nanobot-Pairing-Secret" to challenge.collectorSecret,
                        "X-Nanobot-Installation" to installationId,
                        "X-Nanobot-Label" to label,
                        "X-Nanobot-Device-Key" to Base64.encodeToString(
                            custody.deviceProofPublicKey(),
                            Base64.NO_WRAP,
                        ),
                    ),
                )
                try {
                    custody.savePendingPairing(
                        origin,
                        installationId,
                        challenge.pairingId,
                        PairingChallengeCodec.encode(challenge),
                    )
                } catch (exception: Exception) {
                    // Pairing itself is still valid; native custody is retried on
                    // the next foreground/poll boundary.
                }
                val result = JSObject(collected.toString())
                    .put("origin", origin)
                    .put("installationId", installationId)
                mainHandler.post { call.resolve(result) }
            } catch (exception: Exception) {
                mainHandler.post { call.reject(exception.message ?: "pairing collect failed") }
            }
        }
    }

    @PluginMethod
    fun pairStartCode(call: PluginCall) {
        val origin = call.getString("origin")
        val installationId = call.getString("installationId")
        val code = call.getString("code")
        val requestedLabel = call.getString("label")?.trim().orEmpty()
        val label = if (
            requestedLabel.isEmpty() ||
            requestedLabel.equals("android", ignoreCase = true) ||
            requestedLabel.equals("phone", ignoreCase = true)
        ) {
            defaultDeviceLabel()
        } else {
            requestedLabel.take(64)
        }
        if (origin.isNullOrBlank() || installationId.isNullOrBlank() || code.isNullOrBlank()) {
            call.reject("origin, installationId and code are required")
            return
        }
        if (!Regex("^[0-9]{8}$").matches(code)) {
            call.reject("pairing code must be 8 digits")
            return
        }
        operationExecutor.execute {
            try {
                val existing = custody.load()
                if (
                    transportHandle == 0L ||
                    existing == null ||
                    existing.origin != origin ||
                    existing.installationId != installationId
                ) {
                    startTransport(origin, installationId, "nanobot-" + installationId.takeLast(6))
                }
                waitUntilReady()
                val collected = jsonRequest(
                    "mobile_pair_code",
                    "/mobile/v1/pair/code",
                    body = null,
                    headers = mapOf(
                        "X-Nanobot-Pairing-Code" to code,
                        "X-Nanobot-Installation" to installationId,
                        "X-Nanobot-Label" to label,
                        "X-Nanobot-Device-Key" to Base64.encodeToString(
                            custody.deviceProofPublicKey(),
                            Base64.NO_WRAP,
                        ),
                    ),
                )
                val rawChallenge = collected.optString("challenge")
                activePairing = if (rawChallenge.isNotBlank()) {
                    PairingChallengeCodec.decode(rawChallenge)
                } else {
                    PairingChallenge(
                        collected.getString("pairing_id"),
                        collected.getString("nonce"),
                        collected.getString("collector_secret"),
                    )
                }
                val result = JSObject(collected.toString())
                    .put("origin", origin)
                    .put("installationId", installationId)
                mainHandler.post { call.resolve(result) }
            } catch (exception: Exception) {
                mainHandler.post { call.reject(exception.message ?: "pairing code failed") }
            }
        }
    }

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
                val existing = custody.load()
                if (
                    transportHandle == 0L ||
                    existing == null ||
                    existing.origin != origin ||
                    existing.installationId != installationId
                ) {
                    startTransport(origin, installationId, "nanobot-" + installationId.takeLast(6))
                }
                val current = transportHandle
                if (current == 0L) throw IllegalStateException("transport is not started")
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
    fun pendingPairing(call: PluginCall) {
        val pending = try {
            custody.loadPendingPairing()
        } catch (exception: Exception) {
            call.reject(exception.message ?: "pending pairing is unavailable")
            return
        }
        if (pending == null) {
            call.resolve(JSObject().put("pending", false))
            return
        }
        val challenge = try {
            PairingChallengeCodec.decode(pending.challenge).also { activePairing = it }
        } catch (exception: Exception) {
            try {
                custody.clearPendingPairing()
            } catch (_: Exception) {
                // Corrupt pending state is discarded; a new scan is required.
            }
            call.resolve(JSObject().put("pending", false))
            return
        }
        call.resolve(
            JSObject()
                .put("pending", true)
                .put("origin", pending.origin)
                .put("pairingId", challenge.pairingId)
                .put("challenge", pending.challenge)
        )
    }

    @PluginMethod
    fun pairPoll(call: PluginCall) {
        val pairingId = call.getString("pairingId")
        if (pairingId.isNullOrBlank()) {
            call.reject("pairingId is required")
            return
        }
        val challenge = activePairing
        if (challenge == null || challenge.pairingId != pairingId) {
            call.reject("pairing challenge is not active")
            return
        }
        operationExecutor.execute {
            try {
                val polled = jsonRequest(
                    "mobile_pair_poll",
                    "/mobile/v1/pair/poll",
                    body = null,
                    headers = mapOf(
                        "X-Nanobot-Pairing-Id" to challenge.pairingId,
                        "X-Nanobot-Pairing-Nonce" to challenge.nonce,
                        "X-Nanobot-Pairing-Secret" to challenge.collectorSecret,
                    ),
                )
                val proofChallenge = polled.optJSONObject("proof_challenge")
                val approved = if (
                    polled.optString("status") == "approved" &&
                    proofChallenge != null &&
                    !polled.has("access_token")
                ) {
                    provePairing(challenge.pairingId, proofChallenge)
                } else {
                    polled
                }
                if (approved.optString("status") != "approved") {
                    mainHandler.post { call.resolve(JSObject(approved.toString())) }
                    return@execute
                }
                val record = custody.load()
                    ?: throw IllegalStateException("custody target is missing")
                custody.saveRefreshedTokens(
                    record.origin,
                    record.installationId,
                    approved.getString("access_token"),
                    approved.getString("refresh_token"),
                    approved.optInt("expires_in", 600),
                    approved.optInt("refresh_expires_in", 30 * 24 * 60 * 60),
                )
                activePairing = null
                try {
                    custody.clearPendingPairing()
                } catch (_: Exception) {
                    // Approved credentials are already persisted.
                }
                mainHandler.post {
                    call.resolve(
                        JSObject()
                            .put("status", "approved")
                            .put("installationId", record.installationId)
                    )
                }
            } catch (exception: Exception) {
                mainHandler.post { call.reject(exception.message ?: "pairing poll failed") }
            }
        }
    }

    @PluginMethod
    fun pairRecover(call: PluginCall) {
        operationExecutor.execute {
            try {
                val record = custody.load()
                    ?: throw IllegalStateException("custody target is missing")
                val recovered = recoverAccess(record)
                mainHandler.post {
                    call.resolve(
                        JSObject()
                            .put("status", if (recovered) "approved" else "pending")
                            .put("installationId", record.installationId)
                    )
                }
            } catch (exception: Exception) {
                mainHandler.post { call.reject(exception.message ?: "pairing recovery failed") }
            }
        }
    }

    private fun recoverAccess(record: CustodyRecord): Boolean {
        val installationId = record.installationId
        val begun = jsonRequest(
            "mobile_pair_recover",
            "/mobile/v1/pair/recover",
            body = null,
            headers = mapOf(
                "X-Nanobot-Installation" to installationId,
                "X-Nanobot-Device-Key" to Base64.encodeToString(
                    custody.deviceProofPublicKey(),
                    Base64.NO_WRAP,
                ),
            ),
        )
        val claims = begun.getJSONObject("proof_challenge")
        val canonical = DeviceProofCanonicalizer.canonicalBytes(
            DeviceProofCanonicalizer.fromJson(claims)
        )
        val signature = custody.signDeviceProof(canonical)
        val completed = jsonRequest(
            "mobile_pair_recover_complete",
            "/mobile/v1/pair/recover/complete",
            body = null,
            headers = mapOf(
                "X-Nanobot-Installation" to installationId,
                "X-Nanobot-Device-Signature" to Base64.encodeToString(signature, Base64.NO_WRAP),
                "X-Nanobot-Proof-Claims" to Base64.encodeToString(canonical, Base64.NO_WRAP),
            ),
        )
        custody.saveRefreshedTokens(
            record.origin,
            installationId,
            completed.getString("access_token"),
            completed.optString("refresh_token"),
            completed.optInt("expires_in", 600),
            completed.optInt("refresh_expires_in", 0),
        )
        return true
    }

    @PluginMethod
    fun pendingPairingLink(call: PluginCall) {
        val raw = MainActivity.pendingPairingLink
        MainActivity.pendingPairingLink = null
        if (raw == null) {
            call.resolve(JSObject())
            return
        }
        try {
            val link = PairingLinkParser.parse(raw)
            activePairing = PairingChallengeCodec.decode(link.challenge)
            call.resolve(
                JSObject()
                    .put("origin", link.origin)
                    .put("installationId", link.installationId)
            )
        } catch (exception: Exception) {
            call.reject(exception.message ?: "pairing link is invalid")
        }
    }

    @PluginMethod
    fun consumePairingLink(call: PluginCall) {
        val raw = MainActivity.pendingPairingLink
        MainActivity.pendingPairingLink = null
        if (raw == null) {
            call.reject("no pending pairing link")
            return
        }
        try {
            val link = PairingLinkParser.parse(raw)
            activePairing = PairingChallengeCodec.decode(link.challenge)
            call.resolve(JSObject().put("origin", link.origin).put("installationId", link.installationId))
        } catch (exception: Exception) {
            call.reject(exception.message ?: "pairing link is invalid")
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
        val encodedRequest = request.toString()
        val refusal = NativeRequestLimits.refusal(
            encodedRequest.toByteArray(Charsets.UTF_8).size,
            pendingNative.size,
        )
        if (refusal != null) throw IllegalArgumentException(refusal)
        val requestId = NativeNetwork.http(current, encodedRequest)
        if (requestId == 0L) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve(JSObject().put("requestId", requestId))
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
    fun openLogin(call: PluginCall) {
        val current = transportHandle
        if (current == 0L) {
            call.reject("transport is not started")
            return
        }
        val bytes = NativeNetwork.loginUrl(current)
        if (bytes == null) {
            call.reject("no approved login URL is available")
            return
        }
        val loginURL = JSONObject(String(bytes, StandardCharsets.UTF_8)).optString("login_url")
        if (!isApprovedLoginURL(loginURL)) {
            call.reject("native transport returned an unapproved login URL")
            return
        }
        try {
            val customTabs = CustomTabsIntent.Builder().setShowTitle(true).build()
            customTabs.intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            customTabs.launchUrl(context, Uri.parse(loginURL))
        } catch (exception: Exception) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(loginURL))
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        }
        call.resolve()
    }

    @PluginMethod
    fun stop(call: PluginCall) {
        stopInternal()
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
        integrator.setPrompt("Scan the Nanobot pairing QR code")
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
        MainActivity.pendingPairingLink = raw
        call.resolve(JSObject().put("scanned", true))
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
    fun logout(call: PluginCall) {
        RunKeepAliveRegistry.clear()
        RunKeepAliveService.stop(context.applicationContext)
        val serverReachable = call.getBoolean("serverReachable") ?: false
        val current = transportHandle
        if (current != 0L) {
            NativeNetwork.mediaPurge(current)
        }
        transportHandle = 0L
        pollTask?.cancel(false)
        pollTask = null
        if (current != 0L) {
            NativeNetwork.stop(current)
        }
        val pending = try {
            custody.logout(serverReachable)
        } catch (exception: Exception) {
            call.reject(exception.message ?: "custody erasure failed")
            return
        }
        custody.purgeMedia()
        call.resolve(JSObject().put("serverRevocationPending", pending))
    }

    override fun handleOnDestroy() {
        RunKeepAliveRegistry.clear()
        RunKeepAliveService.stop(context.applicationContext)
        stopInternal()
        pollExecutor.shutdownNow()
        operationExecutor.shutdownNow()
    }

    private fun startPoller() {
        pollTask = pollExecutor.scheduleWithFixedDelay(
            {
                val current = transportHandle
                if (current == 0L) {
                    return@scheduleWithFixedDelay
                }
                val bytes = NativeNetwork.pollEvent(current) ?: return@scheduleWithFixedDelay
                val event = JSONObject(String(bytes, StandardCharsets.UTF_8))
                logTransportEvent(event)
                val requestId = event.optLong("request_id", 0L)
                if (requestId > 0L) {
                    val pending = pendingNative.remove(requestId)
                    if (pending != null) {
                        pending.complete(event)
                        return@scheduleWithFixedDelay
                    }
                }
                mainHandler.post { notifyListeners("networkEvent", JSObject(event, event.keys().asSequence().toList().toTypedArray())) }
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
                Log.i(
                    TAG,
                    "node_state state=${event.optString("state")} login_url=${event.optString("login_url")}",
                )
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

    private fun requestRefresh(record: CustodyRecord): JSONObject = jsonRequest(
        "remote_refresh",
        "/remote/v1/refresh",
        body = JSONObject()
            .put("installation_id", record.installationId)
            .put("refresh_token", record.refreshToken),
    )

    private fun isUnauthorized(exception: Exception): Boolean =
        (exception.message ?: "").contains("HTTP 401")

    private fun httpStatus(exception: Exception): Int {
        val match = Regex("HTTP (\\d{3})").find(exception.message ?: "") ?: return 0
        return match.groupValues.getOrNull(1)?.toIntOrNull() ?: 0
    }

    private fun startTransport(origin: String, installationId: String, hostname: String): CustodyRecord {
        stopInternal()
        val stateRoot = context.filesDir.resolve("network-state")
        val record = custody.loadOrCreate(origin, installationId, hostname)
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
            .put("hostname", record.hostname)
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("max_ws_frame_bytes", 37_748_736)
            .put("max_response_bytes", 8 * 1024 * 1024)
        val newHandle = NativeNetwork.start(config.toString())
        if (newHandle == 0L) {
            throw IllegalStateException(NativeNetwork.lastError())
        }
        transportHandle = newHandle
        NativeNetwork.mediaSetToken(newHandle, record.accessToken)
        NativeMediaInterceptor.transportHandle = newHandle
        startPoller()
        return record
    }

    private fun waitUntilReady(timeoutMs: Long = 90_000) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val current = transportHandle
            if (current == 0L) throw IllegalStateException("native transport stopped")
            val status = JSONObject(String(NativeNetwork.status(current) ?: ByteArray(0), Charsets.UTF_8))
            when (status.optString("state")) {
                "ready" -> return
                "needs_login" -> throw IllegalStateException("tailnet login required")
                "error" -> throw IllegalStateException(status.optString("message", "native transport failed"))
            }
            Thread.sleep(250)
        }
        throw IllegalStateException("native transport readiness timeout")
    }

    private fun provePairing(pairingId: String, claims: JSONObject): JSONObject {
        val canonical = DeviceProofCanonicalizer.canonicalBytes(
            DeviceProofCanonicalizer.fromJson(claims)
        )
        val signature = custody.signDeviceProof(canonical)
        return jsonRequest(
            "mobile_pair_prove",
            "/mobile/v1/pair/prove",
            body = null,
            headers = mapOf(
                "X-Nanobot-Pairing-Id" to pairingId,
                "X-Nanobot-Device-Signature" to Base64.encodeToString(signature, Base64.NO_WRAP),
                "X-Nanobot-Proof-Claims" to Base64.encodeToString(canonical, Base64.NO_WRAP),
            ),
        )
    }

    private fun jsonRequest(
        operation: String,
        path: String,
        body: JSONObject?,
        headers: Map<String, String> = emptyMap(),
        timeoutSeconds: Long = 30,
    ): JSONObject {
        val current = transportHandle
        if (current == 0L) throw IllegalStateException("transport is not started")
        val request = JSONObject()
            .put("operation", operation)
            .put("method", if (body == null) "GET" else "POST")
            .put("path", path)
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
        val requestId = NativeNetwork.http(current, request.toString())
        if (requestId == 0L) throw IllegalStateException(NativeNetwork.lastError())
        val future = CompletableFuture<JSONObject>()
        pendingNative[requestId] = future
        try {
            val event = future.get(timeoutSeconds, TimeUnit.SECONDS)
            if (event.optString("type") == "http_error") {
                throw IllegalStateException(event.optString("message", "native request failed"))
            }
            val status = event.optInt("status")
            if (status < 200 || status >= 300) {
                val detail = try {
                    String(Base64.decode(event.optString("body_base64"), Base64.DEFAULT), Charsets.UTF_8)
                } catch (_: Exception) {
                    ""
                }
                throw IllegalStateException("HTTP $status ${detail.take(200)}".trim())
            }
            val payload = Base64.decode(event.getString("body_base64"), Base64.DEFAULT)
            return JSONObject(String(payload, Charsets.UTF_8))
        } finally {
            pendingNative.remove(requestId)
        }
    }

    private fun stopInternal() {
        pollTask?.cancel(false)
        pollTask = null
        pendingWsFrames.clear()
        val current = transportHandle
        transportHandle = 0L
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
