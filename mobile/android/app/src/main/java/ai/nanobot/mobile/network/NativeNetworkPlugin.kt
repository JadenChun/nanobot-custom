package ai.nanobot.mobile.network

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.os.Handler
import android.os.Looper
import android.util.Base64
import com.getcapacitor.BridgeActivity
import com.getcapacitor.JSObject
import com.getcapacitor.PermissionState
import com.getcapacitor.Plugin
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
    @Volatile private var transportHandle: Long = 0L
    private val pollExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val operationExecutor = Executors.newCachedThreadPool()
    private val pendingNative = ConcurrentHashMap<Long, CompletableFuture<JSONObject>>()
    @Volatile private var activePairing: PairingChallenge? = null
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
        operationExecutor.execute {
            try {
                val refreshed = jsonRequest(
                    "mobile_refresh_get",
                    "/mobile/v1/refresh",
                    body = null,
                    headers = mapOf(
                        "X-Nanobot-Installation" to record.installationId,
                        "X-Nanobot-Refresh-Token" to record.refreshToken,
                    ),
                )
                custody.saveTokens(
                    record.origin,
                    record.installationId,
                    refreshed.getString("access_token"),
                    refreshed.getString("refresh_token"),
                    refreshed.optInt("expires_in", 600),
                    refreshed.optInt("refresh_expires_in", 30 * 24 * 60 * 60),
                )
                mainHandler.post { call.resolve(JSObject().put("refreshed", true)) }
            } catch (exception: Exception) {
                val message = exception.message ?: "token refresh failed"
                if (message.contains("HTTP 401")) {
                    // The owner revoked this installation. Erase local custody and
                    // return to the pairing gate instead of leaving stale access.
                    try {
                        custody.logout(serverReachable = true)
                        custody.purgeMedia()
                    } catch (_: Exception) {
                        // Custody may already be gone; the reload still resets UI.
                    }
                    mainHandler.post {
                        call.reject("pairing was revoked")
                        (activity as? BridgeActivity)?.getBridge()?.getWebView()?.reload()
                    }
                } else {
                    mainHandler.post { call.reject(message) }
                }
            }
        }
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
        val label = call.getString("label") ?: "phone"
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
                    ),
                )
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
                if (polled.optString("status") != "approved") {
                    mainHandler.post { call.resolve(JSObject(polled.toString())) }
                    return@execute
                }
                val record = custody.load()
                    ?: throw IllegalStateException("custody target is missing")
                custody.saveTokens(
                    record.origin,
                    record.installationId,
                    polled.getString("access_token"),
                    polled.getString("refresh_token"),
                    polled.optInt("expires_in", 600),
                    polled.optInt("refresh_expires_in", 30 * 24 * 60 * 60),
                )
                activePairing = null
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
        val requestId = NativeNetwork.http(current, request.toString())
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
    fun wsClose(call: PluginCall) {
        val current = transportHandle
        val wsId = callLong(call, "wsId")
        if (current == 0L || wsId == null) {
            call.reject("transport and wsId are required")
            return
        }
        val status = NativeNetwork.wsClose(current, wsId)
        if (status != 0) {
            call.reject(NativeNetwork.lastError())
            return
        }
        call.resolve()
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
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(loginURL))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
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
        integrator.setOrientationLocked(false)
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
        startActivityForResult(call, intent, "cameraPicked")
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
        val serverReachable = call.getBoolean("serverReachable") ?: false
        val current = transportHandle
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
            .put("max_ws_frame_bytes", 512 * 1024)
        val newHandle = NativeNetwork.start(config.toString())
        if (newHandle == 0L) {
            throw IllegalStateException(NativeNetwork.lastError())
        }
        transportHandle = newHandle
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
        val current = transportHandle
        transportHandle = 0L
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
