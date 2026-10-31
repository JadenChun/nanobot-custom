package ai.nanobot.mobile.network

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import android.os.Handler
import android.os.Looper
import android.util.Base64
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.ActivityCallback
import com.getcapacitor.annotation.CapacitorPlugin
import java.net.URI
import java.nio.charset.StandardCharsets
import ai.nanobot.mobile.attachments.AttachmentLimits
import ai.nanobot.mobile.attachments.AttachmentPolicy
import ai.nanobot.mobile.media.MediaHandleStore
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import androidx.activity.result.ActivityResult

@CapacitorPlugin(name = "NativeNetwork")
class NativeNetworkPlugin : Plugin() {
    @Volatile private var transportHandle: Long = 0L
    private val pollExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private var pollTask: ScheduledFuture<*>? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val custody by lazy { CustodyStore(context) }
    private val mediaHandles by lazy { MediaHandleStore(context) }
    private val attachmentLimits = AttachmentLimits()

    @PluginMethod
    fun abiVersion(call: PluginCall) {
        call.resolve(JSObject().put("version", NativeNetwork.abiVersion()))
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
        stopInternal()
        val stateRoot = context.filesDir.resolve("network-state")
        val record = try {
            custody.loadOrCreate(origin, installationId, hostname)
        } catch (exception: Exception) {
            call.reject(exception.message ?: "native custody rejected the installation")
            return
        }
        val safeInstallation = record.installationId.replace(Regex("[^A-Za-z0-9._-]"), "_").take(64)
        val stateDir = stateRoot.resolve(safeInstallation)
        if (!stateDir.exists() && !stateDir.mkdirs()) {
            call.reject("native state directory could not be created")
            return
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
        val newHandle = NativeNetwork.start(config.toString())
        if (newHandle == 0L) {
            call.reject(NativeNetwork.lastError())
            return
        }
        transportHandle = newHandle
        startPoller()
        call.resolve(JSObject().put("state", "starting"))
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
        val requestId = call.getLong("requestId")
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
        val wsId = call.getLong("wsId")
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
        val wsId = call.getLong("wsId")
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
                mainHandler.post { notifyListeners("networkEvent", JSObject(event, event.keys().asSequence().toList().toTypedArray())) }
            },
            0,
            20,
            TimeUnit.MILLISECONDS
        )
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
