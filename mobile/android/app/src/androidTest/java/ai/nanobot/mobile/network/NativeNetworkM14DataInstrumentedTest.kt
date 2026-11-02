package ai.nanobot.mobile.network

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.security.SecureRandom
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeNetworkM14DataInstrumentedTest {
    private fun waitReady(handle: Long) {
        val deadline = System.currentTimeMillis() + 90_000
        while (System.currentTimeMillis() < deadline) {
            val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0), Charsets.UTF_8))
            when (status.optString("state")) {
                "ready" -> return
                "error", "needs_login" -> throw AssertionError("node not ready: $status")
            }
            Thread.sleep(500)
        }
        throw AssertionError("node readiness timeout")
    }

    private fun request(
        handle: Long,
        operation: String,
        method: String,
        path: String,
        token: String?,
        body: JSONObject? = null,
    ): Pair<Int, JSONObject> {
        val request = JSONObject().put("operation", operation).put("method", method).put("path", path)
        val headers = JSONObject()
        if (token != null) headers.put("Authorization", "Bearer $token")
        if (headers.length() > 0) request.put("headers", headers)
        if (body != null) {
            request.put("body_base64", Base64.encodeToString(body.toString().toByteArray(), Base64.NO_WRAP))
        }
        val requestId = NativeNetwork.http(handle, request.toString())
        assertNotEquals("request rejected: ${NativeNetwork.lastError()}", 0L, requestId)
        val deadline = System.currentTimeMillis() + 30_000
        while (System.currentTimeMillis() < deadline) {
            val bytes = NativeNetwork.pollEvent(handle) ?: continue
            val event = JSONObject(String(bytes, Charsets.UTF_8))
            if (event.optLong("request_id") != requestId) continue
            if (event.optString("type") == "http_error") throw AssertionError("HTTP error: $event")
            if (event.optString("type") == "http_response") {
                val bodyBytes = Base64.decode(event.optString("body_base64"), Base64.NO_WRAP)
                return event.optInt("status") to JSONObject(String(bodyBytes, Charsets.UTF_8))
            }
            Thread.sleep(100)
        }
        throw AssertionError("HTTP response timeout: $path")
    }

    @Test
    fun refreshReplayAndRevokeAgainstLiveOwner() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val tokenFile = File(context.filesDir, "nanobot-m14-tokens.json")
        assumeTrue("live auth key is not installed", keyFile.exists())
        assumeTrue("live data tokens are not installed", tokenFile.exists())
        val authKey = keyFile.readText().trim()
        val tokens = JSONObject(tokenFile.readText())
        val installation = tokens.getString("installation_id")
        val initialAccess = tokens.getString("access_token")
        val initialRefresh = tokens.getString("refresh_token")
        val stateDir = File(context.filesDir, "m14-data-state")
        stateDir.deleteRecursively()
        assertTrue(stateDir.mkdirs())
        val stateKey = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val config = JSONObject()
            .put("origin", "https://fedora.taila1e707.ts.net")
            .put("installation_id", installation)
            .put("hostname", "nanobot-m14d-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", Base64.encodeToString(stateKey, Base64.NO_WRAP))
            .put("auth_key", authKey)
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals("native start failed: ${NativeNetwork.lastError()}", 0L, handle)
        try {
            waitReady(handle)
            val (handshakeStatus, handshake) = request(
                handle, "mobile_handshake", "GET", "/mobile/v1/handshake", initialAccess
            )
            assertEquals(200, handshakeStatus)
            val epoch = handshake.getInt("socket_epoch")
            val generation = handshake.getInt("gateway_generation")
            assertEquals(200, request(handle, "mobile_read", "GET", "/mobile/v1/read", initialAccess).first)

            val digest = "0".repeat(64)
            val envelope = JSONObject()
                .put("request_id", "m14-data-req-1")
                .put("installation_id", installation)
                .put("principal", "device-m14")
                .put("action", "chat.send")
                .put("digest", digest)
                .put("gateway_generation", generation)
                .put("epoch", epoch)
                .put("mode", "submit")
                .put("payload", JSONObject().put("text", "hello"))
            val (mutationStatus, mutation) = request(
                handle, "mobile_mutation", "POST", "/mobile/v1/mutation", initialAccess,
                JSONObject().put("envelope", envelope),
            )
            assertEquals(200, mutationStatus)
            assertEquals("applied", mutation.getString("decision"))
            val replay = request(
                handle, "mobile_mutation", "POST", "/mobile/v1/mutation", initialAccess,
                JSONObject().put("envelope", JSONObject(envelope.toString()).put("mode", "replay_only")),
            )
            assertEquals(200, replay.first)
            assertEquals("replayed", replay.second.getString("decision"))
            assertEquals(1, replay.second.getInt("effect_count"))
            Log.i("NanobotM14", "M14_REPLAY_OK effect_count=${replay.second.getInt("effect_count")}")

            val (refreshStatus, refreshed) = request(
                handle, "mobile_refresh", "POST", "/mobile/v1/refresh", initialAccess,
                JSONObject().put("refresh_token", initialRefresh),
            )
            assertEquals(200, refreshStatus)
            val newAccess = refreshed.getString("access_token")
            assertTrue(newAccess.startsWith("nbm_at_"))
            assertEquals(401, request(handle, "mobile_handshake", "GET", "/mobile/v1/handshake", initialAccess).first)
            assertEquals(200, request(handle, "mobile_handshake", "GET", "/mobile/v1/handshake", newAccess).first)
            Log.i("NanobotM14", "M14_REFRESH_OK")

            assertEquals(200, request(handle, "mobile_logout", "POST", "/mobile/v1/logout", newAccess, JSONObject()).first)
            assertEquals(401, request(handle, "mobile_handshake", "GET", "/mobile/v1/handshake", newAccess).first)
            Log.i("NanobotM14", "M14_REVOKE_OK")
        } finally {
            NativeNetwork.stop(handle)
            stateDir.deleteRecursively()
        }
    }
}
