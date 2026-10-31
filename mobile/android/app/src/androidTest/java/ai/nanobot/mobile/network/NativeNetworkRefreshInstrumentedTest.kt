package ai.nanobot.mobile.network

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeNetworkRefreshInstrumentedTest {
    private fun request(
        handle: Long,
        operation: String,
        path: String,
        access: String? = null,
        headers: Map<String, String> = emptyMap(),
    ): Pair<Int, JSONObject> {
        val encodedHeaders = JSONObject()
        headers.forEach { (key, value) -> encodedHeaders.put(key, value) }
        if (access != null) encodedHeaders.put("Authorization", "Bearer $access")
        val request = JSONObject()
            .put("operation", operation)
            .put("method", "GET")
            .put("path", path)
            .put("headers", encodedHeaders)
        val requestId = NativeNetwork.http(handle, request.toString())
        assertNotEquals(0L, requestId)
        val deadline = System.currentTimeMillis() + 30_000
        while (System.currentTimeMillis() < deadline) {
            val bytes = NativeNetwork.pollEvent(handle) ?: continue
            val event = JSONObject(String(bytes))
            if (event.optLong("request_id") != requestId) continue
            if (event.optString("type") == "http_error") {
                throw AssertionError("HTTP error: $event")
            }
            if (event.optString("type") == "http_response") {
                val body = JSONObject(
                    String(Base64.decode(event.optString("body_base64"), Base64.DEFAULT), Charsets.UTF_8)
                )
                return event.optInt("status") to body
            }
        }
        throw AssertionError("HTTP response timeout")
    }

    @Test
    fun refreshesAccessThroughRealGateway() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val tokenFile = File(context.filesDir, "nanobot-m21-tokens.json")
        assumeTrue(keyFile.exists() && tokenFile.exists())
        val tokens = JSONObject(tokenFile.readText())
        val installation = tokens.getString("installation_id")
        val custody = CustodyStore(context)
        val record = custody.load() ?: throw AssertionError("custody target missing")
        val stateDir = File(context.filesDir, "network-state").resolve(installation)
        val config = JSONObject()
            .put("origin", tokens.getString("origin"))
            .put("installation_id", installation)
            .put("hostname", "nanobot-refresh-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("auth_key", keyFile.readText().trim())
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals(0L, handle)
        try {
            val deadline = System.currentTimeMillis() + 90_000
            while (System.currentTimeMillis() < deadline) {
                val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0)))
                if (status.optString("state") == "ready") break
                Thread.sleep(500)
            }
            val oldAccess = tokens.getString("access_token")
            val refreshToken = tokens.getString("refresh_token")
            val (refreshStatus, refreshed) = request(
                handle,
                "mobile_refresh_get",
                "/mobile/v1/refresh",
                headers = mapOf(
                    "X-Nanobot-Installation" to installation,
                    "X-Nanobot-Refresh-Token" to refreshToken,
                ),
            )
            assertEquals(200, refreshStatus)
            val newAccess = refreshed.getString("access_token")
            assertTrue(newAccess.startsWith("nbm_at_"))
            assertTrue(refreshed.getString("refresh_token").startsWith("nbm_rt_"))
            assertEquals(
                200,
                request(handle, "mobile_handshake", "/mobile/v1/handshake", newAccess).first,
            )
            assertEquals(
                401,
                request(handle, "mobile_handshake", "/mobile/v1/handshake", oldAccess).first,
            )
            Log.i("NanobotM21", "M21_REFRESH_OK access_rotated=true")
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
