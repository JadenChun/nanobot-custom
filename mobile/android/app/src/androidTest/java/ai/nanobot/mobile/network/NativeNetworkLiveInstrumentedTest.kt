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
class NativeNetworkLiveInstrumentedTest {
    @Test
    fun embeddedNodeReachesOwnerHealth() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        assumeTrue("live auth key is not installed", keyFile.exists())
        val authKey = keyFile.readText().trim()
        assumeTrue("live auth key is empty", authKey.isNotEmpty())
        val stateDir = File(context.filesDir, "m05-live-state")
        stateDir.deleteRecursively()
        assertTrue(stateDir.mkdirs())
        val stateKey = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val installation = "m05-live-" + System.currentTimeMillis().toString().takeLast(10)
        val config = JSONObject()
            .put("origin", "https://fedora.taila1e707.ts.net")
            .put("installation_id", installation)
            .put("hostname", "nanobot-m05-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", Base64.encodeToString(stateKey, Base64.NO_WRAP))
            .put("auth_key", authKey)
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals("native start failed: ${NativeNetwork.lastError()}", 0L, handle)
        try {
            val readyDeadline = System.currentTimeMillis() + 90_000
            var ready = false
            while (System.currentTimeMillis() < readyDeadline) {
                val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0), Charsets.UTF_8))
                val state = status.optString("state")
                if (state == "ready") {
                    ready = true
                    break
                }
                if (state == "error" || state == "needs_login") {
                    val diagnostic = NativeNetwork.pollEvent(handle)?.let { String(it, Charsets.UTF_8) }
                    throw AssertionError("embedded node did not become ready: $status; event=$diagnostic")
                }
                val event = NativeNetwork.pollEvent(handle)
                if (event != null) {
                    val body = JSONObject(String(event, Charsets.UTF_8))
                    if (body.optString("type") == "node_state" && body.optString("state") == "ready") {
                        ready = true
                        break
                    }
                }
                Thread.sleep(500)
            }
            assertTrue("embedded node did not become ready", ready)

            val request = JSONObject()
                .put("operation", "remote_health")
                .put("method", "GET")
                .put("path", "/remote/v1/health")
            val requestId = NativeNetwork.http(handle, request.toString())
            assertNotEquals("HTTP request was not admitted: ${NativeNetwork.lastError()}", 0L, requestId)
            val responseDeadline = System.currentTimeMillis() + 30_000
            var status = 0
            while (System.currentTimeMillis() < responseDeadline) {
                val eventBytes = NativeNetwork.pollEvent(handle)
                if (eventBytes != null) {
                    val event = JSONObject(String(eventBytes, Charsets.UTF_8))
                    if (event.optLong("request_id") == requestId && event.optString("type") == "http_response") {
                        status = event.optInt("status")
                        break
                    }
                    if (event.optLong("request_id") == requestId && event.optString("type") == "http_error") {
                        throw AssertionError("embedded HTTP failed: $event")
                    }
                }
                Thread.sleep(100)
            }
            assertEquals(200, status)
            Log.i("NanobotM05", "NANOBOT_M05_LIVE_OK health_status=$status")
        } finally {
            NativeNetwork.stop(handle)
            stateDir.deleteRecursively()
        }
    }
}
