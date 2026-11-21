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
class NativeNetworkPairingInstrumentedTest {
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

    private fun http(handle: Long, operation: String, path: String, body: JSONObject?): JSONObject {
        val request = JSONObject()
            .put("operation", operation)
            .put("method", "POST")
            .put("path", path)
            .put("native", true)
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
                assertEquals(200, event.optInt("status"))
                val responseBody = Base64.decode(event.optString("body_base64"), Base64.NO_WRAP)
                return JSONObject(String(responseBody, Charsets.UTF_8))
            }
            Thread.sleep(100)
        }
        throw AssertionError("HTTP response timeout: $path")
    }

    @Test
    fun pairAgainstLiveOwner() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val challengeFile = File(context.filesDir, "nanobot-m14-challenge.json")
        assumeTrue("live auth key is not installed", keyFile.exists())
        assumeTrue("live pairing challenge is not installed", challengeFile.exists())
        val authKey = keyFile.readText().trim()
        val challenge = JSONObject(challengeFile.readText())
        val pairingId = challenge.getString("pairing_id")
        val nonce = challenge.getString("nonce")
        val collectorSecret = challenge.getString("collector_secret")
        val installation = "m14-live-" + System.currentTimeMillis().toString().takeLast(10)
        val stateDir = File(context.filesDir, "m14-live-state")
        stateDir.deleteRecursively()
        assertTrue(stateDir.mkdirs())
        val stateKey = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val config = JSONObject()
            .put("origin", "https://fedora.taila1e707.ts.net")
            .put("installation_id", installation)
            .put("hostname", "nanobot-m14-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", Base64.encodeToString(stateKey, Base64.NO_WRAP))
            .put("auth_key", authKey)
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals("native start failed: ${NativeNetwork.lastError()}", 0L, handle)
        try {
            waitReady(handle)
            val collected = http(
                handle,
                "remote_pair_collect",
                "/remote/v1/pair/collect",
                JSONObject()
                    .put("pairing_id", pairingId)
                    .put("nonce", nonce)
                    .put("collector_secret", collectorSecret)
                    .put("installation_id", installation)
                    .put("label", "emulator-m14"),
            )
            val code = collected.getString("comparison_code")
            Log.i("NanobotM14", "M14_COLLECTED pairing_id=$pairingId code=$code")
            val deadline = System.currentTimeMillis() + 90_000
            while (System.currentTimeMillis() < deadline) {
                val polled = http(
                    handle,
                    "remote_pair_poll",
                    "/remote/v1/pair/poll",
                    JSONObject()
                        .put("pairing_id", pairingId)
                        .put("nonce", nonce)
                        .put("collector_secret", collectorSecret),
                )
                if (polled.optString("status") == "approved") {
                    assertTrue(polled.getString("access_token").startsWith("nbm_at_"))
                    assertTrue(polled.getString("refresh_token").startsWith("nbm_rt_"))
                    Log.i("NanobotM14", "M14_PAIRED pairing_id=$pairingId")
                    return
                }
                Thread.sleep(500)
            }
            throw AssertionError("pairing approval timeout")
        } finally {
            NativeNetwork.stop(handle)
            stateDir.deleteRecursively()
        }
    }
}
