package ai.nanobot.mobile.network

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.security.SecureRandom
import org.json.JSONObject
import org.junit.Assert.assertNotEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeNetworkWsInstrumentedTest {
    @Test
    fun opensAuthenticatedMobileSocket() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val tokenFile = File(context.filesDir, "nanobot-m16-tokens.json")
        assumeTrue(keyFile.exists() && tokenFile.exists())
        val tokens = JSONObject(tokenFile.readText())
        val installation = tokens.getString("installation_id")
        val custody = CustodyStore(context)
        val record = custody.load() ?: throw AssertionError("custody target missing")
        val stateDir = File(context.filesDir, "network-state").resolve(installation)
        val stateKey = record.stateKeyBase64
        val config = JSONObject()
            .put("origin", "https://fedora.taila1e707.ts.net")
            .put("installation_id", installation)
            .put("hostname", "nanobot-ws-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", stateKey)
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
            val request = JSONObject()
                .put("operation", "mobile_ws")
                .put("path", "/mobile/v1/ws")
                .put("headers", JSONObject().put("Authorization", "Bearer " + tokens.getString("access_token")))
            val wsId = NativeNetwork.wsOpen(handle, request.toString())
            Log.i("NanobotM16", "M16_WS_OPEN id=$wsId error=${NativeNetwork.lastError()}")
            val eventDeadline = System.currentTimeMillis() + 20_000
            while (System.currentTimeMillis() < eventDeadline) {
                val bytes = NativeNetwork.pollEvent(handle) ?: continue
                val event = JSONObject(String(bytes))
                if (event.optString("type").startsWith("ws_")) {
                    Log.i("NanobotM16", "M16_WS_EVENT $event")
                    if (event.optString("type") == "ws_open") return
                    if (event.optString("type") == "ws_error") throw AssertionError(event.toString())
                }
                Thread.sleep(100)
            }
            throw AssertionError("WS event timeout")
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
