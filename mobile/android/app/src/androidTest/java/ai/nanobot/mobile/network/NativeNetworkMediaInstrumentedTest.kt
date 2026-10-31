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
class NativeNetworkMediaInstrumentedTest {
    private fun mobilePath(apiUrl: String): String {
        val parts = apiUrl.trim('/').split('/')
        require(parts.size == 4 && parts[0] == "api" && parts[1] == "media") { "unexpected media URL" }
        return "/mobile/v1/media/${parts[2]}/${parts[3]}"
    }

    private fun request(
        handle: Long,
        token: String,
        path: String,
        range: String? = null,
    ): JSONObject {
        val headers = JSONObject().put("Authorization", "Bearer $token")
        if (range != null) headers.put("Range", range)
        val request = JSONObject()
            .put("operation", "mobile_media")
            .put("method", "GET")
            .put("path", path)
            .put("headers", headers)
        val requestId = NativeNetwork.http(handle, request.toString())
        assertNotEquals(0L, requestId)
        val deadline = System.currentTimeMillis() + 30_000
        while (System.currentTimeMillis() < deadline) {
            val bytes = NativeNetwork.pollEvent(handle) ?: continue
            val event = JSONObject(String(bytes))
            val type = event.optString("type")
            if (event.optLong("request_id") != requestId) continue
            if (type == "http_response" || type == "http_error") return event
            if (type.endsWith("_error") && event.optLong("request_id") == requestId) return event
        }
        throw AssertionError("media response timeout")
    }

    @Test
    fun fetchesRealGatewayMediaRangeAndLimits() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val tokenFile = File(context.filesDir, "nanobot-m15-tokens.json")
        assumeTrue(keyFile.exists() && tokenFile.exists())
        val tokens = JSONObject(tokenFile.readText())
        val installation = tokens.getString("installation_id")
        val custody = CustodyStore(context)
        val record = custody.load() ?: throw AssertionError("custody target missing")
        val stateDir = File(context.filesDir, "network-state").resolve(installation)
        val config = JSONObject()
            .put("origin", tokens.getString("origin"))
            .put("installation_id", installation)
            .put("hostname", "nanobot-media-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("auth_key", keyFile.readText().trim())
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals(0L, handle)
        try {
            var ready = false
            val deadline = System.currentTimeMillis() + 90_000
            while (System.currentTimeMillis() < deadline) {
                val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0)))
                if (status.optString("state") == "ready") { ready = true; break }
                Thread.sleep(500)
            }
            assertTrue("native transport did not become ready", ready)
            val token = tokens.getString("access_token")
            val mediaPath = mobilePath(tokens.getString("media_url"))
            val full = request(handle, token, mediaPath)
            Log.i("NanobotM15", "M15_MEDIA_EVENT $full")
            assertEquals(200, full.optInt("status"))
            val body = Base64.decode(full.getString("body_base64"), Base64.DEFAULT)
            assertTrue(body.size == 68 && body[0] == 0x89.toByte() && body[1] == 0x50.toByte())
            Log.i("NanobotM15", "M15_MEDIA_FULL status=200 bytes=${body.size}")

            val range = request(handle, token, mediaPath, "bytes=0-7")
            assertEquals(206, range.optInt("status"))
            assertEquals("bytes 0-7/68", range.getJSONObject("headers").getString("content-range"))
            assertEquals(8, range.getInt("bytes"))
            Log.i("NanobotM15", "M15_MEDIA_RANGE status=206 content-range=bytes 0-7/68")

            val oversizedPath = mobilePath(tokens.getString("oversized_url"))
            val oversized = request(handle, token, oversizedPath)
            assertEquals(413, oversized.optInt("status"))
            Log.i("NanobotM15", "M15_MEDIA_LIMIT status=413")

            val hostile = request(handle, token, "/mobile/v1/media/AAAAAAAAAAAAAAAAAAAAAA/Z2F0ZXdheS1maXh0dXJlLnBuZw")
            assertEquals(401, hostile.optInt("status"))
            Log.i("NanobotM15", "M15_MEDIA_HOSTILE status=401")
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
