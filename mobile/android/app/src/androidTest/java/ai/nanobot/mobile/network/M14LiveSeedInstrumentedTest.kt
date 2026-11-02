package ai.nanobot.mobile.network

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * M14 live seed: authenticate the app-owned embedded node for the production
 * installation and persist its tsnet state in the same directory the plugin
 * uses. Test-only; no auth key is stored in the product source or logs.
 */
@RunWith(AndroidJUnit4::class)
class M14LiveSeedInstrumentedTest {
    @Test
    fun seedProductionNodeState() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        assumeTrue("live auth key is not installed", keyFile.exists())
        val authKey = keyFile.readText().trim()
        assumeTrue("live auth key is empty", authKey.isNotEmpty())
        val origin = InstrumentationRegistry.getArguments().getString("origin")
            ?: "https://fedora.taila1e707.ts.net"
        val store = CustodyStore(context)
        val record = store.loadOrCreate(
            origin,
            null,
            "nanobot-android",
        )
        val safe = record.installationId.replace(Regex("[^A-Za-z0-9._-]"), "_").take(64)
        val stateDir = context.filesDir.resolve("network-state").resolve(safe)
        stateDir.deleteRecursively()
        assertTrue(stateDir.mkdirs())
        val config = JSONObject()
            .put("origin", record.origin)
            .put("installation_id", record.installationId)
            .put("hostname", record.hostname)
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("auth_key", authKey)
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals("native start failed: ${NativeNetwork.lastError()}", 0L, handle)
        try {
            val deadline = System.currentTimeMillis() + 90_000
            var ready = false
            while (System.currentTimeMillis() < deadline) {
                val status = JSONObject(
                    String(NativeNetwork.status(handle) ?: ByteArray(0), Charsets.UTF_8),
                )
                when (status.optString("state")) {
                    "ready" -> {
                        ready = true
                        break
                    }
                    "error", "needs_login" -> throw AssertionError("node not ready: $status")
                }
                NativeNetwork.pollEvent(handle)
                Thread.sleep(500)
            }
            assertTrue("embedded node did not become ready", ready)
            Log.i(
                "NanobotM14",
                "M14_LIVE_SEED_OK installation=${record.installationId} state=$stateDir",
            )
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
