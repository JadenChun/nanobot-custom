package ai.nanobot.mobile.network

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertNotEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PairingSeedInstrumentedTest {
    @Test
    fun resetTokensAndSeedNodeForProductPairing() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        assumeTrue("live auth key is not installed", keyFile.exists())
        val store = CustodyStore(context)
        val origin = "https://fedora.taila1e707.ts.net"
        store.clear()
        val reset = store.loadOrCreate(origin, null)
        val stateDir = File(context.filesDir, "network-state").resolve(reset.installationId)
        stateDir.mkdirs()
        val config = JSONObject()
            .put("origin", reset.origin)
            .put("installation_id", reset.installationId)
            .put("hostname", reset.hostname)
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", reset.stateKeyBase64)
            .put("auth_key", keyFile.readText().trim())
        val handle = NativeNetwork.start(config.toString())
        assertNotEquals(NativeNetwork.lastError(), 0L, handle)
        try {
            val deadline = System.currentTimeMillis() + 90_000
            while (System.currentTimeMillis() < deadline) {
                val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0)))
                if (status.optString("state") == "ready") {
                    Log.i("NanobotPair", "PAIR_SEED_READY installation=${reset.installationId}")
                    return
                }
                if (status.optString("state") == "error") {
                    throw AssertionError(status.toString())
                }
                Thread.sleep(500)
            }
            throw AssertionError("node readiness timeout")
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
