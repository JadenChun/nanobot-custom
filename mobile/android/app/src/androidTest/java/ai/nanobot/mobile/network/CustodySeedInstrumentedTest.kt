package ai.nanobot.mobile.network

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.security.SecureRandom
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CustodySeedInstrumentedTest {
    @Test
    fun seedCustodyTargetFromLivePairing() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val keyFile = File(context.filesDir, "nanobot-m05-authkey")
        val tokenFile = File(context.filesDir, "nanobot-m16-tokens.json")
        assumeTrue(keyFile.exists())
        assumeTrue(tokenFile.exists())
        val origin = "https://fedora.taila1e707.ts.net"
        val tokens = JSONObject(tokenFile.readText())
        val installation = tokens.getString("installation_id")
        val store = CustodyStore(context)
        val record = store.loadOrCreate(origin, installation)
        val stateDir = File(context.filesDir, "network-state").resolve(installation)
        stateDir.deleteRecursively()
        stateDir.mkdirs()
        val config = JSONObject()
            .put("origin", origin)
            .put("installation_id", installation)
            .put("hostname", "nanobot-m16-" + installation.takeLast(6))
            .put("state_dir", stateDir.absolutePath)
            .put("state_key_base64", record.stateKeyBase64)
            .put("auth_key", keyFile.readText().trim())
        val handle = NativeNetwork.start(config.toString())
        try {
            val deadline = System.currentTimeMillis() + 90_000
            while (System.currentTimeMillis() < deadline) {
                val status = JSONObject(String(NativeNetwork.status(handle) ?: ByteArray(0), Charsets.UTF_8))
                if (status.optString("state") == "ready") break
                if (status.optString("state") == "error") throw AssertionError(status.toString())
                Thread.sleep(500)
            }
            store.saveTokens(
                origin,
                installation,
                tokens.getString("access_token"),
                tokens.getString("refresh_token"),
            )
            Log.i("NanobotM16", "M16_CUSTODY_TARGET_READY installation=$installation")
        } finally {
            NativeNetwork.stop(handle)
        }
    }
}
