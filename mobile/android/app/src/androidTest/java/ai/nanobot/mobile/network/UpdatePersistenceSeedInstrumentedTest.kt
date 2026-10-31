package ai.nanobot.mobile.network

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Seeds encrypted custody and a network-state sentinel. The host script then
 * updates the target APK with `adb install -r` and runs the verify test.
 */
@RunWith(AndroidJUnit4::class)
class UpdatePersistenceSeedInstrumentedTest {
    @Test
    fun seedCustodyAndNetworkState() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = CustodyStore(context)
        store.clear()
        store.loadOrCreate("https://update-persistence.test", "update-persistence")
        store.saveTokens(
            "https://update-persistence.test",
            "update-persistence",
            accessToken = "update-access-token",
            refreshToken = "update-refresh-token",
        )

        val stateDir = File(context.filesDir, "network-state/update-persistence")
        stateDir.mkdirs()
        File(stateDir, "tsnet-sentinel").writeText("persist")
        File(context.filesDir, "update-persistence-sentinel").writeText("persist")
        assertEquals("update-persistence", store.load()?.installationId)
        Log.i("NanobotUpdate", "UPDATE_PERSISTENCE_SEEDED")
    }
}
