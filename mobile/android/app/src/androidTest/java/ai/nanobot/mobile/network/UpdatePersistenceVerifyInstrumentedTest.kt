package ai.nanobot.mobile.network

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Runs after the target APK has been updated in place. */
@RunWith(AndroidJUnit4::class)
class UpdatePersistenceVerifyInstrumentedTest {
    @Test
    fun custodyAndNetworkStateSurviveUpdate() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = CustodyStore(context)
        val record = store.load() ?: error("custody missing after update")
        assertEquals("update-persistence", record.installationId)
        assertEquals("update-access-token", record.accessToken)
        assertEquals("update-refresh-token", record.refreshToken)
        assertTrue(File(context.filesDir, "update-persistence-sentinel").exists())
        assertTrue(
            File(context.filesDir, "network-state/update-persistence/tsnet-sentinel").exists(),
        )
        Log.i("NanobotUpdate", "UPDATE_PERSISTENCE_VERIFIED")
    }
}
