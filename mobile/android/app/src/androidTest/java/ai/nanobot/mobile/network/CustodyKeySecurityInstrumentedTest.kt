package ai.nanobot.mobile.network

import android.security.keystore.KeyInfo
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.security.KeyStore
import javax.crypto.SecretKeyFactory
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CustodyKeySecurityInstrumentedTest {
    @Test
    fun reportsHardwareBackedKeystoreLevel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = CustodyStore(context)
        store.clear()
        try {
            store.loadOrCreate("https://fixture.example:8443", null)
            val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
            val entry = keyStore.getEntry("nanobot.mobile.custody.v1", null) as KeyStore.SecretKeyEntry
            val factory = SecretKeyFactory.getInstance(entry.secretKey.algorithm, "AndroidKeyStore")
            val info = factory.getKeySpec(entry.secretKey, KeyInfo::class.java) as KeyInfo
            assertNotNull(info)
            Log.i("NanobotM21", "M21_KEY_SECURITY level=${info.securityLevel} insideSecureHardware=${info.isInsideSecureHardware}")
        } finally {
            store.clear()
        }
    }
}
