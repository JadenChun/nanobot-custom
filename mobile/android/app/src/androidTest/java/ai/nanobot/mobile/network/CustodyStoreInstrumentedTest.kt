package ai.nanobot.mobile.network

import android.content.Context
import android.content.pm.ApplicationInfo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CustodyStoreInstrumentedTest {
    private lateinit var context: Context
    private lateinit var store: CustodyStore

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        store = CustodyStore(context)
        store.clear()
        store.purgeMedia()
    }

    @After
    fun tearDown() {
        store.clear()
        store.purgeMedia()
    }

    @Test
    fun recordRoundTripKeepsTokensOutOfStatus() {
        val created = store.loadOrCreate("https://fixture.example:8443", null)
        val saved = created.copy(
            refreshToken = "refresh-secret",
            accessToken = "access-secret",
            accessExpiresAt = 10,
            refreshExpiresAt = 20,
            updatedAt = 30,
        )
        store.save(saved)
        val loaded = store.load()
        assertNotNull(loaded)
        assertEquals(saved.installationId, loaded!!.installationId)
        assertEquals("refresh-secret", loaded.refreshToken)
        assertEquals(32, android.util.Base64.decode(loaded.stateKeyBase64, android.util.Base64.NO_WRAP).size)
        val status = store.status()
        assertEquals("ready", status["state"])
        assertFalse(status.toString().contains("refresh-secret"))
        assertFalse(status.toString().contains("access-secret"))
    }

    @Test
    fun ownerRefreshWithoutExpiryStoresZeroDeadline() {
        val saved = store.saveTokens(
            "https://fixture.example:8443",
            "install-owner",
            "access-secret",
            "refresh-secret",
            accessExpiresIn = 600,
            refreshExpiresIn = 0,
        )
        assertEquals(0L, saved.refreshExpiresAt)
        assertEquals(0L, store.load()!!.refreshExpiresAt)
    }

    @Test
    fun installationIdentityIsStableBeforePairing() {
        val first = store.installationIdentity()
        val second = store.installationIdentity()
        assertEquals(first, second)
        assertTrue(Regex("^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$").matches(first))
        store.clear()
        val replacement = store.installationIdentity()
        assertTrue(replacement != first)
    }

    @Test
    fun installationIdentityFollowsCustodyAfterPairing() {
        val pending = store.installationIdentity()
        val created = store.loadOrCreate("https://fixture.example:8443", "install-phone-1")
        assertEquals("install-phone-1", store.installationIdentity())
        assertTrue(created.installationId != pending)
    }

    @Test
    fun originAndInstallationMismatchFailClosed() {
        val created = store.loadOrCreate("https://fixture.example:8443", null)
        assertThrows(CustodyMismatchException::class.java) {
            store.loadOrCreate("https://other.example:8443", created.installationId)
        }
        assertThrows(CustodyMismatchException::class.java) {
            store.loadOrCreate("https://fixture.example:8443", "different-installation")
        }
    }

    @Test
    fun unknownVersionAndCorruptCiphertextFailClosed() {
        store.loadOrCreate("https://fixture.example:8443", null)
        val file = File(context.filesDir, "custody/record.enc")
        val original = file.readBytes()
        original[4] = 99
        file.writeBytes(original)
        assertThrows(CustodyUnknownVersionException::class.java) { store.load() }

        val corruptCiphertext = file.readBytes()
        corruptCiphertext[4] = 1
        corruptCiphertext[corruptCiphertext.size - 1] = (corruptCiphertext[corruptCiphertext.size - 1].toInt() xor 1).toByte()
        file.writeBytes(corruptCiphertext)
        assertThrows(CustodyCorruptException::class.java) { store.load() }
    }

    @Test
    fun missingKeystoreKeyDoesNotSilentlyRecreateCustody() {
        store.loadOrCreate("https://fixture.example:8443", null)
        val method = CustodyStore::class.java.declaredMethods.first { it.name.startsWith("deleteKeyForTest") }
        method.isAccessible = true
        method.invoke(store)
        assertThrows(CustodyKeyInvalidException::class.java) { store.load() }
    }

    @Test
    fun offlineLogoutRetainsLocalErasureAndPendingRevocation() {
        val created = store.loadOrCreate("https://fixture.example:8443", null)
        store.save(created.copy(refreshToken = "refresh-secret", accessToken = "access-secret"))
        assertTrue(store.logout(serverReachable = false))
        val loaded = store.load()
        assertNotNull(loaded)
        assertTrue(loaded!!.revocationPending)
        assertEquals("", loaded.refreshToken)
        assertEquals("", loaded.accessToken)
    }

    @Test
    fun onlineLogoutErasesRecordAndKey() {
        val created = store.loadOrCreate("https://fixture.example:8443", null)
        store.save(created.copy(refreshToken = "refresh-secret"))
        assertFalse(store.logout(serverReachable = true))
        assertNull(store.load())
        assertFalse(File(context.filesDir, "custody/record.enc").exists())
    }

    @Test
    fun backupIsDisabledAndMediaPurgeIsBounded() {
        val flags = context.applicationInfo.flags
        assertEquals(0, flags and ApplicationInfo.FLAG_ALLOW_BACKUP)
        val media = File(context.cacheDir, "nanobot-media")
        assertTrue(media.mkdirs() || media.exists())
        File(media, "private-preview.bin").writeBytes(byteArrayOf(1, 2, 3))
        store.purgeMedia()
        assertFalse(media.exists())
    }
}
