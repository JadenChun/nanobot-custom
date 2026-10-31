package ai.nanobot.mobile.media

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MediaHandleInstrumentedTest {
    @Test
    fun boundedHandlesAreAppPrivateAndPurged() {
        val store = MediaHandleStore(InstrumentationRegistry.getInstrumentation().targetContext, maxBytes = 8)
        val handle = store.put(byteArrayOf(1, 2, 3), "image/png", "png")
        assertNotNull(store.get(handle.id))
        assertEquals(3, handle.bytes.toInt())
        assertTrue(handle.file.absolutePath.contains("/cache/nanobot-media/"))
        assertThrows(MediaHandleError::class.java) { store.put(ByteArray(9), "image/png") }
        store.purge()
        assertNull(store.get(handle.id))
        assertTrue(!File(handle.file.parent!!).exists())
    }
}
