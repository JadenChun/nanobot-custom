package ai.nanobot.mobile.attachments

import java.io.ByteArrayInputStream
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AttachmentPolicyInstrumentedTest {
    @Test
    fun validatesAndCopiesWithinTheBound() {
        val limits = AttachmentLimits(maxBytes = 8)
        val output = File.createTempFile("attachment", ".bin")
        val copied = AttachmentPolicy.copyBounded(
            ByteArrayInputStream(byteArrayOf(1, 2, 3)), output, limits, "image/png", "a.png"
        )
        assertEquals(3L, copied)
        assertThrows(AttachmentRejectedError::class.java) {
            AttachmentPolicy.copyBounded(
                ByteArrayInputStream(ByteArray(9)), output, limits, "image/png", "a.png"
            )
        }
        assertThrows(AttachmentRejectedError::class.java) {
            AttachmentPolicy.validate("a.exe", "application/x-msdownload", 1, limits)
        }
        output.delete()
    }
}
