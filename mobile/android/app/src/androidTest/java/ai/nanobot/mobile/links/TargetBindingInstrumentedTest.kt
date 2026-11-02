package ai.nanobot.mobile.links

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class TargetBindingInstrumentedTest {
    @Test
    fun bindsExactInstallationAndOrigin() {
        val link = PairingLink("https://fixture.example", "install-0001", "0123456789abcdef")
        val bound = TargetBinding.bind(link, "install-0001", "https://fixture.example/")
        assertEquals("https://fixture.example", bound.origin)
        assertThrows(TargetBindingError::class.java) {
            TargetBinding.bind(link, "install-0002", "https://fixture.example")
        }
        assertThrows(TargetBindingError::class.java) {
            TargetBinding.bind(link, "install-0001", "https://other.example")
        }
    }
}
