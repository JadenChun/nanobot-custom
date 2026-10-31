package ai.nanobot.mobile.links

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PairingLinkInstrumentedTest {
    @Test
    fun parsesExactPairingLink() {
        val link = PairingLinkParser.parse(
            "nanobot://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef"
        )
        assertEquals("https://fedora.taila1e707.ts.net", link.origin)
        assertEquals("install-0001", link.installationId)
        assertEquals("0123456789abcdef", link.challenge)
    }

    @Test
    fun rejectsNavigationAndMismatchedInstallation() {
        for (raw in listOf(
            "nanobot://pair?origin=http%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
            "nanobot://pair/evil?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
            "nanobot://pair?origin=https%3A%2F%2Fevil.example&installation=install-0001&challenge=0123456789abcdef&extra=1",
            "nanobot://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=short",
        )) {
            assertThrows(PairingLinkError::class.java) { PairingLinkParser.parse(raw) }
        }
        assertThrows(PairingLinkError::class.java) {
            PairingLinkParser.parse(
                "nanobot://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
                expectedInstallation = "install-other",
            )
        }
    }
}
