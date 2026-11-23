package ai.kohaku.mobile.links

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PairingLinkInstrumentedTest {
    @Test
    fun parsesExactPairingLink() {
        val link = PairingLinkParser.parse(
            "kohaku://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef"
        )
        assertEquals("https://fedora.taila1e707.ts.net", link.origin)
        assertEquals("install-0001", link.installationId)
        assertEquals("0123456789abcdef", link.challenge)
    }

    @Test
    fun acceptsOwnerLinkWithoutDesktopInstallationParameter() {
        val link = PairingLinkParser.parse(
            "kohaku://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&challenge=0123456789abcdef"
        )
        assertEquals("https://fedora.taila1e707.ts.net", link.origin)
        assertEquals("", link.installationId)
        assertEquals("0123456789abcdef", link.challenge)
    }

    @Test
    fun rejectsNavigationAndMismatchedInstallation() {
        for (raw in listOf(
            "kohaku://pair?origin=http%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
            "kohaku://pair/evil?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
            "kohaku://pair?origin=https%3A%2F%2Fevil.example&installation=install-0001&challenge=0123456789abcdef&extra=1",
            "kohaku://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=short",
        )) {
            assertThrows(PairingLinkError::class.java) { PairingLinkParser.parse(raw) }
        }
        assertThrows(PairingLinkError::class.java) {
            PairingLinkParser.parse(
                "kohaku://pair?origin=https%3A%2F%2Ffedora.taila1e707.ts.net&installation=install-0001&challenge=0123456789abcdef",
                expectedInstallation = "install-other",
            )
        }
    }
}
