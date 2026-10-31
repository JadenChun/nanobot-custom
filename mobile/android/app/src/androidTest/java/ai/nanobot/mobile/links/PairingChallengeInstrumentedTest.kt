package ai.nanobot.mobile.links

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class PairingChallengeInstrumentedTest {
    @Test
    fun decodesExactOwnerChallenge() {
        val challenge = PairingChallengeCodec.decode(
            "8a6c5e6c-9d5e-4f8e-9d8a-0123456789ab.nonce-value-000000000000.secret-value-000000000000"
        )
        assertEquals("8a6c5e6c-9d5e-4f8e-9d8a-0123456789ab", challenge.pairingId)
        assertEquals("nonce-value-000000000000", challenge.nonce)
        assertEquals("secret-value-000000000000", challenge.collectorSecret)
        assertEquals(
            "8a6c5e6c-9d5e-4f8e-9d8a-0123456789ab.nonce-value-000000000000.secret-value-000000000000",
            PairingChallengeCodec.encode(challenge),
        )
    }

    @Test
    fun rejectsMalformedOrAmbiguousChallenges() {
        for (value in listOf(
            "",
            "pair-1",
            "pair-1.nonce",
            "pair-1.nonce.secret.extra",
            "pair-1.nonce-value-000000000000",
            "pair-1.nonce-value-000000000000.secret value",
            "pair-1.nonce-value-000000000000.secret-value-000000000000\n",
        )) {
            assertThrows(PairingChallengeError::class.java) { PairingChallengeCodec.decode(value) }
        }
    }
}
