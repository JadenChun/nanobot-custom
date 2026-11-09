package ai.nanobot.mobile.network

import org.junit.Assert.assertEquals
import org.junit.Test

class DeviceProofCanonicalizerTest {
    @Test
    fun canonical_json_sorts_keys_and_omits_whitespace() {
        val claims = linkedMapOf<String, Any?>(
            "nonce" to "n",
            "version" to 2,
            "installation_id" to "install-0001",
            "expires_at" to 1000,
        )

        assertEquals(
            "{\"expires_at\":1000,\"installation_id\":\"install-0001\",\"nonce\":\"n\",\"version\":2}",
            DeviceProofCanonicalizer.canonicalJson(claims),
        )
    }

    @Test
    fun canonical_bytes_are_utf8() {
        val claims = mapOf<String, Any?>("origin" to "https://fixture.example:8443")
        assertEquals(
            "{\"origin\":\"https://fixture.example:8443\"}",
            DeviceProofCanonicalizer.canonicalBytes(claims).toString(Charsets.UTF_8),
        )
    }
}
