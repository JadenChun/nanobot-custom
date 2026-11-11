package ai.nanobot.mobile.network

import java.io.File
import java.security.KeyFactory
import java.security.MessageDigest
import java.security.Signature
import java.security.spec.X509EncodedKeySpec
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class DeviceProofCanonicalizerTest {
    private val corpusBytes = File("../../contracts/device-proof-v2-vectors.json").readBytes()
    private val corpus = JSONObject(String(corpusBytes, Charsets.UTF_8))

    @Test fun shared_corpus_is_pinned() {
        assertEquals("270d3d2715258bb161c97eba520e445068f8fed6f06c01bc57076e4e18d4abfd",
            MessageDigest.getInstance("SHA-256").digest(corpusBytes).joinToString("") { "%02x".format(it) })
    }

    @Test fun shared_bytes_and_signatures() {
        val valid = corpus.getJSONArray("valid")
        for (index in 0 until valid.length()) {
            val vector = valid.getJSONObject(index)
            val bytes = DeviceProofCanonicalizer.canonicalBytes(DeviceProofCanonicalizer.fromJson(vector.getJSONObject("claims")))
            assertEquals(vector.getString("canonical_utf8_hex"), bytes.joinToString("") { "%02x".format(it) })
            val key = KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(DeviceProofCanonicalizer.decode(vector.getString("public_key"))))
            assertTrue(Signature.getInstance("SHA256withECDSA").run {
                initVerify(key); update(bytes); verify(DeviceProofCanonicalizer.decode(vector.getString("signature")))
            })
        }
    }

    @Test fun shared_type_shape_unicode_and_integer_rejections() {
        val rejected = corpus.getJSONArray("rejected_claims")
        for (index in 0 until rejected.length()) {
            val vector = rejected.getJSONObject(index)
            assertThrows(vector.getString("id"), IllegalArgumentException::class.java) {
                DeviceProofCanonicalizer.canonicalBytes(DeviceProofCanonicalizer.fromJson(vector.getJSONObject("claims")))
            }
        }
    }

    @Test fun shared_noncanonical_encodings() {
        val rejected = corpus.getJSONArray("rejected_encodings")
        for (index in 0 until rejected.length()) {
            assertThrows(IllegalArgumentException::class.java) {
                DeviceProofCanonicalizer.decode(rejected.getString(index))
            }
        }
    }
}
