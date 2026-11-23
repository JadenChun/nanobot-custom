package ai.nanobot.mobile.network

import java.io.File
import java.math.BigInteger
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
            val encodedKey = DeviceProofCanonicalizer.decode(vector.getString("public_key"))
            val signatureBytes = DeviceProofCanonicalizer.decode(vector.getString("signature"))
            val key = KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(encodedKey))
            assertArrayEquals(encodedKey, key.encoded)
            assertTrue(isStrictP256Signature(signatureBytes))
            assertTrue(Signature.getInstance("SHA256withECDSA").run {
                initVerify(key); update(bytes); verify(signatureBytes)
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

    @Test fun shared_replay_binding_and_signature_rejections() {
        val valid = corpus.getJSONArray("valid").getJSONObject(0)
        val rejected = corpus.getJSONArray("rejected_verification")
        for (index in 0 until rejected.length()) {
            val vector = rejected.getJSONObject(index)
            val claims = vector.optJSONObject("claims") ?: valid.getJSONObject("claims")
            val keyEncoding = DeviceProofCanonicalizer.decode(
                vector.optString("public_key", valid.getString("public_key")),
            )
            val signature = DeviceProofCanonicalizer.decode(
                vector.optString("signature", valid.getString("signature")),
            )
            val now = vector.optLong("now", valid.getLong("now"))
            assertFalse(vector.getString("id"), verifiesFixture(claims, keyEncoding, signature, now))
        }
        val replay = corpus.getJSONObject("replay")
        assertEquals("ascii", replay.getString("vector"))
        assertEquals("accept", replay.getString("first"))
        assertEquals("reject-consumed", replay.getString("second"))
    }

    private fun verifiesFixture(
        claims: JSONObject,
        keyEncoding: ByteArray,
        signature: ByteArray,
        now: Long,
    ): Boolean = try {
        val bytes = DeviceProofCanonicalizer.canonicalBytes(DeviceProofCanonicalizer.fromJson(claims))
        val key = KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(keyEncoding))
        key.encoded.contentEquals(keyEncoding) &&
            claims.getLong("expires_at") > now &&
            isStrictP256Signature(signature) &&
            Signature.getInstance("SHA256withECDSA").run {
                initVerify(key)
                update(bytes)
                verify(signature)
            }
    } catch (_: Exception) {
        false
    }

    private fun isStrictP256Signature(signature: ByteArray): Boolean {
        if (signature.size < 8 || signature[0] != 0x30.toByte()) return false
        val sequenceLength = signature[1].toInt() and 0xff
        if (sequenceLength >= 0x80 || sequenceLength != signature.size - 2) return false
        val order = BigInteger("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)
        var offset = 2
        repeat(2) {
            if (offset + 2 > signature.size || signature[offset] != 0x02.toByte()) return false
            val length = signature[offset + 1].toInt() and 0xff
            val start = offset + 2
            val end = start + length
            if (length == 0 || length > 33 || end > signature.size) return false
            val first = signature[start].toInt() and 0xff
            if (first and 0x80 != 0) return false
            if (length > 1 && first == 0 && signature[start + 1].toInt() and 0x80 == 0) return false
            val value = BigInteger(1, signature.copyOfRange(start, end))
            if (value.signum() <= 0 || value >= order) return false
            offset = end
        }
        return offset == signature.size
    }
}
