package ai.nanobot.mobile.links

data class PairingChallenge(
    val pairingId: String,
    val nonce: String,
    val collectorSecret: String,
)

class PairingChallengeError(message: String) : Exception(message)

/**
 * Pairing links carry the owner-issued `pairing_id.nonce.collector_secret`
 * tuple in the opaque chain challenge field. The native bridge decodes it so
 * JavaScript never needs to handle the one-time collector material.
 */
object PairingChallengeCodec {
    private val pairingIdPattern = Regex("^[A-Za-z0-9][A-Za-z0-9-]{15,63}$")
    private val secretPattern = Regex("^[A-Za-z0-9_-]{16,128}$")

    fun decode(value: String): PairingChallenge {
        val parts = value.split(".")
        if (parts.size != 3) throw PairingChallengeError("pairing challenge is malformed")
        val (pairingId, nonce, collectorSecret) = parts
        if (!pairingIdPattern.matches(pairingId)) {
            throw PairingChallengeError("pairing id is invalid")
        }
        if (!secretPattern.matches(nonce) || !secretPattern.matches(collectorSecret)) {
            throw PairingChallengeError("pairing secret is invalid")
        }
        return PairingChallenge(pairingId, nonce, collectorSecret)
    }

    fun encode(challenge: PairingChallenge): String =
        "${challenge.pairingId}.${challenge.nonce}.${challenge.collectorSecret}"
}
