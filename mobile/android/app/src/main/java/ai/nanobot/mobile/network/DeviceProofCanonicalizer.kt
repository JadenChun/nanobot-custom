package ai.nanobot.mobile.network

import java.nio.charset.StandardCharsets
import org.json.JSONObject
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

/** Fixed mobile/contracts/device-proof-v2-schema.json; no value coercion. */
@OptIn(ExperimentalEncodingApi::class)
internal object DeviceProofCanonicalizer {
    private val integerKeys = setOf("version", "device_generation", "expires_at")
    private val keys = integerKeys + setOf(
        "purpose", "instance_id", "origin", "installation_id", "device_fingerprint",
        "challenge_id", "nonce",
    )
    private val base64Url = Base64.UrlSafe.withPadding(Base64.PaddingOption.ABSENT)

    fun fromJson(value: JSONObject): Map<String, Any?> = value.keys().asSequence()
        .associateWith { value.get(it) }

    fun canonicalJson(values: Map<String, Any?>): String {
        require(values.keys == keys) { "invalid proof shape" }
        for ((key, value) in values) {
            if (key in integerKeys) {
                require((value is Int || value is Long) && (value as Number).toLong() > 0) {
                    "invalid integer claim"
                }
            } else {
                require(value is String && value.isNotEmpty()) { "invalid text claim" }
                requireValidUnicode(value)
            }
        }
        require((values["version"] as Number).toLong() == 2L) { "unsupported proof version" }
        require(values["purpose"] == "remote.access") { "unsupported proof purpose" }
        return keys.sorted().joinToString(prefix = "{", postfix = "}", separator = ",") { key ->
            quote(key) + ":" + if (key in integerKeys) values[key].toString() else quote(values[key] as String)
        }
    }

    fun canonicalBytes(values: Map<String, Any?>): ByteArray =
        canonicalJson(values).toByteArray(StandardCharsets.UTF_8)

    fun encode(bytes: ByteArray): String = base64Url.encode(bytes)

    fun decode(value: String): ByteArray {
        require(value.matches(Regex("[A-Za-z0-9_-]+"))) { "invalid proof encoding" }
        val bytes = base64Url.decode(value)
        require(encode(bytes) == value) { "non-canonical proof encoding" }
        return bytes
    }

    private fun requireValidUnicode(value: String) {
        var index = 0
        while (index < value.length) {
            val char = value[index++]
            if (Character.isHighSurrogate(char)) {
                require(index < value.length && Character.isLowSurrogate(value[index++])) { "invalid Unicode" }
            } else {
                require(!Character.isLowSurrogate(char)) { "invalid Unicode" }
            }
        }
    }

    private fun quote(value: String): String = buildString {
        append('"')
        for (character in value) {
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000c' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character.code < 0x20) {
                    append("\\u")
                    append(character.code.toString(16).padStart(4, '0'))
                } else append(character)
            }
        }
        append('"')
    }
}
