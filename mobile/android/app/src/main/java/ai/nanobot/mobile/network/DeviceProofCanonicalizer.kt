package ai.nanobot.mobile.network

import java.nio.charset.StandardCharsets
import org.json.JSONObject

/** Exact canonical JSON bytes signed by the Android device key. */
internal object DeviceProofCanonicalizer {
    fun fromJson(value: JSONObject): Map<String, Any?> {
        val result = mutableMapOf<String, Any?>()
        val keys = value.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val raw = value.get(key)
            result[key] = if (raw is JSONObject) fromJson(raw) else raw
        }
        return result
    }

    fun canonicalJson(values: Map<String, Any?>): String {
        val keys = values.keys.sorted()
        return keys.joinToString(prefix = "{", postfix = "}", separator = ",") { key ->
            quote(key) + ":" + canonicalValue(values[key])
        }
    }

    fun canonicalBytes(values: Map<String, Any?>): ByteArray =
        canonicalJson(values).toByteArray(StandardCharsets.UTF_8)

    private fun quote(value: String): String = buildString {
        append('"')
        for (character in value) {
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (character.code < 0x20) {
                        append("\\u")
                        append(character.code.toString(16).padStart(4, '0'))
                    } else {
                        append(character)
                    }
                }
            }
        }
        append('"')
    }

    private fun canonicalValue(value: Any?): String = when (value) {
        is String -> quote(value)
        is Int, is Long, is Boolean -> value.toString()
        is Map<*, *> -> canonicalJson(
            value.entries.associate { (key, item) -> key.toString() to item }
        )
        else -> quote(value?.toString() ?: "")
    }
}
