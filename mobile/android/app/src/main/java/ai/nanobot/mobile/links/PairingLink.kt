package ai.nanobot.mobile.links

import java.net.URI
import java.net.URLDecoder

data class PairingLink(
    val origin: String,
    val installationId: String,
    val challenge: String,
)

class PairingLinkError(message: String) : Exception(message)

object PairingLinkParser {
    private val allowedKeys = setOf("origin", "installation", "challenge", "transport")

    fun parse(raw: String, expectedInstallation: String? = null): PairingLink {
        val uri = try {
            URI(raw)
        } catch (exception: Exception) {
            throw PairingLinkError("invalid pairing link")
        }
        if (uri.scheme != "nanobot" || uri.host != "pair") {
            throw PairingLinkError("pairing link scheme/host is not accepted")
        }
        if (uri.userInfo != null || uri.fragment != null || uri.path?.trim('/')?.isNotEmpty() == true) {
            throw PairingLinkError("pairing link carries unsupported navigation")
        }
        val pairs = (uri.rawQuery ?: "").split("&").filter { it.isNotEmpty() }.map {
            val pieces = it.split("=", limit = 2)
            if (pieces.size != 2) throw PairingLinkError("pairing query is malformed")
            URLDecoder.decode(pieces[0], "UTF-8") to URLDecoder.decode(pieces[1], "UTF-8")
        }
        val values = pairs.groupBy({ it.first }, { it.second }).mapValues { it.value }
        if (!values.keys.all { it in allowedKeys } || values.any { it.value.size != 1 }) {
            throw PairingLinkError("pairing query is not exact")
        }
        if ("origin" !in values || "challenge" !in values) {
            throw PairingLinkError("pairing query is incomplete")
        }
        val origin = values.getValue("origin").single().trimEnd('/')
        val parsedOrigin = try { URI(origin) } catch (exception: Exception) { throw PairingLinkError("origin is invalid") }
        if (parsedOrigin.scheme != "https" || parsedOrigin.host.isNullOrBlank() ||
            parsedOrigin.userInfo != null || parsedOrigin.path?.trim('/')?.isNotEmpty() == true ||
            parsedOrigin.query != null || parsedOrigin.fragment != null
        ) {
            throw PairingLinkError("origin must be exact HTTPS")
        }
        val installation = values["installation"]?.single()
        if (installation != null && !Regex("^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$").matches(installation)) {
            throw PairingLinkError("installation is invalid")
        }
        if (expectedInstallation != null && installation != null && expectedInstallation != installation) {
            throw PairingLinkError("pairing link belongs to another installation")
        }
        val challenge = values.getValue("challenge").single()
        if (challenge.length !in 16..256) throw PairingLinkError("challenge is invalid")
        return PairingLink(
            origin = origin,
            installationId = installation ?: expectedInstallation ?: "",
            challenge = challenge,
        )
    }
}
