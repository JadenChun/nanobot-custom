package ai.kohaku.mobile.network.pairing

import java.util.concurrent.TimeoutException

internal class NativeHttpException(
    val status: Int,
    serverCode: String?,
    val endpoint: String,
) : IllegalStateException("HTTP $status" + (sanitize(serverCode)?.let { " $it" } ?: "")) {
    val serverCode: String? = sanitize(serverCode)

    private companion object {
        val pattern = Regex("[a-z0-9_]{1,64}")
        fun sanitize(code: String?): String? = code?.takeIf(pattern::matches)
    }
}

internal class NativeTransportException(val code: String) : IllegalStateException(code)

internal data class PairingError(
    val kind: String,
    val retryable: Boolean,
    val remediation: String,
    val title: String,
    val message: String,
    val httpStatus: Int? = null,
    val serverCode: String? = null,
    val endpoint: String? = null,
    val technicalDetail: String = "",
)

internal object PairingErrorMapper {
    fun map(failure: Exception, tailscaleState: TailscaleState): PairingError = when (failure) {
        is NativeHttpException -> http(failure, tailscaleState)
        is NativeTransportException -> transport(failure.code)
        is TimeoutException -> transport("timeout")
        else -> custody()
    }

    private fun http(failure: NativeHttpException, tailscaleState: TailscaleState): PairingError {
        val (status, code, endpoint) = Triple(failure.status, failure.serverCode, failure.endpoint)
        fun error(
            remediation: String, title: String, message: String, retryable: Boolean = false,
        ) = PairingError(
            "server_http", retryable, remediation, title, message, status, code, endpoint,
            "${failure.message} ($endpoint)",
        )
        fun serve(title: String, message: String) = error("check_desktop_serve", title, message, true)

        return when {
            status == 410 && code == "pairing_expired" -> error(
                "rescan_qr", "Pairing code expired",
                when {
                    tailscaleState == TailscaleState.READY ->
                        "Tailscale is connected. Scan a fresh pairing QR code from your desktop."
                    else -> "Scan a fresh pairing QR code from your desktop."
                },
            )
            status == 403 && code == "pairing_code_invalid" -> error(
                "edit_code", "Pairing code is incorrect", "Enter the 8-digit code shown on your desktop.",
            )
            status == 403 && code in setOf("pairing_denied", "pairing_rejected") -> error(
                "rescan_qr", "Pairing was denied", "Start a new pairing on your desktop and scan its QR code.",
            )
            status == 403 && code == "origin_mismatch" -> serve(
                "Desktop address mismatch", "Check that Tailscale Serve uses the desktop address shown in Kohaku.",
            )
            (status == 404 && code == "pairing_not_found") || (status == 410 && code == "pairing_consumed") -> error(
                "rescan_qr", "Pairing is no longer available", "Scan a fresh pairing QR code from your desktop.",
            )
            status == 404 && endpoint.startsWith("/mobile/v1/pair/") -> serve(
                "Desktop pairing route is missing",
                "Check the Kohaku gateway and the Tailscale Serve proxy for the /mobile/v1 path.",
            )
            status in setOf(502, 503) -> when (code) {
                "owner_unavailable" -> serve(
                    "Desktop gateway is unavailable", "Start the Kohaku gateway on your desktop and retry.",
                )
                "remote_access_disabled" -> serve(
                    "Remote access is disabled", "Enable remote access in Kohaku on your desktop and retry.",
                )
                "serve_unverified" -> serve(
                    "Tailscale Serve needs verification", "Verify Tailscale Serve in Kohaku on your desktop and retry.",
                )
                else -> serve("Desktop is unavailable", "Check the Kohaku gateway and Tailscale Serve, then retry.")
            }
            status == 409 && code == "pairing_busy" -> error(
                "rescan_qr", "Pairing is busy", "Start a new pairing on your desktop and scan its QR code.",
            )
            status == 429 -> error(
                "retry", "Too many pairing requests", "Wait a moment and retry pairing.", true,
            )
            else -> error("retry", "Desktop request failed", "Retry pairing after you check your desktop.", true)
        }
    }

    private fun transport(code: String): PairingError {
        val message = when (code) {
            "timeout" -> "The desktop took too long to respond. Retry pairing."
            "tls" -> "The secure connection to the desktop failed. Retry pairing."
            else -> "The connection to the desktop failed. Retry pairing."
        }
        return PairingError(
            "transport", true, "retry", "Desktop connection failed", message,
            technicalDetail = code.takeIf { it in setOf("timeout", "network", "tls") } ?: "network",
        )
    }

    fun tailscale(needsLogin: Boolean = false): PairingError = if (needsLogin) {
        PairingError(
            "tailscale", true, "open_tailscale_login", "Tailscale sign-in required",
            "Sign in to Tailscale to continue pairing.",
            technicalDetail = "Tailscale needs sign-in.",
        )
    } else {
        PairingError(
            "tailscale", true, "retry", "Tailscale connection failed", "Retry to connect to Tailscale.",
            technicalDetail = "Tailscale did not become ready.",
        )
    }

    fun invalid(method: PairingMethod): PairingError = if (method == PairingMethod.CODE) {
        PairingError(
            "challenge_invalid", false, "edit_code", "Check the pairing details",
            "Enter the HTTPS desktop address and the 8-digit code shown on your desktop.",
            technicalDetail = "Pairing address or code is invalid.",
        )
    } else {
        PairingError(
            "challenge_invalid", false, "rescan_qr", "Invalid pairing link",
            "Scan a fresh pairing QR code from your desktop.",
            technicalDetail = "Pairing link is invalid.",
        )
    }

    fun custody(): PairingError = PairingError(
        "custody", true, "retry", "Pairing could not be saved", "Retry pairing on this phone.",
        technicalDetail = "Encrypted pairing data is unavailable.",
    )
}
