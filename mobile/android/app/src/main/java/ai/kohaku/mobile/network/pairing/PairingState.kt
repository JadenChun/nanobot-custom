package ai.kohaku.mobile.network.pairing

import com.getcapacitor.JSObject
import java.util.Locale

internal enum class PairingStage {
    IDLE, STARTING_TRANSPORT, TAILSCALE_LOGIN_REQUIRED, COLLECTING, WAITING_APPROVAL,
    EXPIRED_RENEWABLE, PROVING, CANCELLING, PAIRED, FAILED,
}

internal enum class PairingMethod { QR, LINK, CODE }

internal enum class TailscaleState {
    IDLE, STARTING, NEEDS_LOGIN, READY, ERROR;

    companion object {
        fun fromWire(value: String) = entries.firstOrNull { it.name.lowercase(Locale.ROOT) == value } ?: IDLE
    }
}

internal data class PairingSnapshot(
    val stage: PairingStage = PairingStage.IDLE,
    val attemptId: String? = null,
    val method: PairingMethod? = null,
    val origin: String? = null,
    val installationId: String? = null,
    val pairingId: String? = null,
    val tailscaleState: TailscaleState = TailscaleState.IDLE,
    val tailscaleHasLoginUrl: Boolean = false,
    val comparisonCode: String? = null,
    val expiresAt: Long? = null,
    val canCancel: Boolean = false,
    val canRetry: Boolean = false,
    val error: PairingError? = null,
) {
    fun toJSObject(): JSObject = JSObject()
        .put("stage", stage.name.lowercase(Locale.ROOT))
        .put("attemptId", attemptId)
        .put("method", method?.name?.lowercase(Locale.ROOT))
        .put("origin", origin)
        .put("installationId", installationId)
        .put("pairingId", pairingId)
        .put("tailscaleState", tailscaleState.name.lowercase(Locale.ROOT))
        .put("tailscaleHasLoginUrl", tailscaleHasLoginUrl)
        .put("comparisonCode", comparisonCode)
        .apply { expiresAt?.let { put("expiresAt", it) } }
        .put("canCancel", canCancel)
        .put("canRetry", canRetry)
        .apply {
            error?.let {
                put("error", JSObject()
                    .put("kind", it.kind)
                    .apply { it.httpStatus?.let { status -> put("httpStatus", status) } }
                    .put("serverCode", it.serverCode)
                    .put("endpoint", it.endpoint)
                    .put("retryable", it.retryable)
                    .put("remediation", it.remediation)
                    .put("title", it.title)
                    .put("message", it.message)
                    .put("technicalDetail", it.technicalDetail))
            }
        }
}
