package ai.nanobot.mobile.network

/**
 * Classifies refresh failures that end local bearer authority.
 *
 * A confirmed owner revocation and an explicit v1-upgrade requirement both
 * require the user to pair again. Other 401/transport failures keep custody so
 * a transient race cannot orphan encrypted tsnet state.
 */
internal object RefreshAuthority {
    fun requiresRepair(status: Int, message: String): Boolean {
        if (status == 426) return true
        val lower = message.lowercase()
        if (lower.contains("upgrade_required")) return true
        return (status == 401 || status == 403) && lower.contains("device_revoked")
    }
}
