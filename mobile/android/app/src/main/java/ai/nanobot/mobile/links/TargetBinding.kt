package ai.nanobot.mobile.links

data class BoundTarget(val origin: String, val installationId: String, val challenge: String)

class TargetBindingError(message: String) : Exception(message)

object TargetBinding {
    fun bind(link: PairingLink, currentInstallation: String, currentOrigin: String?): BoundTarget {
        if (link.installationId != currentInstallation) {
            throw TargetBindingError("deep link installation does not match custody")
        }
        if (currentOrigin != null && currentOrigin.trimEnd('/') != link.origin) {
            throw TargetBindingError("deep link origin does not match the active target")
        }
        return BoundTarget(link.origin, link.installationId, link.challenge)
    }
}
