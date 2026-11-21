package ai.nanobot.mobile.network

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RefreshAuthorityTest {
    @Test
    fun upgradeRequiredEndsLocalBearerAuthority() {
        assertTrue(RefreshAuthority.requiresRepair(426, "HTTP 426: upgrade_required"))
        assertTrue(RefreshAuthority.requiresRepair(400, "server said upgrade_required"))
    }

    @Test
    fun confirmedRevocationEndsLocalBearerAuthority() {
        assertTrue(RefreshAuthority.requiresRepair(401, "HTTP 401: device_revoked"))
    }

    @Test
    fun transientFailureKeepsCustody() {
        assertFalse(RefreshAuthority.requiresRepair(401, "HTTP 401: invalid_refresh"))
        assertFalse(RefreshAuthority.requiresRepair(503, "HTTP 503: owner_unavailable"))
        assertFalse(RefreshAuthority.requiresRepair(0, "connection lost"))
    }
}
