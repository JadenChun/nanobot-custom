package ai.kohaku.mobile.network

import ai.kohaku.mobile.network.pairing.NativeHttpException
import ai.kohaku.mobile.network.pairing.NativeTransportException
import ai.kohaku.mobile.network.pairing.PairingErrorMapper
import ai.kohaku.mobile.network.pairing.TailscaleState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PairingErrorMapperTest {
    @Test
    fun preservesSafeCodesAndSanitizesOtherContent() {
        val endpoint = "/mobile/v1/pair/collect"
        val safe = PairingErrorMapper.map(NativeHttpException(400, "unknown_code_9", endpoint), TailscaleState.READY)
        assertEquals("server_http", safe.kind)
        assertEquals(400, safe.httpStatus)
        assertEquals("unknown_code_9", safe.serverCode)
        assertEquals(endpoint, safe.endpoint)
        assertEquals("HTTP 400 unknown_code_9 ($endpoint)", safe.technicalDetail)
        for (unsafe in listOf("unexpected text", "https://host", "CODE", "code\n", "a".repeat(65))) {
            val error = PairingErrorMapper.map(NativeHttpException(400, unsafe, endpoint), TailscaleState.READY)
            assertNull(error.serverCode)
            assertEquals("HTTP 400 ($endpoint)", error.technicalDetail)
        }
    }

    @Test
    fun expiredOfferAsksForFreshQrAndShowsReadyTailscale() {
        for (operation in listOf("collect", "code")) {
            val failure = NativeHttpException(410, "pairing_expired", "/mobile/v1/pair/$operation")
            val error = PairingErrorMapper.map(failure, TailscaleState.READY)
            assertEquals("rescan_qr", error.remediation)
            assertFalse(error.retryable)
            assertTrue(error.message.contains("Tailscale is connected"))
            assertTrue(error.message.contains("Scan a fresh pairing QR code"))
            assertFalse(PairingErrorMapper.map(failure, TailscaleState.STARTING).message.contains("Tailscale is connected"))
        }
    }

    @Test
    fun mapsDesktopResponsesToRemediation() {
        data class Case(val status: Int, val code: String, val remediation: String, val retryable: Boolean)
        val cases = listOf(
            Case(403, "pairing_code_invalid", "edit_code", false),
            Case(403, "pairing_denied", "rescan_qr", false),
            Case(403, "pairing_rejected", "rescan_qr", false),
            Case(403, "origin_mismatch", "check_desktop_serve", true),
            Case(409, "pairing_busy", "rescan_qr", false),
            Case(429, "pairing_rate_limited", "retry", true),
        )
        for (case in cases) {
            val failure = NativeHttpException(case.status, case.code, "/mobile/v1/pair/collect")
            val error = PairingErrorMapper.map(failure, TailscaleState.READY)
            assertEquals(case.code, error.serverCode)
            assertEquals(case.status, error.httpStatus)
            assertEquals(case.remediation, error.remediation)
            assertEquals(case.retryable, error.retryable)
            assertTrue(error.title.isNotBlank())
            assertTrue(error.message.isNotBlank())
        }
    }

    @Test
    fun distinguishesMissingServeRouteFromMissingPairing() {
        for (code in listOf(null, "not_found", "pairing_not_found")) {
            val failure = NativeHttpException(404, code, "/mobile/v1/pair/poll")
            val error = PairingErrorMapper.map(failure, TailscaleState.READY)
            if (code == "pairing_not_found") {
                assertNotEquals("check_desktop_serve", error.remediation)
            } else {
                assertEquals("check_desktop_serve", error.remediation)
                assertTrue(error.retryable)
                assertTrue(error.message.contains("Kohaku gateway"))
                assertTrue(error.message.contains("Tailscale Serve"))
                assertTrue(error.message.contains("/mobile/v1"))
            }
        }
    }

    @Test
    fun distinguishesDesktopAvailabilityFailures() {
        val codes = listOf("owner_unavailable", "remote_access_disabled", "serve_unverified")
        for (status in listOf(502, 503)) {
            val errors = codes.map {
                PairingErrorMapper.map(NativeHttpException(status, it, "/mobile/v1/pair/code"), TailscaleState.READY)
            }
            assertEquals(codes.size, errors.map { it.message }.toSet().size)
            assertTrue(errors.all { it.remediation == "check_desktop_serve" && it.retryable })
        }
    }

    @Test
    fun mapsTransportFailuresToRetry() {
        for (code in listOf("timeout", "network", "tls")) {
            val error = PairingErrorMapper.map(NativeTransportException(code), TailscaleState.READY)
            assertEquals("transport", error.kind)
            assertEquals("retry", error.remediation)
            assertTrue(error.retryable)
        }
    }
}
