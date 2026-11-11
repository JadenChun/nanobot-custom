package ai.nanobot.mobile.network

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.FileOutputStream
import java.security.GeneralSecurityException
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.SecureRandom
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import org.json.JSONObject

class CustodyException(message: String, cause: Throwable? = null) : Exception(message, cause)

class CustodyMismatchException(message: String) : Exception(message)

class CustodyUnknownVersionException(message: String) : Exception(message)

class CustodyCorruptException(message: String, cause: Throwable? = null) : Exception(message, cause)

class CustodyKeyInvalidException(message: String, cause: Throwable? = null) : Exception(message, cause)

data class PendingPairingSetup(
    val origin: String,
    val pairingId: String,
    val challenge: String,
    val generation: Long,
    val code: String = "",
    val label: String = "",
)

data class CustodyRecord(
    val version: Int = 1,
    val installationId: String,
    val origin: String,
    val hostname: String,
    val refreshToken: String = "",
    val accessToken: String = "",
    val accessExpiresAt: Long = 0,
    val refreshExpiresAt: Long = 0,
    val stateKeyBase64: String = "",
    val revocationPending: Boolean = false,
    val updatedAt: Long = 0,
    val pendingPairingId: String = "",
    val pendingPairingOrigin: String = "",
    val pendingPairingChallenge: String = "",
    val pendingPairingGeneration: Long = 0,
    val pendingPairingCode: String = "",
    val pendingPairingLabel: String = "",
) {
    fun toJson(): JSONObject = JSONObject()
        .put("version", version)
        .put("installation_id", installationId)
        .put("origin", origin)
        .put("hostname", hostname)
        .put("refresh_token", refreshToken)
        .put("access_token", accessToken)
        .put("access_expires_at", accessExpiresAt)
        .put("refresh_expires_at", refreshExpiresAt)
        .put("state_key_base64", stateKeyBase64)
        .put("revocation_pending", revocationPending)
        .put("updated_at", updatedAt)
        .put("pending_pairing_id", pendingPairingId)
        .put("pending_pairing_origin", pendingPairingOrigin)
        .put("pending_pairing_challenge", pendingPairingChallenge)
        .put("pending_pairing_generation", pendingPairingGeneration)
        .put("pending_pairing_code", pendingPairingCode)
        .put("pending_pairing_label", pendingPairingLabel)

    companion object {
        fun fromJson(json: JSONObject): CustodyRecord {
            val version = json.optInt("version", -1)
            if (version != 1) {
                throw CustodyUnknownVersionException("unsupported custody record version: $version")
            }
            val installationId = json.optString("installation_id")
            val origin = json.optString("origin")
            val stateKey = json.optString("state_key_base64")
            if (installationId.isBlank() || origin.isBlank()) {
                throw CustodyException("custody record is missing its installation binding")
            }
            val decoded = Base64.decode(stateKey, Base64.NO_WRAP)
            if (decoded.size != 32) {
                throw CustodyException("custody state key is not 32 bytes")
            }
            return CustodyRecord(
                version = version,
                installationId = installationId,
                origin = origin,
                hostname = json.optString("hostname", "nanobot-android"),
                refreshToken = json.optString("refresh_token"),
                accessToken = json.optString("access_token"),
                accessExpiresAt = json.optLong("access_expires_at"),
                refreshExpiresAt = json.optLong("refresh_expires_at"),
                stateKeyBase64 = stateKey,
                revocationPending = json.optBoolean("revocation_pending", false),
                updatedAt = json.optLong("updated_at"),
                pendingPairingId = json.optString("pending_pairing_id"),
                pendingPairingOrigin = json.optString("pending_pairing_origin"),
                pendingPairingChallenge = json.optString("pending_pairing_challenge"),
                pendingPairingGeneration = json.optLong("pending_pairing_generation"),
                pendingPairingCode = json.optString("pending_pairing_code"),
                pendingPairingLabel = json.optString("pending_pairing_label"),
            )
        }
    }
}

/**
 * App-private custody for the installation binding, refresh material and the
 * key that encrypts tsnet state. The Android Keystore key never leaves native
 * code; JS sees only sanitized status.
 */
class CustodyStore(context: Context) {
    private val appContext = context.applicationContext
    private val custodyDir = File(appContext.filesDir, "custody")
    private val recordFile = File(custodyDir, "record.enc")
    private val keyStore: KeyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }

    private val attemptFence = PendingAttemptFence(
        read = {
            val file = File(custodyDir, "attempt.generation")
            if (!file.exists()) PendingAttemptFence.State(0, false) else {
                val pieces = file.readText(Charsets.US_ASCII).split(":")
                require(pieces.size == 3 && pieces[0] == "1" && pieces[2] in setOf("0", "1")) { "invalid pending attempt fence" }
                val generation = pieces[1].toLong()
                require(generation >= 0) { "invalid pending generation" }
                PendingAttemptFence.State(generation, pieces[2] == "1")
            }
        },
        persist = { state ->
            ensureDirectory()
            val temporary = File(custodyDir, "attempt.generation.tmp")
            FileOutputStream(temporary).use { stream ->
                stream.write("1:${state.generation}:${if (state.active) 1 else 0}".toByteArray(Charsets.US_ASCII))
                stream.fd.sync()
            }
            check(temporary.renameTo(File(custodyDir, "attempt.generation"))) { "pending fence could not be persisted" }
            syncDirectory()
        },
    )

    fun attemptGeneration(): Long = attemptFence.generation()
    fun cancelPairing() { attemptFence.cancel() }
    fun requirePairing(generation: Long) { attemptFence.withCurrent(generation) {} }
    fun requireAuthority(generation: Long) { attemptFence.withCurrent(generation, requireActive = false) {} }
    fun <T> withPairing(generation: Long, action: () -> T): T = attemptFence.withCurrent(generation, write = action)

    fun beginPairing(origin: String, installationId: String, challenge: String = "", code: String = "", label: String = ""): Long {
        val generation = attemptFence.begin()
        attemptFence.withCurrent(generation) {
            val current = loadOrCreate(origin, installationId)
            save(current.copy(pendingPairingId = "", pendingPairingOrigin = origin,
                pendingPairingChallenge = challenge, pendingPairingGeneration = generation,
                pendingPairingCode = code, pendingPairingLabel = label))
        }
        return generation
    }

    @Synchronized
    fun loadOrCreate(origin: String, installationId: String?, hostname: String = "nanobot-android"): CustodyRecord {
        val existing = load()
        if (existing != null) {
            if (existing.origin != origin) {
                throw CustodyMismatchException("custody is bound to a different origin")
            }
            if (!installationId.isNullOrBlank() && existing.installationId != installationId) {
                throw CustodyMismatchException("custody is bound to a different installation")
            }
            return existing
        }
        val record = CustodyRecord(
            installationId = installationId?.takeIf { it.isNotBlank() } ?: UUID.randomUUID().toString(),
            origin = origin,
            hostname = hostname,
            stateKeyBase64 = Base64.encodeToString(randomBytes(32), Base64.NO_WRAP),
            updatedAt = System.currentTimeMillis(),
        )
        save(record)
        return record
    }

    @Synchronized
    fun load(): CustodyRecord? {
        if (!recordFile.exists()) {
            return null
        }
        val bytes = recordFile.readBytes()
        if (bytes.size < 6 || String(bytes, 0, 4, Charsets.US_ASCII) != MAGIC) {
            throw CustodyCorruptException("custody record magic is invalid")
        }
        val fileVersion = bytes[4].toInt() and 0xff
        if (fileVersion != FILE_VERSION) {
            throw CustodyUnknownVersionException("unsupported custody file version: $fileVersion")
        }
        val nonceLength = bytes[5].toInt() and 0xff
        if (nonceLength != 12 || bytes.size < 6 + nonceLength + 16) {
            throw CustodyCorruptException("custody record is truncated")
        }
        val nonce = bytes.copyOfRange(6, 6 + nonceLength)
        val ciphertext = bytes.copyOfRange(6 + nonceLength, bytes.size)
        val plaintext = try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, getKey(createIfMissing = false), GCMParameterSpec(128, nonce))
            cipher.updateAAD(AAD)
            cipher.doFinal(ciphertext)
        } catch (exception: GeneralSecurityException) {
            throw CustodyCorruptException("custody record cannot be decrypted", exception)
        }
        return try {
            CustodyRecord.fromJson(JSONObject(String(plaintext, Charsets.UTF_8)))
        } catch (exception: CustodyUnknownVersionException) {
            throw exception
        } catch (exception: Exception) {
            throw CustodyCorruptException("custody record is malformed", exception)
        }
    }

    @Synchronized
    fun save(record: CustodyRecord) {
        if (record.version != 1) {
            throw CustodyUnknownVersionException("unsupported custody record version: ${record.version}")
        }
        ensureDirectory()
        val plaintext = record.toJson().toString().toByteArray(Charsets.UTF_8)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, getKey(createIfMissing = !recordFile.exists()))
        cipher.updateAAD(AAD)
        val ciphertext = cipher.doFinal(plaintext)
        val nonce = cipher.iv
        val payload = ByteArray(6 + nonce.size + ciphertext.size)
        MAGIC.toByteArray(Charsets.US_ASCII).copyInto(payload, 0)
        payload[4] = FILE_VERSION.toByte()
        payload[5] = nonce.size.toByte()
        nonce.copyInto(payload, 6)
        ciphertext.copyInto(payload, 6 + nonce.size)

        val temp = File(custodyDir, "record.tmp")
        FileOutputStream(temp).use { stream ->
            stream.write(payload)
            stream.fd.sync()
        }
        if (!temp.renameTo(recordFile)) {
            temp.delete()
            throw CustodyException("custody record could not be replaced atomically")
        }
        syncDirectory()
    }

    @Synchronized
    fun saveTokens(
        origin: String,
        installationId: String,
        accessToken: String,
        refreshToken: String,
        accessExpiresIn: Int = 600,
        refreshExpiresIn: Int = 30 * 24 * 60 * 60,
    ): CustodyRecord {
        if (accessToken.isBlank() || refreshToken.isBlank()) {
            throw CustodyException("paired credentials are empty")
        }
        if (accessToken.length > 512 || refreshToken.length > 512) {
            throw CustodyException("paired credentials exceed the bound")
        }
        val record = loadOrCreate(origin, installationId)
        val now = System.currentTimeMillis()
        val updated = record.copy(
            accessToken = accessToken,
            refreshToken = refreshToken,
            accessExpiresAt = now + accessExpiresIn * 1000L,
            // 0 means "until revoke" (owner device); keep it distinct from a
            // timestamp that is already in the past.
            refreshExpiresAt = if (refreshExpiresIn <= 0) 0L else now + refreshExpiresIn * 1000L,
            updatedAt = now,
        )
        save(updated)
        return updated
    }

    fun savePendingPairing(
        origin: String,
        installationId: String,
        pairingId: String,
        challenge: String,
        generation: Long,
    ): CustodyRecord = attemptFence.withCurrent(generation) {
        synchronized(this) {
        if (origin.isBlank() || pairingId.isBlank() || challenge.isBlank()) {
            throw CustodyException("pending pairing is incomplete")
        }
        if (pairingId.length > 128 || challenge.length > 4096) {
            throw CustodyException("pending pairing exceeds the bound")
        }
        val record = load() ?: loadOrCreate(origin, installationId)
        if (record.origin != origin) {
            throw CustodyMismatchException("custody is bound to a different origin")
        }
        val updated = record.copy(
            pendingPairingId = pairingId,
            pendingPairingOrigin = origin,
            pendingPairingChallenge = challenge,
            pendingPairingGeneration = generation,
            pendingPairingCode = "",
            updatedAt = System.currentTimeMillis(),
        )
        save(updated)
        updated
        }
    }

    fun loadPendingPairing(): PendingPairingSetup? {
        val record = load() ?: return null
        if (
            !attemptFence.isCurrent(record.pendingPairingGeneration) ||
            record.pendingPairingOrigin.isBlank() ||
            (record.pendingPairingChallenge.isBlank() && record.pendingPairingCode.isBlank())
        ) {
            return null
        }
        return PendingPairingSetup(
            origin = record.pendingPairingOrigin,
            pairingId = record.pendingPairingId,
            challenge = record.pendingPairingChallenge,
            generation = record.pendingPairingGeneration,
            code = record.pendingPairingCode,
            label = record.pendingPairingLabel,
        )
    }

    @Synchronized
    fun clearPendingPairing() {
        val record = load() ?: return
        if (
            record.pendingPairingId.isBlank() &&
            record.pendingPairingChallenge.isBlank()
        ) {
            return
        }
        save(
            record.copy(
                pendingPairingId = "",
                pendingPairingOrigin = "",
                pendingPairingChallenge = "",
                updatedAt = System.currentTimeMillis(),
            )
        )
    }

    /**
     * Persist a rotation only while the same live custody binding still exists.
     * A logout/revoke that cleared or fenced custody must win over an older
     * in-flight refresh completion.
     */
    fun saveRefreshedTokens(
        origin: String,
        installationId: String,
        accessToken: String,
        refreshToken: String,
        accessExpiresIn: Int = 600,
        refreshExpiresIn: Int = 30 * 24 * 60 * 60,
        expectedGeneration: Long = attemptGeneration(),
    ): CustodyRecord = attemptFence.withCurrent(expectedGeneration, requireActive = false) {
        synchronized(this) {
        if (accessToken.isBlank()) {
            throw CustodyException("refreshed credentials are empty")
        }
        if (refreshToken.isBlank() && refreshExpiresIn > 0) {
            throw CustodyException("refreshed credentials are empty")
        }
        if (accessToken.length > 512 || refreshToken.length > 512) {
            throw CustodyException("refreshed credentials exceed the bound")
        }
        val current = load() ?: throw CustodyMismatchException("custody was cleared during refresh")
        if (
            current.installationId != installationId ||
            current.origin != origin ||
            current.revocationPending
        ) {
            throw CustodyMismatchException("custody binding changed during refresh")
        }
        val now = System.currentTimeMillis()
        val updated = current.copy(
            accessToken = accessToken,
            refreshToken = refreshToken,
            accessExpiresAt = now + accessExpiresIn * 1000L,
            refreshExpiresAt = if (refreshExpiresIn <= 0) 0L else now + refreshExpiresIn * 1000L,
            updatedAt = now,
        )
        save(updated)
        updated
        }
    }

    fun savePairingTokens(generation: Long, record: CustodyRecord, approved: JSONObject): CustodyRecord =
        attemptFence.withCurrent(generation) {
            val saved = saveRefreshedTokens(record.origin, record.installationId,
                approved.getString("access_token"), approved.optString("refresh_token"),
                approved.optInt("expires_in", 600), approved.optInt("refresh_expires_in", 0),
                expectedGeneration = generation)
            val settled = saved.copy(pendingPairingId = "", pendingPairingOrigin = "",
                pendingPairingChallenge = "", pendingPairingCode = "", pendingPairingLabel = "")
            save(settled)
            settled
        }

    /**
     * Return the installation identity used for pairing. Before the first
     * successful pairing this is a persistent app-private UUID; afterwards it
     * is the installation bound inside the encrypted custody record.
     */
    @Synchronized
    fun installationIdentity(): String {
        val existing = load()
        if (existing != null) return existing.installationId
        val file = pendingInstallationFile()
        if (file.exists()) {
            val value = file.readText(Charsets.UTF_8).trim()
            if (Regex("^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$").matches(value)) return value
            file.delete()
        }
        ensureDirectory()
        val generated = UUID.randomUUID().toString()
        file.writeText(generated, Charsets.UTF_8)
        file.setReadable(false, false)
        file.setWritable(false, false)
        file.setReadable(true, true)
        file.setWritable(true, true)
        return generated
    }

    @Synchronized
    fun status(): Map<String, Any> {
        val record = try {
            load()
        } catch (exception: Exception) {
            return mapOf(
                "keystore" to "android-keystore",
                "state" to "corrupt-or-invalid",
                "error" to (exception.message ?: "custody unavailable"),
            )
        }
        if (record == null) {
            return mapOf("keystore" to "android-keystore", "state" to "absent")
        }
        return mapOf(
            "keystore" to "android-keystore",
            "state" to "ready",
            "installation_id" to record.installationId,
            "origin" to record.origin,
            "version" to record.version,
            "has_refresh" to record.refreshToken.isNotBlank(),
            "revocation_pending" to record.revocationPending,
            "updated_at" to record.updatedAt,
        )
    }

    /** Returns true when server-side revocation remains pending. */
    @Synchronized
    fun logout(serverReachable: Boolean): Boolean {
        val record = load()
        if (!serverReachable) {
            if (record == null) {
                return false
            }
            save(
                record.copy(
                    refreshToken = "",
                    accessToken = "",
                    accessExpiresAt = 0,
                    refreshExpiresAt = 0,
                    revocationPending = true,
                    updatedAt = System.currentTimeMillis(),
                )
            )
            return true
        }
        clear()
        return false
    }

    @Synchronized
    fun clear() {
        recordFile.delete()
        File(custodyDir, "record.tmp").delete()
        pendingInstallationFile().delete()
        try {
            if (keyStore.containsAlias(ALIAS)) {
                keyStore.deleteEntry(ALIAS)
            }
        } catch (_: Exception) {
            // The data is already erased; a stale key cannot recover it.
        }
    }

    @Synchronized
    fun purgeMedia() {
        File(appContext.cacheDir, "nanobot-media").deleteRecursively()
        File(appContext.filesDir, "nanobot-tmp").deleteRecursively()
    }

    /** Return this installation's P-256 SubjectPublicKeyInfo DER bytes. */
    @Synchronized
    fun deviceProofPublicKey(): ByteArray {
        val certificate = keyStore.getCertificate(DEVICE_PROOF_ALIAS)
            ?: generateDeviceProofKey()
        return certificate.publicKey.encoded
    }

    /** Sign canonical proof bytes with the non-exportable Android Keystore key. */
    @Synchronized
    fun signDeviceProof(payload: ByteArray): ByteArray {
        if (payload.isEmpty() || payload.size > 4096) {
            throw CustodyException("device proof payload is outside the bound")
        }
        deviceProofPublicKey()
        val privateKey = (keyStore.getEntry(DEVICE_PROOF_ALIAS, null) as? KeyStore.PrivateKeyEntry)
            ?.privateKey
            ?: throw CustodyKeyInvalidException("device proof key is missing")
        return try {
            Signature.getInstance("SHA256withECDSA").run {
                initSign(privateKey)
                update(payload)
                sign()
            }
        } catch (exception: GeneralSecurityException) {
            throw CustodyKeyInvalidException("device proof signing failed", exception)
        }
    }

    internal fun deleteDeviceProofKeyForTest() {
        if (keyStore.containsAlias(DEVICE_PROOF_ALIAS)) {
            keyStore.deleteEntry(DEVICE_PROOF_ALIAS)
        }
    }

    private fun generateDeviceProofKey(): java.security.cert.Certificate {
        val generator = KeyPairGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_EC,
            "AndroidKeyStore",
        )
        generator.initialize(
            KeyGenParameterSpec.Builder(
                DEVICE_PROOF_ALIAS,
                KeyProperties.PURPOSE_SIGN,
            )
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setAlgorithmParameterSpec(ECGenParameterSpec("secp256r1"))
                .build()
        )
        generator.generateKeyPair()
        return keyStore.getCertificate(DEVICE_PROOF_ALIAS)
            ?: throw CustodyKeyInvalidException("device proof key could not be created")
    }

    internal fun deleteKeyForTest() {
        if (keyStore.containsAlias(ALIAS)) {
            keyStore.deleteEntry(ALIAS)
        }
    }

    internal fun recordFileForTest(): File = recordFile

    private fun pendingInstallationFile(): File = File(custodyDir, "installation.id")

    private fun syncDirectory() {
        val descriptor = Os.open(custodyDir.absolutePath, OsConstants.O_RDONLY or OsConstants.O_DIRECTORY, 0)
        try { Os.fsync(descriptor) } finally { Os.close(descriptor) }
    }

    private fun ensureDirectory() {
        if (!custodyDir.exists() && !custodyDir.mkdirs()) {
            throw CustodyException("custody directory could not be created")
        }
        custodyDir.setReadable(false, false)
        custodyDir.setWritable(false, false)
        custodyDir.setReadable(true, true)
        custodyDir.setWritable(true, true)
    }

    private fun getKey(createIfMissing: Boolean): SecretKey {
        val existing = try {
            keyStore.getEntry(ALIAS, null) as? KeyStore.SecretKeyEntry
        } catch (exception: Exception) {
            throw CustodyKeyInvalidException("Android Keystore key is no longer usable", exception)
        }
        if (existing != null) {
            return existing.secretKey
        }
        if (!createIfMissing) {
            throw CustodyKeyInvalidException("Android Keystore key is missing for an existing custody record")
        }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }

    private fun randomBytes(size: Int): ByteArray {
        val bytes = ByteArray(size)
        SecureRandom().nextBytes(bytes)
        return bytes
    }

    companion object {
        private const val ALIAS = "nanobot.mobile.custody.v1"
        private const val DEVICE_PROOF_ALIAS = "nanobot.mobile.device-proof.v1"
        private const val MAGIC = "NBC1"
        private const val FILE_VERSION = 1
        private val AAD = "nanobot-custody-v1".toByteArray(Charsets.UTF_8)
    }
}
