package ai.nanobot.mobile.network

/**
 * Narrow JNI surface. Every method is implemented by native_network_jni.c,
 * which forwards only to the typed C ABI. There is no URL, dial or LocalAPI
 * entry point here.
 */
object NativeNetwork {
    init {
        System.loadLibrary("nanobot_network_jni")
    }

    external fun abiVersion(): Int
    external fun start(configJson: String): Long
    external fun status(handle: Long): ByteArray?
    external fun loginUrl(handle: Long): ByteArray?
    external fun http(handle: Long, requestJson: String): Long
    external fun cancel(handle: Long, requestId: Long): Int
    external fun wsOpen(handle: Long, requestJson: String): Long
    external fun wsSend(handle: Long, wsId: Long, kind: Int, data: ByteArray): Int
    external fun wsClose(handle: Long, wsId: Long): Int
    external fun pollEvent(handle: Long): ByteArray?
    external fun stop(handle: Long): Int
    external fun lastError(): String
}
