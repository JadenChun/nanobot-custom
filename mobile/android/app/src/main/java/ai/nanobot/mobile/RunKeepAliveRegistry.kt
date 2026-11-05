package ai.nanobot.mobile

import java.util.concurrent.ConcurrentHashMap

/**
 * Chats whose run should keep the Android process alive while the app is in
 * the background. The registry is process-local: if the process dies, the
 * foreground service dies with it and there is nothing left to notify.
 */
object RunKeepAliveRegistry {
    private val chatIds = ConcurrentHashMap.newKeySet<String>()

    @Synchronized
    fun add(chatId: String): Boolean = chatIds.add(chatId)

    @Synchronized
    fun remove(chatId: String): Boolean = chatIds.remove(chatId)

    @Synchronized
    fun clear() {
        chatIds.clear()
    }

    @Synchronized
    fun isEmpty(): Boolean = chatIds.isEmpty()
}
