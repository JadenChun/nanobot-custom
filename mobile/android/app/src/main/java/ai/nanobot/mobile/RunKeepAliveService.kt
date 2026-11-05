package ai.nanobot.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat

/**
 * Keeps the app process out of Android's cached-app freezer while a
 * phone-started run is active. The WebSocket and WebView JavaScript continue
 * to run in the same process, so the completion event can still produce a
 * local notification after the user leaves the app.
 *
 * `dataSync` is subject to Android 15's six-hour-per-24-hour quota. The
 * service stops itself on `onTimeout`; after the quota is exhausted Android
 * rejects another `dataSync` foreground service until the user foregrounds
 * the app. Force-stopping or swiping the app away still ends monitoring.
 */
class RunKeepAliveService : Service() {
    companion object {
        private const val TAG = "NanobotKeepAlive"
        private const val CHANNEL_ID = "nanobot_run_keepalive"
        private const val NOTIFICATION_ID = 0x4E42
        private const val MAX_RUNTIME_MS = 6L * 60L * 60L * 1000L

        fun start(context: Context) {
            val intent = Intent(context, RunKeepAliveService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, RunKeepAliveService::class.java))
        }
    }

    private val handler = Handler(Looper.getMainLooper())
    private val timeout = Runnable {
        Log.w(TAG, "run keepalive reached its runtime limit")
        RunKeepAliveRegistry.clear()
        stopSelf()
    }
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return try {
            val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            } else {
                0
            }
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                buildNotification(),
                type,
            )
            resetWakeLock()
            handler.removeCallbacks(timeout)
            handler.postDelayed(timeout, MAX_RUNTIME_MS)
            START_NOT_STICKY
        } catch (exception: Exception) {
            Log.e(TAG, "could not start the run keepalive foreground service", exception)
            RunKeepAliveRegistry.clear()
            stopSelf()
            START_NOT_STICKY
        }
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        Log.w(TAG, "dataSync run keepalive timed out")
        RunKeepAliveRegistry.clear()
        stopSelf(startId)
    }

    override fun onDestroy() {
        handler.removeCallbacks(timeout)
        releaseWakeLock()
        RunKeepAliveRegistry.clear()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun resetWakeLock() {
        releaseWakeLock()
        val power = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = power.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "Nanobot::RunKeepAlive",
        ).apply {
            setReferenceCounted(false)
            acquire(MAX_RUNTIME_MS)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { lock ->
            if (lock.isHeld) lock.release()
        }
        wakeLock = null
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.run_keepalive_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.run_keepalive_channel_description)
            setShowBadge(false)
            enableVibration(false)
            setSound(null, null)
        }
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle(getString(R.string.run_keepalive_title))
            .setContentText(getString(R.string.run_keepalive_text))
            .setContentIntent(openApp)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setShowWhen(false)
            .setSilent(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }
}
