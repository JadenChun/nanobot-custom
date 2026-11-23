package ai.nanobot.mobile.media

import android.content.Intent
import android.view.ViewGroup
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.nanobot.mobile.MainActivity
import ai.nanobot.mobile.network.mediaBodyForWebView
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WebViewVideoInstrumentedTest {
    @Test
    fun previewsPlaysAndSeeksVideo() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val clip = instrumentation.context.assets.open("red-blue.mp4").use { it.readBytes() }
        val rangeLimit = 8 * 1024 * 1024
        val padding = 2 * rangeLimit
        val media = ByteBuffer.wrap(clip)
        var metadataOffset = 0
        while (media.getInt(metadataOffset + 4) != 0x6d6f6f76) { // moov
            metadataOffset += media.getInt(metadataOffset)
        }
        // Keep the sample offsets and move the metadata past the first two ranges.
        val video = ByteBuffer.allocate(clip.size + padding).apply {
            put(clip, 0, metadataOffset)
            putInt(padding)
            put("free".toByteArray(Charsets.US_ASCII))
            position(metadataOffset + padding)
            put(clip, metadataOffset, clip.size - metadataOffset)
        }.array()
        val requestedOffsets = ConcurrentLinkedQueue<Int>()
        val origin = "https://localhost"
        val path = "/api/media/fixture/${System.nanoTime()}"
        val activity = instrumentation.startActivitySync(
            Intent(instrumentation.targetContext, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        lateinit var webView: WebView
        instrumentation.runOnMainSync {
            webView = WebView(activity)
            activity.addContentView(webView, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT,
            ))
            webView.settings.javaScriptEnabled = true
            webView.settings.mediaPlaybackRequiresUserGesture = false
            webView.webViewClient = object : WebViewClient() {
                override fun shouldInterceptRequest(
                    view: WebView, request: WebResourceRequest,
                ): WebResourceResponse? {
                    if (request.url.toString() != origin + path) return null
                    val range = request.requestHeaders.entries
                        .firstOrNull { it.key.equals("Range", ignoreCase = true) }?.value
                    val offsets = range?.let { Regex("""bytes=(\d+)-(\d*)""").matchEntire(it) }
                    val start = offsets?.groupValues?.get(1)?.toInt() ?: 0
                    val requestedEnd = offsets?.groupValues?.get(2)?.toIntOrNull() ?: video.lastIndex
                    // Match the response that the broker exposes to the WebView.
                    val partial = offsets != null
                    val last = minOf(requestedEnd, video.lastIndex)
                    val length = last - start + 1
                    requestedOffsets.add(start)
                    val headers = mutableMapOf(
                        "Accept-Ranges" to "bytes",
                        "Content-Type" to "video/mp4",
                        "Cache-Control" to "no-store",
                        "Content-Length" to length.toString(),
                    )
                    if (partial) headers["Content-Range"] = "bytes $start-$last/${video.size}"
                    return WebResourceResponse(
                        "video/mp4", null, if (partial) 206 else 200,
                        if (partial) "Partial Content" else "OK", headers,
                        mediaBodyForWebView(ByteArrayInputStream(video, start, length), headers["Content-Range"]),
                    )
                }
            }
            webView.loadDataWithBaseURL(origin, """
                <!doctype html>
                <video id="clip" controls playsinline preload="metadata"></video>
                <script>
                  window.result = "waiting for preview";
                  const video = document.querySelector("video");
                  video.addEventListener("loadedmetadata", () => {
                    window.result = "waiting for decoded data " + video.duration;
                  });
                  const canvas = document.createElement("canvas");
                  canvas.width = canvas.height = 1;
                  const ctx = canvas.getContext("2d");
                  function color() {
                    ctx.drawImage(video, 0, 0, 1, 1);
                    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
                    return r > 180 && b < 60 ? "red" : b > 180 && r < 60 ? "blue" : [r,g,b].join(",");
                  }
                  const next = (name) => new Promise((resolve) =>
                    video.addEventListener(name, resolve, {once: true}));
                  const frameAt = (time) => new Promise((resolve) => {
                    const check = (_, frame) => {
                      if (frame.mediaTime >= time) resolve();
                      else video.requestVideoFrameCallback(check);
                    };
                    video.requestVideoFrameCallback(check);
                  });
                  video.addEventListener("error", () => window.result = "media error " + video.error.code);
                  (async () => {
                    const loaded = next("loadeddata");
                    const firstFrame = frameAt(0);
                    video.src = "$path";
                    await loaded;
                    window.result = "waiting for first frame";
                    await firstFrame;
                    if (!video.videoWidth || color() !== "red") throw Error("initial frame " + video.videoWidth + " " + color());
                    window.result = "waiting for playback";
                    await video.play();
                    while (video.currentTime < 0.5) await next("timeupdate");
                    video.pause();
                    window.result = "waiting for seek";
                    const seeked = next("seeked");
                    const seekFrame = frameAt(2.8);
                    video.currentTime = video.duration - 1;
                    await Promise.all([seeked, seekFrame]);
                    if (Math.abs(video.currentTime - 3) > 0.15 || color() !== "blue") throw Error("seek frame " + video.currentTime + " " + color());
                    window.result = "waiting for end";
                    const ended = next("ended");
                    await video.play();
                    await ended;
                    window.result = "pass";
                  })().catch((error) => window.result = error.message);
                </script>
            """.trimIndent(), "text/html", "UTF-8", null)
        }
        try {
            var result = ""
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(25)
            while (System.nanoTime() < deadline) {
                val value = CompletableFuture<String>()
                instrumentation.runOnMainSync {
                    webView.evaluateJavascript("window.result", value::complete)
                }
                result = value.get(5, TimeUnit.SECONDS)
                if (result != "null" && !result.startsWith("\"waiting for ")) break
                Thread.sleep(100)
            }
            assertEquals("Requested media offsets: $requestedOffsets", "\"pass\"", result)
            assertTrue("WebView did not request a later byte range", requestedOffsets.any { it >= rangeLimit })
        } finally {
            instrumentation.runOnMainSync {
                (webView.parent as ViewGroup).removeView(webView)
                webView.destroy()
                activity.finish()
            }
        }
    }
}
