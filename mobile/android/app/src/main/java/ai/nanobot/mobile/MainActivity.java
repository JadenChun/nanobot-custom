package ai.nanobot.mobile;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.webkit.WebView;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;

import ai.nanobot.mobile.network.NativeNetworkPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    public static volatile String pendingPairingLink = null;

    private static final long EXIT_CONFIRM_WINDOW_MS = 2000;
    private OnBackPressedCallback backCallback;
    private long lastBackPressAt = 0;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(NativeNetworkPlugin.class);
        super.onCreate(savedInstanceState);
        captureLink(getIntent());
        getBridge().getWebView().addJavascriptInterface(new NativeProbe(), "NanobotNativeProbe");
        getBridge().getWebView().setWebViewClient(new NanobotWebViewClient(getBridge()));

        backCallback = new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                dispatchBackToWebApp();
            }
        };
        getOnBackPressedDispatcher().addCallback(this, backCallback);
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        captureLink(intent);
    }

    @Override
    public void onResume() {
        super.onResume();
        lastBackPressAt = 0;
        if (getBridge() != null && getBridge().getWebView() != null) {
            getBridge().getWebView().evaluateJavascript(
                "window.dispatchEvent(new Event('nanobot-native-resume'))",
                null
            );
        }
    }

    /**
     * Give the WebUI the first chance to consume the back press (close the
     * lightbox, leave a folder preview, dismiss the sidebar, ...). Only when
     * no in-app surface handled it do we treat back as an exit request, and
     * even then a second press within two seconds is required.
     */
    private void dispatchBackToWebApp() {
        WebView webView = getBridge() == null ? null : getBridge().getWebView();
        if (webView == null) {
            requestExit();
            return;
        }
        webView.evaluateJavascript(
            "(function(){try{return window.__nanobotHandleBack?window.__nanobotHandleBack():false}catch(e){return false}})()",
            handled -> {
                if ("true".equals(handled)) {
                    lastBackPressAt = 0;
                    return;
                }
                long now = SystemClock.elapsedRealtime();
                if (now - lastBackPressAt <= EXIT_CONFIRM_WINDOW_MS) {
                    requestExit();
                    return;
                }
                lastBackPressAt = now;
                Toast.makeText(this, R.string.back_press_again, Toast.LENGTH_SHORT).show();
            }
        );
    }

    private void requestExit() {
        if (backCallback != null) {
            backCallback.setEnabled(false);
        }
        getOnBackPressedDispatcher().onBackPressed();
    }

    private void captureLink(Intent intent) {
        Uri data = intent == null ? null : intent.getData();
        if (data != null && "nanobot".equals(data.getScheme())) {
            pendingPairingLink = data.toString();
        }
    }
}
