package ai.kohaku.mobile;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.webkit.WebView;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import ai.kohaku.mobile.network.NativeNetworkPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    public static volatile String pendingPairingLink = null;

    private static final long EXIT_CONFIRM_WINDOW_MS = 2000;
    private static final String PERMISSION_PREFERENCES = "kohaku_native_permissions";
    private static final String INITIAL_MICROPHONE_REQUESTED = "initial_microphone_request_requested";
    private static final String INITIAL_NOTIFICATION_REQUESTED = "initial_notification_request_requested";
    private static final String[] MICROPHONE_PERMISSIONS = {
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.MODIFY_AUDIO_SETTINGS
    };

    private OnBackPressedCallback backCallback;
    private long lastBackPressAt = 0;
    private ActivityResultLauncher<String[]> microphonePermissionLauncher;
    private boolean microphonePermissionRequestInFlight;
    private final List<Consumer<Boolean>> pendingMicrophonePermissionCallbacks = new ArrayList<>();

    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(NativeNetworkPlugin.class);
        super.onCreate(savedInstanceState);
        microphonePermissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(),
            result -> completeMicrophonePermissionRequest()
        );
        captureLink(getIntent());
        getBridge().getWebView().addJavascriptInterface(new NativeProbe(), "KohakuNativeProbe");
        getBridge().getWebView().setWebViewClient(new KohakuWebViewClient(getBridge()));
        getBridge().getWebView().setWebChromeClient(
            new KohakuWebChromeClient(getBridge(), this::requestMicrophonePermission)
        );

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
                "window.dispatchEvent(new Event('kohaku-native-resume'))",
                null
            );
        }
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
        requestInitialMicrophonePermission();
    }

    private void requestInitialMicrophonePermission() {
        SharedPreferences preferences = getSharedPreferences(PERMISSION_PREFERENCES, MODE_PRIVATE);
        boolean micRequested = preferences.getBoolean(INITIAL_MICROPHONE_REQUESTED, false);
        boolean notificationRequested =
            Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU
                || preferences.getBoolean(INITIAL_NOTIFICATION_REQUESTED, false)
                || ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    == PackageManager.PERMISSION_GRANTED;
        if (micRequested && notificationRequested) {
            return;
        }
        List<String> toRequest = new ArrayList<>();
        SharedPreferences.Editor editor = preferences.edit();
        if (!micRequested) {
            editor.putBoolean(INITIAL_MICROPHONE_REQUESTED, true);
            for (String permission : MICROPHONE_PERMISSIONS) {
                if (ContextCompat.checkSelfPermission(this, permission)
                        != PackageManager.PERMISSION_GRANTED) {
                    toRequest.add(permission);
                }
            }
        }
        if (!notificationRequested && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            editor.putBoolean(INITIAL_NOTIFICATION_REQUESTED, true);
            toRequest.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!editor.commit() || toRequest.isEmpty() || microphonePermissionRequestInFlight) {
            return;
        }
        microphonePermissionRequestInFlight = true;
        microphonePermissionLauncher.launch(toRequest.toArray(new String[0]));
    }

    private void requestMicrophonePermission(Consumer<Boolean> callback) {
        if (callback != null) {
            pendingMicrophonePermissionCallbacks.add(callback);
        }
        if (hasMicrophonePermission()) {
            completeMicrophonePermissionRequest();
            return;
        }
        if (microphonePermissionRequestInFlight) {
            return;
        }
        microphonePermissionRequestInFlight = true;
        microphonePermissionLauncher.launch(MICROPHONE_PERMISSIONS);
    }

    private boolean hasMicrophonePermission() {
        for (String permission : MICROPHONE_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void completeMicrophonePermissionRequest() {
        microphonePermissionRequestInFlight = false;
        boolean granted = hasMicrophonePermission();
        List<Consumer<Boolean>> callbacks = new ArrayList<>(pendingMicrophonePermissionCallbacks);
        pendingMicrophonePermissionCallbacks.clear();
        for (Consumer<Boolean> callback : callbacks) {
            callback.accept(granted);
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
        if (data != null && "kohaku".equals(data.getScheme())) {
            pendingPairingLink = data.toString();
        }
    }
}
