package ai.nanobot.mobile;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import ai.nanobot.mobile.network.NativeNetworkPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    public static volatile String pendingPairingLink = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(NativeNetworkPlugin.class);
        super.onCreate(savedInstanceState);
        captureLink(getIntent());
        getBridge().getWebView().addJavascriptInterface(new NativeProbe(), "NanobotNativeProbe");
        getBridge().getWebView().setWebViewClient(new NanobotWebViewClient(getBridge()));
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
        if (getBridge() != null && getBridge().getWebView() != null) {
            getBridge().getWebView().evaluateJavascript(
                "window.dispatchEvent(new Event('nanobot-native-resume'))",
                null
            );
        }
    }

    private void captureLink(Intent intent) {
        Uri data = intent == null ? null : intent.getData();
        if (data != null && "nanobot".equals(data.getScheme())) {
            pendingPairingLink = data.toString();
        }
    }
}
