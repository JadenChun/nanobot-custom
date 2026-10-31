package ai.nanobot.mobile;

import android.os.Bundle;
import ai.nanobot.mobile.network.NativeNetworkPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(NativeNetworkPlugin.class);
        super.onCreate(savedInstanceState);
        getBridge().getWebView().addJavascriptInterface(new NativeProbe(), "NanobotNativeProbe");
    }
}
