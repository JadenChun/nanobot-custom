package ai.nanobot.mobile;

import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebView;

import ai.nanobot.mobile.network.NativeMediaInterceptor;
import com.getcapacitor.Bridge;
import com.getcapacitor.BridgeWebViewClient;

/** Keeps Capacitor's routing and streams signed media through the native transport. */
public class NanobotWebViewClient extends BridgeWebViewClient {
    public NanobotWebViewClient(Bridge bridge) {
        super(bridge);
    }

    @Override
    public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
        WebResourceResponse media = NativeMediaInterceptor.INSTANCE.intercept(request);
        if (media != null) {
            return media;
        }
        return super.shouldInterceptRequest(view, request);
    }
}
