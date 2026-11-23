package ai.kohaku.mobile;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.webkit.PermissionRequest;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import com.getcapacitor.Bridge;
import com.getcapacitor.BridgeWebChromeClient;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.function.Consumer;

/** Adds a first-party origin check before Capacitor grants WebView audio capture. */
public class KohakuWebChromeClient extends BridgeWebChromeClient {
    private static final String AUDIO_CAPTURE = PermissionRequest.RESOURCE_AUDIO_CAPTURE;

    interface MicrophonePermissionRequester {
        void requestMicrophonePermission(Consumer<Boolean> callback);
    }

    private final Bridge bridge;
    private final MicrophonePermissionRequester microphonePermissionRequester;
    private final Set<PermissionRequest> pendingAudioRequests = Collections.newSetFromMap(
        new IdentityHashMap<>()
    );

    public KohakuWebChromeClient(
        Bridge bridge,
        MicrophonePermissionRequester microphonePermissionRequester
    ) {
        super(bridge);
        this.bridge = bridge;
        this.microphonePermissionRequester = microphonePermissionRequester;
    }

    @Override
    public void onPermissionRequest(@NonNull PermissionRequest request) {
        if (!isBundledOrigin(request.getOrigin()) || !isAudioOnly(request.getResources())) {
            request.deny();
            return;
        }
        if (!pendingAudioRequests.add(request)) {
            return;
        }

        microphonePermissionRequester.requestMicrophonePermission(granted -> {
            if (!pendingAudioRequests.remove(request)) {
                return;
            }
            if (!granted || !isBundledOrigin(request.getOrigin()) || !hasOsMicrophonePermission()) {
                request.deny();
                return;
            }
            request.grant(new String[] { AUDIO_CAPTURE });
        });
    }

    @Override
    public void onPermissionRequestCanceled(@NonNull PermissionRequest request) {
        pendingAudioRequests.remove(request);
        super.onPermissionRequestCanceled(request);
    }

    private boolean isBundledOrigin(Uri origin) {
        if (
            origin == null || origin.getUserInfo() != null || origin.getHost() == null ||
            origin.getQuery() != null || origin.getFragment() != null
        ) {
            return false;
        }
        String path = origin.getPath();
        if (path != null && !path.isEmpty() && !"/".equals(path)) {
            return false;
        }

        String scheme = bridge.getScheme();
        String host = bridge.getHost();
        String bundledUrl = bridge.getAppUrl();
        String topLevelUrl = bridge.getWebView().getUrl();
        if (scheme == null || host == null || bundledUrl == null || topLevelUrl == null) {
            return false;
        }

        Uri capacitorOrigin = Uri.parse(scheme + "://" + host);
        return sameOrigin(origin, capacitorOrigin) &&
            sameOrigin(origin, Uri.parse(bundledUrl)) &&
            sameOrigin(origin, Uri.parse(topLevelUrl));
    }

    private static boolean sameOrigin(Uri left, Uri right) {
        if (
            left == null || right == null || left.getUserInfo() != null || right.getUserInfo() != null ||
            left.getHost() == null || right.getHost() == null ||
            left.getScheme() == null || right.getScheme() == null ||
            !left.getScheme().equalsIgnoreCase(right.getScheme()) ||
            !left.getHost().equalsIgnoreCase(right.getHost())
        ) {
            return false;
        }
        int leftPort = left.getPort() == -1 ? defaultPort(left.getScheme()) : left.getPort();
        int rightPort = right.getPort() == -1 ? defaultPort(right.getScheme()) : right.getPort();
        return leftPort == rightPort;
    }

    private static int defaultPort(String scheme) {
        if ("https".equalsIgnoreCase(scheme)) {
            return 443;
        }
        if ("http".equalsIgnoreCase(scheme)) {
            return 80;
        }
        return -1;
    }

    private static boolean isAudioOnly(String[] resources) {
        return resources != null && resources.length == 1 && AUDIO_CAPTURE.equals(resources[0]);
    }

    private boolean hasOsMicrophonePermission() {
        return ContextCompat.checkSelfPermission(bridge.getContext(), Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                bridge.getContext(),
                Manifest.permission.MODIFY_AUDIO_SETTINGS
            ) == PackageManager.PERMISSION_GRANTED;
    }
}
