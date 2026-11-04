package ai.nanobot.mobile;

import android.os.Build;
import android.util.Log;
import android.webkit.JavascriptInterface;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * M01 fixture-only native probe. It proves the packaged JNI/CMake path and
 * reports the executing ABI; it grants no transport or filesystem authority.
 */
public final class NativeProbe {
    private static final String TAG = "NanobotProbe";

    static {
        System.loadLibrary("nanobot_probe");
    }

    private native String nativeIdentity();

    @JavascriptInterface
    public void log(String message) {
        Log.i("NanobotWeb", String.valueOf(message));
    }

    @JavascriptInterface
    public String probe() {        String machine = "unknown";
        long pageSize = -1;
        try {
            String identity = nativeIdentity();
            int separator = identity.lastIndexOf(':');
            if (separator > 0) {
                machine = identity.substring(0, separator);
                pageSize = Long.parseLong(identity.substring(separator + 1));
            }
        } catch (Throwable throwable) {
            Log.e(TAG, "NANOBOT_MOBILE_PROBE_FAIL", throwable);
        }

        JSONObject result = new JSONObject();
        try {
            JSONArray abis = new JSONArray();
            for (String abi : Build.SUPPORTED_ABIS) {
                abis.put(abi);
            }
            result.put("abi", Build.SUPPORTED_ABIS.length > 0 ? Build.SUPPORTED_ABIS[0] : "unknown");
            result.put("architecture", Build.CPU_ABI);
            result.put("androidApi", Build.VERSION.SDK_INT);
            result.put("pageSize", pageSize);
            result.put("machine", machine);
            result.put("supportedAbis", abis);
        } catch (JSONException exception) {
            Log.e(TAG, "NANOBOT_MOBILE_PROBE_FAIL", exception);
            return "{\"error\":\"probe serialization failed\"}";
        }

        Log.i(TAG, "NANOBOT_MOBILE_PROBE_OK " + result);
        return result.toString();
    }
}
