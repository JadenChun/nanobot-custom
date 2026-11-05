/**
 * Android keepalive bridge used while a phone-started run is active.
 *
 * The native shell runs a `dataSync` foreground service so Android does not
 * freeze the process and stop the WebSocket/WebView while the user is away.
 * This is intentionally best-effort: browsers, desktop shells, older native
 * builds, background-start restrictions, or an exhausted dataSync quota must
 * never interfere with chat.
 */
import { readLocalPreferences } from "@/lib/local-preferences";

interface MobileRunKeepAliveBridge {
  start?: (chatId: string) => Promise<void> | void;
  stop?: (chatId?: string) => Promise<void> | void;
}

declare global {
  interface Window {
    nanobotMobileRunKeepAlive?: MobileRunKeepAliveBridge;
  }
}

function mobileBridge(): MobileRunKeepAliveBridge | undefined {
  if (typeof window === "undefined") return undefined;
  return window.nanobotMobileRunKeepAlive;
}

export function startMobileRunKeepAlive(chatId: string): void {
  if (typeof window === "undefined") return;
  if (!readLocalPreferences().browserNotifications) return;
  const bridge = mobileBridge();
  if (!bridge?.start) return;
  try {
    void Promise.resolve(bridge.start(chatId)).catch(() => {
      // Best-effort: a rejected foreground-service start must not affect chat.
    });
  } catch {
    // Best-effort, same as the asynchronous rejection path.
  }
}

export function stopMobileRunKeepAlive(chatId?: string): void {
  if (typeof window === "undefined") return;
  const bridge = mobileBridge();
  if (!bridge?.stop) return;
  try {
    void Promise.resolve(bridge.stop(chatId)).catch(() => {
      // The service may already have been stopped by the system or teardown.
    });
  } catch {
    // Best-effort, same as the asynchronous rejection path.
  }
}
