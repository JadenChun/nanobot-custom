/**
 * Turn-completion notifications shared by the browser, desktop shell, and
 * mobile client.
 *
 * The desktop shell handles WebKit's `show-notification` signal natively and
 * the mobile bootstrap installs a `nanobotMobileNotifications` bridge backed by
 * Android local notifications. Regular browsers fall back to the Notification
 * API, then to the service worker path required by mobile browsers.
 */
interface MobileTaskNotificationBridge {
  requestPermission?: () => Promise<boolean> | boolean;
  notify?: (options: {
    title: string;
    body: string;
    tag?: string;
  }) => Promise<void> | void;
}

declare global {
  interface Window {
    nanobotMobileNotifications?: MobileTaskNotificationBridge;
    __TAURI_INTERNALS__?: unknown;
  }
}

const DEFAULT_NOTIFICATION_TITLE = "nanobot";
const DEFAULT_NOTIFICATION_ICON = "/brand/nanobot_icon_192.png";

function mobileBridge(): MobileTaskNotificationBridge | undefined {
  if (typeof window === "undefined") return undefined;
  return window.nanobotMobileNotifications;
}

function isDesktopShell(): boolean {
  return typeof window !== "undefined"
    && typeof window.__TAURI_INTERNALS__ !== "undefined";
}

export function taskNotificationsSupported(): boolean {
  if (isDesktopShell() || mobileBridge()?.notify) return true;
  return typeof Notification !== "undefined";
}

export async function requestTaskNotificationPermission(): Promise<boolean> {
  if (isDesktopShell()) return true;
  const bridge = mobileBridge();
  if (bridge?.requestPermission) {
    try {
      return Boolean(await bridge.requestPermission());
    } catch {
      return false;
    }
  }
  if (typeof Notification === "undefined") return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;
  try {
    return (await Notification.requestPermission()) === "granted";
  } catch {
    return false;
  }
}

function notificationOptions(body: string, tag?: string): NotificationOptions {
  return {
    body,
    tag,
    icon: DEFAULT_NOTIFICATION_ICON,
    badge: DEFAULT_NOTIFICATION_ICON,
    data: { url: "/" },
  };
}

function showViaServiceWorker(
  title: string,
  body: string,
  tag?: string,
): void {
  if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;
  void navigator.serviceWorker.getRegistration().then((registration) => {
    if (!registration) return undefined;
    return registration.showNotification(title, notificationOptions(body, tag));
  }).catch(() => {
    // Notifications are optional; a blocked registration must not affect chat.
  });
}

export function showTaskNotification({
  body,
  tag,
  title = DEFAULT_NOTIFICATION_TITLE,
}: {
  body: string;
  tag?: string;
  title?: string;
}): void {
  if (isDesktopShell()) {
    const query = new URLSearchParams({ title, body });
    if (tag) query.set("tag", tag);
    void fetch(`/api/desktop/notify?${query.toString()}`, {
      credentials: "same-origin",
    }).catch(() => {
      // Desktop notifications are best-effort and must not affect chat.
    });
    return;
  }

  const bridge = mobileBridge();
  if (bridge?.notify) {
    try {
      void bridge.notify({ title, body, tag });
    } catch {
      // The native bridge is best-effort, like the browser API below.
    }
    return;
  }

  if (typeof Notification === "undefined" || Notification.permission !== "granted") {
    return;
  }

  try {
    const notification = new Notification(title, notificationOptions(body, tag));
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
    return;
  } catch {
    // Android/iOS browsers do not expose the non-persistent constructor.
  }
  showViaServiceWorker(title, body, tag);
}
