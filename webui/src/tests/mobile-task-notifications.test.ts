import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const bootstrapSource = readFileSync(
  resolve(process.cwd(), "../mobile/mobile-bootstrap.js"),
  "utf8",
);
const androidRoot = resolve(process.cwd(), "../mobile/android/app/src/main");
const notificationVectorPath = resolve(
  androidRoot,
  "res/drawable/ic_stat_nanobot.xml",
);
const notificationBitmapPath = resolve(
  androidRoot,
  "res/drawable-nodpi/ic_nanobot_notification.png",
);
const canonicalBitmapPath = resolve(
  process.cwd(),
  "public/brand/nanobot_icon_192.png",
);
const keepAliveSource = readFileSync(
  resolve(
    androidRoot,
    "java/ai/nanobot/mobile/RunKeepAliveService.kt",
  ),
  "utf8",
);

interface LocalNotification {
  id: number;
  title: string;
  body: string;
  isExactNotification: boolean;
  smallIcon?: string;
  largeIcon?: string;
  schedule?: unknown;
}

interface LocalNotificationsPlugin {
  requestPermissions: () => Promise<{ display: string }>;
  schedule: (options: { notifications: LocalNotification[] }) => Promise<unknown>;
}

interface MobileNotificationBridge {
  requestPermission: () => Promise<boolean>;
  notify: (options: { title: string; body: string }) => Promise<void>;
}

interface CapacitorWindow extends Window {
  Capacitor?: {
    Plugins?: {
      NativeNetwork?: {
        target: () => Promise<null>;
        pairingIdentity: () => Promise<{ installationId: string }>;
        pendingPairingLink: () => Promise<null>;
      };
      LocalNotifications?: LocalNotificationsPlugin;
    };
    registerPlugin?: (name: string) => LocalNotificationsPlugin | undefined;
  };
  nanobotMobileNotifications?: MobileNotificationBridge;
}

function installBootstrap(localNotifications?: LocalNotificationsPlugin): void {
  const capacitorWindow = window as CapacitorWindow;
  capacitorWindow.Capacitor = {
    Plugins: {
      NativeNetwork: {
        target: async () => null,
        pairingIdentity: async () => ({ installationId: "phone-local-id" }),
        pendingPairingLink: async () => null,
      },
      ...(localNotifications ? { LocalNotifications: localNotifications } : {}),
    },
  };
  window.eval(bootstrapSource);
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  document.body.innerHTML = "";
  const capacitorWindow = window as CapacitorWindow;
  delete capacitorWindow.Capacitor;
  delete capacitorWindow.nanobotMobileNotifications;
  delete (window as Window & { nanobotMobileTransport?: unknown }).nanobotMobileTransport;
  delete (window as Window & { nanobotMobileReady?: unknown }).nanobotMobileReady;
  delete (window as Window & { nanobotMobileRunKeepAlive?: unknown }).nanobotMobileRunKeepAlive;
});

describe("mobile task notification bridge", () => {
  it("schedules the original completion text with branded Android resources", async () => {
    const schedule = vi.fn<LocalNotificationsPlugin["schedule"]>(async () => ({}));
    const requestPermissions = vi.fn(async () => ({ display: "granted" }));
    installBootstrap({ requestPermissions, schedule });
    const bridge = (window as CapacitorWindow).nanobotMobileNotifications;

    expect(bridge).toBeDefined();
    await expect(bridge?.requestPermission()).resolves.toBe(true);
    await bridge?.notify({ title: "Task finished", body: "Your task is complete." });

    expect(schedule).toHaveBeenCalledTimes(1);
    const [notification] = schedule.mock.calls[0][0].notifications;
    expect(notification).toMatchObject({
      title: "Task finished",
      body: "Your task is complete.",
      isExactNotification: false,
      smallIcon: "ic_stat_nanobot",
      largeIcon: "ic_nanobot_notification",
    });
    expect(notification).not.toHaveProperty("schedule");
    expect(requestPermissions).toHaveBeenCalledTimes(1);
  });

  it("reports denied permission and swallows local scheduling failures", async () => {
    const schedule = vi.fn<LocalNotificationsPlugin["schedule"]>(async () => {
      throw new Error("notifications denied");
    });
    const requestPermissions = vi.fn(async () => ({ display: "denied" }));
    installBootstrap({ requestPermissions, schedule });
    const bridge = (window as CapacitorWindow).nanobotMobileNotifications;

    expect(bridge).toBeDefined();
    await expect(bridge?.requestPermission()).resolves.toBe(false);
    await expect(
      bridge?.notify({ title: "Task failed", body: "The task did not finish." }),
    ).resolves.toBeUndefined();
    expect(schedule).toHaveBeenCalledTimes(1);
  });

  it("does not install a notification bridge without the Capacitor plugin", () => {
    installBootstrap();

    expect((window as CapacitorWindow).nanobotMobileNotifications).toBeUndefined();
  });

  it("contains valid icon resources and points keepalive at the same small drawable", () => {
    const hasVector = existsSync(notificationVectorPath);
    const hasBitmap = existsSync(notificationBitmapPath);
    expect(hasVector).toBe(true);
    expect(hasBitmap).toBe(true);
    expect(keepAliveSource).toContain("R.drawable.ic_stat_nanobot");
    expect(keepAliveSource).not.toContain("android.R.drawable.stat_notify_sync");
    if (!hasVector || !hasBitmap) return;

    const vectorSource = readFileSync(notificationVectorPath, "utf8");
    const parsed = new DOMParser().parseFromString(vectorSource, "application/xml");
    const vector = parsed.documentElement;
    expect(vector.localName).toBe("vector");
    expect(vector.getAttribute("android:width")).toBe("24dp");
    expect(vector.getAttribute("android:height")).toBe("24dp");
    expect(vector.getAttribute("android:viewportWidth")).toBe("24");
    expect(vector.getAttribute("android:viewportHeight")).toBe("24");
    expect(vector.querySelectorAll("path").length).toBeGreaterThan(0);
    for (const path of vector.querySelectorAll("path")) {
      expect(path.getAttribute("android:fillColor")).toBe("#FFFFFFFF");
      expect(path.getAttribute("android:pathData")).toBeTruthy();
    }
    expect(vector.querySelector("path")?.getAttribute("android:fillType")).toBe("evenOdd");

    const bitmap = readFileSync(notificationBitmapPath);
    const canonical = readFileSync(canonicalBitmapPath);
    expect(bitmap).toEqual(canonical);
    expect(bitmap.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
    expect(bitmap.readUInt32BE(16)).toBe(192);
    expect(bitmap.readUInt32BE(20)).toBe(192);
  });
});
