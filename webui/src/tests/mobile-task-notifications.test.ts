import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const bootstrapSource = readFileSync(
  resolve(process.cwd(), "../mobile/mobile-bootstrap.js"),
  "utf8",
);
const androidRoot = resolve(process.cwd(), "../mobile/android/app/src/main");
const notificationSmallDensities = [
  ["res/drawable-mdpi/ic_stat_nanobot.png", 24],
  ["res/drawable-hdpi/ic_stat_nanobot.png", 36],
  ["res/drawable-xhdpi/ic_stat_nanobot.png", 48],
  ["res/drawable-xxhdpi/ic_stat_nanobot.png", 72],
  ["res/drawable-xxxhdpi/ic_stat_nanobot.png", 96],
] as const;
const canonicalBadgePath = resolve(
  process.cwd(),
  "public/brand/kohaku_notification_badge.png",
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
  extra?: { chatId: string };
  channelId?: string;
}

interface LocalNotificationsPlugin {
  createChannel?: (options: unknown) => Promise<void>;
  addListener?: (name: string, listener: (action: { notification: LocalNotification }) => void) => Promise<void>;
  requestPermissions: () => Promise<{ display: string }>;
  schedule: (options: { notifications: LocalNotification[] }) => Promise<unknown>;
}

interface MobileNotificationBridge {
  requestPermission: () => Promise<boolean>;
  notify: (options: { title: string; body: string; chatId?: string; sound?: boolean }) => Promise<boolean | void>;
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
      ...(localNotifications ? { LocalNotifications: { createChannel: async () => {}, addListener: async () => {}, ...localNotifications } } : {}),
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
    ).resolves.toBe(false);
    expect(schedule).toHaveBeenCalledTimes(1);
  });

  it("posts alerts per thread, uses the sound setting, and opens the tapped chat", async () => {
    const createChannel = vi.fn().mockResolvedValue(undefined);
    const addListener = vi.fn().mockResolvedValue(undefined);
    const schedule = vi.fn<LocalNotificationsPlugin["schedule"]>().mockResolvedValue({});
    installBootstrap({ createChannel, addListener, schedule, requestPermissions: async () => ({ display: "granted" }) });
    const bridge = window.nanobotMobileNotifications!;
    for (const [chatId, sound] of [["chat-17", true], ["chat-17", false], ["chat-18", true]] as const) {
      await bridge.notify!({ title: "Kohaku", body: "Task completed", chatId, sound });
    }
    expect(createChannel).toHaveBeenCalledWith(expect.objectContaining({ id: "kohaku_task_alerts", importance: 4, vibration: true }));
    const [first, second, third] = schedule.mock.calls.map(([payload]) => payload.notifications[0]);
    expect(first.id).toBeGreaterThan(0);
    expect(first.id).toBeLessThan(2147483648);
    expect(second.id).toBe(first.id);
    expect(third.id).not.toBe(first.id);
    expect(first.channelId).toBe("kohaku_task_alerts");
    expect(second.channelId).toBe("kohaku_task_alerts_silent");
    expect(first.extra).toEqual({ chatId: "chat-17" });
    expect(addListener.mock.calls[0][0]).toBe("localNotificationActionPerformed");
    addListener.mock.calls[0][1]({ notification: first });
    expect(window.location.hash).toBe("#/chat/websocket%3Achat-17");
    window.history.replaceState(null, "", "/");
  });

  it("does not install a notification bridge without the Capacitor plugin", () => {
    installBootstrap();

    expect((window as CapacitorWindow).nanobotMobileNotifications).toBeUndefined();
  });

  it("contains valid icon resources and points keepalive at the same small drawable", () => {
    const hasBitmap = existsSync(notificationBitmapPath);
    expect(hasBitmap).toBe(true);
    expect(keepAliveSource).toContain("R.drawable.ic_stat_nanobot");
    expect(keepAliveSource).not.toContain("android.R.drawable.stat_notify_sync");

    const pngHeader = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    for (const [relativePath, expectedSize] of notificationSmallDensities) {
      const iconPath = resolve(androidRoot, relativePath);
      expect(existsSync(iconPath)).toBe(true);
      const bytes = readFileSync(iconPath);
      expect(bytes.subarray(0, 8)).toEqual(pngHeader);
      expect(bytes.readUInt32BE(16)).toBe(expectedSize);
      expect(bytes.readUInt32BE(20)).toBe(expectedSize);
    }
    expect(
      readFileSync(resolve(androidRoot, "res/drawable-xxxhdpi/ic_stat_nanobot.png")),
    ).toEqual(readFileSync(canonicalBadgePath));

    if (!hasBitmap) return;
    const bitmap = readFileSync(notificationBitmapPath);
    const canonical = readFileSync(canonicalBitmapPath);
    expect(bitmap).toEqual(canonical);
    expect(bitmap.subarray(0, 8)).toEqual(pngHeader);
    expect(bitmap.readUInt32BE(16)).toBe(192);
    expect(bitmap.readUInt32BE(20)).toBe(192);
  });
});
