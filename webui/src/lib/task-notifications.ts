/**
 * Turn-completion notifications shared by the browser, desktop shell, and
 * mobile client.
 *
 * The desktop shell routes completion alerts through the local gateway notifier
 * and the mobile bootstrap installs a `nanobotMobileNotifications` bridge backed
 * by Android local notifications. Regular browsers fall back to the Notification
 * API, then to the service worker path required by mobile browsers.
 */
import type { TFunction } from "i18next";

import type { TaskNotificationEvent } from "@/lib/kohaku-client";
import { resolveModelRequestFailureCopy } from "@/lib/model-request-failure";

interface MobileTaskNotificationBridge {
  requestPermission?: () => Promise<boolean> | boolean;
  notify?: (options: {
    title: string;
    body: string;
    tag?: string;
    chatId?: string;
    sound?: boolean;
    temporary?: boolean;
  }) => Promise<boolean | void> | void;
}

declare global {
  interface Window {
    nanobotMobileNotifications?: MobileTaskNotificationBridge;
    __TAURI_INTERNALS__?: unknown;
  }
}

const DEFAULT_NOTIFICATION_TITLE = "Kohaku";
const DEFAULT_NOTIFICATION_ICON = "/brand/kohaku_icon_192.png";
const DESKTOP_HIDE_EVENT = "nanobot:desktop-hide";
const DESKTOP_SHOW_EVENT = "nanobot:desktop-show";

let desktopWindowOpen = true;

function mobileBridge(): MobileTaskNotificationBridge | undefined {
  if (typeof window === "undefined") return undefined;
  return window.nanobotMobileNotifications;
}

function isDesktopShell(): boolean {
  return typeof window !== "undefined"
    && typeof window.__TAURI_INTERNALS__ !== "undefined";
}

/** Whether this client currently has a window the user can actually see. */
export function clientWindowIsOpen(): boolean {
  if (!isDesktopShell()) return true;
  return desktopWindowOpen;
}

if (typeof window !== "undefined") {
  window.addEventListener(DESKTOP_HIDE_EVENT, () => {
    desktopWindowOpen = false;
  });
  window.addEventListener(DESKTOP_SHOW_EVENT, () => {
    desktopWindowOpen = true;
  });

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

/** Completion/failure/interruption text shared by every client surface. */
export function buildTaskNotificationBody(
  event: TaskNotificationEvent,
  t: TFunction,
): string {
  if (event.kind === "recovery") {
    return event.status === "failed"
      ? t("recovery.failed", { defaultValue: "Task recovery failed" })
      : t("recovery.interrupted", { defaultValue: "Task interrupted" });
  }
  if (event.outcome === "failed" && event.failureKind === "model") {
    return resolveModelRequestFailureCopy(
      {
        ...(event.failureErrorKind ? { errorKind: event.failureErrorKind } : {}),
        ...(event.failureAttempts !== undefined
          ? { attempts: event.failureAttempts }
          : {}),
      },
      t,
    ).body;
  }
  if (event.outcome === "failed") {
    return event.failureMessage || "This turn failed and has ended.";
  }
  return t("recovery.completed", { defaultValue: "Task completed" });
}

async function chime(): Promise<void> {
  let audio: AudioContext | undefined;
  let resumeTimeout: ReturnType<typeof setTimeout> | undefined;
  try {
    audio = new AudioContext();
    if (audio.state === "suspended") {
      // Autoplay policies can leave resume pending until the next user gesture.
      await Promise.race([
        audio.resume(),
        new Promise<void>((resolve) => { resumeTimeout = setTimeout(resolve, 1000); }),
      ]);
    }
    if (audio.state !== "running") return;
    const tone = audio.createOscillator();
    const gain = audio.createGain();
    const now = audio.currentTime;
    tone.frequency.setValueAtTime(660, now);
    tone.frequency.setValueAtTime(880, now + 0.12);
    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.09, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    tone.connect(gain).connect(audio.destination);
    const ended = new Promise<void>((resolve) => { tone.onended = () => resolve(); });
    tone.start(now);
    tone.stop(now + 0.36);
    await ended;
  } catch {
    // Notification delivery can succeed when the browser blocks audio.
  } finally {
    clearTimeout(resumeTimeout);
    if (audio) await audio.close().catch(() => {});
  }
}

export async function showTaskNotification({
  event, t, requestMutation, threadTitle, temporary, sound = true,
}: {
  event?: TaskNotificationEvent;
  t: TFunction;
  threadTitle?: string;
  temporary?: boolean;
  sound?: boolean;
  requestMutation?: (action: string, payload: Record<string, unknown>) => Promise<unknown>;
}): Promise<boolean> {
  if (!clientWindowIsOpen()) return false;
  const chatId = event?.chatId;
  const title = threadTitle?.trim() ? `Kohaku · ${threadTitle.trim()}` : DEFAULT_NOTIFICATION_TITLE;
  const body = event ? buildTaskNotificationBody(event, t) : t("settings.rows.testNotificationBody");
  const tag = chatId ? `nanobot-turn-${chatId}` : "kohaku-test";
  try {
    if (isDesktopShell()) {
      if (!requestMutation) return false;
      if (event?.kind === "recovery" && event.status !== "failed") return false;
      if (event?.kind === "turn_end" && event.outcome && !["completed", "failed"].includes(event.outcome)) return false;
      const result = await requestMutation("desktop.notify", { title, body, tag, sound });
      if (sound && !(result && typeof result === "object" && "nativeSound" in result && result.nativeSound)) void chime();
      return true;
    }
    const bridge = mobileBridge();
    if (bridge?.notify) return await bridge.notify({ title, body, tag, chatId, sound, temporary }) !== false;
    if (typeof Notification === "undefined" || Notification.permission !== "granted") return false;
    const hash = !chatId ? "" : temporary ? "#/temporary/" + encodeURIComponent(chatId)
      : "#/chat/" + encodeURIComponent("websocket:" + chatId);
    const options = { body, tag, silent: true, icon: DEFAULT_NOTIFICATION_ICON,
      badge: "/brand/kohaku_notification_badge.png", data: { chatId, temporary } };
    try {
      const notification = new Notification(title, options);
      notification.onclick = () => {
        window.focus();
        if (hash) window.location.hash = hash;
        notification.close();
      };
    } catch {
      const registration = await navigator.serviceWorker?.getRegistration();
      if (!registration) return false;
      await registration.showNotification(title, options);
    }
    if (sound) void chime();
    return true;
  } catch { return false; }
}
