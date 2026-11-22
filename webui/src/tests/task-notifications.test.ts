import type { TFunction } from "i18next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildTaskNotificationBody, showTaskNotification } from "@/lib/task-notifications";

const t = ((key: string, options?: { defaultValue?: string }) => {
  if (key === "errors.modelRequestFailed.body") return "model failed";
  return options?.defaultValue ?? key;
}) as unknown as TFunction;

const completed = { kind: "turn_end", chatId: "chat", outcome: "completed" } as const;

describe("desktop task notifications", () => {
  beforeEach(() => {
    (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = {};
    window.dispatchEvent(new Event("nanobot:desktop-show"));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
    window.dispatchEvent(new Event("nanobot:desktop-show"));
  });

  it("does not notify while the desktop shell is hidden", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({} as Response);
    const mutate = vi.fn().mockResolvedValue({ ok: true });

    window.dispatchEvent(new Event("nanobot:desktop-hide"));
    showTaskNotification({ event: completed, t, requestMutation: mutate });
    expect(mutate).not.toHaveBeenCalled();

    window.dispatchEvent(new Event("nanobot:desktop-show"));
    showTaskNotification({ event: completed, t, requestMutation: mutate });
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate).toHaveBeenCalledWith("desktop.notify", {
      title: "Kohaku",
      body: "Task completed",
      tag: "nanobot-turn-chat",
      sound: true,
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("delivers titled alerts while minimized and honors mute", async () => {
    vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    document.dispatchEvent(new Event("visibilitychange"));
    const mutate = vi.fn().mockResolvedValue({ ok: true });
    await showTaskNotification({ event: completed, t, threadTitle: "Release review", sound: false, requestMutation: mutate });
    expect(mutate).toHaveBeenCalledWith("desktop.notify", {
      title: "Kohaku · Release review", body: "Task completed", tag: "nanobot-turn-chat", sound: false,
    });
  });

  it("sends no unauthenticated request without the mutation control", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({} as Response);

    window.dispatchEvent(new Event("nanobot:desktop-show"));
    showTaskNotification({ event: completed, t });

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("task notification copy", () => {
  it("uses completion copy for a successful turn", () => {
    expect(buildTaskNotificationBody({
      kind: "turn_end",
      chatId: "chat",
      outcome: "completed",
    }, t)).toBe("Task completed");
  });

  it("uses the failure message for a failed turn", () => {
    expect(buildTaskNotificationBody({
      kind: "turn_end",
      chatId: "chat",
      outcome: "failed",
      failureMessage: "boom",
    }, t)).toBe("boom");
  });

  it("uses model failure copy for a model failure", () => {
    expect(buildTaskNotificationBody({
      kind: "turn_end",
      chatId: "chat",
      outcome: "failed",
      failureKind: "model",
      failureErrorKind: "timeout",
    }, t)).toBe("model failed");
  });

  it("uses recovery copy for interrupted and failed recovery", () => {
    expect(buildTaskNotificationBody({
      kind: "recovery",
      chatId: "chat",
      status: "awaiting_user",
    }, t)).toBe("Task interrupted");
    expect(buildTaskNotificationBody({
      kind: "recovery",
      chatId: "chat",
      status: "failed",
    }, t)).toBe("Task recovery failed");
  });
});
