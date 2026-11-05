import type { TFunction } from "i18next";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildTaskNotificationBody, showTaskNotification } from "@/lib/task-notifications";

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

    window.dispatchEvent(new Event("nanobot:desktop-hide"));
    showTaskNotification({ body: "hidden window" });
    expect(fetchSpy).not.toHaveBeenCalled();

    window.dispatchEvent(new Event("nanobot:desktop-show"));
    showTaskNotification({ body: "visible window" });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0]?.[0]).toContain("/api/desktop/notify");
  });
});

describe("task notification copy", () => {
  const t = ((key: string, options?: { defaultValue?: string }) => {
    if (key === "errors.modelRequestFailed.body") return "model failed";
    return options?.defaultValue ?? key;
  }) as unknown as TFunction;

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
