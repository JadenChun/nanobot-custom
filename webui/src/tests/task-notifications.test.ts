import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { showTaskNotification } from "@/lib/task-notifications";

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
