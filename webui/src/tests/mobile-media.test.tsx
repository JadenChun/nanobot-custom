import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AttachmentTile } from "@/components/AttachmentTile";
import {
  installMobileRuntime,
  resetMobileRuntime,
  type MobileRuntime,
} from "@/lib/mobile-runtime";

const PNG_BASE64 = "iVBORw0KGgo=";

afterEach(() => {
  resetMobileRuntime();
  vi.unstubAllGlobals();
});

function runtime(): MobileRuntime {
  return {
    transport: {} as MobileRuntime["transport"],
    request: vi.fn(async () => ({
      requestId: 1,
      status: 200,
      headers: { "content-type": "image/png" },
      bodyBase64: PNG_BASE64,
    })),
    handshake: vi.fn(async () => ({})),
    read: vi.fn(async () => ({})),
    onEvent: vi.fn(async () => () => undefined),
    createWebSocket: vi.fn(() => ({}) as WebSocket),
  };
}

describe("mobile media binding", () => {
  it("passes browser media URLs through unchanged", () => {
    render(<AttachmentTile attachment={{ kind: "image", url: "/api/media/sig/payload", name: "p.png" }} />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("/api/media/sig/payload");
  });

  it("keeps the signed URL on the same origin for the native interceptor", () => {
    const instance = runtime();
    installMobileRuntime(instance);
    render(<AttachmentTile attachment={{ kind: "image", url: "/api/media/sig/payload", name: "p.png" }} />);
    // The Android WebView intercepts /api/media/... and streams it natively, so
    // the client never fetches media through the JS bridge.
    expect(screen.getByRole("img").getAttribute("src")).toBe("/api/media/sig/payload");
    expect(instance.request).not.toHaveBeenCalled();
  });
});
