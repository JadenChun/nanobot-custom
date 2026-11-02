import { render, screen, waitFor } from "@testing-library/react";
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

  it("resolves signed media through the mobile runtime into a blob URL", async () => {
    const instance = runtime();
    installMobileRuntime(instance);
    const createObjectURL = vi.fn(() => "blob:mobile-media");
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL: vi.fn() });
    render(<AttachmentTile attachment={{ kind: "image", url: "/api/media/sig/payload", name: "p.png" }} />);
    await waitFor(() => expect(screen.getByRole("img").getAttribute("src")).toBe("blob:mobile-media"));
    expect(instance.request).toHaveBeenCalledWith(expect.objectContaining({
      operation: "mobile_media",
      path: "/mobile/v1/media/sig/payload",
    }));
    expect(createObjectURL).toHaveBeenCalled();
  });
});
