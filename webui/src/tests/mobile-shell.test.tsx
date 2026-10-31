import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { MobileConnectionNotice } from "@/components/MobileConnectionNotice";
import {
  installMobileRuntime,
  resetMobileRuntime,
  type MobileRuntime,
} from "@/lib/mobile-runtime";

function runtime(): MobileRuntime {
  return {
    transport: {} as MobileRuntime["transport"],
    request: async () => ({ requestId: 1, status: 200, headers: {}, bodyBase64: "" }),
    handshake: async () => ({}),
    read: async () => ({}),
    onEvent: async () => () => undefined,
    createWebSocket: () => ({}) as WebSocket,
  };
}

afterEach(() => resetMobileRuntime());

describe("mobile shell", () => {
  it("hides connection notices in the browser runtime", () => {
    render(<MobileConnectionNotice status="error" />);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows connecting, reconnecting and error states on mobile", () => {
    installMobileRuntime(runtime());
    const { rerender } = render(<MobileConnectionNotice status="connecting" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    rerender(<MobileConnectionNotice status="reconnecting" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    rerender(<MobileConnectionNotice status="error" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("hides the notice when the socket is open", () => {
    installMobileRuntime(runtime());
    render(<MobileConnectionNotice status="open" />);
    expect(screen.queryByRole("status")).toBeNull();
  });
});
