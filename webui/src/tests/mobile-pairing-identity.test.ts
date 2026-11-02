import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

const bootstrapSource = readFileSync(
  resolve(process.cwd(), "../mobile/mobile-bootstrap.js"),
  "utf8",
);

interface PairingArgs {
  origin: string;
  installationId: string;
  challenge: string;
  label: string;
}

function installPlugin() {
  let releaseIdentity: (identity: { installationId: string }) => void = () => {};
  const identity = new Promise<{ installationId: string }>((resolve) => {
    releaseIdentity = resolve;
  });
  const pairStart = vi.fn(async (args: PairingArgs) => ({
    pairing_id: "pairing-1",
    comparison_code: "123456",
    installationId: args.installationId,
  }));
  const pairPoll = vi.fn(async () => ({ status: "approved" }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(() => identity),
    pairStart,
    pairPoll,
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { plugin, pairStart, releaseIdentity };
}

afterEach(() => {
  document.body.innerHTML = "";
  delete (window as unknown as { Capacitor?: unknown }).Capacitor;
  delete (window as unknown as { nanobotMobileReady?: unknown }).nanobotMobileReady;
  delete (window as unknown as { nanobotMobileTransport?: unknown }).nanobotMobileTransport;
});

function installPairedPlugin() {
  let emit: ((event: Record<string, unknown>) => void) | null = null;
  const refresh = vi.fn(async () => {
    throw new Error("pairing was revoked");
  });
  const logout = vi.fn(async () => ({ serverRevocationPending: false }));
  const plugin = {
    target: vi.fn(async () => ({
      origin: "https://desktop.example",
      installationId: "phone-local-id",
      hasAccess: true,
    })),
    start: vi.fn(async () => ({})),
    status: vi.fn(async () => ({ state: "ready" })),
    addListener: vi.fn(async (_name: string, listener: (event: Record<string, unknown>) => void) => {
      emit = listener;
      return { remove: vi.fn(async () => {}) };
    }),
    http: vi.fn(async () => ({ requestId: 1 })),
    refresh,
    logout,
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  setTimeout(() => {
    emit?.({ type: "http_response", request_id: 1, status: 401, body_base64: "" });
  }, 0);
  return { plugin, refresh, logout };
}

describe("mobile pairing identity", () => {
  it("waits for the local identity when a link is submitted early", async () => {
    const { pairStart, releaseIdentity } = installPlugin();
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    const input = document.querySelector<HTMLInputElement>(".mobile-pairing-input");
    expect(input).not.toBeNull();
    input!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");

    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    expect(pairStart).not.toHaveBeenCalled();

    releaseIdentity({ installationId: "phone-local-id" });
    await vi.waitFor(() => expect(pairStart).toHaveBeenCalledTimes(1));
    expect(pairStart.mock.calls[0][0]).toMatchObject({
      installationId: "phone-local-id",
      challenge: "pairing-1.nonce.secret",
    });
  });

  it("clears a revoked pairing before the app shell opens", async () => {
    const reload = vi.spyOn(window.location, "reload").mockImplementation(() => {});
    const { refresh, logout } = installPairedPlugin();
    window.eval(bootstrapSource);

    await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(logout).toHaveBeenCalledWith({ serverReachable: true }));
    expect(reload).toHaveBeenCalledTimes(1);
    reload.mockRestore();
  });
});
