import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PairingSnapshot } from "@/lib/mobile-pairing-bridge";
import type { MobileTransport } from "@/lib/mobile-runtime";

const bootstrap = readFileSync(resolve(process.cwd(), "../mobile/mobile-bootstrap.js"), "utf8");
const origin = "https://desktop.example";
const installationId = "phone-local-id";
const snapshot: PairingSnapshot = {
  stage: "idle", attemptId: null, method: null, origin, installationId: null, pairingId: null,
  tailscaleState: "idle", tailscaleHasLoginUrl: false, comparisonCode: null, expiresAt: null,
  canCancel: false, canRetry: false, error: null,
};
const nativeSnapshot = {
  stage: "idle", origin, tailscaleState: "idle", tailscaleHasLoginUrl: false,
  canCancel: false, canRetry: false,
};
const target = { origin, installationId, hasAccess: true };
const nativeStatus = { state: "ready" };

function install() {
  const listeners = new Map<string, (value: object) => void>();
  const plugin = {
    target: vi.fn(async (): Promise<object | null> => target),
    pairingState: vi.fn(async () => nativeSnapshot),
    pairWithLink: vi.fn(async () => nativeSnapshot),
    pairWithCode: vi.fn(async () => nativeSnapshot),
    pairRetry: vi.fn(async () => nativeSnapshot),
    pairCancel: vi.fn(async () => nativeSnapshot),
    pairResetToIdle: vi.fn(async () => nativeSnapshot),
    pairCheckResume: vi.fn(async () => nativeSnapshot),
    scanPairingQr: vi.fn(async () => ({ scanned: true })),
    status: vi.fn(async () => nativeStatus),
    openLogin: vi.fn(async () => undefined),
    refresh: vi.fn(async () => undefined),
    logout: vi.fn(async () => undefined),
    start: vi.fn(async () => undefined),
    http: vi.fn(async () => ({ requestId: 41 })),
    addListener: vi.fn(async (name: string, listener: (value: object) => void) => {
      listeners.set(name, listener);
      return { remove: async () => { listeners.delete(name); } };
    }),
  };
  vi.stubGlobal("Capacitor", { Plugins: { NativeNetwork: plugin } });
  window.eval(bootstrap);
  const transport = (window as Window & { nanobotMobileTransport?: MobileTransport }).nanobotMobileTransport!;
  return { plugin, listeners, pairing: window.kohakuMobilePairing!, transport };
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.kohakuMobilePairing;
  delete (window as Window & { nanobotMobileTransport?: unknown }).nanobotMobileTransport;
  delete (window as Window & { nanobotMobileRunKeepAlive?: unknown }).nanobotMobileRunKeepAlive;
  document.body.replaceChildren();
});

describe("native mobile bootstrap", () => {
  it("exposes native bridges without starting pairing or replacing the splash", () => {
    document.body.innerHTML = '<div id="root"><div class="boot-splash">Loading Kohaku…</div></div>';
    const before = document.body.innerHTML;
    const { plugin, pairing, transport } = install();

    expect(pairing).toBeDefined();
    expect(transport).toBeDefined();
    expect(document.body.innerHTML).toBe(before);
    for (const nativeMethod of Object.values(plugin)) expect(nativeMethod).not.toHaveBeenCalled();
    expect(window).not.toHaveProperty("nanobotMobileReady");
  });

  it("forwards pairing commands and returns public snapshots", async () => {
    const { plugin, pairing } = install();
    const rawLink = "kohaku://pair?example=%2B";
    const code = "12345678";

    expect(await pairing.pairingState()).toEqual(snapshot);
    expect(plugin.pairingState).toHaveBeenCalledWith();
    expect(await pairing.pairWithLink(rawLink)).toEqual(snapshot);
    expect(plugin.pairWithLink).toHaveBeenCalledWith({ rawLink });
    expect(await pairing.pairWithCode(origin, code)).toEqual(snapshot);
    expect(plugin.pairWithCode).toHaveBeenCalledWith({ origin, code });
    for (const method of ["pairRetry", "pairCancel", "pairResetToIdle", "pairCheckResume"] as const) {
      expect(await pairing[method]()).toEqual(snapshot);
      expect(plugin[method]).toHaveBeenCalledWith();
    }
    expect(await pairing.scanPairingQr()).toEqual({ scanned: true });
    expect(plugin.scanPairingQr).toHaveBeenCalledWith();
    await pairing.openLogin();
    expect(plugin.openLogin).toHaveBeenCalledWith();
    expect(await pairing.status()).toEqual(nativeStatus);
    expect(plugin.status).toHaveBeenCalledWith();
    await pairing.refresh();
    expect(plugin.refresh).toHaveBeenCalledWith();
    await pairing.logout(true);
    await pairing.logout(false);
    expect(plugin.logout.mock.calls).toEqual([[{ serverReachable: true }], [{ serverReachable: false }]]);
  });

  it("returns public target fields and distinguishes a missing target from a custody error", async () => {
    const { plugin, pairing } = install();
    plugin.target.mockResolvedValueOnce({ ...target, hasRefresh: false, internal: "native-only" });
    expect(await pairing.target()).toEqual({ ...target, hasRefresh: false });
    expect(await pairing.target()).toEqual(target);
    plugin.target.mockRejectedValueOnce(new Error("no paired target"));
    expect(await pairing.target()).toBeNull();
    plugin.target.mockRejectedValueOnce(new Error("custody is unavailable"));
    await expect(pairing.target()).rejects.toThrow("custody is unavailable");
  });

  it("subscribes and unsubscribes pairing and transport events and preserves transport return values", async () => {
    const { plugin, listeners, pairing, transport } = install();
    const onPairing = vi.fn();
    const stopPairing = await pairing.onPairingStateChange(onPairing);
    listeners.get("pairingStateChanged")?.(nativeSnapshot);
    expect(onPairing).toHaveBeenCalledWith(snapshot);
    stopPairing();
    expect(listeners.has("pairingStateChanged")).toBe(false);

    await transport.start(origin, installationId, "kohaku-phone");
    expect(plugin.start).toHaveBeenCalledWith({ origin, installationId, hostname: "kohaku-phone" });
    const onNetwork = vi.fn();
    const stopNetwork = await transport.onEvent(onNetwork);
    const event = { type: "http_response", request_id: 41, status: 200 };
    listeners.get("networkEvent")?.(event);
    expect(onNetwork).toHaveBeenCalledWith(event);
    stopNetwork();
    expect(listeners.has("networkEvent")).toBe(false);
    const request = { operation: "mobile_handshake", method: "GET" as const, path: "/mobile/v1/handshake" };
    expect(await transport.http(request)).toBe(41);
    expect(plugin.http).toHaveBeenCalledWith(request);
    expect(await transport.refresh()).toBe(true);
    plugin.refresh.mockRejectedValueOnce(new Error("offline"));
    expect(await transport.refresh()).toBe(false);
  });
});
