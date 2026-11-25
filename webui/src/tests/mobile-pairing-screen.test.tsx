import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import MobileAuthGate from "@/components/mobile/MobileAuthGate";
import { MobilePairingScreen } from "@/components/mobile/MobilePairingScreen";
import type {
  MobilePairingBridge, MobilePairingTarget, PairingError, PairingRemediation, PairingSnapshot, PairingStage,
} from "@/lib/mobile-pairing-bridge";
import type { MobileEvent, MobileTransport } from "@/lib/mobile-runtime";

const origin = "https://desktop.example";
const target = { origin, installationId: "phone-local-id", hasAccess: true };
const defaultError: PairingError = {
  kind: "server_http", retryable: true, remediation: "check_desktop_serve",
  title: "Desktop unavailable", message: "Check Tailscale Serve on your desktop.",
  technicalDetail: "HTTP 503 from the desktop",
};

function snapshot(stage: PairingStage, fields: Partial<PairingSnapshot> = {}): PairingSnapshot {
  return {
    stage, attemptId: null, method: null, origin, installationId: null, pairingId: null,
    tailscaleState: "idle", tailscaleHasLoginUrl: false, comparisonCode: null, expiresAt: null,
    canCancel: false, canRetry: false, error: null, ...fields,
  };
}

function pairingFixture(initial = snapshot("idle"), saved: MobilePairingTarget | null = null) {
  let current = initial;
  let listener: ((state: PairingSnapshot) => void) | undefined;
  const unsubscribe = vi.fn(() => { listener = undefined; });
  const bridge = {
    target: vi.fn<MobilePairingBridge["target"]>(async () => saved),
    pairingState: vi.fn<MobilePairingBridge["pairingState"]>(async () => current),
    onPairingStateChange: vi.fn<MobilePairingBridge["onPairingStateChange"]>(async (callback) => {
      listener = callback;
      return unsubscribe;
    }),
    pairWithLink: vi.fn<MobilePairingBridge["pairWithLink"]>(async () => snapshot("starting_transport")),
    pairWithCode: vi.fn<MobilePairingBridge["pairWithCode"]>(async () => snapshot("starting_transport")),
    pairRetry: vi.fn<MobilePairingBridge["pairRetry"]>(async () => snapshot("starting_transport")),
    pairCancel: vi.fn<MobilePairingBridge["pairCancel"]>(async () => snapshot("cancelling")),
    pairResetToIdle: vi.fn<MobilePairingBridge["pairResetToIdle"]>(async () => snapshot("idle", { origin: null })),
    pairCheckResume: vi.fn<MobilePairingBridge["pairCheckResume"]>(async () => current),
    scanPairingQr: vi.fn<MobilePairingBridge["scanPairingQr"]>(async () => ({ scanned: true })),
    openLogin: vi.fn<MobilePairingBridge["openLogin"]>(async () => {}),
    status: vi.fn<MobilePairingBridge["status"]>(async () => ({ state: "ready" })),
    refresh: vi.fn<MobilePairingBridge["refresh"]>(async () => {}),
    logout: vi.fn<MobilePairingBridge["logout"]>(async () => {}),
  };
  return { bridge, unsubscribe, emit(next: PairingSnapshot) { current = next; listener?.(next); } };
}

function transportFixture(...statuses: number[]) {
  let nextId = 0;
  const listeners = new Set<(event: MobileEvent) => void>();
  const emit = (event: MobileEvent) => { for (const listener of listeners) listener(event); };
  const transport = {
    start: vi.fn<MobileTransport["start"]>(async () => {}),
    status: vi.fn<MobileTransport["status"]>(async () => ({ state: "ready" })),
    http: vi.fn<MobileTransport["http"]>(async () => {
      const id = ++nextId;
      emit({ type: "http_response", request_id: id, status: statuses.shift() ?? 200 });
      return id;
    }),
    cancel: vi.fn<MobileTransport["cancel"]>(async () => {}),
    onEvent: vi.fn<MobileTransport["onEvent"]>(async (listener) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    }),
    openLogin: vi.fn<MobileTransport["openLogin"]>(async () => {}),
    refresh: vi.fn<MobileTransport["refresh"]>(async () => true),
    wsOpen: vi.fn<MobileTransport["wsOpen"]>(async () => 1),
    wsSend: vi.fn<MobileTransport["wsSend"]>(async () => {}),
    wsClose: vi.fn<MobileTransport["wsClose"]>(async () => {}),
    stop: vi.fn<MobileTransport["stop"]>(async () => {}),
  };
  return { transport, emit };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("MobilePairingScreen", () => {
  it.each([
    ["idle", "Scan QR code", {}],
    ["starting_transport", "Connecting to Tailscale…", { tailscaleState: "starting" }],
    ["tailscale_login_required", "Sign in to Tailscale", { tailscaleState: "needs_login", tailscaleHasLoginUrl: true }],
    ["collecting", "Contacting your desktop…", {}],
    ["waiting_approval", "Approve owner access", { comparisonCode: "832156" }],
    ["expired_renewable", "Renew setup", {}],
    ["proving", "Verifying this phone…", {}],
    ["cancelling", "Cancelling pairing…", {}],
    ["paired", "Connected — loading Kohaku…", {}],
    ["failed", defaultError.title, { error: defaultError }],
  ] satisfies Array<[PairingStage, string, Partial<PairingSnapshot>]>)(
    "shows the %s stage",
    async (stage, text, fields) => {
      const { bridge } = pairingFixture(snapshot(stage, fields));
      render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);

      expect(await screen.findByText(text)).toBeVisible();
      if (stage === "tailscale_login_required") expect(screen.getByRole("button", { name: "Back" })).toBeVisible();
      if (stage === "waiting_approval") {
        expect(screen.getByText("832156")).toBeVisible();
        expect(screen.getByText("Approve owner access").parentElement).toHaveTextContent(
          "Tap Approve owner access on your desktop.",
        );
      }
      if (stage === "expired_renewable") expect(screen.getByRole("button", { name: "Scan fresh QR" })).toBeVisible();
      if (stage === "failed") expect(screen.getByText(defaultError.message)).toBeVisible();
    },
  );

  it("shows a live expiry for the comparison code", async () => {
    const now = Date.UTC(2026, 10, 25);
    vi.spyOn(Date, "now").mockReturnValue(now);
    const { bridge } = pairingFixture(snapshot("waiting_approval", { comparisonCode: "832156", expiresAt: now + 65_000 }));
    render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
    expect(await screen.findByText("Expires in 1:05")).toBeVisible();
    vi.mocked(Date.now).mockReturnValue(now + 2_000);
    expect(await screen.findByText("Expires in 1:03", {}, { timeout: 2_000 })).toBeVisible();
  });

  it("prefills the desktop address, keeps edits, and formats the setup code", async () => {
    const { bridge, emit } = pairingFixture();
    render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "8-digit code" }));
    const address = screen.getByLabelText("Desktop address");
    const code = screen.getByLabelText("Setup code");
    expect(address).toHaveValue(origin);
    fireEvent.change(address, { target: { value: "https://new.example" } });
    fireEvent.change(code, { target: { value: "12345678" } });
    act(() => emit(snapshot("idle", { origin: "https://updated.example" })));
    expect(address).toHaveValue("https://new.example");
    expect(code).toHaveValue("1234 5678");
    fireEvent.click(screen.getByRole("button", { name: "Pair with code" }));
    await waitFor(() => expect(bridge.pairWithCode).toHaveBeenCalledWith("https://new.example", "12345678"));
  });

  it("pastes and forwards the opaque setup link", async () => {
    const { bridge } = pairingFixture();
    const link = "kohaku://pair?example=%2B";
    vi.spyOn(navigator.clipboard, "readText").mockResolvedValue(link);
    render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Paste link" }));
    fireEvent.click(screen.getByRole("button", { name: "Paste" }));
    await waitFor(() => expect(screen.getByLabelText("Pairing link")).toHaveValue(link));
    fireEvent.click(screen.getByRole("button", { name: "Pair with link" }));
    await waitFor(() => expect(bridge.pairWithLink).toHaveBeenCalledWith(link));
  });

  it.each([
    ["rescan_qr", "Scan fresh QR", "scanPairingQr"],
    ["edit_code", "Edit 8-digit code", "pairResetToIdle"],
    ["check_desktop_serve", "Retry", "pairRetry"],
    ["open_tailscale_login", "Sign in to Tailscale", "openLogin"],
    ["switch_desktop", "Use a different desktop", "pairResetToIdle"],
    ["switch_desktop", "Start over", "pairResetToIdle"],
  ] satisfies Array<[PairingRemediation, string, keyof MobilePairingBridge]>)(
    "applies %s remediation via %s",
    async (remediation, label, method) => {
      const { bridge } = pairingFixture(snapshot("failed", { error: { ...defaultError, remediation } }));
      render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
      fireEvent.click(await screen.findByRole("button", { name: label }));

      await waitFor(() => expect(bridge[method]).toHaveBeenCalledOnce());
      if (remediation === "rescan_qr") {
        expect(bridge.pairResetToIdle.mock.invocationCallOrder[0])
          .toBeLessThan(bridge.scanPairingQr.mock.invocationCallOrder[0]);
        expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
      }
      if (remediation === "edit_code") {
        await waitFor(() => expect(screen.getByLabelText("Desktop address")).toBeVisible());
        expect(screen.getByLabelText("Desktop address")).toHaveValue(origin);
      }
      if (remediation === "switch_desktop") {
        expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
      }
    },
  );

  it("expands and copies the technical detail", async () => {
    const { bridge } = pairingFixture(snapshot("failed", { error: defaultError }));
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
    render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
    const details = await screen.findByRole("button", { name: "Technical details" });
    expect(details).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText(defaultError.technicalDetail)).not.toBeVisible();

    fireEvent.click(details);
    expect(details).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(defaultError.technicalDetail)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(defaultError.technicalDetail));
  });

  it("receives native snapshots, checks pairing on resume and online, and unsubscribes", async () => {
    const { bridge, emit, unsubscribe } = pairingFixture();
    const { unmount } = render(<MobilePairingScreen pairing={bridge} onPaired={vi.fn()} />);
    await screen.findByRole("button", { name: "Scan QR code" });
    act(() => emit(snapshot("tailscale_login_required", { tailscaleHasLoginUrl: true })));
    expect(screen.getByRole("button", { name: "Sign in to Tailscale" })).toBeVisible();

    bridge.pairCheckResume.mockResolvedValueOnce(snapshot("collecting"));
    fireEvent(window, new Event("kohaku-native-resume"));
    expect(await screen.findByText("Contacting your desktop…")).toBeVisible();
    bridge.pairCheckResume.mockResolvedValueOnce(snapshot("proving"));
    fireEvent(window, new Event("online"));
    expect(await screen.findByText("Verifying this phone…")).toBeVisible();
    expect(bridge.pairCheckResume).toHaveBeenCalledTimes(2);
    unmount();
    expect(unsubscribe).toHaveBeenCalledOnce();
  });
});

describe("MobileAuthGate", () => {
  it.each([null, { ...target, hasAccess: false }])("opens pairing without usable saved access", async (saved) => {
    const { bridge } = pairingFixture(snapshot("idle"), saved);
    const { transport } = transportFixture();
    const onReady = vi.fn();
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);

    expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
    expect(onReady).not.toHaveBeenCalled();
    expect(transport.start).not.toHaveBeenCalled();
  });

  it("shows the paired confirmation before opening the app once", async () => {
    const { bridge, emit } = pairingFixture();
    const onReady = vi.fn();
    render(<MobileAuthGate pairing={bridge} onReady={onReady} />);
    await screen.findByRole("button", { name: "Scan QR code" });
    act(() => emit(snapshot("paired")));
    expect(screen.getByText("Connected — loading Kohaku…")).toBeVisible();
    expect(onReady).not.toHaveBeenCalled();
    act(() => emit(snapshot("paired")));
    await waitFor(() => expect(onReady).toHaveBeenCalledOnce(), { timeout: 1_500 });
  });

  it.each(["ready", "starting"])("waits for Tailscale when initial status is %s", async (state) => {
    const { bridge } = pairingFixture(snapshot("idle"), target);
    const { transport, emit } = transportFixture(200);
    bridge.status.mockResolvedValue({ state });
    const onReady = vi.fn();
    const reload = vi.spyOn(window.location, "reload").mockImplementation(() => {});
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);
    await waitFor(() => expect(bridge.status).toHaveBeenCalled());
    expect(transport.start).toHaveBeenCalledWith(origin, target.installationId, expect.any(String));
    if (state === "starting") {
      expect(transport.http).not.toHaveBeenCalled();
      act(() => emit({ type: "node_state", state: "ready" }));
    }
    await waitFor(() => expect(onReady).toHaveBeenCalledOnce());
    expect(transport.http).toHaveBeenCalledWith({
      operation: "mobile_handshake", method: "GET", path: "/mobile/v1/handshake",
    });
    expect(reload).not.toHaveBeenCalled();
  });

  it("uses only the first terminal event for the handshake's native request ID", async () => {
    const { bridge } = pairingFixture(snapshot("idle"), target);
    const { transport, emit } = transportFixture(200);
    let admit!: (id: number) => void;
    transport.http.mockImplementationOnce(() => new Promise((resolve) => { admit = resolve; }));
    const onReady = vi.fn();
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);
    await waitFor(() => expect(transport.http).toHaveBeenCalledTimes(1));
    act(() => {
      emit({ type: "http_response", request_id: 7, status: 500 });
      emit({ type: "http_response", request_id: 41, status: 401 });
      emit({ type: "http_response", request_id: 41, status: 500 });
      admit(41);
    });
    await waitFor(() => expect(onReady).toHaveBeenCalledOnce());
    expect(bridge.refresh).toHaveBeenCalledOnce();
    expect(transport.http).toHaveBeenCalledTimes(2);
  });

  it.each(["HTTP 400", "HTTP 401", "HTTP 403", "HTTP 410"])(
    "accepts a successful second handshake after %s from refresh",
    async (message) => {
      const { bridge } = pairingFixture(snapshot("idle"), target);
      const { transport } = transportFixture(401, 200);
      bridge.refresh.mockRejectedValueOnce(new Error(message));
      const onReady = vi.fn();
      render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);

      await waitFor(() => expect(onReady).toHaveBeenCalledOnce());
      expect(transport.http).toHaveBeenCalledTimes(2);
      expect(bridge.logout).not.toHaveBeenCalled();
    },
  );

  it("clears a pairing that needs repair after the second handshake fails", async () => {
    const { bridge } = pairingFixture(snapshot("idle"), target);
    const { transport } = transportFixture(401, 401);
    bridge.refresh.mockRejectedValueOnce(new Error("HTTP 403: proof rejected"));
    const onReady = vi.fn();
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);

    expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
    expect(bridge.logout).toHaveBeenCalledWith(true);
    expect(onReady).not.toHaveBeenCalled();
  });

  it.each(["pairing was revoked", "pairing upgrade required"])(
    "opens pairing when native refresh reports '%s'",
    async (message) => {
      const { bridge } = pairingFixture(snapshot("idle"), target);
      const { transport } = transportFixture(401);
      bridge.refresh.mockRejectedValueOnce(new Error(message));
      render(<MobileAuthGate pairing={bridge} transport={transport} onReady={vi.fn()} />);

      expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
      expect(transport.http).toHaveBeenCalledOnce();
      expect(bridge.logout).not.toHaveBeenCalled();
    },
  );

  it.each(["refresh_error", "transport_error", "needs_login", "server_error"])(
    "preserves the saved pairing on %s",
    async (failure) => {
      const { bridge } = pairingFixture(snapshot("idle"), target);
      const { transport } = transportFixture(failure === "refresh_error" ? 401 : 500);
      if (failure === "refresh_error") bridge.refresh.mockRejectedValueOnce(new Error("offline"));
      if (failure === "transport_error") transport.start.mockRejectedValueOnce(new Error("start failed"));
      if (failure === "needs_login") bridge.status.mockResolvedValueOnce({ state: "needs_login" });
      const onReady = vi.fn();
      render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);

      expect(await screen.findByRole("heading", { name: "Reconnect this phone" })).toBeVisible();
      expect(screen.getByText("Your pairing with desktop.example is saved.")).toBeVisible();
      expect(bridge.logout).not.toHaveBeenCalled();
      expect(onReady).not.toHaveBeenCalled();
    },
  );

  it.each(["Retry", "kohaku-native-resume", "online"])("reconnects on %s", async (trigger) => {
    const { bridge } = pairingFixture(snapshot("idle"), target);
    const { transport } = transportFixture(500, 200);
    const onReady = vi.fn();
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={onReady} />);
    await screen.findByRole("heading", { name: "Reconnect this phone" });

    if (trigger === "Retry") fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    else fireEvent(window, new Event(trigger));
    await waitFor(() => expect(onReady).toHaveBeenCalledOnce());
    expect(bridge.logout).not.toHaveBeenCalled();
  });

  it("opens Tailscale sign-in and can unpair from the reconnect screen", async () => {
    const { bridge } = pairingFixture(snapshot("idle"), target);
    const { transport } = transportFixture();
    bridge.status.mockResolvedValueOnce({ state: "needs_login" });
    render(<MobileAuthGate pairing={bridge} transport={transport} onReady={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Sign in to Tailscale" }));
    await waitFor(() => expect(bridge.openLogin).toHaveBeenCalledOnce());

    const unpair = screen.getByRole("button", { name: "Unpair / Switch desktop" });
    await waitFor(() => expect(unpair).toBeEnabled());
    fireEvent.click(unpair);
    expect(await screen.findByRole("button", { name: "Scan QR code" })).toBeVisible();
    expect(bridge.logout).toHaveBeenCalledWith(false);
  });
});
