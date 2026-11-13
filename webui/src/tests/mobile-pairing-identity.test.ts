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

interface PairingCodeArgs {
  origin: string;
  installationId: string;
  code: string;
  label: string;
}

function installPlugin(
  pairStartCodeImpl?: (args: PairingCodeArgs) => Promise<Record<string, unknown>>,
) {
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
  const pairStartCode = vi.fn(
    pairStartCodeImpl ??
      (async (args: PairingCodeArgs) => ({
        pairing_id: "pairing-code",
        comparison_code: "87654321",
        nonce: "nonce-value-000000000000",
        collector_secret: "secret-value-000000000000",
        installationId: args.installationId,
      })),
  );
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(() => identity),
    pairStart,
    pairStartCode,
    pairPoll,
    pairCancel: vi.fn(async () => ({ status: "cancelled" })),
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { plugin, pairStart, pairStartCode, releaseIdentity };
}

function installPreflightPlugin() {
  const connectTailscale = vi.fn(async () => ({ state: "needs_login" }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({
      installationId: "phone-local-id",
      origin: "https://desktop.example",
      hasAccess: false,
    })),
    connectTailscale,
    status: vi.fn(async () => ({ state: "needs_login" })),
    pairStart: vi.fn(),
    pairStartCode: vi.fn(),
    pairPoll: vi.fn(),
    pairCancel: vi.fn(async () => ({ status: "cancelled" })),
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { plugin, connectTailscale };
}

afterEach(() => {
  document.body.innerHTML = "";
  delete (window as unknown as { Capacitor?: unknown }).Capacitor;
  delete (window as unknown as { nanobotMobileReady?: unknown }).nanobotMobileReady;
  delete (window as unknown as { nanobotMobileTransport?: unknown }).nanobotMobileTransport;
});

function installPairedPlugin(
  mode:
    | "revoked"
    | "unreachable"
    | "server_error"
    | "transport_error"
    | "start_error"
    | "needs_login" = "revoked",
) {
  let emit: ((event: Record<string, unknown>) => void) | null = null;
  let paired = true;
  const refresh = vi.fn(async () => {
    if (mode === "revoked") {
      paired = false;
      window.location.reload();
      throw new Error("pairing was revoked");
    }
    throw new Error("network");
  });
  const logout = vi.fn(async () => ({ serverRevocationPending: false }));
  const plugin = {
    target: vi.fn(async () => paired ? ({
      origin: "https://desktop.example",
      installationId: "phone-local-id",
      hasAccess: true,
    }) : null),
    start: vi.fn(async () => {
      if (mode === "start_error") throw new Error("native start failed");
      return {};
    }),
    status: vi.fn(async () => ({
      state: mode === "transport_error"
        ? "error"
        : mode === "needs_login"
          ? "needs_login"
          : "ready",
    })),
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
  if (mode === "revoked" || mode === "unreachable" || mode === "server_error") {
    setTimeout(() => {
      emit?.(
        mode === "unreachable"
          ? { type: "http_error", request_id: 1, code: "network" }
          : {
            type: "http_response",
            request_id: 1,
            status: mode === "server_error" ? 500 : 401,
            body_base64: "",
          },
      );
    }, 0);
  }
  return { plugin, refresh, logout };
}

function installCancelPlugin() {
  let resolvePairStart: (value: Record<string, unknown>) => void = () => {};
  const pairStart = vi.fn(() => new Promise<Record<string, unknown>>((resolve) => {
    resolvePairStart = resolve;
  }));
  const pairPoll = vi.fn(async () => ({ status: "waiting" }));
  const pairCancel = vi.fn(async () => ({ status: "cancelled" }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({ installationId: "phone-local-id" })),
    pairStart,
    pairPoll,
    pairCancel,
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return {
    pairStart,
    pairPoll,
    pairCancel,
    resolvePairStart: (value: Record<string, unknown>) => resolvePairStart(value),
  };
}

function installHeldPairingPlugin(hold: "collect" | "poll") {
  let releaseCollect: (value: Record<string, unknown>) => void = () => {};
  let releasePoll: (value: Record<string, unknown>) => void = () => {};
  let resolveCancel: (value: Record<string, unknown>) => void = () => {};
  const collected = { pairing_id: "pairing-1", comparison_code: "123456" };
  const pairStart = vi.fn(() => hold === "collect"
    ? new Promise<Record<string, unknown>>((resolve) => { releaseCollect = resolve; })
    : Promise.resolve(collected));
  const pairStartCode = vi.fn(() => hold === "collect"
    ? new Promise<Record<string, unknown>>((resolve) => { releaseCollect = resolve; })
    : Promise.resolve(collected));
  const pairPoll = vi.fn(() => hold === "poll"
    ? new Promise<Record<string, unknown>>((resolve) => { releasePoll = resolve; })
    : Promise.resolve({ status: "waiting" }));
  const pairCancel = vi.fn(() => new Promise<Record<string, unknown>>((resolve) => {
    resolveCancel = resolve;
  }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({ installationId: "phone-local-id" })),
    pairStart,
    pairStartCode,
    pairPoll,
    pairCancel,
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return {
    pairStart,
    pairStartCode,
    pairPoll,
    pairCancel,
    resolveCancel: () => resolveCancel({ status: "cancelled" }),
    releaseHeld: () => {
      if (hold === "collect") releaseCollect({
        pairing_id: "late-pairing",
        comparison_code: "999999",
        installationId: "phone-local-id",
      });
      else releasePoll({ status: "approved", installationId: "phone-local-id" });
    },
  };
}

function installHandshakeRacePlugin() {
  let listener: ((event: Record<string, unknown>) => void) | null = null;
  let releaseHandshake: (value: { requestId: number }) => void = () => {};
  const refresh = vi.fn(async () => true);
  const plugin = {
    target: vi.fn(async () => ({
      origin: "https://desktop.example",
      installationId: "phone-local-id",
      hasAccess: true,
    })),
    start: vi.fn(async () => undefined),
    status: vi.fn(async () => ({ state: "ready" })),
    addListener: vi.fn(async (
      _name: string,
      callback: (event: Record<string, unknown>) => void,
    ) => {
      listener = callback;
      return { remove: vi.fn(async () => { if (listener === callback) listener = null; }) };
    }),
    http: vi.fn(() => new Promise<{ requestId: number }>((resolve) => { releaseHandshake = resolve; })),
    cancel: vi.fn(async () => undefined),
    refresh,
    logout: vi.fn(async () => ({ serverRevocationPending: false })),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return {
    plugin,
    refresh,
    emit: (event: Record<string, unknown>) => listener?.(event),
    releaseHandshake: (requestId: number) => releaseHandshake({ requestId }),
  };
}

function installStartupRejectPlugin() {
  let rejectIdentity: (error: Error) => void = () => {};
  const pairingIdentity = vi.fn(() => new Promise<{ installationId: string }>((_resolve, reject) => {
    rejectIdentity = reject;
  }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity,
    pairStart: vi.fn(async () => ({ pairing_id: "pairing-1", comparison_code: "123456" })),
    pairPoll: vi.fn(async () => ({ status: "waiting" })),
    pairCancel: vi.fn(async () => ({ status: "cancelled" })),
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { pairingIdentity, rejectIdentity: (error: Error) => rejectIdentity(error) };
}

function installRejectedPairStartPlugin() {
  let rejectPairStart: (error: Error) => void = () => {};
  const pairStart = vi.fn(() => new Promise<Record<string, unknown>>((_resolve, reject) => {
    rejectPairStart = reject;
  }));
  const pairPoll = vi.fn(async () => ({ status: "waiting" }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({ installationId: "phone-local-id" })),
    pairStart,
    pairPoll,
    pairCancel: vi.fn(async () => ({ status: "cancelled" })),
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { pairStart, pairPoll, rejectPairStart: (error: Error) => rejectPairStart(error) };
}

function installRejectedPollPlugin() {
  let rejectPoll: (error: Error) => void = () => {};
  const pairStart = vi.fn(async () => ({
    pairing_id: "pairing-1",
    comparison_code: "123456",
    installationId: "phone-local-id",
  }));
  const pairPoll = vi.fn(() => new Promise<Record<string, unknown>>((_resolve, reject) => {
    rejectPoll = reject;
  }));
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({ installationId: "phone-local-id" })),
    pairStart,
    pairPoll,
    pairCancel: vi.fn(async () => ({ status: "cancelled" })),
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { pairPoll, rejectPoll: (error: Error) => rejectPoll(error) };
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

  it("starts manual pairing from an HTTPS origin and 8-digit code", async () => {
    const { pairStartCode, releaseIdentity } = installPlugin();
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-pairing-manual")).not.toBeNull();
    });
    const manual = document.querySelector<HTMLDetailsElement>(".mobile-pairing-manual");
    manual!.open = true;
    const origin = document.querySelector<HTMLInputElement>("#mobile-pairing-origin");
    const code = document.querySelector<HTMLInputElement>("#mobile-pairing-code-input");
    const button = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((item) => item.textContent === "Pair with code");
    origin!.value = "https://desktop.example";
    code!.value = "12345678";
    button!.click();
    releaseIdentity({ installationId: "phone-local-id" });
    await vi.waitFor(() => expect(pairStartCode).toHaveBeenCalledTimes(1));
    expect(pairStartCode.mock.calls[0][0]).toMatchObject({
      installationId: "phone-local-id",
      code: "12345678",
    });
    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-pairing")).toBeNull();
    });
  });

  it("checks a saved Tailscale target before showing pairing controls", async () => {
    const { connectTailscale } = installPreflightPlugin();
    window.eval(bootstrapSource);
    await vi.waitFor(() => {
      expect(document.body.textContent).toContain("Tailscale sign-in is required before pairing");
    });
    expect(connectTailscale).toHaveBeenCalledWith({
      origin: "https://desktop.example",
      installationId: "phone-local-id",
    });
    const cancel = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((item) => item.textContent === "Use a different link");
    cancel?.click();
  });

  it("retries the manual code after a Tailscale sign-in is required", async () => {
    let attempts = 0;
    const { pairStartCode, releaseIdentity } = installPlugin(async (args) => {
      attempts += 1;
      if (attempts === 1) throw new Error("tailnet login required");
      return {
        pairing_id: "pairing-code",
        nonce: "nonce-value-000000000000",
        collector_secret: "secret-value-000000000000",
        installationId: args.installationId,
      };
    });
    window.eval(bootstrapSource);
    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-pairing-manual")).not.toBeNull();
    });
    const origin = document.querySelector<HTMLInputElement>("#mobile-pairing-origin");
    const code = document.querySelector<HTMLInputElement>("#mobile-pairing-code-input");
    const button = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((item) => item.textContent === "Pair with code");
    origin!.value = "https://desktop.example";
    code!.value = "12345678";
    button!.click();
    releaseIdentity({ installationId: "phone-local-id" });
    await vi.waitFor(() => expect(pairStartCode).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => {
      expect(document.body.textContent).toContain("Tailscale sign-in is required");
    });
    const retry = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((item) => item.textContent === "Retry");
    retry!.click();
    await vi.waitFor(() => expect(pairStartCode).toHaveBeenCalledTimes(2));
    expect(pairStartCode.mock.calls[1][0]).toMatchObject({
      origin: "https://desktop.example",
      code: "12345678",
    });
  });

  it("ignores a startup identity rejection after cancel", async () => {
    const { rejectIdentity } = installStartupRejectPlugin();
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    const input = document.querySelector<HTMLInputElement>(".mobile-pairing-input");
    input!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    const cancel = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Use a different link");
    cancel?.click();
    rejectIdentity(new Error("startup identity failed"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.body.textContent).not.toContain("startup identity failed");
  });

  it("ignores a pairing result that arrives after cancel", async () => {
    const { pairStart, pairPoll, pairCancel, resolvePairStart } = installCancelPlugin();
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    const input = document.querySelector<HTMLInputElement>(".mobile-pairing-input");
    input!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(pairStart).toHaveBeenCalledTimes(1));

    const cancel = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Use a different link");
    cancel?.click();
    expect(pairCancel).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    });

    resolvePairStart({
      pairing_id: "stale-pairing",
      comparison_code: "999999",
      installationId: "phone-local-id",
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.body.textContent).not.toContain("999999");
    expect(pairPoll).not.toHaveBeenCalled();
  });

  it("ignores a pairStart rejection that arrives after cancel", async () => {
    const { pairStart, rejectPairStart } = installRejectedPairStartPlugin();
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    const input = document.querySelector<HTMLInputElement>(".mobile-pairing-input");
    input!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(pairStart).toHaveBeenCalledTimes(1));

    const cancel = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Use a different link");
    cancel?.click();
    rejectPairStart(new Error("late pairing failure"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.body.textContent).not.toContain("late pairing failure");
  });

  it("ignores a pairPoll rejection that arrives after cancel", async () => {
    const { rejectPoll } = installRejectedPollPlugin();
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    const input = document.querySelector<HTMLInputElement>(".mobile-pairing-input");
    input!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("waiting");
    });

    const cancel = [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Cancel");
    cancel?.click();
    rejectPoll(new Error("late poll failure"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.body.textContent).not.toContain("late poll failure");
  });

  it.each([
    { flow: "QR", hold: "collect" as const },
    { flow: "manual", hold: "collect" as const },
    { flow: "QR", hold: "poll" as const },
    { flow: "manual", hold: "poll" as const },
  ])("fences a held $flow $hold result before returning to pairing", async ({ flow, hold }) => {
    const fixture = installHeldPairingPlugin(hold);
    window.eval(bootstrapSource);
    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });

    if (flow === "QR") {
      document.querySelector<HTMLInputElement>(".mobile-pairing-input")!.value = [
        "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
        "installation=desktop-generated-id",
        "challenge=pairing-1.nonce.secret",
      ].join("&");
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    } else {
      document.querySelector<HTMLDetailsElement>(".mobile-pairing-manual")!.open = true;
      document.querySelector<HTMLInputElement>("#mobile-pairing-origin")!.value = "https://desktop.example";
      document.querySelector<HTMLInputElement>("#mobile-pairing-code-input")!.value = "12345678";
      [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
        .find((button) => button.textContent === "Pair with code")!.click();
    }

    const collect = flow === "QR" ? fixture.pairStart : fixture.pairStartCode;
    await vi.waitFor(() => expect(collect).toHaveBeenCalledTimes(1));
    if (hold === "poll") await vi.waitFor(() => expect(fixture.pairPoll).toHaveBeenCalledTimes(1));

    [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Cancel" || button.textContent === "Use a different link")!
      .click();
    expect(fixture.pairCancel).toHaveBeenCalledTimes(1);
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).not.toBe("idle");
    expect(document.querySelector<HTMLElement>(".mobile-pairing-status")?.textContent).toBe("Cancelling pairing...");

    fixture.resolveCancel();
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    });
    fixture.releaseHeld();
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.querySelectorAll<HTMLElement>(".mobile-pairing-pill")[1]?.textContent)
      .toBe("Connects when pairing");
    expect(document.body.textContent).not.toContain("Paired. Loading Nanobot...");
    expect(document.body.textContent).not.toContain("999999");
  });

  it("shows a native pairing cancellation persistence failure", async () => {
    const fixture = installHeldPairingPlugin("collect");
    fixture.pairCancel.mockRejectedValueOnce(new Error("custody fence sync failed"));
    window.eval(bootstrapSource);

    const form = await vi.waitFor(() => {
      const element = document.querySelector<HTMLFormElement>(".mobile-pairing-form");
      expect(element).not.toBeNull();
      return element as HTMLFormElement;
    });
    document.querySelector<HTMLInputElement>(".mobile-pairing-input")!.value = [
      "nanobot://pair?origin=https%3A%2F%2Fdesktop.example",
      "installation=desktop-generated-id",
      "challenge=pairing-1.nonce.secret",
    ].join("&");
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(fixture.pairStart).toHaveBeenCalledTimes(1));

    [...document.querySelectorAll<HTMLButtonElement>(".mobile-pairing-button")]
      .find((button) => button.textContent === "Use a different link")!.click();
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("error");
    });
    expect(document.querySelector<HTMLElement>(".mobile-pairing-status")?.textContent)
      .toContain("Pairing cancellation failed: custody fence sync failed");

    fixture.releaseHeld();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("error");
    expect(document.body.textContent).not.toContain("Paired. Loading Nanobot...");
  });

  it("joins the bootstrap handshake when its terminal event arrives before the ID", async () => {
    const fixture = installHandshakeRacePlugin();
    window.eval(bootstrapSource);
    const ready = (window as unknown as { nanobotMobileReady: Promise<void> }).nanobotMobileReady;
    let readyCount = 0;
    void ready.then(() => { readyCount += 1; });
    await vi.waitFor(() => expect(fixture.plugin.http).toHaveBeenCalledTimes(1));

    fixture.emit({ type: "http_response", request_id: 7, status: 401, body_base64: "other-id" });
    fixture.emit({ type: "http_response", request_id: 41, status: 200, body_base64: "handshake" });
    fixture.emit({ type: "http_response", request_id: 41, status: 500, body_base64: "duplicate" });
    fixture.releaseHandshake(41);
    await ready;
    await Promise.resolve();

    expect(readyCount).toBe(1);
    expect(fixture.refresh).not.toHaveBeenCalled();
    expect(fixture.plugin.logout).not.toHaveBeenCalled();
    expect(document.querySelector(".mobile-reauth")).toBeNull();
  });

  it("clears a revoked pairing before the app shell opens", async () => {
    const reload = vi.spyOn(window.location, "reload").mockImplementation(() => {});
    const { plugin, refresh } = installPairedPlugin();
    window.eval(bootstrapSource);

    await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    expect(await plugin.target()).toBeNull();
    expect(reload).toHaveBeenCalledTimes(1);
    reload.mockRestore();
  });

  it("shows reconnect instead of the app shell when the desktop is unreachable", async () => {
    const { logout } = installPairedPlugin("unreachable");
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-reauth")).not.toBeNull();
    });
    expect(document.body.textContent).toContain("Can't reach this desktop right now");
    expect(logout).not.toHaveBeenCalled();
  });

  it("shows reconnect when the native transport cannot start", async () => {
    installPairedPlugin("transport_error");
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-reauth")).not.toBeNull();
    });
    expect(document.body.textContent).toContain("native transport failed");
  });

  it("shows reconnect on a non-200, non-401 handshake", async () => {
    installPairedPlugin("server_error");
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-reauth")).not.toBeNull();
    });
    expect(document.body.textContent).toContain("Can't reach this desktop right now");
  });

  it("shows reconnect immediately when Tailscale needs a fresh login", async () => {
    installPairedPlugin("needs_login");
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-reauth")).not.toBeNull();
    });
    expect(document.body.textContent).toContain("tailnet login required");
  });

  it("shows reconnect when native transport start rejects", async () => {
    installPairedPlugin("start_error");
    window.eval(bootstrapSource);

    await vi.waitFor(() => {
      expect(document.querySelector(".mobile-reauth")).not.toBeNull();
    });
    expect(document.body.textContent).toContain("native start failed");
  });
});
