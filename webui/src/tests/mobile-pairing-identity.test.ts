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
  const refresh = vi.fn(async () => {
    throw new Error(mode === "revoked" ? "pairing was revoked" : "network");
  });
  const logout = vi.fn(async () => ({ serverRevocationPending: false }));
  const plugin = {
    target: vi.fn(async () => ({
      origin: "https://desktop.example",
      installationId: "phone-local-id",
      hasAccess: true,
    })),
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
  const plugin = {
    target: vi.fn(async () => ({ hasAccess: false })),
    pairingIdentity: vi.fn(async () => ({ installationId: "phone-local-id" })),
    pairStart,
    pairPoll,
    pendingPairingLink: vi.fn(async () => null),
  };
  (window as unknown as { Capacitor?: unknown }).Capacitor = {
    Plugins: { NativeNetwork: plugin },
  };
  return { pairStart, pairPoll, resolvePairStart: (value: Record<string, unknown>) => resolvePairStart(value) };
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
    const { pairStart, pairPoll, resolvePairStart } = installCancelPlugin();
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
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");

    resolvePairStart({
      pairing_id: "stale-pairing",
      comparison_code: "999999",
      installationId: "phone-local-id",
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.querySelector<HTMLElement>(".mobile-pairing")?.dataset.state).toBe("idle");
    expect(document.querySelector(".mobile-pairing-code")?.textContent).toBe("------");
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

  it("clears a revoked pairing before the app shell opens", async () => {
    const reload = vi.spyOn(window.location, "reload").mockImplementation(() => {});
    const { refresh, logout } = installPairedPlugin();
    window.eval(bootstrapSource);

    await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(logout).toHaveBeenCalledWith({ serverReachable: true }));
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
