import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchBootstrap } from "@/lib/bootstrap";
import { fetchWithTimeout } from "@/lib/http";
import {
  installMobileRuntime,
  resetMobileRuntime,
  shouldRegisterServiceWorker,
  type MobileRuntime,
  type MobileTransport,
} from "@/lib/mobile-runtime";
import { createRuntimeHost, isNativeRuntime } from "@/lib/runtime";

function fakeRuntime(): MobileRuntime {
  let next = 0;
  const transport: MobileTransport = {
    start: vi.fn(async () => undefined),
    status: vi.fn(async () => ({ state: "ready" })),
    http: vi.fn(async () => {
      next += 1;
      return next;
    }),
    cancel: vi.fn(async () => undefined),
    onEvent: vi.fn(async () => () => undefined),
    openLogin: vi.fn(async () => undefined),
    refresh: vi.fn(async () => true),
    wsOpen: vi.fn(async () => 1),
    wsSend: vi.fn(async () => undefined),
    wsClose: vi.fn(async () => undefined),
    stop: vi.fn(async () => undefined),
  };
  const runtime: MobileRuntime = {
    transport,
    request: vi.fn(async (request) => {
      if (request.path === "/mobile/v1/read") {
        return { requestId: 1, status: 200, headers: {}, bodyBase64: btoa("{\"sessions\":[]}") };
      }
      return { requestId: 2, status: 200, headers: { "content-type": "image/png" }, bodyBase64: btoa("PNG") };
    }),
    handshake: vi.fn(async () => ({ mobile_protocol: 1 })),
    read: vi.fn(async () => ({ sessions: [] })),
    onEvent: vi.fn(async () => () => undefined),
    createWebSocket: vi.fn(() => ({}) as WebSocket),
  };
  return runtime;
}

afterEach(() => {
  resetMobileRuntime();
  vi.unstubAllGlobals();
});

it("registers the browser service worker only without mobile runtime", () => {
  expect(shouldRegisterServiceWorker()).toBe(true);
});

describe("mobile runtime wiring", () => {
  it("uses the native handshake for bootstrap without browser fetch", async () => {
    const runtime = fakeRuntime();
    installMobileRuntime(runtime);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const boot = await fetchBootstrap();
    expect(boot.ws_path).toBe("/mobile/v1/ws");
    expect(boot.runtime_surface).toBe("native");
    expect(runtime.handshake).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("maps session reads and signed media through the runtime", async () => {
    const runtime = fakeRuntime();
    installMobileRuntime(runtime);
    const sessions = await fetchWithTimeout("/api/sessions");
    expect(sessions.status).toBe(200);
    expect(await sessions.json()).toEqual({ sessions: [] });
    const media = await fetchWithTimeout("/api/media/sig/payload");
    expect(media.status).toBe(200);
    expect(await media.text()).toBe("PNG");
    expect(runtime.request).toHaveBeenCalledWith(expect.objectContaining({
      operation: "mobile_media",
      path: "/mobile/v1/media/sig/payload",
    }));
  });

  it("denies unsupported browser operations and exposes a native socket factory", async () => {
    const runtime = fakeRuntime();
    installMobileRuntime(runtime);
    await expect(fetchWithTimeout("/not-an-api")).rejects.toThrow("mobile operation unavailable");
    const version = await fetchWithTimeout("/api/settings/version-check");
    expect(version.status).toBe(200);
    expect(runtime.request).toHaveBeenCalledWith(expect.objectContaining({
      operation: "mobile_feature",
      path: "/mobile/v1/feature?path=" + encodeURIComponent("/api/settings/version-check"),
    }));
    expect(isNativeRuntime()).toBe(true);
    expect(createRuntimeHost("browser").socketFactory).toBeTypeOf("function");
    expect(shouldRegisterServiceWorker()).toBe(false);
  });
});
