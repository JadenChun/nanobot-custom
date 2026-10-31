import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchWithTimeout } from "@/lib/http";
import {
  installMobileRuntime,
  resetMobileRuntime,
  type MobileHttpResponse,
  type MobileRuntime,
} from "@/lib/mobile-runtime";

function encode(value: unknown): string {
  return btoa(JSON.stringify(value));
}

function runtimeWith(
  responses: Record<string, MobileHttpResponse>,
): { runtime: MobileRuntime; request: ReturnType<typeof vi.fn> } {
  const request = vi.fn(async (input: { path: string }) => {
    const response = responses[input.path];
    if (!response) throw new Error(`unexpected mobile path: ${input.path}`);
    return response;
  });
  const runtime: MobileRuntime = {
    transport: {} as MobileRuntime["transport"],
    request,
    handshake: async () => ({}),
    read: async () => ({}),
    onEvent: async () => () => undefined,
    createWebSocket: () => ({}) as WebSocket,
  };
  return { runtime, request };
}

afterEach(() => resetMobileRuntime());

describe("mobile HTTP bridge", () => {
  it("maps the model settings read to the model-only mobile route", async () => {
    const { runtime, request } = runtimeWith({
      "/mobile/v1/model-settings": {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ agent: { model_preset: "default" } }),
      },
    });
    installMobileRuntime(runtime);

    const response = await fetchWithTimeout("/api/settings");
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ agent: { model_preset: "default" } });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_model_settings",
      method: "GET",
      path: "/mobile/v1/model-settings",
    });
  });

  it("maps the workspaces read and preserves JSON", async () => {
    const { runtime } = runtimeWith({
      "/mobile/v1/workspaces": {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ controls: { permission_v2: true } }),
      },
    });
    installMobileRuntime(runtime);

    const response = await fetchWithTimeout("/api/workspaces");
    await expect(response.json()).resolves.toEqual({ controls: { permission_v2: true } });
  });

  it("maps thread reads, keeps the query, and carries 404 through", async () => {
    const { runtime, request } = runtimeWith({
      "/mobile/v1/sessions/websocket%3Achat-1/thread?limit=2": {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ messages: ["hello"] }),
      },
      "/mobile/v1/sessions/websocket%3Amissing/thread": {
        requestId: 2,
        status: 404,
        headers: {},
        bodyBase64: encode({ error: "thread_not_found" }),
      },
    });
    installMobileRuntime(runtime);

    const thread = await fetchWithTimeout("/api/sessions/websocket%3Achat-1/webui-thread?limit=2");
    await expect(thread.json()).resolves.toEqual({ messages: ["hello"] });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_thread",
      method: "GET",
      path: "/mobile/v1/sessions/websocket%3Achat-1/thread?limit=2",
    });

    const missing = await fetchWithTimeout("/api/sessions/websocket%3Amissing/webui-thread");
    expect(missing.status).toBe(404);
  });
});
