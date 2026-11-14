import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchWithTimeout } from "@/lib/http";
import {
  installMobileRuntime,
  resetMobileRuntime,
  setMobileOwnerCapability,
  setMobileStandardDispatchCapability,
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
    recover: async () => undefined,
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

  it("maps allowlisted chat API reads through the generic proxy", async () => {
    const encodedCommands = "/mobile/v1/api?path=" + encodeURIComponent("/api/commands");
    const encodedPreview = "/mobile/v1/api?path=" + encodeURIComponent(
      "/api/sessions/websocket%3Achat-1/file-preview?path=notes%2Fready.md&probe=1",
    );
    const { runtime, request } = runtimeWith({
      [encodedCommands]: {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ commands: [] }),
      },
      [encodedPreview]: {
        requestId: 2,
        status: 200,
        headers: {},
        bodyBase64: encode({ available: true }),
      },
    });
    installMobileRuntime(runtime);

    const commands = await fetchWithTimeout("/api/commands");
    await expect(commands.json()).resolves.toEqual({ commands: [] });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_api",
      method: "GET",
      path: encodedCommands,
    });

    const preview = await fetchWithTimeout(
      "/api/sessions/websocket%3Achat-1/file-preview?path=notes%2Fready.md&probe=1",
    );
    await expect(preview.json()).resolves.toEqual({ available: true });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_api",
      method: "GET",
      path: encodedPreview,
    });
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

  it("maps delegated-child pages through their chat.read operation", async () => {
    const path = "/mobile/v1/sessions/websocket%3Achat-1/delegated-children?limit=120&before=cursor%2B1";
    const { runtime, request } = runtimeWith({
      [path]: {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ children: [], page: { has_more_before: false } }),
      },
    });
    installMobileRuntime(runtime);

    const response = await fetchWithTimeout(
      "/api/sessions/websocket%3Achat-1/delegated-children?limit=120&before=cursor%2B1",
    );
    await expect(response.json()).resolves.toEqual({
      children: [],
      page: { has_more_before: false },
    });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_delegated_children",
      method: "GET",
      path,
    });
  });
});

describe("owner mobile feature bridge", () => {
  it("uses the owner feature route for full settings when owner is negotiated", async () => {
    const { runtime, request } = runtimeWith({
      ["/mobile/v1/feature?path=" + encodeURIComponent("/api/settings")]: {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ owners: true }),
      },
    });
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);

    const response = await fetchWithTimeout("/api/settings");
    await expect(response.json()).resolves.toEqual({ owners: true });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_feature",
      method: "GET",
      path: "/mobile/v1/feature?path=" + encodeURIComponent("/api/settings"),
    });
  });

  it("routes unknown API reads through the feature route and surfaces owner denial", async () => {
    const providerPath = "/mobile/v1/feature?path=" + encodeURIComponent("/api/settings/provider-models?provider=openai");
    const deniedPath = "/mobile/v1/feature?path=" + encodeURIComponent("/api/webui/automations");
    const { runtime } = runtimeWith({
      [providerPath]: {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ models: [] }),
      },
      [deniedPath]: {
        requestId: 2,
        status: 403,
        headers: {},
        bodyBase64: encode({ error: "owner_required" }),
      },
    });
    installMobileRuntime(runtime);

    const models = await fetchWithTimeout("/api/settings/provider-models?provider=openai");
    await expect(models.json()).resolves.toEqual({ models: [] });

    await expect(fetchWithTimeout("/api/webui/automations")).rejects.toThrow("owner_required");
  });

  it("keeps the legacy model-only settings read without owner capability", async () => {
    const { runtime } = runtimeWith({
      "/mobile/v1/model-settings": {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ agent: {} }),
      },
    });
    installMobileRuntime(runtime);
    setMobileOwnerCapability(false);

    const response = await fetchWithTimeout("/api/settings");
    await expect(response.json()).resolves.toEqual({ agent: {} });
  });
});

describe("standard owner dispatcher bridge", () => {
  it("uses the negotiated standard dispatch route for owner reads", async () => {
    const dispatchPath = "/mobile/v1/dispatch?path=" + encodeURIComponent("/api/settings");
    const { runtime, request } = runtimeWith({
      [dispatchPath]: {
        requestId: 1,
        status: 200,
        headers: {},
        bodyBase64: encode({ standard: true }),
      },
    });
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);
    setMobileStandardDispatchCapability(true);

    const response = await fetchWithTimeout("/api/settings");
    await expect(response.json()).resolves.toEqual({ standard: true });
    expect(request).toHaveBeenCalledWith({
      operation: "mobile_dispatch",
      method: "GET",
      path: dispatchPath,
    });
  });

  it("falls back to the wrapper for an older server without dispatch", async () => {
    const dispatchPath = "/mobile/v1/dispatch?path=" + encodeURIComponent("/api/settings");
    const featurePath = "/mobile/v1/feature?path=" + encodeURIComponent("/api/settings");
    const { runtime, request } = runtimeWith({
      [dispatchPath]: { requestId: 1, status: 404, headers: {}, bodyBase64: encode({ error: "not_found" }) },
      [featurePath]: {
        requestId: 2,
        status: 200,
        headers: {},
        bodyBase64: encode({ owners: true }),
      },
    });
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);
    setMobileStandardDispatchCapability(true);

    const response = await fetchWithTimeout("/api/settings");
    await expect(response.json()).resolves.toEqual({ owners: true });
    expect(request).toHaveBeenCalledTimes(2);
  });
});
