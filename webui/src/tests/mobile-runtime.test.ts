import { describe, expect, it, vi } from "vitest";

import {
  createMobileRuntime,
  mobileOwnerCapability,
  resetMobileRuntime,
  type MobileEvent,
  type MobileTransport,
} from "@/lib/mobile-runtime";

function transportFixture() {
  let listener: ((event: MobileEvent) => void) | null = null;
  let nextRequest = 0;
  const http = vi.fn(async () => {
    nextRequest += 1;
    return nextRequest;
  });
  const cancel = vi.fn(async () => undefined);
  const transport: MobileTransport = {
    start: vi.fn(async () => undefined),
    status: vi.fn(async () => ({ state: "ready" })),
    http,
    cancel,
    onEvent: vi.fn(async (callback) => {
      listener = callback;
      return () => { listener = null; };
    }),
    openLogin: vi.fn(async () => undefined),
    refresh: vi.fn(async () => true),
    wsOpen: vi.fn(async () => 1),
    wsSend: vi.fn(async () => undefined),
    wsClose: vi.fn(async () => undefined),
    stop: vi.fn(async () => undefined),
  };
  return {
    transport,
    emit: (event: MobileEvent) => listener?.(event),
    http,
    cancel,
    refresh: transport.refresh,
  };
}

describe("mobile runtime transport", () => {
  it("correlates named requests with native events", async () => {
    const fixture = transportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const promise = runtime.request({
      operation: "mobile_read",
      method: "GET",
      path: "/mobile/v1/read",
    });
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalledTimes(1));
    fixture.emit({
      type: "http_response",
      request_id: 1,
      status: 200,
      headers: { "content-type": "application/json" },
      body_base64: btoa("{\"sessions\":[]}"),
    });
    await expect(promise).resolves.toMatchObject({ requestId: 1, status: 200 });
  });

  it("rejects native errors and cancels timed-out requests", async () => {
    const fixture = transportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const errorPromise = runtime.request({ operation: "mobile_read", method: "GET", path: "/mobile/v1/read" });
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalled());
    fixture.emit({ type: "http_error", request_id: 1, code: "scope_denied" });
    await expect(errorPromise).rejects.toThrow("scope_denied");

    const timeoutPromise = runtime.request(
      { operation: "mobile_read", method: "GET", path: "/mobile/v1/read" },
      50,
    );
    const timeoutExpectation = expect(timeoutPromise).rejects.toThrow("timed out");
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalledTimes(2));
    await timeoutExpectation;
    expect(fixture.cancel).toHaveBeenCalledWith(2);
  });

  it("uses the named handshake operation and no browser fetch", async () => {
    const fixture = transportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const promise = runtime.handshake();
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalled());
    fixture.emit({
      type: "http_response",
      request_id: 1,
      status: 200,
      headers: {},
      body_base64: btoa("{\"mobile_protocol\":1}"),
    });
    await expect(promise).resolves.toEqual({ mobile_protocol: 1 });
  });

  it("refreshes once after a 401 and retries the named request", async () => {
    const fixture = transportFixture();
    let calls = 0;
    fixture.http.mockImplementation(async () => {
      calls += 1;
      return calls;
    });
    const runtime = createMobileRuntime(fixture.transport);
    const promise = runtime.request({
      operation: "mobile_media",
      method: "GET",
      path: "/mobile/v1/media/sig/payload",
    });
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalledTimes(1));
    fixture.emit({ type: "http_response", request_id: 1, status: 401, headers: {}, body_base64: "" });
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalledTimes(2));
    expect(fixture.refresh).toHaveBeenCalledTimes(1);
    fixture.emit({ type: "http_response", request_id: 2, status: 200, headers: {}, body_base64: "" });
    await expect(promise).resolves.toMatchObject({ status: 200 });
  });
});


describe("mobile owner capability", () => {
  it("tracks the negotiated owner capability and resets on teardown", async () => {
    const fixture = transportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    expect(mobileOwnerCapability()).toBe(false);

    const promise = runtime.handshake();
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalled());
    fixture.emit({
      type: "http_response",
      request_id: 1,
      status: 200,
      headers: { "content-type": "application/json" },
      body_base64: btoa(JSON.stringify({
        mobile_protocol: 1,
        capabilities: { chat: true, owner: true },
      })),
    });
    await promise;
    expect(mobileOwnerCapability()).toBe(true);

    resetMobileRuntime();
    expect(mobileOwnerCapability()).toBe(false);
  });
});

function socketTransportFixture() {
  let listener: ((event: MobileEvent) => void) | null = null;
  let nextId = 0;
  const wsOpen = vi.fn(async () => {
    nextId += 1;
    return nextId;
  });
  const wsSend = vi.fn(async () => undefined);
  const wsClose = vi.fn(async () => undefined);
  const wsPing = vi.fn(async () => true);
  const transport: MobileTransport = {
    start: vi.fn(async () => undefined),
    status: vi.fn(async () => ({})),
    http: vi.fn(async () => 1),
    cancel: vi.fn(async () => undefined),
    onEvent: vi.fn(async (callback) => {
      listener = callback;
      return () => {
        if (listener === callback) listener = null;
      };
    }),
    openLogin: vi.fn(async () => undefined),
    refresh: vi.fn(async () => true),
    wsOpen,
    wsSend,
    wsClose,
    wsPing,
    stop: vi.fn(async () => undefined),
  };
  return {
    transport,
    wsOpen,
    wsSend,
    wsClose,
    wsPing,
    emit: (event: MobileEvent) => listener?.(event),
    subscribed: () => listener !== null,
  };
}

function base64Of(text: string): string {
  return btoa(String.fromCharCode(...new TextEncoder().encode(text)));
}

describe("mobile socket lifecycle", () => {
  it("emits one close and no open when the native open fails", async () => {
    const fixture = socketTransportFixture();
    fixture.wsOpen.mockRejectedValueOnce(new Error("open failed"));
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const closed = vi.fn();
    socket.onopen = opened;
    socket.onclose = closed;
    await vi.waitFor(() => expect(closed).toHaveBeenCalledTimes(1));
    expect(opened).not.toHaveBeenCalled();
    const messages = vi.fn();
    socket.onmessage = messages;
    fixture.emit({ type: "ws_frame", ws_id: 1, data_base64: base64Of("late") });
    fixture.emit({ type: "ws_closed", ws_id: 1, code: 1000 });
    expect(messages).not.toHaveBeenCalled();
    expect(closed).toHaveBeenCalledTimes(1);
  });

  it("ignores late frames and closes after a remote close", async () => {
    const fixture = socketTransportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const messages: string[] = [];
    const closed = vi.fn();
    socket.onmessage = (event) => messages.push(String(event.data));
    socket.onclose = closed;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalled());
    await vi.waitFor(() => expect(messages).toEqual([]));
    fixture.emit({ type: "ws_frame", ws_id: 1, data_base64: base64Of("first") });
    fixture.emit({ type: "ws_closed", ws_id: 1, code: 1000 });
    expect(messages).toEqual(["first"]);
    expect(closed).toHaveBeenCalledTimes(1);
    fixture.emit({ type: "ws_frame", ws_id: 1, data_base64: base64Of("late") });
    fixture.emit({ type: "ws_closed", ws_id: 1, code: 1000 });
    expect(messages).toEqual(["first"]);
    expect(closed).toHaveBeenCalledTimes(1);
  });

  it("closes a late native open once after close during opening", async () => {
    const fixture = socketTransportFixture();
    let resolveOpen: ((id: number) => void) | null = null;
    fixture.wsOpen.mockImplementationOnce(() => new Promise((resolve) => {
      resolveOpen = resolve;
    }));
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const closed = vi.fn();
    socket.onopen = opened;
    socket.onclose = closed;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalled());
    socket.close();
    expect(closed).toHaveBeenCalledTimes(1);
    resolveOpen?.(9);
    await vi.waitFor(() => expect(fixture.wsClose).toHaveBeenCalledWith(9));
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(opened).not.toHaveBeenCalled();
    expect(closed).toHaveBeenCalledTimes(1);
  });

  it("decodes UTF-8 text frames without losing characters", async () => {
    const fixture = socketTransportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const messages: string[] = [];
    socket.onmessage = (event) => messages.push(String(event.data));
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalled());
    fixture.emit({ type: "ws_frame", ws_id: 1, data_base64: base64Of("你好 👋") });
    expect(messages).toEqual(["你好 👋"]);
  });

  it("routes events only to the owning socket identity", async () => {
    const fixture = socketTransportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const firstMessages: string[] = [];
    const secondMessages: string[] = [];
    first.onmessage = (event) => firstMessages.push(String(event.data));
    second.onmessage = (event) => secondMessages.push(String(event.data));
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(2));
    fixture.emit({ type: "ws_frame", ws_id: 1, data_base64: base64Of("old") });
    fixture.emit({ type: "ws_frame", ws_id: 2, data_base64: base64Of("new") });
    expect(firstMessages).toEqual(["old"]);
    expect(secondMessages).toEqual(["new"]);
  });

  it("keeps a healthy open socket during foreground recovery", async () => {
    const fixture = socketTransportFixture();
    fixture.wsPing.mockResolvedValue(true);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    socket.onopen = opened;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    await runtime.recover();
    expect(fixture.wsPing).toHaveBeenCalledWith(1);
    expect(fixture.wsClose).not.toHaveBeenCalled();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(opened).toHaveBeenCalledTimes(1);
  });

  it("renews credentials and replaces a stale socket once", async () => {
    const fixture = socketTransportFixture();
    fixture.wsPing.mockResolvedValue(false);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    socket.onopen = opened;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    await Promise.all([runtime.recover(), runtime.recover()]);
    expect(fixture.wsPing).toHaveBeenCalledTimes(1);
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).toHaveBeenCalledWith(1);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    expect(opened).toHaveBeenCalledTimes(2);
  });

  it("rejects sends before open and forwards text bytes after open", async () => {
    const fixture = socketTransportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    expect(() => socket.send("early")).toThrow("text frames only");
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalled());
    socket.send("你好");
    expect(fixture.wsSend).toHaveBeenCalledWith(
      1,
      new TextEncoder().encode("你好"),
      "text",
    );
  });
});

describe("native completion admission", () => {
  const input = { operation: "mobile_read", method: "GET" as const, path: "/mobile/v1/read" };
  it("joins a completion emitted before http returns its ID", async () => {
    const fixture = transportFixture();
    fixture.http.mockImplementation(async () => {
      fixture.emit({ type: "http_response", request_id: 41, status: 200, body_base64: "" });
      return 41;
    });
    const runtime = createMobileRuntime(fixture.transport);
    await expect(runtime.request(input, 40)).resolves.toMatchObject({ requestId: 41, status: 200 });
  });
  it("joins concurrent early results without crossing request identities", async () => {
    const fixture = transportFixture();
    let nextId = 0;
    fixture.http.mockImplementation(async () => {
      const id = ++nextId;
      fixture.emit({ type: "http_response", request_id: id, status: 200 + id, body_base64: "" });
      await Promise.resolve();
      return id;
    });
    const runtime = createMobileRuntime(fixture.transport);
    const results = await Promise.all(Array.from({ length: 64 }, () => runtime.request(input, 100)));
    expect(new Set(results.map((result) => result.requestId)).size).toBe(64);
    for (const result of results) expect(result.status).toBe(200 + result.requestId);
  });
  it("reserves finite capacity before launch and cancels/disposes each request once", async () => {
    const fixture = transportFixture();
    const runtime = createMobileRuntime(fixture.transport);
    const requests = Array.from({ length: 64 }, () => runtime.request(input, 1000).catch((error: Error) => error.message));
    await vi.waitFor(() => expect(fixture.http).toHaveBeenCalledTimes(64));
    await expect(runtime.request(input, 1000)).rejects.toThrow("queue is full");
    runtime.dispose(); runtime.dispose();
    await Promise.all(requests);
    expect(fixture.cancel).toHaveBeenCalledTimes(64);
    fixture.emit({ type: "http_response", request_id: 1, status: 200 });
    await expect(runtime.request(input)).rejects.toThrow("disposed");
  });
  it("cancels an ID returned after timeout once and ignores late results", async () => {
    const fixture = transportFixture();
    let release: ((id: number) => void) | undefined;
    fixture.http.mockImplementation(() => new Promise<number>((resolve) => { release = resolve; }));
    const runtime = createMobileRuntime(fixture.transport);
    const request = runtime.request(input, 30);
    const expectation = expect(request).rejects.toThrow("timed out");
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    await expectation; release!(99);
    await vi.waitFor(() => expect(fixture.cancel).toHaveBeenCalledWith(99));
    fixture.emit({ type: "http_response", request_id: 99, status: 200 });
    runtime.dispose();
    expect(fixture.cancel).toHaveBeenCalledTimes(1);
  });
});
