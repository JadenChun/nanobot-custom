import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

async function flushSocketWork(): Promise<void> {
  for (let index = 0; index < 20; index += 1) await Promise.resolve();
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

  it("refreshes a stale socket once and leaves replacement to the owning client", async () => {
    const fixture = socketTransportFixture();
    fixture.wsPing.mockResolvedValue(false);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const closed = vi.fn();
    socket.onopen = opened;
    socket.onclose = closed;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    await Promise.all([runtime.recover(), runtime.recover()]);
    expect(fixture.wsPing).toHaveBeenCalledTimes(1);
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).toHaveBeenCalledWith(1);
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(closed).toHaveBeenCalledTimes(1);
    expect(socket.readyState).toBe(3);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(opened).toHaveBeenCalledTimes(1);
    const replacement = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(replacement.readyState).toBe(1));
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    expect(opened).toHaveBeenCalledTimes(1);
    runtime.dispose();
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

describe("mobile socket startup ordering and bounds", () => {
  beforeEach(() => {
    // happy-dom aliases CloseEvent to Event and discards the browser's code.
    vi.stubGlobal("CloseEvent", class extends Event {
      readonly code: number;
      constructor(type: string, init: CloseEventInit = {}) {
        super(type, init);
        this.code = init.code ?? 0;
      }
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("delivers only matching early UTF-8/ready frames after open and makes close terminal", async () => {
    const fixture = socketTransportFixture();
    const opening = deferred<number>();
    fixture.wsOpen.mockImplementationOnce(() => opening.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const events: string[] = [];
    socket.onopen = () => events.push("open");
    socket.onmessage = (event) => events.push(String(event.data));
    socket.onclose = (event) => events.push(`close:${event.code}`);
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    expect(fixture.subscribed()).toBe(true);
    for (let index = 0; index < 80; index += 1) {
      for (const ws_id of [0, -1, NaN, Infinity, "41"]) {
        fixture.emit({ type: "ws_frame", ws_id, data_base64: base64Of("invalid") });
      }
    }
    fixture.emit({ type: "http_response", ws_id: 41, request_id: 88,
      body_base64: "A".repeat(4 * 1024 * 1024 + 4) });
    fixture.emit({ type: "ws_frame", ws_id: 42, data_base64: base64Of("unrelated") });
    fixture.emit({ type: "ws_closed", ws_id: 42, code: "1006" });
    fixture.emit({ type: "ws_frame", ws_id: 41, data_base64: base64Of("你好 👋") });
    fixture.emit({ type: "ws_frame", ws_id: 41, data_base64: base64Of('{"event":"ready"}') });
    fixture.emit({ type: "ws_closed", ws_id: 41, code: "1000" });
    fixture.emit({ type: "ws_frame", ws_id: 41, data_base64: base64Of("late") });
    expect(events).toEqual([]);
    opening.resolve(41);
    await vi.waitFor(() => expect(socket.readyState).toBe(3));
    expect(events).toEqual(["open", "你好 👋", '{"event":"ready"}', "close:1000"]);
    fixture.emit({ type: "ws_closed", ws_id: 41, code: "1000" });
    expect(events).toHaveLength(4);
    expect(fixture.wsClose).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it("accepts the 64-event startup boundary without dropping or duplicating frames", async () => {
    const fixture = socketTransportFixture();
    fixture.wsOpen.mockImplementationOnce(async () => {
      for (let index = 0; index < 64; index += 1) {
        fixture.emit({ type: "ws_frame", ws_id: 7, data_base64: base64Of(String(index)) });
      }
      return 7;
    });
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const messages: string[] = [];
    socket.onmessage = (event) => messages.push(String(event.data));
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    expect(messages).toEqual(Array.from({ length: 64 }, (_, index) => String(index)));
    runtime.dispose();
  });

  it("accepts exactly 4 MiB of encoded startup payload", async () => {
    const fixture = socketTransportFixture();
    fixture.wsOpen.mockImplementationOnce(async () => {
      fixture.emit({ type: "ws_frame", ws_id: 7, data_base64: "YWFh".repeat(1024 * 1024) });
      return 7;
    });
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const messages = vi.fn();
    socket.onmessage = messages;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    expect(messages).toHaveBeenCalledTimes(1);
    expect(messages.mock.calls[0][0].data).toHaveLength(3 * 1024 * 1024);
    runtime.dispose();
  });

  it.each(["event count", "encoded payload bytes", "UTF-8 encoded payload bytes"])("fails closed on startup %s overflow and releases the eventual ID once", async (bound) => {
    const fixture = socketTransportFixture();
    const opening = deferred<number>();
    fixture.wsOpen.mockImplementationOnce(() => opening.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const messages = vi.fn();
    const errored = vi.fn();
    const closed = vi.fn();
    socket.onopen = opened;
    socket.onmessage = messages;
    socket.onerror = errored;
    socket.onclose = closed;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    for (let index = 0; index < (bound === "event count" ? 65 : 1); index += 1) {
      fixture.emit({ type: "ws_frame", ws_id: 7, data_base64:
        bound === "event count" ? base64Of("ready")
          : bound === "encoded payload bytes" ? "YWFh".repeat(1024 * 1024 + 1)
            : "🙂".repeat(1024 * 1024 + 1) });
    }
    expect(socket.readyState).toBe(3);
    expect(errored).toHaveBeenCalledTimes(1);
    expect(closed).toHaveBeenCalledTimes(1);
    expect(closed.mock.calls[0][0].code).toBe(1006);
    socket.close();
    runtime.dispose();
    opening.resolve(7);
    await vi.waitFor(() => expect(fixture.wsClose).toHaveBeenCalledWith(7));
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(opened).not.toHaveBeenCalled();
    expect(messages).not.toHaveBeenCalled();
    expect(errored).toHaveBeenCalledTimes(1);
    expect(closed).toHaveBeenCalledTimes(1);
  });
});

describe("ordered native socket release", () => {
  it("waits for a delayed close before opening the owning client's replacement", async () => {
    const fixture = socketTransportFixture();
    const closing = deferred<void>();
    fixture.wsClose.mockImplementationOnce(() => closing.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    const closed = vi.fn();
    first.onclose = closed;
    first.close();
    first.close();
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await flushSocketWork();
    expect(first.readyState).toBe(3);
    expect(closed).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    closing.resolve();
    await vi.waitFor(() => expect(second.readyState).toBe(1));
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    runtime.dispose();
  });

  it("orders a retired late-open result and its delayed cleanup before the next open", async () => {
    const fixture = socketTransportFixture();
    const opening = deferred<number>();
    const closing = deferred<void>();
    fixture.wsOpen.mockImplementationOnce(() => opening.promise).mockResolvedValueOnce(10);
    fixture.wsClose.mockImplementationOnce(() => closing.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const closed = vi.fn();
    first.onopen = opened;
    first.onclose = closed;
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    first.close();
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    opening.resolve(9);
    await vi.waitFor(() => expect(fixture.wsClose).toHaveBeenCalledWith(9));
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    closing.resolve();
    await vi.waitFor(() => expect(second.readyState).toBe(1));
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(opened).not.toHaveBeenCalled();
    expect(closed).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it("cannot resurrect a disposed runtime, including pending or queued opens", async () => {
    const fixture = socketTransportFixture();
    const opening = deferred<number>();
    fixture.wsOpen.mockImplementationOnce(() => opening.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(fixture.wsOpen).toHaveBeenCalledTimes(1));
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    first.onopen = opened;
    second.onopen = opened;
    runtime.dispose();
    runtime.dispose();
    opening.resolve(9);
    await vi.waitFor(() => expect(fixture.wsClose).toHaveBeenCalledWith(9));
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(opened).not.toHaveBeenCalled();
    expect(first.readyState).toBe(3);
    expect(second.readyState).toBe(3);
    expect(() => runtime.createWebSocket("wss://host/mobile/v1/ws")).toThrow("disposed");
    await runtime.recover();
    await runtime.renew();
    expect(fixture.transport.refresh).not.toHaveBeenCalled();
  });

  it("waits for close rejection without claiming release and accepts broker refusal as terminal", async () => {
    const fixture = socketTransportFixture();
    const closing = deferred<void>();
    fixture.wsClose.mockImplementationOnce(() => closing.promise);
    fixture.wsOpen.mockResolvedValueOnce(1).mockRejectedValueOnce(new Error("broker still busy"));
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const firstError = vi.fn();
    const firstClose = vi.fn();
    first.onerror = firstError;
    first.onclose = firstClose;
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    first.close();
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const secondOpen = vi.fn();
    const secondError = vi.fn();
    const secondClose = vi.fn();
    second.onopen = secondOpen;
    second.onerror = secondError;
    second.onclose = secondClose;
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    closing.reject(new Error("release not established"));
    await vi.waitFor(() => expect(secondClose).toHaveBeenCalledTimes(1));
    expect(firstError).toHaveBeenCalledTimes(1);
    expect(firstClose).toHaveBeenCalledTimes(1);
    expect(secondOpen).not.toHaveBeenCalled();
    expect(secondError).toHaveBeenCalledTimes(1);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    runtime.dispose();
  });
});

describe("generation-scoped mobile socket maintenance", () => {
  it("coalesces renewals, refreshes first, and emits one close without reopening the adapter", async () => {
    const fixture = socketTransportFixture();
    const refresh = deferred<boolean>();
    vi.mocked(fixture.transport.refresh).mockImplementationOnce(() => refresh.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    const closed = vi.fn();
    socket.onopen = opened;
    socket.onclose = closed;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    const renewing = Promise.all([runtime.renew(), runtime.renew()]);
    await flushSocketWork();
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).not.toHaveBeenCalled();
    refresh.resolve(true);
    await renewing;
    expect(closed).toHaveBeenCalledTimes(1);
    expect(opened).toHaveBeenCalledTimes(1);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    runtime.dispose();
  });

  it.each(["ping", "refresh"])("fences overlapping recover/renew after explicit replacement during %s", async (boundary) => {
    const fixture = socketTransportFixture();
    const ping = deferred<boolean>();
    const refresh = deferred<boolean>();
    fixture.wsPing.mockImplementationOnce(() => ping.promise);
    vi.mocked(fixture.transport.refresh).mockImplementationOnce(() => refresh.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    const maintenance = Promise.all([runtime.recover(), runtime.renew(), runtime.recover()]);
    await flushSocketWork();
    if (boundary === "refresh") {
      ping.resolve(false);
      await vi.waitFor(() => expect(fixture.transport.refresh).toHaveBeenCalledTimes(1));
    }
    first.close();
    const replacement = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const replacementClose = vi.fn();
    replacement.onclose = replacementClose;
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    ping.resolve(false);
    refresh.resolve(true);
    await maintenance;
    await vi.waitFor(() => expect(replacement.readyState).toBe(1));
    expect(fixture.wsPing).toHaveBeenCalledTimes(1);
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(boundary === "refresh" ? 1 : 0);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    expect(replacementClose).not.toHaveBeenCalled();
    expect(replacement.readyState).toBe(1);
    runtime.dispose();
  });

  it("does not repeat renewal for a generation already retired by stale recovery", async () => {
    const fixture = socketTransportFixture();
    fixture.wsPing.mockResolvedValue(false);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    let replacement: WebSocket | undefined;
    first.onclose = () => { replacement = runtime.createWebSocket("wss://host/mobile/v1/ws"); };
    await Promise.all([runtime.recover(), runtime.renew()]);
    await vi.waitFor(() => expect(replacement?.readyState).toBe(1));
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    runtime.dispose();
  });

  it("opens a replacement only after pending renewal refresh without retiring it", async () => {
    const fixture = socketTransportFixture();
    const refresh = deferred<boolean>();
    vi.mocked(fixture.transport.refresh).mockImplementationOnce(() => refresh.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    const renewing = runtime.renew();
    await vi.waitFor(() => expect(fixture.transport.refresh).toHaveBeenCalledTimes(1));
    first.close();
    const replacement = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await flushSocketWork();
    expect(replacement.readyState).toBe(0);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    refresh.resolve(true);
    await renewing;
    await vi.waitFor(() => expect(replacement.readyState).toBe(1));
    expect(replacement.readyState).toBe(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(fixture.wsOpen).toHaveBeenCalledTimes(2);
    runtime.dispose();
  });

  it.each([
    ["renew", "refresh first"], ["renew", "close first"],
    ["recover", "refresh first"], ["recover", "close first"],
  ] as const)("orders %s credentials and delayed native close before replacement (%s)", async (operation, releaseOrder) => {
    const fixture = socketTransportFixture();
    const refresh = deferred<boolean>();
    const closing = deferred<void>();
    let credentialsReady = false;
    const credentialsAtOpen: boolean[] = [];
    fixture.wsOpen.mockImplementation(async () => {
      credentialsAtOpen.push(credentialsReady);
      return credentialsAtOpen.length;
    });
    vi.mocked(fixture.transport.refresh).mockImplementationOnce(async () => {
      const refreshed = await refresh.promise;
      credentialsReady = refreshed;
      return refreshed;
    });
    fixture.wsClose.mockImplementationOnce(() => closing.promise);
    fixture.wsPing.mockResolvedValue(false);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    const repairing = runtime[operation]();
    await vi.waitFor(() => expect(fixture.transport.refresh).toHaveBeenCalledTimes(1));
    const overlapping = operation === "renew" ? runtime.recover() : runtime.renew();
    first.close();
    const replacement = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const replacementClose = vi.fn();
    replacement.onclose = replacementClose;
    await flushSocketWork();
    if (releaseOrder === "refresh first") refresh.resolve(true);
    else closing.resolve();
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(replacement.readyState).toBe(0);
    refresh.resolve(true);
    closing.resolve();
    await Promise.all([repairing, overlapping]);
    await vi.waitFor(() => expect(replacement.readyState).toBe(1));
    expect(credentialsAtOpen).toEqual([false, true]);
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(replacementClose).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it.each(["renew", "recover"] as const)("does not resurrect a queued replacement disposed during %s refresh", async (operation) => {
    const fixture = socketTransportFixture();
    const refresh = deferred<boolean>();
    vi.mocked(fixture.transport.refresh).mockImplementationOnce(() => refresh.promise);
    fixture.wsPing.mockResolvedValue(false);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    const repairing = runtime[operation]();
    await vi.waitFor(() => expect(fixture.transport.refresh).toHaveBeenCalledTimes(1));
    first.close();
    const replacement = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const opened = vi.fn();
    replacement.onopen = opened;
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    runtime.dispose();
    refresh.resolve(true);
    await repairing;
    await flushSocketWork();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(replacement.readyState).toBe(3);
    expect(opened).not.toHaveBeenCalled();
  });

  it("rotates access tokens in-band via wsRefreshAuth without closing a healthy socket", async () => {
    const fixture = socketTransportFixture();
    const wsRefreshAuth = vi.fn(async () => true);
    fixture.transport.wsRefreshAuth = wsRefreshAuth;
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const closed = vi.fn();
    socket.onclose = closed;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));

    await runtime.renew();

    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(wsRefreshAuth).toHaveBeenCalledWith(1);
    expect(fixture.wsClose).not.toHaveBeenCalled();
    expect(closed).not.toHaveBeenCalled();
    expect(socket.readyState).toBe(1);
    runtime.dispose();
  });

  it.each([false, "rejected"])("keeps a healthy socket open on renew when credential refresh is %s", async (result) => {
    const fixture = socketTransportFixture();
    if (result === false) vi.mocked(fixture.transport.refresh).mockResolvedValue(false);
    else vi.mocked(fixture.transport.refresh).mockRejectedValue(new Error("refresh refused"));
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const closed = vi.fn();
    socket.onclose = closed;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    await runtime.renew();
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).not.toHaveBeenCalled();
    expect(fixture.wsOpen).toHaveBeenCalledTimes(1);
    expect(closed).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it.each([false, "rejected"])("closes a ping-failed socket on recover even when credential refresh is %s", async (result) => {
    const fixture = socketTransportFixture();
    fixture.wsPing.mockResolvedValue(false);
    if (result === false) vi.mocked(fixture.transport.refresh).mockResolvedValue(false);
    else vi.mocked(fixture.transport.refresh).mockRejectedValue(new Error("refresh refused"));
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const closed = vi.fn();
    socket.onclose = closed;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    await runtime.recover();
    expect(fixture.transport.refresh).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose).toHaveBeenCalledTimes(1);
    expect(closed).toHaveBeenCalledTimes(1);
    expect(socket.readyState).toBe(3);
    runtime.dispose();
  });
});

describe("mobile native send refusal", () => {
  it("retires and releases the captured native socket exactly once on asynchronous refusal", async () => {
    const fixture = socketTransportFixture();
    const sending = deferred<void>();
    fixture.wsSend.mockImplementation(() => sending.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const socket = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const errored = vi.fn();
    const closed = vi.fn();
    socket.onerror = errored;
    socket.onclose = closed;
    await vi.waitFor(() => expect(socket.readyState).toBe(1));
    socket.send("first");
    socket.send("second");
    sending.reject(new Error("native queue refused"));
    await vi.waitFor(() => expect(closed).toHaveBeenCalledTimes(1));
    expect(errored).toHaveBeenCalledTimes(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    expect(socket.readyState).toBe(3);
    fixture.emit({ type: "ws_closed", ws_id: 1, code: "1006" });
    socket.close();
    expect(closed).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it("ignores an old asynchronous refusal after explicit replacement", async () => {
    const fixture = socketTransportFixture();
    const sending = deferred<void>();
    fixture.wsSend.mockImplementationOnce(() => sending.promise);
    const runtime = createMobileRuntime(fixture.transport);
    const first = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const firstError = vi.fn();
    const firstClose = vi.fn();
    first.onerror = firstError;
    first.onclose = firstClose;
    await vi.waitFor(() => expect(first.readyState).toBe(1));
    first.send("original");
    first.close();
    const second = runtime.createWebSocket("wss://host/mobile/v1/ws");
    const secondError = vi.fn();
    const secondClose = vi.fn();
    second.onerror = secondError;
    second.onclose = secondClose;
    await vi.waitFor(() => expect(second.readyState).toBe(1));
    sending.reject(new Error("late native refusal"));
    await flushSocketWork();
    expect(firstClose).toHaveBeenCalledTimes(1);
    expect(firstError).not.toHaveBeenCalled();
    expect(secondError).not.toHaveBeenCalled();
    expect(secondClose).not.toHaveBeenCalled();
    expect(second.readyState).toBe(1);
    expect(fixture.wsClose.mock.calls).toEqual([[1]]);
    runtime.dispose();
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
