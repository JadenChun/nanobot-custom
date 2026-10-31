import { describe, expect, it, vi } from "vitest";

import { createMobileRuntime, type MobileEvent, type MobileTransport } from "@/lib/mobile-runtime";

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
});
