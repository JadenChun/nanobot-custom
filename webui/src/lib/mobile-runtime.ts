/**
 * Shared mobile runtime transport contract (M16).
 *
 * The Android transport supplies named operations only. This module owns
 * request/event correlation so the shared React client never needs browser
 * fetch, bootstrap secrets in storage, or a raw URL/dial surface.
 */

export interface MobileHttpRequest {
  operation: string;
  method: "GET" | "POST";
  path: string;
  headers?: Record<string, string>;
  bodyBase64?: string;
  timeoutMs?: number;
}

export interface MobileHttpResponse {
  requestId: number;
  status: number;
  headers: Record<string, string>;
  bodyBase64: string;
}

export interface MobileEvent {
  type: string;
  request_id?: number;
  status?: number;
  headers?: Record<string, string>;
  body_base64?: string;
  code?: string;
  message?: string;
  [key: string]: unknown;
}

export interface MobileTransport {
  start(origin: string, installationId: string, hostname: string): Promise<void>;
  status(): Promise<Record<string, unknown>>;
  http(request: MobileHttpRequest): Promise<number>;
  cancel(requestId: number): Promise<void>;
  onEvent(listener: (event: MobileEvent) => void): Promise<() => void>;
  openLogin(): Promise<void>;
  wsOpen(path: string, headers?: Record<string, string>): Promise<number>;
  wsSend(wsId: number, data: Uint8Array, kind?: "text" | "binary"): Promise<void>;
  wsClose(wsId: number): Promise<void>;
  stop(): Promise<void>;
}

export interface MobileRuntime {
  transport: MobileTransport;
  request(request: MobileHttpRequest, timeoutMs?: number): Promise<MobileHttpResponse>;
  handshake(): Promise<Record<string, unknown>>;
  read(path?: string): Promise<Record<string, unknown>>;
  onEvent(listener: (event: MobileEvent) => void): Promise<() => void>;
}

class MobileRuntimeError extends Error {}

export function createMobileRuntime(transport: MobileTransport): MobileRuntime {
  const pending = new Map<number, {
    resolve: (response: MobileHttpResponse) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();
  const listeners = new Set<(event: MobileEvent) => void>();
  let listening: Promise<() => void> | null = null;

  const ensureListener = (): Promise<() => void> => {
    if (listening === null) {
      listening = transport.onEvent((event) => {
        if (event.type === "http_response" && typeof event.request_id === "number") {
          const entry = pending.get(event.request_id);
          if (entry) {
            pending.delete(event.request_id);
            clearTimeout(entry.timer);
            entry.resolve({
              requestId: event.request_id,
              status: event.status ?? 0,
              headers: event.headers ?? {},
              bodyBase64: event.body_base64 ?? "",
            });
          }
        } else if (event.type === "http_error" && typeof event.request_id === "number") {
          const entry = pending.get(event.request_id);
          if (entry) {
            pending.delete(event.request_id);
            clearTimeout(entry.timer);
            entry.reject(new MobileRuntimeError(event.code ?? "http_error"));
          }
        }
        for (const listener of listeners) listener(event);
      });
    }
    return listening;
  };

  const request = async (input: MobileHttpRequest, timeoutMs = 30_000): Promise<MobileHttpResponse> => {
    await ensureListener();
    const requestId = await transport.http(input);
    if (!requestId) throw new MobileRuntimeError("mobile request was not admitted");
    return new Promise<MobileHttpResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(requestId);
        void transport.cancel(requestId);
        reject(new MobileRuntimeError("mobile request timed out"));
      }, timeoutMs);
      pending.set(requestId, { resolve, reject, timer });
    });
  };

  return {
    transport,
    request,
    async handshake() {
      const response = await request({ operation: "mobile_handshake", method: "GET", path: "/mobile/v1/handshake" });
      if (response.status !== 200) throw new MobileRuntimeError(`handshake_status_${response.status}`);
      return JSON.parse(decodeBase64(response.bodyBase64)) as Record<string, unknown>;
    },
    async read(path = "/mobile/v1/read") {
      const response = await request({ operation: "mobile_read", method: "GET", path });
      if (response.status !== 200) throw new MobileRuntimeError(`read_status_${response.status}`);
      return JSON.parse(decodeBase64(response.bodyBase64)) as Record<string, unknown>;
    },
    async onEvent(listener) {
      await ensureListener();
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

function decodeBase64(value: string): string {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new TextDecoder().decode(bytes);
}
