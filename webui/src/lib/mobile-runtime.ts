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

export interface MobilePickedCamera {
  name: string;
  mediaType: string;
  dataBase64: string;
}

export interface MobileTransport {
  start(origin: string, installationId: string, hostname: string): Promise<void>;
  status(): Promise<Record<string, unknown>>;
  http(request: MobileHttpRequest): Promise<number>;
  cancel(requestId: number): Promise<void>;
  onEvent(listener: (event: MobileEvent) => void): Promise<() => void>;
  openLogin(): Promise<void>;
  refresh(): Promise<boolean>;
  pickCamera?(): Promise<MobilePickedCamera>;
  wsOpen(path: string, headers?: Record<string, string>): Promise<number>;
  wsSend(wsId: number, data: Uint8Array, kind?: "text" | "binary"): Promise<void>;
  wsClose(wsId: number): Promise<void>;
  wsPing?(wsId: number): Promise<boolean>;
  stop(): Promise<void>;
}

export interface MobileRuntime {
  transport: MobileTransport;
  request(request: MobileHttpRequest, timeoutMs?: number): Promise<MobileHttpResponse>;
  handshake(): Promise<Record<string, unknown>>;
  read(path?: string): Promise<Record<string, unknown>>;
  onEvent(listener: (event: MobileEvent) => void): Promise<() => void>;
  createWebSocket(url: string): WebSocket;
  /** Restore foreground connectivity once; no-op when no runtime is installed. */
  recover(): Promise<void>;
}

let installedMobileRuntime: MobileRuntime | null = null;
let ownerCapability = false;

/** Whether the paired device negotiated the owner capability (M30/M31). */
export function mobileOwnerCapability(): boolean {
  return ownerCapability;
}

export function setMobileOwnerCapability(value: boolean): void {
  ownerCapability = value;
}

export function installMobileRuntime(runtime: MobileRuntime): void {
  installedMobileRuntime = runtime;
}

export function getMobileRuntime(): MobileRuntime | null {
  return installedMobileRuntime;
}

export function isMobileRuntimeActive(): boolean {
  return installedMobileRuntime !== null;
}

export function recoverMobileRuntime(): Promise<void> {
  return installedMobileRuntime?.recover() ?? Promise.resolve();
}

export function resetMobileRuntime(): void {
  installedMobileRuntime = null;
  ownerCapability = false;
}

export function shouldRegisterServiceWorker(): boolean {
  return !isMobileRuntimeActive();
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
  const sockets = new Set<MobileWebSocket>();
  let recovery: Promise<void> | null = null;

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

  const execute = async (input: MobileHttpRequest, timeoutMs: number): Promise<MobileHttpResponse> => {
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

  const request = async (input: MobileHttpRequest, timeoutMs = 30_000): Promise<MobileHttpResponse> => {
    let response = await execute(input, timeoutMs);
    if (response.status === 401) {
      const refreshed = await transport.refresh().catch(() => false);
      if (refreshed) response = await execute(input, timeoutMs);
    }
    return response;
  };

  const runtime: MobileRuntime = {
    transport,
    request,
    async handshake() {
      const response = await request({ operation: "mobile_handshake", method: "GET", path: "/mobile/v1/handshake" });
      if (response.status !== 200) throw new MobileRuntimeError(`handshake_status_${response.status}`);
      const payload = JSON.parse(decodeBase64(response.bodyBase64)) as Record<string, unknown>;
      const capabilities = payload.capabilities;
      setMobileOwnerCapability(
        typeof capabilities === "object"
          && capabilities !== null
          && (capabilities as Record<string, unknown>).owner === true,
      );
      return payload;
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
    createWebSocket(url) {
      const socket = new MobileWebSocket(runtime, url);
      sockets.add(socket);
      return socket as unknown as WebSocket;
    },
    async recover() {
      if (recovery !== null) return recovery;
      recovery = (async () => {
        const active = [...sockets].filter((socket) => !socket.isRetired());
        for (const socket of active) {
          await socket.recover();
          if (socket.isRetired()) sockets.delete(socket);
        }
      })().finally(() => {
        recovery = null;
      });
      return recovery;
    },
  };
  return runtime;
}

class MobileWebSocket {
  binaryType: BinaryType = "blob";
  onclose: ((this: WebSocket, ev: CloseEvent) => unknown) | null = null;
  onerror: ((this: WebSocket, ev: Event) => unknown) | null = null;
  onmessage: ((this: WebSocket, ev: MessageEvent) => unknown) | null = null;
  onopen: ((this: WebSocket, ev: Event) => unknown) | null = null;
  readyState = 0;
  readonly url: string;
  private wsId = 0;
  private unsubscribe: (() => void) | null = null;
  private retired = false;
  private closeEmitted = false;

  constructor(private runtime: MobileRuntime, url: string) {
    this.url = url;
    const parsed = new URL(url);
    if (parsed.pathname !== "/mobile/v1/ws") {
      throw new Error("mobile socket path is not allowlisted");
    }
    void this.open();
  }

  isRetired(): boolean {
    return this.retired;
  }

  async recover(): Promise<void> {
    if (this.retired || this.readyState !== 1 || !this.wsId) return;
    const ping = this.runtime.transport.wsPing;
    if (!ping) return;
    let healthy = false;
    try {
      healthy = await ping.call(this.runtime.transport, this.wsId);
    } catch {
      healthy = false;
    }
    if (healthy && this.readyState === 1) return;
    await this.runtime.transport.refresh().catch(() => false);
    await this.reopen();
  }

  private async reopen(): Promise<void> {
    const staleId = this.wsId;
    this.wsId = 0;
    this.readyState = 0;
    this.unsubscribe?.();
    this.unsubscribe = null;
    if (staleId) void this.runtime.transport.wsClose(staleId);
    await this.open();
  }

  private async open(): Promise<void> {
    try {
      const unsubscribe = await this.runtime.onEvent((event) => this.handleEvent(event));
      if (this.retired) {
        unsubscribe();
        return;
      }
      this.unsubscribe = unsubscribe;
      const wsId = await this.runtime.transport.wsOpen("/mobile/v1/ws");
      if (this.retired) {
        if (wsId) void this.runtime.transport.wsClose(wsId);
        return;
      }
      if (!wsId) throw new Error("mobile socket was not admitted");
      this.wsId = wsId;
      this.readyState = 1;
      this.onopen?.call(this as unknown as WebSocket, new Event("open"));
    } catch {
      if (this.retired) return;
      this.onerror?.call(this as unknown as WebSocket, new Event("error"));
      this.emitClose(1006);
    }
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    if (this.readyState !== 1 || typeof data !== "string") {
      throw new Error("mobile socket accepts text frames only");
    }
    void this.runtime.transport.wsSend(this.wsId, new TextEncoder().encode(data), "text");
  }

  close(): void {
    if (this.retired) return;
    this.retired = true;
    this.readyState = 2;
    const wsId = this.wsId;
    this.wsId = 0;
    if (wsId) void this.runtime.transport.wsClose(wsId);
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.emitClose(1000);
  }

  private handleEvent(event: MobileEvent): void {
    if (this.retired || !this.wsId) return;
    if (event.type === "ws_frame" && event.ws_id === this.wsId) {
      const raw = typeof event.data_base64 === "string" ? decodeBase64(event.data_base64) : "";
      this.onmessage?.call(this as unknown as WebSocket, new MessageEvent("message", { data: raw }));
    } else if (event.type === "ws_closed" && event.ws_id === this.wsId) {
      this.wsId = 0;
      this.emitClose(Number(event.code ?? 1000));
    }
  }

  private emitClose(code: number): void {
    if (this.closeEmitted) return;
    this.closeEmitted = true;
    this.retired = true;
    this.wsId = 0;
    this.readyState = 3;
    const unsubscribe = this.unsubscribe;
    this.unsubscribe = null;
    unsubscribe?.();
    this.onclose?.call(this as unknown as WebSocket, new CloseEvent("close", { code }));
  }
}

function decodeBase64(value: string): string {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new TextDecoder().decode(bytes);
}
