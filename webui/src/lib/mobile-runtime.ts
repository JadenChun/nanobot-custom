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
  logout?(): Promise<void>;
  remoteStatus?(): Promise<Record<string, unknown>>;
  remoteDevices?(): Promise<Record<string, unknown>>;
  remoteRevoke?(installationId: string): Promise<Record<string, unknown>>;
  pickCamera?(): Promise<MobilePickedCamera>;
  wsOpen(path: string, headers?: Record<string, string>): Promise<number>;
  wsSend(wsId: number, data: Uint8Array, kind?: "text" | "binary"): Promise<void>;
  wsClose(wsId: number): Promise<void>;
  wsPing?(wsId: number): Promise<boolean>;
  wsRefreshAuth?(wsId: number): Promise<boolean>;
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
  /** Refresh access and retire live sockets for the owning client's reconnect. */
  renew(): Promise<void>;
  dispose?(): void;
}

let installedMobileRuntime: MobileRuntime | null = null;
let ownerCapability = false;
let standardDispatchCapability = false;

/** Whether the paired device negotiated the owner capability (M30/M31). */
export function mobileOwnerCapability(): boolean {
  return ownerCapability;
}

export function setMobileOwnerCapability(value: boolean): void {
  ownerCapability = value;
}

/** Whether the paired server advertises the standard owner dispatch route. */
export function mobileStandardDispatchCapability(): boolean {
  return standardDispatchCapability;
}

export function setMobileStandardDispatchCapability(value: boolean): void {
  standardDispatchCapability = value;
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
  installedMobileRuntime?.dispose?.();
  installedMobileRuntime = null;
  ownerCapability = false;
  standardDispatchCapability = false;
}

export function shouldRegisterServiceWorker(): boolean {
  return !isMobileRuntimeActive();
}

class MobileRuntimeError extends Error {}

export function createMobileRuntime(transport: MobileTransport): MobileRuntime & { dispose(): void } {
  type Entry = {
    id: number | null;
    done: boolean;
    resolve: (response: MobileHttpResponse) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  };
  const pending = new Map<number, Entry>();
  const admitted = new Set<Entry>();
  // Native returns its request ID after launch. Join terminal events arriving
  // during that interval, within the same finite admission and byte bounds.
  const early = new Map<number, MobileEvent>();
  const maxPending = 64;
  const maxEarlyBytes = 64 * 1024 * 1024;
  let earlyBytes = 0;
  let launching = 0;
  let disposed = false;
  const cancel = (id: number): void => { void transport.cancel(id).catch(() => undefined); };
  const settle = (entry: Entry, event?: MobileEvent, error?: Error): void => {
    if (entry.done) return;
    entry.done = true;
    admitted.delete(entry);
    if (entry.id !== null) pending.delete(entry.id);
    clearTimeout(entry.timer);
    if (error) {
      if (entry.id !== null) cancel(entry.id);
      entry.reject(error);
    } else if (event?.type === "http_error") {
      entry.reject(new MobileRuntimeError(event.code ?? "http_error"));
    } else if (event) {
      entry.resolve({ requestId: entry.id!, status: event.status ?? 0,
        headers: event.headers ?? {}, bodyBase64: event.body_base64 ?? "" });
    }
  };
  const listeners = new Set<(event: MobileEvent) => void>();
  let listening: Promise<() => void> | null = null;
  const sockets = new Set<MobileWebSocket>();
  let recovery: Promise<void> | null = null;
  let renewal: Promise<void> | null = null;
  // Register credential work synchronously when requested. New adapters capture
  // this phase only, never the later wait for ordered native socket cleanup.
  let credentialMaintenance = Promise.resolve();
  let socketWork = Promise.resolve();
  // Keep native admission behind prior closes, including cleanup of an ID
  // returned after its adapter was retired. HTTP traffic is independent.
  const orderSocketWork = (work: () => Promise<void>): Promise<void> => {
    const result = socketWork.then(work);
    // Settlement, not release proof: each adapter reports its own failure and
    // the broker still decides whether a subsequent open can be admitted.
    socketWork = result.catch(() => undefined);
    return result;
  };
  let renewalTimer: ReturnType<typeof setTimeout> | null = null;
  let lastRefreshedAt = 0;

  const refreshCredentials = async (): Promise<boolean> => {
    const ok = await transport.refresh().catch(() => false);
    if (ok) lastRefreshedAt = Date.now();
    return ok;
  };

  const scheduleRenewal = (payload: Record<string, unknown>): void => {
    if (disposed) return;
    const limits = payload.limits;
    const rawTtl = limits && typeof limits === "object"
      ? Number((limits as Record<string, unknown>).access_ttl_seconds)
      : NaN;
    const ttlSeconds = Number.isFinite(rawTtl) && rawTtl >= 120 ? rawTtl : 600;
    if (renewalTimer !== null) clearTimeout(renewalTimer);
    renewalTimer = setTimeout(() => {
      renewalTimer = null;
      void runtime.renew().finally(() => scheduleRenewal(payload));
    }, Math.max(60_000, Math.floor(ttlSeconds * 0.7) * 1000));
  };

  const ensureListener = (): Promise<() => void> => {
    if (listening === null) {
      listening = transport.onEvent((event) => {
        if (disposed) return;
        if ((event.type === "http_response" || event.type === "http_error")
          && typeof event.request_id === "number") {
          const entry = pending.get(event.request_id);
          if (entry) settle(entry, event);
          else if (launching > 0 && !early.has(event.request_id)) {
            const bytes = (event.body_base64?.length ?? 0) + (event.message?.length ?? 0) + 8192;
            if (early.size < maxPending && earlyBytes + bytes <= maxEarlyBytes) {
              early.set(event.request_id, event);
              earlyBytes += bytes;
            }
          }
        }
        for (const listener of listeners) listener(event);
      });
    }
    return listening;
  };

  const execute = (input: MobileHttpRequest, timeoutMs: number): Promise<MobileHttpResponse> => {
    if (disposed) return Promise.reject(new MobileRuntimeError("mobile runtime disposed"));
    if (admitted.size >= maxPending) return Promise.reject(new MobileRuntimeError("mobile request queue is full"));
    return new Promise<MobileHttpResponse>((resolve, reject) => {
      const entry: Entry = { id: null, done: false, resolve, reject,
        timer: setTimeout(() => settle(entry, undefined, new MobileRuntimeError("mobile request timed out")), timeoutMs) };
      admitted.add(entry);
      void (async () => {
        try {
          await ensureListener();
          if (entry.done || disposed) return;
          launching += 1;
          try {
            const id = await transport.http(input);
            if (!id) throw new MobileRuntimeError("mobile request was not admitted");
            entry.id = id;
            const event = early.get(id);
            if (event) {
              earlyBytes -= (event.body_base64?.length ?? 0) + (event.message?.length ?? 0) + 8192;
              early.delete(id);
            }
            if (entry.done || disposed) { cancel(id); return; }
            if (pending.has(id)) throw new MobileRuntimeError("duplicate mobile request ID");
            pending.set(id, entry);
            if (event) settle(entry, event);
          } finally {
            launching -= 1;
            if (launching === 0) { early.clear(); earlyBytes = 0; }
          }
        } catch (error) {
          settle(entry, undefined, error instanceof Error ? error : new MobileRuntimeError(String(error)));
        }
      })();
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

  const runtime: MobileRuntime & { dispose(): void } = {
    transport,
    request,
    dispose() {
      if (disposed) return;
      disposed = true;
      if (renewalTimer !== null) clearTimeout(renewalTimer);
      for (const entry of [...admitted]) settle(entry, undefined, new MobileRuntimeError("mobile runtime disposed"));
      early.clear(); earlyBytes = 0;
      listeners.clear();
      for (const socket of sockets) socket.close();
      sockets.clear();
      if (listening) void listening.then((remove) => remove()).catch(() => undefined);
    },
    async handshake() {
      // Rotate the access token unless a recent refresh already succeeded
      // (e.g., during renew()/recover() right before reconnect), avoiding
      // redundant double-rotation round-trips.
      if (Date.now() - lastRefreshedAt >= 15_000) {
        await refreshCredentials();
      }
      const response = await request({ operation: "mobile_handshake", method: "GET", path: "/mobile/v1/handshake" });
      if (response.status !== 200) throw new MobileRuntimeError(`handshake_status_${response.status}`);
      const payload = JSON.parse(decodeBase64(response.bodyBase64)) as Record<string, unknown>;
      const capabilities = payload.capabilities;
      setMobileOwnerCapability(
        typeof capabilities === "object"
          && capabilities !== null
          && (capabilities as Record<string, unknown>).owner === true,
      );
      setMobileStandardDispatchCapability(
        payload.request_settings_version === 1
        || (
          typeof capabilities === "object"
          && capabilities !== null
          && (capabilities as Record<string, unknown>).standard_dispatch === true
        ),
      );
      scheduleRenewal(payload);
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
      if (disposed) throw new MobileRuntimeError("mobile runtime disposed");
      const socket = new MobileWebSocket(runtime, url, orderSocketWork, credentialMaintenance);
      sockets.add(socket);
      return socket as unknown as WebSocket;
    },
    async recover() {
      if (disposed) return;
      if (recovery !== null) return recovery;
      const active = [...sockets].filter((socket) => !socket.isRetired());
      const work = credentialMaintenance.then(async () => {
        if (disposed) return;
        for (const socket of active) {
          await socket.recover();
          if (socket.isRetired()) sockets.delete(socket);
        }
      });
      credentialMaintenance = work.catch(() => undefined);
      recovery = work.then(async () => {
        for (const socket of active) await socket.waitForClose();
      }).finally(() => {
        recovery = null;
      });
      return recovery;
    },
    async renew() {
      if (disposed) return;
      if (renewal !== null) return renewal;
      // Capture adapter generations before refresh/serialization. A client
      // replacement created during either wait does not belong to this repair.
      const active = [...sockets].filter((socket) => !socket.isRetired());
      const work = credentialMaintenance.then(async () => {
        if (disposed || active.every((socket) => socket.isRetired())) return;
        const refreshed = await refreshCredentials();
        if (!refreshed || disposed) return;
        let keptOpen = false;
        for (const socket of active) {
          const rotatedInBand = await socket.refreshAuthInBand();
          if (rotatedInBand) {
            keptOpen = true;
            continue;
          }
          socket.close();
          if (socket.isRetired()) sockets.delete(socket);
        }
        if (keptOpen && !disposed) {
          scheduleRenewal({});
        }
      });
      credentialMaintenance = work.catch(() => undefined);
      renewal = work.then(async () => {
        for (const socket of active) await socket.waitForClose();
      }).finally(() => {
        renewal = null;
      });
      return renewal;
    },
  };
  return runtime;
}

type MobileSocketEvent = {
  type: "ws_frame" | "ws_closed";
  wsId: number;
  payload: string;
  payloadBytes: number;
  code: number;
};

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
  private errorEmitted = false;
  private openingEvents = true;
  private earlyEvents: MobileSocketEvent[] = [];
  private earlyBytes = 0;
  private opening: Promise<void>;
  private closing: Promise<void> | null = null;

  constructor(
    private runtime: MobileRuntime,
    url: string,
    private orderSocketWork: (work: () => Promise<void>) => Promise<void>,
    private credentialsReady: Promise<void>,
  ) {
    this.url = url;
    const parsed = new URL(url);
    if (parsed.pathname !== "/mobile/v1/ws") {
      throw new Error("mobile socket path is not allowlisted");
    }
    this.opening = orderSocketWork(() => this.open()).catch(() => this.failTransport());
  }

  isRetired(): boolean {
    return this.retired;
  }

  async refreshAuthInBand(): Promise<boolean> {
    if (this.retired || this.readyState !== 1 || !this.wsId) return false;
    const refreshAuth = this.runtime.transport.wsRefreshAuth;
    if (!refreshAuth) return false;
    try {
      return Boolean(await refreshAuth.call(this.runtime.transport, this.wsId));
    } catch {
      return false;
    }
  }

  async recover(): Promise<void> {
    if (this.retired) return;
    const wsId = this.wsId;
    let confirmedDead = false;
    if (this.readyState === 1 && wsId) {
      const ping = this.runtime.transport.wsPing;
      if (!ping) return;
      const healthy = await ping.call(this.runtime.transport, wsId).catch(() => false);
      if (this.retired || this.wsId !== wsId) return;
      if (healthy) return;
      confirmedDead = true;
    }
    const refreshed = await this.runtime.transport.refresh().catch(() => false);
    if (this.retired) return;
    // If ping already proved the native socket is dead, close it even when
    // refresh fails (e.g., Tailscale still waking up) so the client transitions
    // to reconnecting instead of staying stuck in a false "open" zombie state.
    if (!refreshed && !confirmedDead) return;
    this.close();
  }

  private async open(): Promise<void> {
    if (this.retired) return;
    await this.credentialsReady;
    if (this.retired) return;
    const unsubscribe = await this.runtime.onEvent((event) => this.handleEvent(event));
    if (this.retired) {
      unsubscribe();
      return;
    }
    this.unsubscribe = unsubscribe;
    const wsId = await this.runtime.transport.wsOpen("/mobile/v1/ws");
    if (!Number.isSafeInteger(wsId) || wsId <= 0) throw new Error("mobile socket was not admitted");
    if (this.retired) {
      // This cleanup is part of the opening barrier, not queued behind itself.
      await this.runtime.transport.wsClose(wsId);
      return;
    }
    this.wsId = wsId;
    this.readyState = 1;
    this.onopen?.call(this as unknown as WebSocket, new Event("open"));
    // Keep buffering through onopen so any synchronous native responses remain
    // behind frames that arrived while the ID was still unknown.
    while (!this.retired && this.earlyEvents.length > 0) {
      const event = this.earlyEvents.shift()!;
      this.earlyBytes -= event.payloadBytes;
      this.deliverEvent(event);
    }
    this.openingEvents = false;
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    if (this.readyState !== 1 || typeof data !== "string") {
      throw new Error("mobile socket accepts text frames only");
    }
    const wsId = this.wsId;
    // Native queue acceptance is not server admission or delivery proof. A
    // refusal retires only this adapter; the client retains outcome uncertainty.
    void this.runtime.transport
      .wsSend(wsId, new TextEncoder().encode(data), "text")
      .catch(() => {
        if (!this.retired && this.wsId === wsId) this.failTransport();
      });
  }

  /** Cleanup must not hold the credential phase that a queued replacement needs. */
  async waitForClose(): Promise<void> {
    await this.opening;
    await this.closing;
  }

  private failTransport(): void {
    this.emitError();
    this.retire(1006);
  }

  close(): void {
    this.retire(1000);
  }

  private retire(code: number): void {
    if (this.retired) return;
    this.retired = true;
    this.readyState = 2;
    const wsId = this.wsId;
    this.wsId = 0;
    if (wsId) {
      this.closing = this.orderSocketWork(() => this.runtime.transport.wsClose(wsId))
        .catch(() => this.emitError());
    }
    this.emitClose(code);
  }

  private handleEvent(event: MobileEvent): void {
    if (this.retired || (event.type !== "ws_frame" && event.type !== "ws_closed")
      || typeof event.ws_id !== "number" || !Number.isSafeInteger(event.ws_id) || event.ws_id <= 0) return;
    const socketEvent: MobileSocketEvent = {
      type: event.type,
      wsId: event.ws_id,
      payload: event.type === "ws_frame" && typeof event.data_base64 === "string" ? event.data_base64 : "",
      payloadBytes: 0,
      code: Number(event.code ?? 1000),
    };
    if (this.openingEvents) {
      const remaining = 4 * 1024 * 1024 - this.earlyBytes;
      // Valid base64 is ASCII; malformed input must not bypass the byte bound.
      socketEvent.payloadBytes = socketEvent.payload.length > remaining
        ? socketEvent.payload.length : new TextEncoder().encode(socketEvent.payload).byteLength;
      if (this.earlyEvents.length >= 64 || socketEvent.payloadBytes > remaining) {
        this.failTransport();
        return;
      }
      this.earlyEvents.push(socketEvent);
      this.earlyBytes += socketEvent.payloadBytes;
      return;
    }
    this.deliverEvent(socketEvent);
  }

  private deliverEvent(event: MobileSocketEvent): void {
    if (this.retired || event.wsId !== this.wsId) return;
    if (event.type === "ws_frame") {
      try {
        const raw = decodeBase64(event.payload);
        this.onmessage?.call(this as unknown as WebSocket, new MessageEvent("message", { data: raw }));
      } catch {
        this.failTransport();
      }
    } else {
      this.wsId = 0;
      this.emitClose(event.code);
    }
  }

  private emitError(): void {
    if (this.errorEmitted) return;
    this.errorEmitted = true;
    this.onerror?.call(this as unknown as WebSocket, new Event("error"));
  }

  private emitClose(code: number): void {
    if (this.closeEmitted) return;
    this.closeEmitted = true;
    this.retired = true;
    this.wsId = 0;
    this.readyState = 3;
    this.earlyEvents = [];
    this.earlyBytes = 0;
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
