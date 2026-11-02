import { Capacitor, registerPlugin, type PluginListenerHandle } from '@capacitor/core';

export type NativeOperation =
  | 'fixture_bootstrap'
  | 'fixture_read'
  | 'fixture_refresh'
  | 'fixture_mutation'
  | 'fixture_counter'
  | 'fixture_media'
  | 'fixture_attachment';

export type NativeMethod = 'GET' | 'POST';

export interface NativeHttpRequest {
  operation: NativeOperation;
  method: NativeMethod;
  path: string;
  headers?: Record<string, string>;
  bodyBase64?: string;
  timeoutMs?: number;
}

export interface NativeNodeStateEvent {
  type: 'node_state';
  state: 'new' | 'starting' | 'needs_login' | 'ready' | 'stopping' | 'stopped' | 'error';
  login_url?: string;
  message?: string;
  tailscale_ips?: string[];
}

export interface NativeHttpResponseEvent {
  type: 'http_response';
  request_id: number;
  status: number;
  headers: Record<string, string>;
  body_base64: string;
  bytes: number;
}

export interface NativeHttpErrorEvent {
  type: 'http_error';
  request_id: number;
  code: 'cancelled' | 'timeout' | 'tls' | 'network' | 'too_large' | 'redirect_denied' | 'invalid' | 'unavailable';
  message: string;
  status: number;
}

export interface NativeWsOpenEvent {
  type: 'ws_open';
  ws_id: number;
}

export interface NativeWsFrameEvent {
  type: 'ws_frame';
  ws_id: number;
  kind: 'text' | 'binary';
  data_base64: string;
  bytes: number;
}

export interface NativeWsClosedEvent {
  type: 'ws_closed';
  ws_id: number;
  code: number;
  reason: string;
}

export interface NativeWsErrorEvent {
  type: 'ws_error';
  ws_id: number;
  code: 'handshake' | 'too_large' | 'queue_full' | 'write_failed';
  message?: string;
  status?: number;
}

export type NativeNetworkEvent =
  | NativeNodeStateEvent
  | NativeHttpResponseEvent
  | NativeHttpErrorEvent
  | NativeWsOpenEvent
  | NativeWsFrameEvent
  | NativeWsClosedEvent
  | NativeWsErrorEvent;

interface NativeNetworkPlugin {
  abiVersion(): Promise<{ version: number }>;
  start(options: { origin: string; installationId: string; hostname: string }): Promise<{ state: string }>;
  status(): Promise<NativeNodeStateEvent & { origin: string; installation_id: string }>;
  http(options: NativeHttpRequest): Promise<{ requestId: number }>;
  cancel(options: { requestId: number }): Promise<void>;
  wsOpen(options: { operation: 'fixture_ws'; path: string; headers?: Record<string, string> }): Promise<{ wsId: number }>;
  wsSend(options: { wsId: number; kind: 'text' | 'binary'; dataBase64: string }): Promise<void>;
  wsSendBegin(options: { wsId: number; kind: 'text' | 'binary' }): Promise<void>;
  wsSendChunk(options: { wsId: number; dataBase64: string }): Promise<void>;
  wsSendEnd(options: { wsId: number; discard?: boolean }): Promise<void>;
  wsClose(options: { wsId: number }): Promise<void>;
  openLogin(): Promise<void>;
  stop(): Promise<void>;
  addListener(
    eventName: 'networkEvent',
    listener: (event: NativeNetworkEvent) => void,
  ): Promise<PluginListenerHandle>;
}

const plugin = registerPlugin<NativeNetworkPlugin>('NativeNetwork');

function requireNative(): void {
  if (!Capacitor.isNativePlatform()) {
    throw new Error('NativeNetwork is available only in the packaged Android client');
  }
}

export const nativeBroker = {
  async start(origin: string, installationId: string, hostname: string): Promise<void> {
    requireNative();
    await plugin.start({ origin, installationId, hostname });
  },

  async status(): Promise<NativeNodeStateEvent & { origin: string; installation_id: string }> {
    requireNative();
    return plugin.status();
  },

  async http(request: NativeHttpRequest): Promise<number> {
    requireNative();
    const response = await plugin.http(request);
    return response.requestId;
  },

  async cancel(requestId: number): Promise<void> {
    requireNative();
    await plugin.cancel({ requestId });
  },

  async wsOpen(path = '/fixture/v1/ws', headers?: Record<string, string>): Promise<number> {
    requireNative();
    const response = await plugin.wsOpen({ operation: 'fixture_ws', path, headers });
    return response.wsId;
  },

  async wsSend(wsId: number, data: Uint8Array, kind: 'text' | 'binary' = 'text'): Promise<void> {
    requireNative();
    const chunkBytes = 256 * 1024;
    if (data.length <= chunkBytes) {
      await plugin.wsSend({ wsId, kind, dataBase64: bytesToBase64(data) });
      return;
    }
    await plugin.wsSendBegin({ wsId, kind });
    try {
      for (let offset = 0; offset < data.length; offset += chunkBytes) {
        const chunk = data.subarray(offset, Math.min(offset + chunkBytes, data.length));
        await plugin.wsSendChunk({ wsId, dataBase64: bytesToBase64(chunk) });
      }
      await plugin.wsSendEnd({ wsId });
    } catch (error) {
      await plugin.wsSendEnd({ wsId, discard: true }).catch(() => undefined);
      throw error;
    }
  },

  async wsClose(wsId: number): Promise<void> {
    requireNative();
    await plugin.wsClose({ wsId });
  },

  async openLogin(): Promise<void> {
    requireNative();
    await plugin.openLogin();
  },

  async stop(): Promise<void> {
    requireNative();
    await plugin.stop();
  },

  async onEvent(listener: (event: NativeNetworkEvent) => void): Promise<PluginListenerHandle> {
    requireNative();
    return plugin.addListener('networkEvent', listener);
  },
};

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

export function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
