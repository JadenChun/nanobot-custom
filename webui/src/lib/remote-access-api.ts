import { fetchWithTimeout } from "./http";
import { getMobileRuntime, mobileOwnerCapability } from "./mobile-runtime";

export interface RemoteAccessMutationTransport {
  requestMutation<T>(action: string, payload: Record<string, unknown>): Promise<T>;
}

export interface RemoteAccessDevice {
  installation_id: string;
  label: string;
  scopes: string[];
  status: "active" | "revoked";
  created_at: number;
  last_seen_at: number;
}

export interface RemoteTailscaleStatus {
  available: boolean;
  connected: boolean;
  serve_mapped: boolean;
  dns_name: string;
  origin: string;
  error?: string;
}

export interface RemoteAccessStatus {
  enabled: boolean;
  installation_id: string;
  devices: RemoteAccessDevice[];
  owner_available?: boolean;
}

export interface RemotePairingOffer {
  pairing_id: string;
  pairing_code?: string;
  comparison_code: string;
  nonce: string;
  collector_secret: string;
  expires_at: number;
  expires_in?: number;
  expiration?: number;
  scopes?: string[];
  transport?: "tailscale";
  origin?: string;
  qr_link?: string;
}

export interface RemotePairingPending {
  pairing_id: string;
  state:
    | "offer_created"
    | "pending_approval"
    | "approved"
    | "denied"
    | "expired";
  expires_at: number;
  expires_in?: number;
  requested_at?: number;
  pairing_code?: string;
  origin?: string;
  created_at?: number;
  renewable?: boolean;
  transport?: "tailscale";
  scopes?: string[];
  device_fingerprint?: string;
  device_generation?: number;
  device?: {
    installation_id: string;
    label: string;
    platform?: string;
    model?: string;
  };
}

export interface RemotePairingApproval {
  pairing_id?: string;
  installation_id?: string;
  scopes?: string[];
}

/** Build the phone deep link for one owner-issued pairing challenge. */
export function buildMobilePairingLink(
  origin: string,
  offer: RemotePairingOffer,
): string {
  const challenge = `${offer.pairing_id}.${offer.nonce}.${offer.collector_secret}`;
  const params = new URLSearchParams({
    origin: origin.trim().replace(/\/+$/, ""),
    challenge,
  });
  return `nanobot://pair?${params.toString()}`;
}

async function request<T>(
  token: string,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetchWithTimeout(path, {
    ...init,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(`remote access request failed: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

function mobileOwnerTransport() {
  const runtime = getMobileRuntime();
  if (!runtime || !mobileOwnerCapability()) return null;
  return runtime.transport;
}

async function ownerWithRefresh<T>(
  operation: () => Promise<T>,
  refresh: () => Promise<boolean>,
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (!(error instanceof Error) || !error.message.includes("HTTP 401")) throw error;
    const refreshed = await refresh().catch(() => false);
    if (!refreshed) throw error;
    return operation();
  }
}

function ownerDeviceRows(value: unknown): RemoteAccessDevice[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item !== "object" || item === null) return [];
    const row = item as Record<string, unknown>;
    const installationId = typeof row.installation_id === "string"
      ? row.installation_id
      : "";
    if (!installationId) return [];
    return [{
      installation_id: installationId,
      label: typeof row.label === "string" ? row.label : "",
      scopes: Array.isArray(row.scopes)
        ? row.scopes.filter((scope): scope is string => typeof scope === "string")
        : [],
      status: row.status === "revoked" ? "revoked" as const : "active" as const,
      created_at: typeof row.created_at === "number" ? row.created_at : 0,
      last_seen_at: typeof row.last_seen_at === "number" ? row.last_seen_at : 0,
    }];
  });
}

export async function fetchRemoteAccessStatus(token: string): Promise<RemoteAccessStatus> {
  const transport = mobileOwnerTransport();
  if (transport?.remoteStatus && transport.remoteDevices) {
    const [status, devices] = await Promise.all([
      ownerWithRefresh(
        () => transport.remoteStatus!(),
        () => transport.refresh(),
      ),
      ownerWithRefresh(
        () => transport.remoteDevices!(),
        () => transport.refresh(),
      ),
    ]);
    return {
      enabled: status.enabled === true,
      installation_id: typeof status.installation_id === "string"
        ? status.installation_id
        : "",
      devices: ownerDeviceRows(devices.devices),
      owner_available: true,
    };
  }
  return request<RemoteAccessStatus>(token, "/api/settings/remote-access");
}

export async function fetchRemoteTailscaleStatus(
  token: string,
): Promise<RemoteTailscaleStatus> {
  return request<RemoteTailscaleStatus>(token, "/api/settings/remote-access/tailscale");
}

export async function fetchRemotePairingPendingList(
  token: string,
): Promise<RemotePairingPending[]> {
  try {
    const response = await request<
      RemotePairingPending | { pending?: RemotePairingPending[] }
    >(token, "/api/settings/remote-access/pairing/pending");
    if ("pending" in response && Array.isArray(response.pending)) {
      return response.pending;
    }
    return [response as RemotePairingPending];
  } catch (exception) {
    // Older servers do not expose the pending route; the waiting screen still
    // works through the active offer held by this component.
    if (exception instanceof Error && exception.message.includes("HTTP 404")) {
      return [];
    }
    throw exception;
  }
}

export async function fetchRemotePairingPending(
  token: string,
  pairingId?: string,
): Promise<RemotePairingPending | null> {
  const pending = await fetchRemotePairingPendingList(token);
  if (pairingId) {
    return pending.find((item) => item.pairing_id === pairingId) ?? null;
  }
  return pending[0] ?? null;
}

export async function setRemoteAccessEnabled(
  transport: RemoteAccessMutationTransport,
  enabled: boolean,
): Promise<boolean> {
  const result = await transport.requestMutation<{ enabled: boolean }>(
    "settings.remote_access.enable",
    { enabled },
  );
  return result.enabled;
}

export async function revokeRemoteDevice(
  transport: RemoteAccessMutationTransport,
  installationId: string,
): Promise<void> {
  try {
    // A live gateway owns in-process Full consents, so prefer its revoke path
    // when the socket is available. It clears those consents through the
    // existing callback.
    await transport.requestMutation("settings.remote_access.revoke", {
      installation_id: installationId,
    });
    return;
  } catch (error) {
    const owner = mobileOwnerTransport();
    if (!owner?.remoteRevoke) throw error;
    // Gateway stopped/unreachable: the independent owner remains authoritative.
    await ownerWithRefresh(
      () => owner.remoteRevoke!(installationId),
      () => owner.refresh(),
    );
  }
}

export async function startRemotePairing(
  transport: RemoteAccessMutationTransport,
  input: {
    origin: string;
    label: string;
    mediaConsent?: boolean;
    fullControlConsent?: boolean;
    sharedHistoryConsent?: boolean;
  },
): Promise<RemotePairingOffer> {
  return transport.requestMutation<RemotePairingOffer>(
    "settings.remote_access.pairing.start",
    {
      origin: input.origin,
      label: input.label,
      chatConsent: true,
      mediaConsent: input.mediaConsent ?? true,
      fullControlConsent: input.fullControlConsent ?? true,
      sharedHistoryConsent: input.sharedHistoryConsent ?? true,
    },
  );
}

export async function renewRemotePairing(
  transport: RemoteAccessMutationTransport,
  pairingId: string,
): Promise<RemotePairingPending> {
  return transport.requestMutation<RemotePairingPending>(
    "settings.remote_access.pairing.renew",
    { pairingId },
  );
}

export async function approveRemotePairing(
  transport: RemoteAccessMutationTransport,
  input: { pairingId: string; comparisonCode?: string },
): Promise<RemotePairingApproval> {
  return transport.requestMutation<RemotePairingApproval>(
    "settings.remote_access.pairing.approve",
    {
      pairingId: input.pairingId,
      ...(input.comparisonCode ? { comparisonCode: input.comparisonCode } : {}),
    },
  );
}

export async function denyRemotePairing(
  transport: RemoteAccessMutationTransport,
  pairingId: string,
): Promise<void> {
  await transport.requestMutation("settings.remote_access.pairing.deny", {
    pairingId,
  });
}
