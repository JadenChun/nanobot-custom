import { fetchWithTimeout } from "./http";

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

export interface RemoteAccessStatus {
  enabled: boolean;
  installation_id: string;
  devices: RemoteAccessDevice[];
}

export interface RemotePairingOffer {
  pairing_id: string;
  comparison_code: string;
  nonce: string;
  collector_secret: string;
  expires_at: number;
  expiration?: number;
  scopes?: string[];
}

export interface RemotePairingApproval {
  pairing_id?: string;
  installation_id?: string;
  scopes?: string[];
}

/** Build the phone deep link for one owner-issued pairing challenge. */
export function buildMobilePairingLink(
  origin: string,
  installationId: string,
  offer: RemotePairingOffer,
): string {
  const challenge = `${offer.pairing_id}.${offer.nonce}.${offer.collector_secret}`;
  const params = new URLSearchParams({
    origin: origin.trim().replace(/\/+$/, ""),
    installation: installationId.trim(),
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

export async function fetchRemoteAccessStatus(token: string): Promise<RemoteAccessStatus> {
  return request<RemoteAccessStatus>(token, "/api/settings/remote-access");
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
  await transport.requestMutation("settings.remote_access.revoke", {
    installation_id: installationId,
  });
}

export async function startRemotePairing(
  transport: RemoteAccessMutationTransport,
  input: {
    origin: string;
    label: string;
    chatConsent: boolean;
    mediaConsent: boolean;
    fullControlConsent: boolean;
    sharedHistoryConsent: boolean;
  },
): Promise<RemotePairingOffer> {
  return transport.requestMutation<RemotePairingOffer>(
    "settings.remote_access.pairing.start",
    input,
  );
}

export async function approveRemotePairing(
  transport: RemoteAccessMutationTransport,
  input: { pairingId: string; comparisonCode: string },
): Promise<RemotePairingApproval> {
  return transport.requestMutation<RemotePairingApproval>(
    "settings.remote_access.pairing.approve",
    input,
  );
}
