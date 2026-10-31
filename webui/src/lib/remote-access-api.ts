import { fetchWithTimeout } from "./http";

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
  expiration?: number;
  scopes?: string[];
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
      "X-Nanobot-Auth": token,
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

export async function setRemoteAccessEnabled(token: string, enabled: boolean): Promise<boolean> {
  const result = await request<{ enabled: boolean }>(token, "/api/settings/remote-access/enable", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
  return result.enabled;
}

export async function revokeRemoteDevice(token: string, installationId: string): Promise<void> {
  await request(token, "/api/settings/remote-access/revoke", {
    method: "POST",
    body: JSON.stringify({ installation_id: installationId }),
  });
}

export async function startRemotePairing(
  token: string,
  input: { origin: string; label: string; chatConsent: boolean; mediaConsent: boolean },
): Promise<RemotePairingOffer> {
  return request<RemotePairingOffer>(token, "/api/settings/remote-access/pairing/start", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function approveRemotePairing(
  token: string,
  input: { pairingId: string; comparisonCode: string },
): Promise<void> {
  await request(token, "/api/settings/remote-access/pairing/approve", {
    method: "POST",
    body: JSON.stringify(input),
  });
}
