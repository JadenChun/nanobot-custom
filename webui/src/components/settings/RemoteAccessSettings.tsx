import { useCallback, useEffect, useState } from "react";

import {
  approveRemotePairing,
  fetchRemoteAccessStatus,
  revokeRemoteDevice,
  setRemoteAccessEnabled,
  startRemotePairing,
  type RemoteAccessStatus,
} from "@/lib/remote-access-api";

interface RemoteAccessSettingsProps {
  token: string;
  initialStatus?: RemoteAccessStatus | null;
}

export function RemoteAccessSettings({ token, initialStatus = null }: RemoteAccessSettingsProps) {
  const [status, setStatus] = useState<RemoteAccessStatus | null>(initialStatus);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [origin, setOrigin] = useState("");
  const [label, setLabel] = useState("phone");
  const [pairingId, setPairingId] = useState("");
  const [code, setCode] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchRemoteAccessStatus(token));
      setError(null);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : "remote access unavailable");
    }
  }, [token]);

  useEffect(() => {
    if (!initialStatus) void refresh();
  }, [initialStatus, refresh]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
      await refresh();
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : "remote access action failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="space-y-6" aria-labelledby="remote-access-heading">
      <header>
        <h1 id="remote-access-heading" className="text-lg font-medium">Remote access</h1>
        <p className="text-sm text-muted-foreground">
          Manage this installation&apos;s phone access. Changes stay on this server.
        </p>
      </header>
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
      <div className="flex items-center justify-between rounded-xl border p-4">
        <div>
          <p className="text-sm font-medium">Remote access</p>
          <p className="text-xs text-muted-foreground">
            {status?.enabled ? "Enabled for approved devices" : "Disabled"}
          </p>
        </div>
        <button
          type="button"
          disabled={busy || !status}
          onClick={() => void run(async () => {
            await setRemoteAccessEnabled(token, !status?.enabled);
          })}
          className="rounded-md border px-3 py-1.5 text-sm"
        >
          {status?.enabled ? "Disable" : "Enable"}
        </button>
      </div>

      <div className="space-y-2">
        <h2 className="text-sm font-medium">Devices</h2>
        {status?.devices.length ? (
          <ul className="divide-y rounded-xl border">
            {status.devices.map((device) => (
              <li key={device.installation_id} className="flex items-center justify-between gap-3 p-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{device.label}</p>
                  <p className="truncate text-xs text-muted-foreground">
                    {device.status} · {device.scopes.join(", ")}
                  </p>
                </div>
                {device.status === "active" ? (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void run(async () => {
                      await revokeRemoteDevice(token, device.installation_id);
                    })}
                    className="rounded-md border px-3 py-1 text-xs"
                  >
                    Revoke
                  </button>
                ) : null}
              </li>
            ))}
          </ul>
        ) : <p className="text-sm text-muted-foreground">No paired devices.</p>}
      </div>

      <div className="space-y-3">
        <h2 className="text-sm font-medium">Pair a phone</h2>
        <label className="block text-xs">
          Exact HTTPS origin
          <input
            aria-label="Exact HTTPS origin"
            value={origin}
            onChange={(event) => setOrigin(event.target.value)}
            className="mt-1 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
          />
        </label>
        <label className="block text-xs">
          Label
          <input
            aria-label="Device label"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            className="mt-1 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
          />
        </label>
        <button
          type="button"
          disabled={busy || !origin.trim()}
          onClick={() => void run(async () => {
            const offer = await startRemotePairing(token, {
              origin: origin.trim(),
              label,
              chatConsent: true,
              mediaConsent: false,
            });
            setPairingId(offer.pairing_id);
            setCode(offer.comparison_code);
          })}
          className="rounded-md border px-3 py-1.5 text-sm"
        >
          Create pairing challenge
        </button>
        {pairingId ? (
          <div className="grid gap-2 rounded-xl border p-3 text-sm">
            <p>Comparison code: <strong>{code}</strong></p>
            <label className="text-xs">
              Comparison code from the phone
              <input
                aria-label="Approval comparison code"
                value={code}
                onChange={(event) => setCode(event.target.value)}
                className="mt-1 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
              />
            </label>
            <button
              type="button"
              disabled={busy}
              onClick={() => void run(async () => {
                await approveRemotePairing(token, { pairingId, comparisonCode: code });
                setPairingId("");
              })}
              className="w-fit rounded-md border px-3 py-1.5 text-sm"
            >
              Approve pairing
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
