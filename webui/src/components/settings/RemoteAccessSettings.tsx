import QRCode from "qrcode";
import { useCallback, useEffect, useState } from "react";

import {
  approveRemotePairing,
  buildMobilePairingLink,
  fetchRemoteAccessStatus,
  revokeRemoteDevice,
  setRemoteAccessEnabled,
  startRemotePairing,
  type RemoteAccessStatus,
} from "@/lib/remote-access-api";

interface RemoteAccessSettingsProps {
  token: string;
  initialStatus?: RemoteAccessStatus | null;
  /** Initial enable/re-enable and new-device approval are server-local only. */
  localSetupAvailable?: boolean;
}

export function RemoteAccessSettings({
  token,
  initialStatus = null,
  localSetupAvailable = true,
}: RemoteAccessSettingsProps) {
  const [status, setStatus] = useState<RemoteAccessStatus | null>(initialStatus);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [origin, setOrigin] = useState("");
  const [installationId, setInstallationId] = useState("");
  const [label, setLabel] = useState("phone");
  const [pairingId, setPairingId] = useState("");
  const [code, setCode] = useState("");
  const [approvalCode, setApprovalCode] = useState("");
  const [mediaConsent, setMediaConsent] = useState(false);
  const [pairingLink, setPairingLink] = useState("");
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState<{ installationId: string; label: string } | null>(null);

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

  useEffect(() => {
    if (!origin && typeof window !== "undefined" && window.location.protocol === "https:") {
      setOrigin(window.location.origin);
    }
  }, [origin]);

  useEffect(() => {
    if (!pairingLink) {
      setQrDataUrl("");
      return;
    }
    let cancelled = false;
    void QRCode.toDataURL(pairingLink, {
      width: 224,
      margin: 1,
      color: { dark: "#111827", light: "#ffffff" },
    }).then((url) => {
      if (!cancelled) setQrDataUrl(url);
    }).catch(() => {
      if (!cancelled) setQrDataUrl("");
    });
    return () => {
      cancelled = true;
    };
  }, [pairingLink]);

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
        {localSetupAvailable ? (
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
        ) : (
          <span className="text-xs text-muted-foreground">Local server only</span>
        )}
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
                    onClick={() => setRevoking({
                      installationId: device.installation_id,
                      label: device.label,
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
        {revoking ? (
          <div
            role="alertdialog"
            aria-label={`Revoke ${revoking.label}`}
            className="rounded-xl border border-destructive/40 bg-destructive/5 p-3"
          >
            <p className="text-sm font-medium">Revoke {revoking.label}?</p>
            <p className="mt-1 text-xs text-muted-foreground">
              This device loses access immediately. It must be paired again locally.
            </p>
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                className="rounded-md border px-3 py-1 text-xs"
                onClick={() => setRevoking(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="rounded-md border border-destructive/50 px-3 py-1 text-xs text-destructive"
                disabled={busy}
                onClick={() => void run(async () => {
                  await revokeRemoteDevice(token, revoking.installationId);
                  setRevoking(null);
                })}
              >
                Confirm revoke
              </button>
            </div>
          </div>
        ) : null}
      </div>

      {!localSetupAvailable ? (
        <p className="rounded-xl border p-3 text-sm text-muted-foreground">
          New-device pairing and remote on/off stay on the server-local desktop or CLI.
          Existing devices can still be reviewed and revoked here.
        </p>
      ) : (
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
          Phone installation ID (optional)
          <input
            aria-label="Phone installation ID"
            value={installationId}
            onChange={(event) => setInstallationId(event.target.value)}
            placeholder="Leave blank to generate one for the QR"
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
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            aria-label="Allow chat media and attachments"
            checked={mediaConsent}
            onChange={(event) => setMediaConsent(event.target.checked)}
          />
          Allow chat media and attachments
        </label>
        <button
          type="button"
          disabled={busy || !origin.trim()}
          onClick={() => void run(async () => {
            const offer = await startRemotePairing(token, {
              origin: origin.trim(),
              label,
              chatConsent: true,
              mediaConsent,
            });
            const targetInstallation = installationId.trim() || crypto.randomUUID();
            setInstallationId(targetInstallation);
            setPairingId(offer.pairing_id);
            setCode(offer.comparison_code);
            setApprovalCode("");
            setPairingLink(buildMobilePairingLink(origin, targetInstallation, offer));
          })}
          className="rounded-md border px-3 py-1.5 text-sm"
        >
          Create pairing challenge
        </button>
        {pairingLink ? (
          <div className="grid gap-2 rounded-xl border p-3 text-sm">
            <p className="text-xs text-muted-foreground">
              Open this link on the phone. Keep it private until the codes match.
            </p>
            {qrDataUrl ? (
              <img
                src={qrDataUrl}
                alt="Phone pairing QR code"
                className="h-44 w-44 rounded-lg border bg-white p-2"
              />
            ) : null}
            <textarea
              aria-label="Phone pairing link"
              readOnly
              value={pairingLink}
              rows={4}
              className="w-full break-all rounded-md border bg-transparent px-2 py-1.5 font-mono text-xs"
            />
            <button
              type="button"
              onClick={() => {
                void navigator.clipboard?.writeText(pairingLink).then(() => {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1500);
                }).catch(() => undefined);
              }}
              className="w-fit rounded-md border px-3 py-1.5 text-sm"
            >
              {copied ? "Copied" : "Copy link"}
            </button>
          </div>
        ) : null}
        {pairingId ? (
          <div className="grid gap-2 rounded-xl border p-3 text-sm">
            <p>Comparison code: <strong>{code}</strong></p>
            <label className="text-xs">
              Comparison code from the phone
              <input
                aria-label="Approval comparison code"
                value={approvalCode}
                onChange={(event) => setApprovalCode(event.target.value)}
                className="mt-1 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
              />
            </label>
            <button
              type="button"
              disabled={busy || !approvalCode.trim()}
              onClick={() => void run(async () => {
                await approveRemotePairing(token, { pairingId, comparisonCode: approvalCode });
                setPairingId("");
                setPairingLink("");
              })}
              className="w-fit rounded-md border px-3 py-1.5 text-sm"
            >
              Approve pairing
            </button>
          </div>
        ) : null}
      </div>
      )}
    </section>
  );
}
