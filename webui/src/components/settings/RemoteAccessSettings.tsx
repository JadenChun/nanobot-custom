import QRCode from "qrcode";
import { useCallback, useEffect, useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import {
  approveRemotePairing,
  buildMobilePairingLink,
  fetchRemoteAccessStatus,
  revokeRemoteDevice,
  setRemoteAccessEnabled,
  startRemotePairing,
  type RemoteAccessMutationTransport,
  type RemoteAccessStatus,
} from "@/lib/remote-access-api";

interface RemoteAccessSettingsProps {
  token: string;
  initialStatus?: RemoteAccessStatus | null;
  /** WebSocket mutation transport; settings mutations are never plain HTTP. */
  mutate: RemoteAccessMutationTransport;
  /** Initial enable/re-enable and new-device approval are server-local only. */
  localSetupAvailable?: boolean;
}

export function RemoteAccessSettings({
  token,
  initialStatus = null,
  mutate,
  localSetupAvailable = true,
}: RemoteAccessSettingsProps) {
  const [status, setStatus] = useState<RemoteAccessStatus | null>(initialStatus);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [setupOpen, setSetupOpen] = useState(false);
  const [step, setStep] = useState<"intro" | "waiting" | "connected">("intro");
  const [origin, setOrigin] = useState(
    typeof window !== "undefined" && window.location.protocol === "https:"
      ? window.location.origin
      : "",
  );
  const [fullControl, setFullControl] = useState(true);
  const [grantedFullControl, setGrantedFullControl] = useState<boolean | null>(null);
  const [pairingId, setPairingId] = useState("");
  const [code, setCode] = useState("");
  const [pairingLink, setPairingLink] = useState("");
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [copied, setCopied] = useState<"link" | "code" | null>(null);
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
              await setRemoteAccessEnabled(mutate, !status?.enabled);
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
                  await revokeRemoteDevice(mutate, revoking.installationId);
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
        <div className="flex items-center justify-between gap-3 rounded-xl border p-4">
          <div>
            <h2 className="text-sm font-medium">Pair a phone</h2>
            <p className="text-xs text-muted-foreground">
              Scan a QR code and approve this phone. Full owner control is enabled by default.
            </p>
          </div>
          <button
            type="button"
            disabled={busy || !status}
            onClick={() => {
              setStep("intro");
              setPairingId("");
              setCode("");
              setPairingLink("");
              setSetupOpen(true);
            }}
            className="shrink-0 rounded-md border px-3 py-1.5 text-sm"
          >
            Set up remote access
          </button>
        </div>
      )}

      <Dialog
        open={setupOpen}
        onOpenChange={(open) => {
          setSetupOpen(open);
          if (!open) {
            setStep("intro");
            setGrantedFullControl(null);
            setPairingId("");
            setCode("");
            setPairingLink("");
          }
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {step === "connected" ? "You're connected" : "Connect your phone"}
            </DialogTitle>
            <DialogDescription>
              {step === "intro"
                ? fullControl
                  ? "Continue chats and control this desktop from your phone."
                  : "Continue chats and transfer media from your phone."
                : step === "waiting"
                  ? "Scan the QR code, then confirm the same code on both devices."
                  : grantedFullControl === false
                    ? "This phone can continue chats and transfer media. Full owner control stays off."
                    : "This phone can now use the same chats, settings, apps, skills and workspace actions as the desktop."}
            </DialogDescription>
          </DialogHeader>

          {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}

          {step === "intro" ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-amber-500/35 bg-amber-500/10 p-3 text-xs text-muted-foreground">
                {fullControl
                  ? "Nanobot can access this desktop, including files, apps and workspace actions, to complete tasks you send from your phone. Only connect devices you own and trust."
                  : "This phone can read and continue the same chats and transfer media from this desktop. Full owner control is off, so settings, apps, skills, automations and workspace actions stay on this desktop. Only connect devices you own and trust."}
              </div>
              <ul className="space-y-2 text-sm">
                <li>Continue any chat or project from your phone.</li>
                <li>Send and receive photos and files in chats.</li>
                {fullControl
                  ? <li>Use settings, apps, skills, automations and workspace actions remotely.</li>
                  : null}
                <li>Approve this phone once; the connection stays until you revoke it.</li>
              </ul>
              <label className="block text-xs">
                Server address (HTTPS)
                <input
                  aria-label="Server address"
                  value={origin}
                  onChange={(event) => setOrigin(event.target.value)}
                  placeholder="https://your-tailnet-host.ts.net"
                  className="mt-1 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
                />
              </label>
              <label className="flex items-start gap-2 text-xs">
                <input
                  type="checkbox"
                  aria-label="Allow full owner control"
                  checked={fullControl}
                  onChange={(event) => setFullControl(event.target.checked)}
                  className="mt-0.5"
                />
                <span>
                  Allow full owner control. The phone can use the same settings, apps, skills,
                  automations and workspace actions as this desktop.
                </span>
              </label>
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setSetupOpen(false)}
                  className="rounded-md border px-3 py-1.5 text-sm"
                >
                  Later
                </button>
                <button
                  type="button"
                  disabled={busy || !origin.trim()}
                  onClick={() => void run(async () => {
                    const offer = await startRemotePairing(mutate, {
                      origin: origin.trim(),
                      label: "phone",
                      chatConsent: true,
                      mediaConsent: true,
                      fullControlConsent: fullControl,
                      sharedHistoryConsent: fullControl,
                    });
                    setGrantedFullControl(fullControl);
                    setPairingId(offer.pairing_id);
                    setCode(offer.comparison_code);
                    setPairingLink(
                      buildMobilePairingLink(origin.trim(), crypto.randomUUID(), offer),
                    );
                    setStep("waiting");
                  })}
                  className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                >
                  Get started
                </button>
              </div>
            </div>
          ) : null}

          {step === "waiting" ? (
            <div className="space-y-4">
              <div className="flex flex-col items-center gap-2">
                {qrDataUrl ? (
                  <img
                    src={qrDataUrl}
                    alt="Pairing QR code"
                    className="h-48 w-48 rounded-lg border bg-white p-2"
                  />
                ) : (
                  <div className="grid h-48 w-48 place-items-center rounded-lg border text-xs text-muted-foreground">
                    Loading QR code
                  </div>
                )}
                <p className="text-xs text-muted-foreground">Scan this code with the phone.</p>
              </div>
              <div className="rounded-lg border p-3 text-center">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">Pairing code</p>
                <p className="mt-1 font-mono text-3xl font-semibold tracking-[0.25em]">{code}</p>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard?.writeText(code).then(() => {
                      setCopied("code");
                      window.setTimeout(() => setCopied(null), 1500);
                    }).catch(() => undefined);
                  }}
                  className="mt-2 rounded-md border px-3 py-1 text-xs"
                >
                  {copied === "code" ? "Copied" : "Copy code"}
                </button>
              </div>
              <p className="text-center text-xs text-muted-foreground">
                Waiting for the phone to connect. Approve once the code matches.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard?.writeText(pairingLink).then(() => {
                      setCopied("link");
                      window.setTimeout(() => setCopied(null), 1500);
                    }).catch(() => undefined);
                  }}
                  className="rounded-md border px-3 py-1.5 text-sm"
                >
                  {copied === "link" ? "Copied" : "Copy link"}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void run(async () => {
                    const offer = await startRemotePairing(mutate, {
                      origin: origin.trim(),
                      label: "phone",
                      chatConsent: true,
                      mediaConsent: true,
                      fullControlConsent: fullControl,
                      sharedHistoryConsent: fullControl,
                    });
                    setGrantedFullControl(fullControl);
                    setPairingId(offer.pairing_id);
                    setCode(offer.comparison_code);
                    setPairingLink(
                      buildMobilePairingLink(origin.trim(), crypto.randomUUID(), offer),
                    );
                  })}
                  className="rounded-md border px-3 py-1.5 text-sm"
                >
                  Refresh code
                </button>
                <button
                  type="button"
                  disabled={busy || !pairingId}
                  onClick={() => void run(async () => {
                    const approval = await approveRemotePairing(mutate, {
                      pairingId,
                      comparisonCode: code,
                    });
                    if (Array.isArray(approval.scopes)) {
                      setGrantedFullControl(approval.scopes.includes("owner"));
                    }
                    setStep("connected");
                    await refresh();
                  })}
                  className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                >
                  Approve phone
                </button>
              </div>
              <textarea
                aria-label="Phone pairing link"
                readOnly
                value={pairingLink}
                rows={3}
                className="w-full break-all rounded-md border bg-transparent px-2 py-1.5 font-mono text-xs"
              />
            </div>
          ) : null}

          {step === "connected" ? (
            <div className="space-y-4">
              <p className="text-sm">
                {grantedFullControl === false
                  ? "The phone is paired for chats and media only. Full owner control is off, so settings, apps, skills, automations and workspace actions stay on this desktop. You can revoke it at any time from the device list."
                  : "The phone is paired with full owner control. You can revoke it at any time from the device list."}
              </p>
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => setSetupOpen(false)}
                  className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                >
                  Done
                </button>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </section>
  );
}
