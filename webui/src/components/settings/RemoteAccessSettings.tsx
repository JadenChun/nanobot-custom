import QRCode from "qrcode";
import { useCallback, useEffect, useMemo, useState } from "react";

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
  denyRemotePairing,
  fetchRemoteAccessStatus,
  fetchRemotePairingPending,
  fetchRemotePairingPendingList,
  fetchRemoteTailscaleStatus,
  renewRemotePairing,
  revokeRemoteDevice,
  setRemoteAccessEnabled,
  startRemotePairing,
  type RemoteAccessMutationTransport,
  type RemoteAccessStatus,
  type RemotePairingOffer,
  type RemotePairingPending,
} from "@/lib/remote-access-api";

interface RemoteAccessSettingsProps {
  token: string;
  initialStatus?: RemoteAccessStatus | null;
  /** WebSocket mutation transport; settings mutations are never plain HTTP. */
  mutate: RemoteAccessMutationTransport;
  /** New-device pairing is local-only. */
  localSetupAvailable?: boolean;
}

const OFFER_SECONDS = 300;

function currentHttpsOrigin(): string {
  if (typeof window === "undefined" || window.location.protocol !== "https:") return "";
  return window.location.origin;
}

function formatLastSeen(timestamp: number): string {
  if (!timestamp) return "never";
  const date = new Date(timestamp * 1000);
  return Number.isNaN(date.getTime()) ? "unknown" : date.toLocaleString();
}

function formatRemaining(seconds: number): string {
  const bounded = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(bounded / 60);
  const rest = bounded % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

function DeviceFingerprint({ value }: { value?: string }) {
  if (!value) return null;
  return (
    <p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
      Key fingerprint: {value}
    </p>
  );
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
  const [step, setStep] = useState<"loading" | "waiting" | "pending" | "connected">("loading");
  const [origin, setOrigin] = useState(currentHttpsOrigin);
  const [offer, setOffer] = useState<RemotePairingOffer | null>(null);
  const [pending, setPending] = useState<RemotePairingPending | null>(null);
  const [now, setNow] = useState(() => Date.now());
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

  const restorePending = useCallback(async () => {
    try {
      const items = await fetchRemotePairingPendingList(token);
      const nowMs = Date.now();
      const active = items.find(
        (item) =>
          (item.state === "pending_approval" || item.state === "expired") &&
          (item.expires_at * 1000 > nowMs || item.renewable === true),
      );
      if (!active) return;
      setOffer({
        pairing_id: active.pairing_id,
        pairing_code: active.pairing_code,
        comparison_code: active.pairing_code ?? "",
        nonce: "",
        collector_secret: "",
        expires_at: active.expires_at,
        origin: active.origin,
        transport: "tailscale",
        scopes: active.scopes,
      });
      setPending(active);
      setNow(Date.now());
      setStep("pending");
      if (active.expires_at * 1000 <= Date.now()) {
        setError(
          active.renewable
            ? "The pairing code expired. Renew setup on this desktop."
            : "The pairing code expired. Generate a new one.",
        );
      }
    } catch {
      /* The pending route is optional for older servers. */
    }
  }, [token]);

  useEffect(() => {
    if (!localSetupAvailable) return;
    void restorePending();
  }, [localSetupAvailable, restorePending]);

  useEffect(() => {
    if (!setupOpen || offer || !localSetupAvailable) return;
    void restorePending();
  }, [setupOpen, offer, localSetupAvailable, restorePending]);

  useEffect(() => {
    if (!offer) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [offer]);

  useEffect(() => {
    if (!offer || (step !== "waiting" && step !== "pending")) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await fetchRemotePairingPending(token, offer.pairing_id);
        if (cancelled || !next || next.pairing_id !== offer.pairing_id) return;
        setPending(next);
        if (next.state === "pending_approval") setStep("pending");
        if (next.state === "approved") setStep("connected");
        if (next.state === "denied" || next.state === "expired") {
          setError(
            next.state === "denied"
              ? "The phone request was denied."
              : next.renewable
                ? "The pairing code expired. Renew setup on this desktop."
                : "The pairing code expired. Generate a new one.",
          );
        }
      } catch {
        /* Older servers may not support pending status yet. */
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [offer, step, token]);

  const expiresAt = offer
    ? (offer.expires_at ?? offer.expiration ?? Math.floor(now / 1000) + OFFER_SECONDS) * 1000
    : 0;
  const remaining = offer ? Math.max(0, (expiresAt - now) / 1000) : 0;
  const expired = !!offer && remaining <= 0;

  const pairingLink = useMemo(
    () => (offer && origin ? buildMobilePairingLink(origin, offer) : ""),
    [offer, origin],
  );
  const pairingCode = offer
    ? String(offer.pairing_code || offer.comparison_code || "").replace(/(\d{4})(?=\d)/g, "$1 ")
    : "";
  const hasFreshChallenge = Boolean(offer?.nonce && offer?.collector_secret);

  const [qrDataUrl, setQrDataUrl] = useState("");
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

  const openSetup = async () => {
    if (offer && (step === "pending" || step === "waiting")) {
      setSetupOpen(true);
      return;
    }
    resetSetup();
    setSetupOpen(true);
    if (!status?.enabled) {
      setError("Enable remote access first.");
      return;
    }
    try {
      const readiness = await fetchRemoteTailscaleStatus(token);
      const detected = readiness.connected && readiness.serve_mapped ? readiness.origin : "";
      const target = detected || currentHttpsOrigin();
      if (!target) {
        setError(
          readiness.error ||
            "Tailscale Serve is not ready. Run `tailscale serve --bg 18793` on this desktop.",
        );
        return;
      }
      setOrigin(target);
      await begin(target);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : "pairing is unavailable");
    }
  };

  const resetSetup = () => {
    setStep("loading");
    setOffer(null);
    setPending(null);
    setCopied(null);
    setError(null);
  };

  const renew = async () => {
    if (!offer) return;
    await run(async () => {
      const renewed = await renewRemotePairing(mutate, offer.pairing_id);
      setPending(renewed);
      setOffer((current) => (
        current && current.pairing_id === renewed.pairing_id
          ? { ...current, expires_at: renewed.expires_at }
          : current
      ));
      setNow(Date.now());
      setError(null);
      setStep("pending");
    });
  };

  const begin = async (targetOrigin = origin) => {
    await run(async () => {
      if (!targetOrigin.trim()) return;
      const next = await startRemotePairing(mutate, {
        origin: targetOrigin.trim(),
        label: "phone",
        mediaConsent: true,
        fullControlConsent: true,
        sharedHistoryConsent: true,
      });
      setOffer(next);
      setPending(null);
      setNow(Date.now());
      setStep("waiting");
    });
  };

  const approve = async () => {
    if (!offer) return;
    await run(async () => {
      const approval = await approveRemotePairing(mutate, {
        pairingId: offer.pairing_id,
        comparisonCode: offer.pairing_code || offer.comparison_code,
      });
      if (Array.isArray(approval.scopes) && !approval.scopes.includes("owner")) {
        throw new Error("The server did not grant owner access.");
      }
      setStep("connected");
    });
  };

  const deny = async () => {
    if (!offer) return;
    await run(async () => {
      await denyRemotePairing(mutate, offer.pairing_id);
      resetSetup();
    });
  };

  return (
    <section className="space-y-6" aria-labelledby="remote-access-heading">
      <header>
        <h1 id="remote-access-heading" className="text-lg font-medium">Remote access</h1>
        <p className="text-sm text-muted-foreground">
          Pair this owner phone once. It stays connected until you revoke it.
        </p>
      </header>
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}

      {step === "pending" && pending?.device ? (
        <div className="rounded-xl border border-emerald-600/30 p-4">
          <p className="text-sm font-medium">
            {pending.device.label || "Phone"} is waiting for owner approval
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {pending.device.platform || "Android"} · files, commands, settings, existing chats
          </p>
          <DeviceFingerprint value={pending.device_fingerprint} />
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => void deny()}
              className="rounded-md border px-3 py-1.5 text-sm"
            >
              Deny
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void approve()}
              className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
            >
              Approve owner access
            </button>
          </div>
        </div>
      ) : null}

      <div className="flex items-center justify-between rounded-xl border p-4">
        <div>
          <p className="text-sm font-medium">Remote access</p>
          <p className="text-xs text-muted-foreground">
            {status?.enabled ? "Enabled for the owner phone" : "Disabled"}
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
        <h2 className="text-sm font-medium">Owner device</h2>
        {status?.devices.length ? (
          <ul className="divide-y rounded-xl border">
            {status.devices.map((device) => (
              <li key={device.installation_id} className="flex items-center justify-between gap-3 p-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="truncate text-sm font-medium">{device.label}</p>
                    {device.scopes.includes("owner") ? (
                      <span className="rounded border border-emerald-600/40 px-1.5 py-0.5 text-[10px] text-emerald-700">
                        Owner
                      </span>
                    ) : null}
                  </div>
                  <p className="truncate text-xs text-muted-foreground">
                    {device.status === "active" ? "Connected until revoked" : "Revoked"}
                    {" · last seen "}
                    {formatLastSeen(device.last_seen_at)}
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
        ) : <p className="text-sm text-muted-foreground">No owner phone paired.</p>}
        {revoking ? (
          <div role="alertdialog" aria-label={`Revoke ${revoking.label}`} className="rounded-xl border border-destructive/40 bg-destructive/5 p-3">
            <p className="text-sm font-medium">Revoke {revoking.label}?</p>
            <p className="mt-1 text-xs text-muted-foreground">
              The phone loses owner access immediately and must be paired again here.
            </p>
            <div className="mt-3 flex gap-2">
              <button type="button" className="rounded-md border px-3 py-1 text-xs" onClick={() => setRevoking(null)}>Cancel</button>
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
          New-phone pairing and remote on/off stay on the server-local desktop or CLI.
        </p>
      ) : (
        <div className="flex items-center justify-between gap-3 rounded-xl border p-4">
          <div>
            <h2 className="text-sm font-medium">Pair owner phone</h2>
            <p className="text-xs text-muted-foreground">
              {status?.enabled
                ? "Full owner control. The phone stays connected until revoked."
                : "Enable remote access first."}
            </p>
          </div>
          <button
            type="button"
            disabled={busy || !status?.enabled}
            onClick={() => void openSetup()}
            className="shrink-0 rounded-md border px-3 py-1.5 text-sm"
          >
            Pair owner phone
          </button>
        </div>
      )}

      <Dialog open={setupOpen} onOpenChange={(open) => {
        setSetupOpen(open);
      }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {step === "connected" ? "Owner phone connected" : "Pair owner phone"}
            </DialogTitle>
            <DialogDescription>
              {step === "loading"
                ? "Checking Tailscale and preparing a 5-minute pairing code."
                : step === "waiting"
                  ? "Scan the QR code or enter the 8-digit code on the phone."
                  : step === "pending"
                    ? "Approve this phone to grant owner access."
                    : "This phone stays authorized until you revoke it."}
            </DialogDescription>
          </DialogHeader>

          {step === "loading" ? (
            <div className="space-y-4">
              <p
                role={error ? "alert" : undefined}
                className={error ? "text-sm text-destructive" : "text-sm text-muted-foreground"}
              >
                {error || "Preparing pairing…"}
              </p>
              {error ? (
                <p className="text-xs text-muted-foreground">
                  Make sure Tailscale Serve is running on this desktop, then retry.
                </p>
              ) : null}
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setSetupOpen(false)}
                  className="rounded-md border px-3 py-1.5 text-sm"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void openSetup()}
                  className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : null}

          {step === "waiting" || step === "pending" ? (
            <div className="space-y-4">
              <p className="text-xs text-muted-foreground">
                Only pair your own phone. Full owner control; the desktop must approve it.
              </p>

              {hasFreshChallenge ? (
                <div className="grid gap-4 sm:grid-cols-[224px_1fr]">
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
                    <p className="text-xs text-muted-foreground">Scan with the phone</p>
                  </div>
                  <div className="space-y-3">
                    <div className="rounded-lg border p-3 text-center">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        8-digit code
                      </p>
                      <p className="mt-1 whitespace-nowrap font-mono text-2xl font-semibold tracking-[0.12em]">
                        {pairingCode}
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          void navigator.clipboard
                            ?.writeText(offer?.pairing_code || offer?.comparison_code || "")
                            .then(() => {
                              setCopied("code");
                              window.setTimeout(() => setCopied(null), 1500);
                            })
                            .catch(() => undefined);
                        }}
                        className="mt-2 rounded-md border px-3 py-1 text-xs"
                      >
                        {copied === "code" ? "Copied" : "Copy code"}
                      </button>
                    </div>
                    <p className="text-center text-xs text-muted-foreground">
                      {expired ? "Code expired." : `Expires in ${formatRemaining(remaining)}`}
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          void navigator.clipboard
                            ?.writeText(pairingLink)
                            .then(() => {
                              setCopied("link");
                              window.setTimeout(() => setCopied(null), 1500);
                            })
                            .catch(() => undefined);
                        }}
                        className="rounded-md border px-3 py-1.5 text-sm"
                      >
                        {copied === "link" ? "Copied" : "Copy link"}
                      </button>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => void begin()}
                        className="rounded-md border px-3 py-1.5 text-sm"
                      >
                        New code
                      </button>
                      {expired && pending?.renewable ? (
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() => void renew()}
                          className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                        >
                          Renew setup
                        </button>
                      ) : null}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 text-center">
                  <p className="text-xs text-muted-foreground">
                    This request was created earlier.
                  </p>
                  {expired && pending?.renewable ? (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void renew()}
                      className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                    >
                      Renew setup
                    </button>
                  ) : null}
                </div>
              )}

              {step === "pending" && pending?.device ? (
                <div className="rounded-xl border border-emerald-600/30 p-3">
                  <p className="text-sm font-medium">
                    {pending.device.label || "Phone"} wants owner access
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {pending.device.platform || "Android"} · files, commands, settings, existing chats
                  </p>
                  <DeviceFingerprint value={pending.device_fingerprint} />
                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void deny()}
                      className="rounded-md border px-3 py-1.5 text-sm"
                    >
                      Deny
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void approve()}
                      className="rounded-md border border-foreground bg-foreground px-3 py-1.5 text-sm text-background"
                    >
                      Approve owner access
                    </button>
                  </div>
                </div>
              ) : (
                <p className="text-center text-xs text-muted-foreground">
                  Waiting for the phone to connect…
                </p>
              )}
            </div>
          ) : null}

          {step === "connected" ? (
            <div className="space-y-4">
              <p className="text-sm">
                Owner phone connected. It stays authorized until revoked.
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
