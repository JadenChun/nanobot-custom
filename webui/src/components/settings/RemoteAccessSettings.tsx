import { Check, ChevronDown, CircleCheck, CircleX, Clock, Copy, RefreshCw, Smartphone } from "lucide-react";
import QRCode from "qrcode";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
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
  type RemoteTailscaleStatus,
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
type CopyTarget = "link" | "code" | "command";

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
  const [expanded, setExpanded] = useState(false);
  if (!value) return null;
  return (
    <div className="mt-2 text-xs">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
        className="inline-flex max-w-full flex-wrap items-center gap-x-2 rounded-sm text-left text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        Key fingerprint
        {!expanded ? <code>{value.slice(0, 8)}…{value.slice(-4)}</code> : null}
        <ChevronDown aria-hidden="true" className={`h-3 w-3 ${expanded ? "rotate-180" : ""}`} />
      </button>
      {expanded ? <code className="mt-1 block break-all text-foreground">{value}</code> : null}
    </div>
  );
}

function ReadinessCheck({ ready, label }: { ready: boolean; label: string }) {
  const Icon = ready ? CircleCheck : CircleX;
  return (
    <li className="flex items-center gap-2.5 px-3 py-2.5 text-sm text-foreground">
      <Icon
        aria-hidden="true"
        className={`h-4 w-4 shrink-0 ${ready ? "text-emerald-700 dark:text-emerald-400" : "text-destructive"}`}
      />
      <span>{label}</span>
      <span className="sr-only">{ready ? "Ready" : "Not ready"}</span>
    </li>
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
  const [readiness, setReadiness] = useState<RemoteTailscaleStatus | null>(null);
  const [origin, setOrigin] = useState("");
  const [offer, setOffer] = useState<RemotePairingOffer | null>(null);
  const [pending, setPending] = useState<RemotePairingPending | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [copied, setCopied] = useState<CopyTarget | null>(null);
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
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(null), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  useEffect(() => {
    if (!offer || (step !== "waiting" && step !== "pending")) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await fetchRemotePairingPending(token, offer.pairing_id);
        if (cancelled || !next || next.pairing_id !== offer.pairing_id) return;
        setPending(next);
        if (next.state === "pending_approval" || (next.state === "expired" && next.device)) {
          setStep("pending");
        }
        if (next.state === "approved") setStep("connected");
        if (next.state === "denied") setError("The phone request was denied.");
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
  const expired = !!offer && (remaining <= 0 || pending?.state === "expired");
  const pairingSecret = offer?.pairing_code || offer?.comparison_code || "";
  const pairingCode = pairingSecret.replace(/(\d{4})(?=\d)/g, "$1 ");
  const hasFreshChallenge = Boolean(offer?.nonce && offer?.collector_secret);
  const canRenew = step === "pending" && pending?.renewable;
  const preflight = readiness && !(readiness.connected && readiness.serve_mapped && readiness.origin)
    ? readiness : null;
  const command = preflight?.recommended_command;

  const pairingLink = useMemo(
    () => (offer && origin && hasFreshChallenge ? buildMobilePairingLink(origin, offer) : ""),
    [offer, origin, hasFreshChallenge],
  );

  const [qr, setQr] = useState<{ link: string; dataUrl: string } | null>(null);
  useEffect(() => {
    if (!pairingLink) return;
    let cancelled = false;
    void QRCode.toDataURL(pairingLink, {
      width: 224,
      margin: 1,
      color: { dark: "#111827", light: "#ffffff" },
    }).then((dataUrl) => {
      if (!cancelled) setQr({ link: pairingLink, dataUrl });
    }).catch(() => {
      if (!cancelled) setQr(null);
    });
    return () => {
      cancelled = true;
    };
  }, [pairingLink]);

  const copy = async (target: CopyTarget, value: string) => {
    if (!navigator.clipboard) {
      setError("Clipboard access is unavailable.");
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setCopied(target);
    } catch {
      setError("Could not copy to the clipboard.");
    }
  };

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

  const checkReadiness = async () => {
    if (!status?.enabled) {
      setError("Enable remote access first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const next = await fetchRemoteTailscaleStatus(token);
      setReadiness(next);
      if (!next.connected || !next.serve_mapped || !next.origin) return;
      setOrigin(next.origin);
      await begin(next.origin);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : "pairing is unavailable");
    } finally {
      setBusy(false);
    }
  };

  const openSetup = async () => {
    setSetupOpen(true);
    if (offer && (step === "pending" || step === "waiting")) return;
    resetSetup();
    await checkReadiness();
  };

  const resetSetup = () => {
    setStep("loading");
    setReadiness(null);
    setOrigin("");
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

  const begin = async (targetOrigin: string) => {
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
      setCopied(null);
      setNow(Date.now());
      setStep("waiting");
    });
  };

  const newCode = async () => {
    setStep("loading");
    await checkReadiness();
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

  const activeDevices = (status?.devices ?? []).filter((device) => device.status === "active");
  const pendingDevice = step === "pending" ? pending?.device : undefined;
  const approvalCard = pendingDevice ? (
    <div className="rounded-xl border border-emerald-600/25 bg-emerald-50 p-4 text-foreground shadow-sm dark:bg-emerald-950/30">
      <div className="flex items-start gap-3">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300">
          <Smartphone aria-hidden="true" className="h-5 w-5" />
        </span>
        <div className="min-w-0">
          <p className="break-words text-sm font-medium">
            {pendingDevice.label || "Phone"}{setupOpen ? " wants owner access" : " is waiting for owner approval"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {pendingDevice.platform || "Android"} · files, commands, settings, existing chats
          </p>
          <DeviceFingerprint value={pending?.device_fingerprint} />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap justify-end gap-2">
        <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => void deny()}>
          Deny
        </Button>
        <Button type="button" size="sm" disabled={busy || expired} onClick={() => void approve()}>
          Approve owner access
        </Button>
      </div>
    </div>
  ) : null;

  return (
    <section className="space-y-6" aria-labelledby="remote-access-heading">
      <header>
        <h1 id="remote-access-heading" className="text-lg font-medium">Remote access</h1>
        <p className="text-sm text-muted-foreground">
          Pair this owner phone once. It stays connected until you revoke it.
        </p>
      </header>
      {!setupOpen && error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
      {!setupOpen ? approvalCard : null}

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
        {activeDevices.length ? (
          <ul className="divide-y rounded-xl border">
            {activeDevices.map((device) => (
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
                    Authorized until revoked
                    {" · last seen "}
                    {formatLastSeen(device.last_seen_at)}
                  </p>
                </div>
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
        <DialogContent className="max-h-[calc(100dvh-2rem)] max-w-lg overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {step === "connected" ? "Owner phone connected" : "Pair owner phone"}
            </DialogTitle>
            <DialogDescription>
              {step === "loading"
                ? "Verify Tailscale Serve on this desktop to start pairing."
                : step === "waiting"
                  ? "Scan the QR code or enter the 8-digit code on the phone."
                  : step === "pending"
                    ? "Approve this phone to grant owner access."
                    : "This phone stays authorized until you revoke it."}
            </DialogDescription>
          </DialogHeader>

          {step === "loading" ? (
            <div className="space-y-4">
              {preflight ? (
                <div className="space-y-3 rounded-xl border border-border bg-card p-4">
                  <ul aria-label="Tailscale readiness" className="divide-y divide-border rounded-lg border border-border bg-muted/40">
                    <ReadinessCheck ready={preflight.connected} label="Tailscale connected" />
                    <ReadinessCheck ready={preflight.serve_mapped} label="/mobile/v1 Serve target verified" />
                  </ul>
                  <p role="status" className="text-sm text-destructive">
                    {preflight.error || "Tailscale Serve is not ready."}
                  </p>
                  {command ? (
                    <div className="flex items-start gap-2 rounded-lg border border-border bg-muted p-2">
                      <code className="min-w-0 flex-1 select-all break-all p-1 font-mono text-xs text-foreground">{command}</code>
                      <Button type="button" variant="ghost" size="sm" onClick={() => void copy("command", command)} className="h-7 gap-1.5 px-2 text-xs">
                        {copied === "command" ? <Check aria-hidden="true" className="h-3.5 w-3.5" /> : <Copy aria-hidden="true" className="h-3.5 w-3.5" />}
                        {copied === "command" ? "Copied" : "Copy"}
                      </Button>
                    </div>
                  ) : null}
                </div>
              ) : null}
              {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
              {busy ? <p role="status" className="text-sm text-muted-foreground">Checking Tailscale…</p> : null}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" size="sm" onClick={() => setSetupOpen(false)}>
                  Cancel
                </Button>
                <Button type="button" size="sm" disabled={busy} onClick={() => void checkReadiness()} className="gap-2">
                  <RefreshCw aria-hidden="true" className="h-4 w-4" />
                  Re-check
                </Button>
              </div>
            </div>
          ) : null}

          {step === "waiting" || step === "pending" ? (
            <div className="space-y-4">
              {error ? (
                <p role="alert" className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                  {error}
                </p>
              ) : null}
              {expired ? (
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-card p-4">
                  <p role="status" className="flex items-center gap-2 text-sm text-destructive">
                    <Clock aria-hidden="true" className="h-4 w-4" />
                    Pairing code expired.
                  </p>
                  <Button type="button" size="sm" disabled={busy} onClick={() => void (canRenew ? renew() : newCode())}>
                    {canRenew ? "Renew setup" : "New code"}
                  </Button>
                </div>
              ) : hasFreshChallenge ? (
                <>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
                    <div className="mx-auto grid h-40 w-40 place-items-center rounded-xl border border-border bg-white p-2 shadow-sm sm:h-44 sm:w-44">
                      {qr && qr.link === pairingLink ? (
                        <img src={qr.dataUrl} alt="Pairing QR code" className="block h-full w-full" />
                      ) : (
                        <span className="text-xs text-gray-700">Loading QR code</span>
                      )}
                    </div>
                    <div className="flex flex-col justify-center gap-2 rounded-xl border border-border bg-card p-4 text-foreground">
                      <p className="text-xs font-medium text-muted-foreground">8-digit code</p>
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <p className="whitespace-nowrap font-mono text-2xl font-semibold tabular-nums">
                          {pairingCode}
                        </p>
                        <Button type="button" variant="ghost" size="sm" onClick={() => void copy("code", pairingSecret)} className="h-7 gap-1.5 px-2 text-xs">
                          {copied === "code" ? <Check aria-hidden="true" className="h-3.5 w-3.5" /> : <Copy aria-hidden="true" className="h-3.5 w-3.5" />}
                          {copied === "code" ? "Copied" : "Copy code"}
                        </Button>
                      </div>
                      <p role="timer" className="flex items-center gap-1.5 text-xs tabular-nums text-muted-foreground">
                        <Clock aria-hidden="true" className="h-3.5 w-3.5" />
                        Expires in {formatRemaining(remaining)}
                      </p>
                    </div>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    First time on this phone? Signing into Tailscale may take a minute — if this code expires, click <strong>New code</strong> and scan again.
                  </p>
                  <div className="flex flex-wrap justify-end gap-2">
                    <Button type="button" variant="outline" size="sm" onClick={() => void copy("link", pairingLink)}>
                      {copied === "link" ? "Copied" : "Copy link"}
                    </Button>
                    <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => void newCode()}>
                      New code
                    </Button>
                  </div>
                </>
              ) : null}

              {approvalCard || (!expired ? (
                <p className="text-center text-xs text-muted-foreground">
                  Waiting for the phone to connect…
                </p>
              ) : null)}
            </div>
          ) : null}

          {step === "connected" ? (
            <div className="space-y-4">
              <p className="text-sm">
                Owner phone connected. It stays authorized until revoked.
              </p>
              <div className="flex justify-end">
                <Button type="button" size="sm" onClick={() => setSetupOpen(false)}>
                  Done
                </Button>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </section>
  );
}
