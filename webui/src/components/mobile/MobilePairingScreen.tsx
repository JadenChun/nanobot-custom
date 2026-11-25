import {
  AlertCircle, ArrowLeft, Check, ChevronDown, Clock3, Copy, ExternalLink, LoaderCircle,
  Monitor, QrCode, ShieldCheck, Smartphone,
} from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  desktopHost, onMobileResume, type MobilePairingBridge, type PairingError, type PairingSnapshot,
} from "@/lib/mobile-pairing-bridge";
import { cn } from "@/lib/utils";

import { MobileAuthLayout, MobileStatusStrip, TailscaleLoginInstructions } from "./MobileReauthScreen";

const inputStyle = "min-h-11 w-full min-w-0 rounded-lg border border-input bg-background px-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
const buttonStyle = "min-h-11 gap-2";

function currentStep(snapshot: PairingSnapshot | null) {
  switch (snapshot?.stage) {
    case "paired": return 3;
    case "collecting":
    case "waiting_approval":
    case "expired_renewable":
    case "proving": return 2;
    case "starting_transport":
    case "tailscale_login_required": return 1;
    case "failed":
    case "cancelling":
      return snapshot.tailscaleState === "ready" ? 2 : snapshot.tailscaleState === "idle" ? 0 : 1;
    default: return 0;
  }
}

function PairingStepper({ step }: { step: number }) {
  return (
    <ol className="grid grid-cols-3 gap-1 rounded-xl bg-muted/40 p-1" aria-label="Pairing progress">
      {["Scan", "Tailscale", "Approve"].map((label, index) => (
        <li key={label} aria-current={index === step ? "step" : undefined} className={cn(
          "flex items-center justify-center gap-1.5 rounded-lg px-1 py-2 text-xs font-medium",
          index === step && "bg-primary text-primary-foreground",
          index > step && "text-muted-foreground",
        )}>
          {index < step ? (
            <span className="inline-flex items-center justify-center rounded-full bg-emerald-500/15 p-0.5 text-emerald-700 dark:text-emerald-300">
              <Check size={12} aria-hidden="true" /><span className="sr-only">Complete</span>
            </span>
          ) : <span className="tabular-nums">{index + 1}.</span>}
          {label}
        </li>
      ))}
    </ol>
  );
}

function Countdown({ expiresAt }: { expiresAt: number }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, []);

  // The gateway publishes Unix seconds. Native snapshots can also use milliseconds.
  const expiry = expiresAt < 1e12 ? expiresAt * 1_000 : expiresAt;
  const seconds = Math.max(0, Math.ceil((expiry - now) / 1_000));
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs font-medium tabular-nums">
      <Clock3 size={13} aria-hidden="true" />
      Expires in {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
    </span>
  );
}

function DesktopApproval({ snapshot, host, onScan, busy }: {
  snapshot: PairingSnapshot;
  host: string | null;
  onScan: () => void;
  busy: boolean;
}) {
  const expired = snapshot.stage === "expired_renewable";
  return (
    <div className="space-y-4 text-center">
      <div className="grid grid-cols-[minmax(0,1fr)_2rem_minmax(0,1.6fr)] items-start gap-2 rounded-xl border border-border/60 bg-muted/40 p-3">
        <div className="grid justify-items-center gap-1.5 text-xs font-medium">
          <span className="rounded-xl border bg-card p-2.5"><Smartphone size={22} aria-hidden="true" /></span>
          This phone
        </div>
        <div className="relative mt-5 h-px bg-border" aria-hidden="true">
          <span className="absolute left-1/2 top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-500 motion-safe:animate-pulse" />
        </div>
        <div className="grid min-w-0 justify-items-center gap-1.5 break-words text-xs font-medium">
          <span className="rounded-xl border bg-card p-2.5"><Monitor size={22} aria-hidden="true" /></span>
          {host ?? "Desktop"}
        </div>
      </div>
      {snapshot.stage === "collecting" && <p role="status" className="text-sm font-medium">Contacting your desktop…</p>}
      {snapshot.stage === "proving" && <p role="status" className="text-sm font-medium">Verifying this phone…</p>}
      {snapshot.stage === "waiting_approval" && (
        <p className="text-sm text-muted-foreground">Tap <strong className="text-foreground">Approve owner access</strong> on your desktop.</p>
      )}
      {snapshot.comparisonCode && (
        <div className="rounded-xl border bg-background px-3 py-2.5">
          <p className="text-xs text-muted-foreground">Check that the codes match</p>
          <p className="mt-1 font-mono text-2xl font-semibold tracking-widest">{snapshot.comparisonCode}</p>
        </div>
      )}
      {snapshot.expiresAt !== null && <Countdown expiresAt={snapshot.expiresAt} />}
      {expired && (
        <div className="space-y-2 rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-3 text-left text-sm text-amber-900 dark:text-amber-100">
          <p>This setup expired. Tap <strong>Renew setup</strong> on your desktop to continue.</p>
          <Button type="button" className={buttonStyle} variant="outline" disabled={busy} onClick={onScan}>
            <QrCode size={17} aria-hidden="true" /> Scan fresh QR
          </Button>
        </div>
      )}
    </div>
  );
}

const remediationLabels = {
  rescan_qr: "Scan fresh QR",
  edit_code: "Edit 8-digit code",
  retry: "Retry",
  open_tailscale_login: "Sign in to Tailscale",
  check_desktop_serve: "Retry",
  switch_desktop: "Use a different desktop",
} satisfies Record<PairingError["remediation"], string>;

function PairingFailure({ error, onRemediate, onReset, busy }: {
  error: PairingError;
  onRemediate: () => void;
  onReset: () => void;
  busy: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const detailsId = useId();
  const warning = error.kind === "transport" || error.kind === "tailscale" || error.httpStatus === 410
    || error.serverCode?.includes("expired");

  return (
    <div className="space-y-3">
      <div role="alert" className={cn(
        "flex gap-3 rounded-xl border p-3 text-sm",
        warning
          ? "border-amber-500/25 bg-amber-500/10 text-amber-900 dark:text-amber-100"
          : "border-destructive/25 bg-destructive/10 text-red-900 dark:text-red-100",
      )}>
        <AlertCircle size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
        <div><p className="font-semibold">{error.title}</p><p className="mt-1">{error.message}</p></div>
      </div>
      <div className="grid gap-2">
        <Button type="button" className={buttonStyle} disabled={busy} onClick={onRemediate}>
          {remediationLabels[error.remediation]}
        </Button>
        <Button type="button" variant="ghost" className={buttonStyle} disabled={busy} onClick={onReset}>
          Start over
        </Button>
      </div>
      <div className="border-t">
        <Button type="button" variant="ghost" className="min-h-11 w-full justify-between px-2"
          aria-expanded={expanded} aria-controls={detailsId} onClick={() => setExpanded(!expanded)}>
          Technical details <ChevronDown size={16} aria-hidden="true" className={cn(expanded && "rotate-180")} />
        </Button>
        <div id={detailsId} hidden={!expanded} className="space-y-2 rounded-lg bg-muted/60 p-3">
          <p className="whitespace-pre-wrap break-all font-mono text-xs">{error.technicalDetail}</p>
          <Button type="button" variant="outline" className={buttonStyle} onClick={() => {
            void (async () => {
              try {
                await navigator.clipboard.writeText(error.technicalDetail);
                setCopyStatus("Copied");
              } catch {
                setCopyStatus("Couldn’t copy");
              }
            })();
          }}>
            <Copy size={15} aria-hidden="true" /> Copy
          </Button>
          <span role="status" className="ml-2 text-xs text-muted-foreground">{copyStatus}</span>
        </div>
      </div>
    </div>
  );
}

export function MobilePairingScreen({ pairing, onPaired, initialOrigin = "" }: {
  pairing: MobilePairingBridge;
  onPaired: () => void;
  initialOrigin?: string;
}) {
  const [snapshot, setSnapshot] = useState<PairingSnapshot | null>(null);
  const [lastOrigin, setLastOrigin] = useState(initialOrigin);
  const [draftOrigin, setDraftOrigin] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [tab, setTab] = useState<"code" | "link" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<"submit" | "action" | null>(null);
  const operation = useRef<symbol | null>(null);
  const revision = useRef(0);
  const linkInput = useRef<HTMLInputElement>(null);
  const formId = useId();

  const publish = useCallback((next: PairingSnapshot) => {
    revision.current += 1;
    setSnapshot(next);
    if (next.origin) setLastOrigin(next.origin);
    setError(null);
  }, []);

  const readSnapshot = useCallback(async (read: () => Promise<PairingSnapshot>) => {
    const before = ++revision.current;
    let next: PairingSnapshot;
    try {
      next = await read();
    } catch (error) {
      if (revision.current === before) throw error;
      return null;
    }
    if (revision.current === before) publish(next);
    return next;
  }, [publish]);

  useEffect(() => {
    let active = true;
    let unsubscribe: (() => void) | undefined;
    const read = (resume = false) => {
      void readSnapshot(() => resume ? pairing.pairCheckResume() : pairing.pairingState()).catch(() => {
        if (active) setError("Couldn’t check pairing. Try again.");
      });
    };
    const removeResume = onMobileResume(() => read(true));
    void (async () => {
      const remove = await pairing.onPairingStateChange((next) => { if (active) publish(next); });
      if (!active) {
        remove();
        return;
      }
      unsubscribe = remove;
      read();
    })().catch(() => { if (active) read(); });
    return () => {
      active = false;
      revision.current += 1;
      operation.current = null;
      unsubscribe?.();
      removeResume();
    };
  }, [pairing, publish, readSnapshot]);

  const stage = snapshot?.stage;
  useEffect(() => {
    if (stage !== "paired") return;
    const timer = window.setTimeout(onPaired, 550);
    return () => window.clearTimeout(timer);
  }, [stage, onPaired]);

  const origin = draftOrigin ?? lastOrigin;
  const run = async (
    action: () => Promise<unknown>,
    message: string,
    kind: "submit" | "action" = "action",
    force = false,
  ) => {
    if (operation.current && !force) return;
    const token = Symbol();
    operation.current = token;
    setPending(kind);
    setError(null);
    try {
      await action();
    } catch {
      if (operation.current === token) setError(message);
    } finally {
      if (operation.current === token) {
        operation.current = null;
        setPending(null);
      }
    }
  };
  const scan = () => void run(() => pairing.scanPairingQr(), "Couldn’t open the QR scanner.", "submit");
  const reset = async () => readSnapshot(() => pairing.pairResetToIdle());
  const freshScan = () => void run(async () => {
    if ((await reset())?.stage === "idle") await pairing.scanPairingQr();
  }, "Couldn’t scan a fresh QR code.", "action", pending === "submit");
  const login = () => void run(
    () => pairing.openLogin(), "Couldn’t open Tailscale sign-in.", "action", pending === "submit",
  );
  const back = () => void run(
    () => readSnapshot(() => pairing.pairCancel()), "Couldn’t cancel pairing.", "action", true,
  );
  const startOver = () => void run(async () => {
    setTab(null);
    setDraftOrigin(null);
    setCode("");
    await reset();
  }, "Couldn’t reset pairing.");
  const remediate = (remediation: PairingError["remediation"]) => {
    switch (remediation) {
      case "rescan_qr": freshScan(); break;
      case "edit_code":
        void run(async () => {
          setDraftOrigin(null);
          setTab("code");
          await reset();
        }, "Couldn’t return to the code form.");
        break;
      case "open_tailscale_login": login(); break;
      case "retry":
      case "check_desktop_serve":
        void run(() => readSnapshot(() => pairing.pairRetry()), "Couldn’t retry pairing.", "submit");
        break;
      case "switch_desktop": startOver(); break;
    }
  };

  const host = desktopHost(snapshot?.origin ?? null);
  const busy = pending !== null;
  const cancellable = snapshot?.canCancel && pending !== "action";
  const nativeError = snapshot?.error;
  return (
    <MobileAuthLayout title="Connect to Kohaku" subtitle="Pair this phone with your desktop.">
      <PairingStepper step={currentStep(snapshot)} />
      <MobileStatusStrip host={host} tailscale={snapshot?.tailscaleState ?? "idle"} />
      {!snapshot && (error ? (
        <Button type="button" className={buttonStyle} disabled={busy} onClick={() => void run(
          () => readSnapshot(() => pairing.pairingState()), "Couldn’t check pairing. Try again.",
        )}>Retry</Button>
      ) : (
        <p role="status" className="text-sm text-muted-foreground">Checking pairing…</p>
      ))}

      {stage === "idle" && (
        <div className="space-y-3">
          <Button type="button" className="h-12 w-full gap-2" disabled={busy} onClick={scan}>
            <QrCode size={19} aria-hidden="true" /> Scan QR code
          </Button>
          <div className="grid grid-cols-2 gap-1 rounded-xl bg-muted/60 p-1">
            {(["code", "link"] as const).map((option) => (
              <Button key={option} type="button" className="min-h-11 px-2" variant={tab === option ? "outline" : "ghost"}
                aria-expanded={tab === option} aria-controls={formId + option}
                onClick={() => { setError(null); setTab(tab === option ? null : option); }}>
                {option === "code" ? "8-digit code" : "Paste link"}
              </Button>
            ))}
          </div>

          <form id={formId + "code"} hidden={tab !== "code"} className="space-y-3" onSubmit={(event) => {
            event.preventDefault();
            const address = origin.trim();
            if (!/^https:\/\//i.test(address) || !desktopHost(address)) {
              setError("Enter a desktop address that starts with https://.");
              return;
            }
            void run(() => readSnapshot(() => pairing.pairWithCode(address, code)), "Couldn’t submit the setup code.", "submit");
          }}>
            <label className="block space-y-1.5 text-sm font-medium">
              <span>Desktop address</span>
              <input type="url" className={inputStyle} value={origin} placeholder="https://desktop.example"
                onChange={(event) => setDraftOrigin(event.target.value)} autoCapitalize="none" spellCheck={false} required />
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              <span>Setup code</span>
              <input className={cn(inputStyle, "font-mono tracking-wider")} placeholder="0000 0000"
                inputMode="numeric" autoComplete="one-time-code" value={code.replace(/^(\d{4})(\d+)/, "$1 $2")}
                onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 8))} />
            </label>
            <Button className={cn(buttonStyle, "w-full")} type="submit" disabled={busy || code.length !== 8}>
              Pair with code
            </Button>
          </form>

          <form id={formId + "link"} hidden={tab !== "link"} className="space-y-3" onSubmit={(event) => {
            event.preventDefault();
            const rawLink = linkInput.current?.value;
            if (!rawLink) return;
            if (linkInput.current) linkInput.current.value = "";
            void run(() => readSnapshot(() => pairing.pairWithLink(rawLink)), "Couldn’t submit the setup link.", "submit");
          }}>
            <label htmlFor={formId + "pairing-link"} className="block text-sm font-medium">Pairing link</label>
            <div className="flex gap-2">
              <input className={inputStyle} id={formId + "pairing-link"} ref={linkInput} placeholder="kohaku://pair?…"
                autoComplete="off" autoCapitalize="none" spellCheck={false} required />
              <Button type="button" className={buttonStyle} variant="outline" disabled={busy} onClick={() => void run(async () => {
                const value = await navigator.clipboard.readText();
                if (linkInput.current) linkInput.current.value = value;
              }, "Couldn’t read the clipboard. Paste the link into the field.")}>Paste</Button>
            </div>
            <Button type="submit" className={cn(buttonStyle, "w-full")} disabled={busy}>Pair with link</Button>
          </form>
        </div>
      )}

      {(stage === "starting_transport" || stage === "tailscale_login_required") && (
        <div className="space-y-3 text-center">
          {stage === "starting_transport" ? (
            <div role="status" className="space-y-2">
              <LoaderCircle className="mx-auto size-12 rounded-2xl bg-muted p-3 motion-safe:animate-spin" aria-hidden="true" />
              <p className="text-sm font-medium">Connecting to Tailscale…</p>
            </div>
          ) : (
            <>
              <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/10 text-amber-700 ring-1 ring-inset ring-amber-500/25 dark:text-amber-300">
                <ShieldCheck aria-hidden="true" />
              </span>
              <TailscaleLoginInstructions />
              <Button type="button" className={cn(buttonStyle, "w-full")} disabled={pending === "action" || !snapshot?.tailscaleHasLoginUrl}
                onClick={login}><ExternalLink size={17} aria-hidden="true" /> Sign in to Tailscale</Button>
            </>
          )}
          <Button type="button" variant="ghost" className={buttonStyle} disabled={!cancellable} onClick={back}>
            <ArrowLeft size={16} aria-hidden="true" /> Back
          </Button>
        </div>
      )}

      {snapshot && (stage === "collecting" || stage === "waiting_approval" || stage === "expired_renewable" || stage === "proving") && (
        <>
          <DesktopApproval snapshot={snapshot} host={host} onScan={freshScan} busy={pending === "action"} />
          {snapshot.canCancel && (
            <Button type="button" variant="ghost" className={cn(buttonStyle, "w-full")} disabled={!cancellable} onClick={back}>
              Cancel pairing
            </Button>
          )}
        </>
      )}

      {stage === "cancelling" && (
        <p role="status" className="flex items-center justify-center gap-2 text-sm font-medium">
          <LoaderCircle size={17} className="motion-safe:animate-spin" aria-hidden="true" /> Cancelling pairing…
        </p>
      )}

      {stage === "paired" && (
        <div role="status" className="space-y-2 py-2 text-center">
          <Check className="mx-auto size-12 rounded-2xl bg-emerald-500/15 p-3 text-emerald-700 dark:text-emerald-300" aria-hidden="true" />
          <p className="text-sm font-medium">Connected — loading Kohaku…</p>
        </div>
      )}

      {stage === "failed" && (nativeError ? (
        <PairingFailure key={nativeError.technicalDetail} error={nativeError} busy={busy}
          onRemediate={() => remediate(nativeError.remediation)} onReset={startOver} />
      ) : (
        <Button type="button" className={buttonStyle} disabled={busy} onClick={startOver}>Start over</Button>
      ))}
      {error && <p role="alert" className="text-sm text-amber-800 dark:text-amber-200">{error}</p>}
    </MobileAuthLayout>
  );
}
