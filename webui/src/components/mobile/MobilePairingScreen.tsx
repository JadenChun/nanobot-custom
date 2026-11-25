import {
  Check, ChevronDown, Copy, ExternalLink, LoaderCircle, QrCode,
} from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  desktopHost, onMobileResume, type MobilePairingBridge, type PairingError, type PairingSnapshot,
} from "@/lib/mobile-pairing-bridge";
import { cn } from "@/lib/utils";

import { MobileAuthLayout } from "./MobileReauthScreen";

const inputStyle =
  "h-11 w-full min-w-0 rounded-xl border border-input bg-transparent px-3.5 text-base placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
const primaryButton = "h-12 w-full gap-2 rounded-xl";
const quietButton = "min-h-11 w-full text-muted-foreground";

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
    <p className="text-xs tabular-nums text-muted-foreground">
      Expires in {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
    </p>
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

  return (
    <div className="space-y-2 pt-1">
      <Button type="button" className={primaryButton} disabled={busy} onClick={onRemediate}>
        {remediationLabels[error.remediation]}
      </Button>
      <Button type="button" variant="ghost" className={quietButton} disabled={busy} onClick={onReset}>
        Start over
      </Button>
      <div className="pt-3">
        <button
          type="button"
          className="inline-flex min-h-10 items-center gap-1 px-2 text-xs text-muted-foreground hover:text-foreground"
          aria-expanded={expanded}
          aria-controls={detailsId}
          onClick={() => setExpanded(!expanded)}
        >
          Technical details
          <ChevronDown size={14} aria-hidden="true" className={cn("transition-transform", expanded && "rotate-180")} />
        </button>
        <div
          id={detailsId}
          hidden={!expanded}
          className="mt-2 space-y-2 rounded-xl bg-muted/50 px-3 py-2.5 text-left"
        >
          <p className="whitespace-pre-wrap break-all font-mono text-xs text-muted-foreground">
            {error.technicalDetail}
          </p>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 gap-1.5 px-2 text-xs"
              onClick={() => {
                void (async () => {
                  try {
                    await navigator.clipboard.writeText(error.technicalDetail);
                    setCopyStatus("Copied");
                  } catch {
                    setCopyStatus("Couldn’t copy");
                  }
                })();
              }}
            >
              <Copy size={13} aria-hidden="true" /> Copy
            </Button>
            {copyStatus && <span role="status" className="text-xs text-muted-foreground">{copyStatus}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

function stageHeader(
  stage: PairingSnapshot["stage"] | undefined,
  host: string | null,
  manualMode: "code" | "link" | null,
  nativeError: PairingError | null | undefined,
): { title: string; subtitle: string | null } {
  switch (stage) {
    case "starting_transport":
      return { title: "Connecting to Tailscale…", subtitle: host };
    case "tailscale_login_required":
      return {
        title: "Sign in to Tailscale",
        subtitle: "One-time sign-in to reach your desktop privately. Tap × or the notification when done.",
      };
    case "collecting":
      return { title: "Contacting your desktop…", subtitle: host };
    case "waiting_approval":
      return { title: "Approve on desktop", subtitle: host };
    case "expired_renewable":
      return { title: "Setup expired", subtitle: "Tap Renew setup on your desktop to continue." };
    case "proving":
      return { title: "Verifying this phone…", subtitle: host };
    case "cancelling":
      return { title: "Cancelling pairing…", subtitle: null };
    case "paired":
      return { title: "Connected", subtitle: "Loading Kohaku…" };
    case "failed":
      return {
        title: nativeError?.title ?? "Couldn’t pair",
        subtitle: nativeError?.message ?? "Start over to try again.",
      };
    case "idle":
      if (manualMode) {
        return {
          title: manualMode === "code" ? "Enter setup code" : "Paste pairing link",
          subtitle: "Copy it from Kohaku desktop settings.",
        };
      }
      return {
        title: "Connect to Kohaku",
        subtitle: "Scan the QR code in Kohaku desktop settings.",
      };
    default:
      return { title: "Connect to Kohaku", subtitle: null };
  }
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
  const [manualMode, setManualMode] = useState<"code" | "link" | null>(null);
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
    } catch (caught) {
      if (revision.current === before) throw caught;
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
    ignoreCancelled = false,
  ) => {
    if (operation.current && !force) return;
    const token = Symbol();
    operation.current = token;
    setPending(kind);
    setError(null);
    try {
      await action();
    } catch (caught) {
      const text = (caught as { message?: string } | null)?.message ?? "";
      if (operation.current === token && !(ignoreCancelled && /cancelled/i.test(text))) {
        setError(message);
      }
    } finally {
      if (operation.current === token) {
        operation.current = null;
        setPending(null);
      }
    }
  };

  const scan = () => void run(
    () => pairing.scanPairingQr(), "Couldn’t open the QR scanner.", "submit", false, true,
  );
  const reset = async () => readSnapshot(() => pairing.pairResetToIdle());
  const freshScan = () => void run(async () => {
    if ((await reset())?.stage === "idle") await pairing.scanPairingQr();
  }, "Couldn’t scan a fresh QR code.", "action", pending === "submit", true);
  const login = () => void run(
    () => pairing.openLogin(), "Couldn’t open Tailscale sign-in.", "action", pending === "submit",
  );
  const back = () => void run(
    () => readSnapshot(() => pairing.pairCancel()), "Couldn’t cancel pairing.", "action", true,
  );
  const startOver = () => void run(async () => {
    setManualMode(null);
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
          setManualMode("code");
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
  const { title, subtitle } = stageHeader(stage, host, manualMode, nativeError);

  const headerIcon = stage === "paired" ? (
    <span className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
      <Check size={24} aria-hidden="true" />
    </span>
  ) : (stage === "starting_transport" || stage === "collecting" || stage === "proving" || stage === "cancelling") ? (
    <LoaderCircle size={32} className="text-muted-foreground motion-safe:animate-spin" aria-hidden="true" />
  ) : undefined;

  return (
    <MobileAuthLayout title={title} subtitle={subtitle} icon={headerIcon}>
      {!snapshot && (error ? (
        <Button
          type="button"
          className={primaryButton}
          disabled={busy}
          onClick={() => void run(() => readSnapshot(() => pairing.pairingState()), "Couldn’t check pairing. Try again.")}
        >
          Retry
        </Button>
      ) : (
        <p role="status" className="text-sm text-muted-foreground">Checking pairing…</p>
      ))}

      {stage === "idle" && !manualMode && (
        <div className="space-y-3 pt-1">
          <Button type="button" className={primaryButton} disabled={busy} onClick={scan}>
            <QrCode size={18} aria-hidden="true" /> Scan QR code
          </Button>
          <button
            type="button"
            className="min-h-11 px-3 text-sm text-muted-foreground hover:text-foreground"
            onClick={() => { setError(null); setManualMode("code"); }}
          >
            Enter code or link manually
          </button>
          {snapshot?.tailscaleState === "ready" && (
            <p className="pt-2 text-xs text-emerald-700 dark:text-emerald-400">
              Tailscale connected
            </p>
          )}
        </div>
      )}

      {stage === "idle" && manualMode && (
        <div className="space-y-4 text-left">
          <div className="flex justify-center gap-6 border-b border-border/60 text-sm">
            {(["code", "link"] as const).map((option) => (
              <button
                key={option}
                type="button"
                aria-pressed={manualMode === option}
                className={cn(
                  "-mb-px min-h-10 border-b-2 px-1 font-medium transition-colors",
                  manualMode === option
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground",
                )}
                onClick={() => { setError(null); setManualMode(option); }}
              >
                {option === "code" ? "8-digit code" : "Paste link"}
              </button>
            ))}
          </div>

          {manualMode === "code" ? (
            <form
              id={formId + "code"}
              className="space-y-3"
              onSubmit={(event) => {
                event.preventDefault();
                const address = origin.trim();
                if (!/^https:\/\//i.test(address) || !desktopHost(address)) {
                  setError("Enter a desktop address that starts with https://.");
                  return;
                }
                void run(() => readSnapshot(() => pairing.pairWithCode(address, code)), "Couldn’t submit the setup code.", "submit");
              }}
            >
              <label className="block space-y-1.5 text-xs font-medium text-muted-foreground">
                <span>Desktop address</span>
                <input
                  type="url"
                  className={inputStyle}
                  value={origin}
                  placeholder="https://desktop.ts.net"
                  onChange={(event) => setDraftOrigin(event.target.value)}
                  autoCapitalize="none"
                  spellCheck={false}
                  required
                />
              </label>
              <label className="block space-y-1.5 text-xs font-medium text-muted-foreground">
                <span>Setup code</span>
                <input
                  className={cn(inputStyle, "text-center font-mono text-lg tracking-widest")}
                  placeholder="0000 0000"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  value={code.replace(/^(\d{4})(\d+)/, "$1 $2")}
                  onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 8))}
                />
              </label>
              <Button className={primaryButton} type="submit" disabled={busy || code.length !== 8}>
                Pair with code
              </Button>
            </form>
          ) : (
            <form
              id={formId + "link"}
              className="space-y-3"
              onSubmit={(event) => {
                event.preventDefault();
                const rawLink = linkInput.current?.value;
                if (!rawLink) return;
                if (linkInput.current) linkInput.current.value = "";
                void run(() => readSnapshot(() => pairing.pairWithLink(rawLink)), "Couldn’t submit the setup link.", "submit");
              }}
            >
              <label htmlFor={formId + "pairing-link"} className="block text-xs font-medium text-muted-foreground">
                Pairing link
              </label>
              <div className="flex gap-2">
                <input
                  className={inputStyle}
                  id={formId + "pairing-link"}
                  ref={linkInput}
                  placeholder="kohaku://pair?…"
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  required
                />
                <Button
                  type="button"
                  variant="outline"
                  className="h-11 shrink-0 rounded-xl px-3.5"
                  disabled={busy}
                  onClick={() => void run(async () => {
                    const value = await navigator.clipboard.readText();
                    if (linkInput.current) linkInput.current.value = value;
                  }, "Couldn’t read the clipboard. Paste the link into the field.")}
                >
                  Paste
                </Button>
              </div>
              <Button type="submit" className={primaryButton} disabled={busy}>
                Pair with link
              </Button>
            </form>
          )}

          <Button
            type="button"
            variant="ghost"
            className={quietButton}
            onClick={() => { setError(null); setManualMode(null); }}
          >
            Back
          </Button>
        </div>
      )}

      {stage === "starting_transport" && (
        <div className="pt-2">
          <Button type="button" variant="ghost" className={quietButton} disabled={!cancellable} onClick={back}>
            Back
          </Button>
        </div>
      )}

      {stage === "tailscale_login_required" && (
        <div className="space-y-2 pt-1">
          <Button
            type="button"
            className={primaryButton}
            disabled={pending === "action" || !snapshot?.tailscaleHasLoginUrl}
            onClick={login}
          >
            <ExternalLink size={17} aria-hidden="true" /> Sign in to Tailscale
          </Button>
          <Button type="button" variant="ghost" className={quietButton} disabled={!cancellable} onClick={back}>
            Back
          </Button>
        </div>
      )}

      {snapshot && (stage === "collecting" || stage === "waiting_approval" || stage === "expired_renewable" || stage === "proving") && (
        <div className="space-y-4">
          {stage === "waiting_approval" && (
            <p className="text-sm text-muted-foreground">
              Tap <strong className="font-medium text-foreground">Approve owner access</strong> on your desktop.
            </p>
          )}
          {snapshot.comparisonCode && (
            <p className="py-1 font-mono text-3xl font-semibold tracking-[0.2em]">
              {snapshot.comparisonCode}
            </p>
          )}
          {snapshot.expiresAt !== null && <Countdown expiresAt={snapshot.expiresAt} />}
          {stage === "expired_renewable" && (
            <Button type="button" className={primaryButton} disabled={pending === "action"} onClick={freshScan}>
              <QrCode size={17} aria-hidden="true" /> Scan fresh QR
            </Button>
          )}
          {snapshot.canCancel && (
            <Button type="button" variant="ghost" className={quietButton} disabled={!cancellable} onClick={back}>
              Cancel
            </Button>
          )}
        </div>
      )}

      {stage === "failed" && (nativeError ? (
        <PairingFailure
          key={nativeError.technicalDetail}
          error={nativeError}
          busy={busy}
          onRemediate={() => remediate(nativeError.remediation)}
          onReset={startOver}
        />
      ) : (
        <Button type="button" className={primaryButton} disabled={busy} onClick={startOver}>
          Start over
        </Button>
      ))}

      {error && (
        <p role="alert" className="text-sm text-amber-700 dark:text-amber-300">
          {error}
        </p>
      )}
    </MobileAuthLayout>
  );
}
