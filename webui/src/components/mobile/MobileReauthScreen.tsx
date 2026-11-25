import { ExternalLink, LoaderCircle, Monitor, RotateCw, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { desktopHost, type TailscaleNodeState } from "@/lib/mobile-pairing-bridge";

export function MobileAuthLayout({ title, subtitle, children }: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <main className="mobile-auth-layout bg-background text-foreground">
      <div className="w-full max-w-sm space-y-5 rounded-2xl border bg-card p-5 text-card-foreground shadow-sm sm:p-6">
        <header className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border bg-muted/40">
            <img src="./brand/kohaku_icon_transparent.png" alt="" width={28} height={28} />
          </div>
          <div className="min-w-0">
            <h1 className="text-xl font-semibold leading-tight tracking-tight">{title}</h1>
            <p className="mt-1 break-words text-sm text-muted-foreground">{subtitle}</p>
          </div>
        </header>
        {children}
      </div>
    </main>
  );
}

const tailscaleLabels: Record<Exclude<TailscaleNodeState, "idle">, string> = {
  starting: "Connecting…",
  needs_login: "Sign in needed",
  ready: "Ready",
  error: "Needs attention",
};

export function MobileStatusStrip({ host, tailscale }: {
  host: string | null;
  tailscale: TailscaleNodeState;
}) {
  if (!host && tailscale === "idle") return null;
  return (
    <dl className="flex flex-wrap gap-x-3 gap-y-2 rounded-xl border border-border/60 bg-muted/40 px-3 py-2.5">
      {host && (
        <div className="min-w-0 flex-1">
          <dt className="text-xs text-muted-foreground">Desktop</dt>
          <dd className="break-words text-sm font-medium">{host}</dd>
        </div>
      )}
      {tailscale !== "idle" && (
        <div>
          <dt className="text-xs text-muted-foreground">Tailscale</dt>
          <dd className={tailscale === "ready" ? "text-sm font-medium text-emerald-700 dark:text-emerald-300" : "text-sm font-medium"}>
            {tailscaleLabels[tailscale]}
          </dd>
        </div>
      )}
    </dl>
  );
}

export function TailscaleLoginInstructions() {
  return (
    <p className="text-sm leading-relaxed text-muted-foreground">
      Sign in to your Tailscale account once. When Tailscale shows “Login successful”, tap{" "}
      <strong>×</strong> or the notification to return to Kohaku. Kohaku continues automatically.
    </p>
  );
}

interface MobileReauthScreenProps {
  origin: string;
  needsLogin: boolean;
  pending: "retry" | "login" | "unpair" | null;
  error: string | null;
  onRetry: () => void;
  onLogin: () => void;
  onUnpair: () => void;
}

export function MobileReauthScreen({
  origin, needsLogin, pending, error, onRetry, onLogin, onUnpair,
}: MobileReauthScreenProps) {
  const host = desktopHost(origin) ?? origin;
  return (
    <MobileAuthLayout title="Reconnect this phone" subtitle={`Your pairing with ${host} is saved.`}>
      <MobileStatusStrip host={host} tailscale={pending === "retry" ? "starting" : needsLogin ? "needs_login" : "idle"} />
      <div className="space-y-3">
        <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/10 text-amber-700 ring-1 ring-inset ring-amber-500/25 dark:text-amber-300">
          {needsLogin ? <ShieldCheck aria-hidden="true" /> : <Monitor aria-hidden="true" />}
        </span>
        <h2 className="text-base font-semibold">
          {needsLogin ? "Tailscale needs you to sign in" : "Can’t reach this desktop right now"}
        </h2>
        {needsLogin ? <TailscaleLoginInstructions /> : (
          <p className="text-sm leading-relaxed text-muted-foreground">
            Make sure Kohaku is open on your desktop and that your phone can connect to Tailscale.
          </p>
        )}
        {pending && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <LoaderCircle size={16} className="motion-safe:animate-spin" aria-hidden="true" />
            {pending === "retry" ? "Connecting to your desktop…" : pending === "login" ? "Opening Tailscale…" : "Clearing saved pairing…"}
          </p>
        )}
        {error && <p className="text-sm text-amber-800 dark:text-amber-200" role="alert">{error}</p>}
      </div>
      <div className="grid gap-2">
        <Button type="button" className="min-h-11 gap-2" variant={needsLogin ? "outline" : "default"} disabled={pending !== null} onClick={onRetry}>
          <RotateCw size={17} aria-hidden="true" /> Retry
        </Button>
        <Button type="button" className="min-h-11 gap-2" variant={needsLogin ? "default" : "outline"} disabled={pending !== null} onClick={onLogin}>
          <ExternalLink size={17} aria-hidden="true" /> Sign in to Tailscale
        </Button>
        <Button type="button" className="min-h-11 text-muted-foreground" variant="ghost" disabled={pending !== null} onClick={onUnpair}>
          Unpair / Switch desktop
        </Button>
      </div>
    </MobileAuthLayout>
  );
}
