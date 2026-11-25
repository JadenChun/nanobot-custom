import { ExternalLink, LoaderCircle, RotateCw } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { desktopHost } from "@/lib/mobile-pairing-bridge";

export function MobileAuthLayout({ title, subtitle, icon, children }: {
  title: string;
  subtitle?: string | null;
  icon?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main className="mobile-auth-layout bg-background text-foreground">
      <div className="flex w-full max-w-[21rem] flex-col items-center text-center">
        <div className="mb-5 flex h-12 w-12 items-center justify-center">
          {icon ?? (
            <img
              src="./brand/kohaku_icon_transparent.png"
              alt=""
              width={44}
              height={44}
              className="h-11 w-11 object-contain"
            />
          )}
        </div>
        <h1 className="text-[1.375rem] font-semibold leading-snug tracking-tight">{title}</h1>
        {subtitle && (
          <p className="mt-1.5 max-w-[18.5rem] break-words text-sm leading-relaxed text-muted-foreground">
            {subtitle}
          </p>
        )}
        <div className="mt-7 w-full space-y-3">{children}</div>
      </div>
    </main>
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
  const subtitle = needsLogin
    ? `Sign in to Tailscale to reconnect to ${host}. Tap × or the notification when done.`
    : `Your pairing with ${host} is saved, but Kohaku can’t reach it right now.`;

  return (
    <MobileAuthLayout
      title={needsLogin ? "Sign in to Tailscale" : "Reconnect this phone"}
      subtitle={subtitle}
    >
      {pending && (
        <p className="flex items-center justify-center gap-2 text-sm text-muted-foreground" role="status">
          <LoaderCircle size={15} className="motion-safe:animate-spin" aria-hidden="true" />
          {pending === "retry"
            ? "Connecting to your desktop…"
            : pending === "login"
              ? "Opening Tailscale…"
              : "Clearing saved pairing…"}
        </p>
      )}
      {error && (
        <p className="text-sm text-amber-700 dark:text-amber-300" role="alert">
          {error}
        </p>
      )}
      <div className="space-y-2 pt-1">
        {needsLogin ? (
          <>
            <Button
              type="button"
              className="h-12 w-full gap-2 rounded-xl"
              disabled={pending !== null}
              onClick={onLogin}
            >
              <ExternalLink size={17} aria-hidden="true" /> Sign in to Tailscale
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="min-h-11 w-full gap-2 text-muted-foreground"
              disabled={pending !== null}
              onClick={onRetry}
            >
              <RotateCw size={15} aria-hidden="true" /> Retry
            </Button>
          </>
        ) : (
          <>
            <Button
              type="button"
              className="h-12 w-full gap-2 rounded-xl"
              disabled={pending !== null}
              onClick={onRetry}
            >
              <RotateCw size={17} aria-hidden="true" /> Retry
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="min-h-11 w-full gap-2 text-muted-foreground"
              disabled={pending !== null}
              onClick={onLogin}
            >
              <ExternalLink size={15} aria-hidden="true" /> Sign in to Tailscale
            </Button>
          </>
        )}
      </div>
      <div className="pt-3">
        <button
          type="button"
          className="min-h-11 px-3 text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline disabled:opacity-50"
          disabled={pending !== null}
          onClick={onUnpair}
        >
          Switch desktop
        </button>
      </div>
    </MobileAuthLayout>
  );
}
