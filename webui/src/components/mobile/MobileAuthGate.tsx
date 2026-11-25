import { LoaderCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  checkMobileAccess, onMobileResume, waitForMobileTransport,
  type MobilePairingBridge, type MobilePairingTarget,
} from "@/lib/mobile-pairing-bridge";
import type { MobileTransport } from "@/lib/mobile-runtime";

import { MobilePairingScreen } from "./MobilePairingScreen";
import { MobileAuthLayout, MobileReauthScreen } from "./MobileReauthScreen";

type ReauthView = {
  kind: "reauth";
  target: MobilePairingTarget;
  needsLogin: boolean;
  error?: string;
};

type GateView =
  | { kind: "checking" | "ready" | "target_error" }
  | { kind: "pairing"; origin?: string }
  | ReauthView;

async function verifyTarget(
  pairing: MobilePairingBridge,
  transport: MobileTransport | undefined,
  target: MobilePairingTarget,
  signal: AbortSignal,
): Promise<GateView> {
  const reauth: ReauthView = { kind: "reauth", target, needsLogin: false };
  if (!transport) return reauth;
  const connection = await waitForMobileTransport(pairing, transport, target, signal);
  if (connection !== "ready") return { ...reauth, needsLogin: connection === "needs_login" };

  const access = await checkMobileAccess(pairing, transport, signal);
  if (signal.aborted || access === "unreachable") return reauth;
  if (access === "ok") return { kind: "ready" };
  if (access === "repair") {
    try {
      await pairing.logout(true);
    } catch {
      return { ...reauth, error: "Couldn’t clear the saved pairing. Try again." };
    }
  }
  return { kind: "pairing", origin: target.origin };
}

interface MobileAuthGateProps {
  pairing: MobilePairingBridge;
  transport?: MobileTransport;
  onReady: () => void;
}

type ReauthAction = "retry" | "login" | "unpair";

export default function MobileAuthGate({ pairing, transport, onReady }: MobileAuthGateProps) {
  const [view, setView] = useState<GateView>({ kind: "checking" });
  const [targetAttempt, setTargetAttempt] = useState(0);
  const [pending, setPending] = useState<ReauthAction | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const lifecycle = useRef<AbortController | null>(null);
  const activeAction = useRef<ReauthAction | null>(null);
  const notified = useRef(false);
  const ready = useCallback(() => {
    if (notified.current) return;
    notified.current = true;
    onReady();
  }, [onReady]);

  useEffect(() => {
    const controller = new AbortController();
    lifecycle.current = controller;
    const { signal } = controller;
    void (async () => {
      try {
        const target = await pairing.target();
        if (signal.aborted) return;
        const next = target?.hasAccess
          ? await verifyTarget(pairing, transport, target, signal)
          : { kind: "pairing" as const, origin: target?.origin };
        if (!signal.aborted) setView(next);
      } catch {
        if (!signal.aborted) setView({ kind: "target_error" });
      }
    })();
    return () => controller.abort();
  }, [pairing, targetAttempt, transport]);

  useEffect(() => {
    if (view.kind === "ready") ready();
  }, [ready, view.kind]);

  const target = view.kind === "reauth" ? view.target : null;
  const run = useCallback(async (action: ReauthAction) => {
    const signal = lifecycle.current?.signal;
    if (!target || !signal || signal.aborted || activeAction.current) return;
    activeAction.current = action;
    setPending(action);
    setActionError(null);
    try {
      if (action === "login") {
        await pairing.openLogin();
      } else if (action === "unpair") {
        await pairing.logout(false);
        if (!signal.aborted) setView({ kind: "pairing", origin: target.origin });
      } else {
        const next = await verifyTarget(pairing, transport, target, signal);
        if (!signal.aborted) setView(next);
      }
    } catch (error) {
      if (signal.aborted) return;
      const message = (error as { message?: string } | null)?.message;
      if (action === "login") {
        setActionError(message?.startsWith("no approved login URL")
          ? "Tailscale is already signed in. Check your desktop and try again."
          : "Couldn’t open Tailscale sign-in.");
      } else {
        setActionError(action === "unpair"
          ? "Couldn’t clear the saved pairing. Try again."
          : "Couldn’t reconnect to your desktop. Try again.");
      }
    } finally {
      activeAction.current = null;
      if (!signal.aborted) setPending(null);
    }
  }, [pairing, target, transport]);

  useEffect(() => {
    if (!target) return;
    const reconnect = () => { void run("retry"); };
    const removeResume = onMobileResume(reconnect);
    let disposed = false;
    let unsubscribe: (() => void) | undefined;
    // The native readiness event also covers sign-in that finishes after resume.
    void transport?.onEvent((event) => {
      if (!disposed && event.type === "node_state" && event.state === "ready") reconnect();
    }).then((remove) => {
      if (disposed) remove();
      else unsubscribe = remove;
    }).catch(() => {});
    return () => {
      disposed = true;
      removeResume();
      unsubscribe?.();
    };
  }, [run, target, transport]);

  if (view.kind === "pairing") {
    return <MobilePairingScreen pairing={pairing} onPaired={ready} initialOrigin={view.origin} />;
  }
  if (view.kind === "reauth") {
    return (
      <MobileReauthScreen origin={view.target.origin} needsLogin={view.needsLogin}
        pending={pending} error={actionError ?? view.error ?? null}
        onRetry={() => void run("retry")} onLogin={() => void run("login")} onUnpair={() => void run("unpair")} />
    );
  }
  return (
    <MobileAuthLayout
      title={view.kind === "target_error" ? "Couldn’t check pairing" : "Connecting to Kohaku…"}
      subtitle={view.kind === "target_error" ? "Check your connection and try again." : null}
      icon={view.kind === "target_error" ? undefined : (
        <LoaderCircle size={32} className="text-muted-foreground motion-safe:animate-spin" aria-hidden="true" />
      )}
    >
      {view.kind === "target_error" ? (
        <Button type="button" className="h-12 w-full rounded-xl" onClick={() => {
          setView({ kind: "checking" });
          setTargetAttempt((attempt) => attempt + 1);
        }}>Retry</Button>
      ) : (
        <span role="status" className="sr-only">Connecting to Kohaku…</span>
      )}
    </MobileAuthLayout>
  );
}
