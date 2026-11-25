import type { MobileTransport } from "./mobile-runtime";

export type PairingStage =
  | "idle"
  | "starting_transport"
  | "tailscale_login_required"
  | "collecting"
  | "waiting_approval"
  | "expired_renewable"
  | "proving"
  | "cancelling"
  | "paired"
  | "failed";

export type TailscaleNodeState = "idle" | "starting" | "needs_login" | "ready" | "error";

export type PairingErrorKind =
  | "server_http"
  | "transport"
  | "tailscale"
  | "challenge_invalid"
  | "custody"
  | "cancelled";

export type PairingRemediation =
  | "rescan_qr"
  | "edit_code"
  | "retry"
  | "open_tailscale_login"
  | "check_desktop_serve"
  | "switch_desktop";

export interface PairingError {
  kind: PairingErrorKind;
  httpStatus?: number | null;
  serverCode?: string | null;
  endpoint?: string | null;
  retryable: boolean;
  remediation: PairingRemediation;
  title: string;
  message: string;
  technicalDetail: string;
}

export interface PairingSnapshot {
  stage: PairingStage;
  attemptId: string | null;
  method: "qr" | "link" | "code" | null;
  origin: string | null;
  installationId: string | null;
  pairingId: string | null;
  tailscaleState: TailscaleNodeState;
  tailscaleHasLoginUrl: boolean;
  comparisonCode: string | null;
  expiresAt: number | null;
  canCancel: boolean;
  canRetry: boolean;
  error: PairingError | null;
}

export interface MobilePairingTarget {
  origin: string;
  installationId: string;
  hasAccess: boolean;
  hasRefresh?: boolean;
}

export interface MobilePairingBridge {
  target(): Promise<MobilePairingTarget | null>;
  pairingState(): Promise<PairingSnapshot>;
  onPairingStateChange(listener: (snapshot: PairingSnapshot) => void): Promise<() => void>;
  pairWithLink(rawLink: string): Promise<PairingSnapshot>;
  pairWithCode(origin: string, code: string): Promise<PairingSnapshot>;
  pairRetry(): Promise<PairingSnapshot>;
  pairCancel(): Promise<PairingSnapshot>;
  pairResetToIdle(): Promise<PairingSnapshot>;
  pairCheckResume(): Promise<PairingSnapshot>;
  scanPairingQr(): Promise<{ scanned: boolean }>;
  openLogin(): Promise<void>;
  status(): Promise<Record<string, unknown>>;
  refresh(): Promise<void>;
  logout(serverReachable: boolean): Promise<void>;
}

declare global {
  interface Window {
    kohakuMobilePairing?: MobilePairingBridge;
  }
}

export function desktopHost(origin: string | null): string | null {
  try {
    return origin ? new URL(origin).host : null;
  } catch {
    return null;
  }
}

export function onMobileResume(listener: () => void): () => void {
  window.addEventListener("kohaku-native-resume", listener);
  window.addEventListener("online", listener);
  return () => {
    window.removeEventListener("kohaku-native-resume", listener);
    window.removeEventListener("online", listener);
  };
}

type TransportResult = "ready" | "needs_login" | "unreachable";

export function waitForMobileTransport(
  pairing: MobilePairingBridge,
  transport: MobileTransport,
  target: MobilePairingTarget,
  signal: AbortSignal,
): Promise<TransportResult> {
  return new Promise((resolve) => {
    let settled = false;
    let unsubscribe: (() => void) | undefined;
    let latestState: unknown;
    let readingStatus = true;
    const finish = (result: TransportResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      unsubscribe?.();
      resolve(result);
    };
    const abort = () => finish("unreachable");
    const timer = setTimeout(abort, 20_000);
    const check = (state: unknown) => {
      if (state === "ready" || state === "needs_login") finish(state);
      if (state === "error") finish("unreachable");
    };

    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      abort();
      return;
    }
    void (async () => {
      const remove = await transport.onEvent((event) => {
        if (event.type !== "node_state") return;
        latestState = event.state;
        if (!readingStatus) check(latestState);
      });
      if (settled) {
        remove();
        return;
      }
      unsubscribe = remove;
      const { origin, installationId } = target;
      await transport.start(origin, installationId, "kohaku-" + installationId.slice(-6));
      if (settled) return;

      // Capture a transition that occurs while the native status call is pending.
      latestState = undefined;
      const status = await pairing.status();
      readingStatus = false;
      check(latestState ?? status.state);
    })().catch(abort);
  });
}

function probeHandshake(transport: MobileTransport, signal: AbortSignal): Promise<number | null> {
  return new Promise((resolve) => {
    let settled = false;
    let requestId: number | undefined;
    let unsubscribe: (() => void) | undefined;
    const early = new Map<number, number | null>();
    const cancel = (id: number) => { void transport.cancel(id).catch(() => {}); };
    const finish = (status: number | null, terminal = false) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      unsubscribe?.();
      early.clear();
      if (!terminal && requestId !== undefined) cancel(requestId);
      resolve(status);
    };
    const abort = () => finish(null);
    const timer = setTimeout(abort, 8_000);

    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      abort();
      return;
    }
    void (async () => {
      const remove = await transport.onEvent((event) => {
        if (settled || (event.type !== "http_response" && event.type !== "http_error")) return;
        const id = event.request_id;
        if (id === undefined || !Number.isSafeInteger(id) || id <= 0) return;
        const status = event.type === "http_response" ? event.status ?? null : null;
        if (requestId !== undefined) {
          if (id === requestId) finish(status, true);
        } else if (!early.has(id)) {
          // A response can cross the bridge before the native request ID.
          if (early.size < 64) early.set(id, status);
          else abort();
        }
      });
      if (settled) {
        remove();
        return;
      }
      unsubscribe = remove;
      const id = await transport.http({
        operation: "mobile_handshake", method: "GET", path: "/mobile/v1/handshake",
      });
      if (!Number.isSafeInteger(id) || id <= 0) {
        abort();
        return;
      }
      if (settled) {
        cancel(id);
        return;
      }
      requestId = id;
      if (early.has(id)) finish(early.get(id) ?? null, true);
      early.clear();
    })().catch(abort);
  });
}

export async function checkMobileAccess(
  pairing: MobilePairingBridge,
  transport: MobileTransport,
  signal: AbortSignal,
): Promise<"ok" | "revoked" | "repair" | "unreachable"> {
  const status = await probeHandshake(transport, signal);
  if (status !== 401) return status === 200 ? "ok" : "unreachable";
  if (signal.aborted) return "unreachable";

  let repair = false;
  try {
    await pairing.refresh();
  } catch (error) {
    const message = (error as { message?: string } | null)?.message ?? String(error);
    // Native has already cleared custody for these two errors.
    if (message === "pairing was revoked" || message === "pairing upgrade required") return "revoked";
    if (!/^HTTP (400|401|403|410)\b/.test(message)) return "unreachable";
    repair = true;
  }
  if (await probeHandshake(transport, signal) === 200) return "ok";
  return repair ? "repair" : "unreachable";
}
