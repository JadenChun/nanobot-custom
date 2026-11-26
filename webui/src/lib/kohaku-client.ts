import { decodeNotification } from "../../../packages/client-events/notifications";
import type {
  ApprovalDecisionPayload,
  ChatHarness,
  HarnessBinding,
  ConnectionStatus,
  InboundEvent,
  MessageDeliveryErrorKind,
  Outbound,
  OutboundCliAppMention,
  OutboundMcpPresetMention,
  OutboundMedia,
  SessionMention,
  SidebarStatePayload,
  GoalStateWsPayload,
  FolderAccessRequest,
  WorkspaceScopePayload,
  WorkspacePermissionProjection,
  WebUIMutationOutcome,
  StopExpectation,
  InputAttemptBinding,
  InputAttemptCapability,
  InputAttemptProof,
  NextTurnSettingsAcknowledgement,
} from "./types";
import { createHostWebSocket } from "./runtime";
import {
  inputAttemptFingerprint,
  matchingAttemptProof,
  matchingAttemptResult,
  parseAttemptCapability,
  validAttemptTurnId,
} from "./input-attempt";
import {
  startMobileRunKeepAlive,
  stopMobileRunKeepAlive,
} from "./mobile-run-keep-alive";

/** WebSocket readyState constants, referenced by value to stay portable
 * across runtimes that don't expose a global ``WebSocket`` (tests, SSR). */
const WS_OPEN = 1;
const WS_CLOSING = 2;
const HOST_SOCKET_URL_PREFIX = "nanobot-host://";

function createDefaultSocket(url: string): WebSocket {
  if (url.startsWith(HOST_SOCKET_URL_PREFIX)) {
    return createHostWebSocket(url);
  }
  return new WebSocket(url);
}

/** Inbound WebSocket ``console.log`` / parse-failure ``console.warn``.
 *
 * - **Dev** (non-production bundle): **on by default** — messages appear at default log level.
 * - **Production**: off unless ``localStorage.setItem('nanobot_debug_ws','1')`` (or ``true``).
 * - **Silence anywhere**: ``localStorage.setItem('nanobot_debug_ws','0')`` (or ``false`` / ``off``).
 * Values are read on every frame; no reload needed.
 */
function wsInboundDebugEnabled(): boolean {
  if (typeof globalThis === "undefined") return false;
  try {
    if (import.meta.env.MODE === "test") return false;
    const ls = (globalThis as unknown as { localStorage?: Storage }).localStorage;
    const raw = ls?.getItem("nanobot_debug_ws")?.trim().toLowerCase() ?? "";
    if (raw === "0" || raw === "false" || raw === "off" || raw === "no") {
      return false;
    }
    if (raw === "1" || raw === "true" || raw === "on" || raw === "yes") {
      return true;
    }
    return !import.meta.env.PROD;
  } catch {
    return !import.meta.env.PROD;
  }
}

/** Shorten streaming text fields so logging stays usable for huge deltas. */
function summarizeInboundWsPayload(ev: InboundEvent): unknown {
  const kind = (ev as { event?: string }).event;
  const safe = { ...ev } as Record<string, unknown>;
  delete safe.input_attempt_source;
  delete safe.input_attempt_result;
  delete safe.input_attempt_proof;
  if (kind !== "delta" && kind !== "reasoning_delta") return safe;
  const row = { ...(ev as object) } as Record<string, unknown>;
  const text = typeof row.text === "string" ? row.text : "";
  const max = 240;
  if (text.length > max) {
    row.text = `${text.slice(0, max)}… (${text.length} chars)`;
  }
  return row;
}

type Unsubscribe = () => void;
/** Client-qualified disposition for this delivery only; never decoded from wire. */
export type LocalInputAdmission = {
  readonly turnId: string;
  readonly disposition: "unresolved" | "admitted";
};
type EventHandler = (
  ev: InboundEvent,
  admission?: LocalInputAdmission,
  liveNaturalCompletion?: true,
) => void;
type StatusHandler = (status: ConnectionStatus) => void;
type RuntimeModelHandler = (modelName: string | null, modelPreset?: string | null) => void;
type SessionUpdateScope = "metadata" | "thread" | string;
type SessionUpdateHandler = (
  chatId: string,
  scope?: SessionUpdateScope,
  workspaceScope?: WorkspaceScopePayload,
  workspaceAccessRequest?: FolderAccessRequest,
  workspacePermissions?: WorkspacePermissionProjection,
  nextTurn?: NextTurnSessionUpdate,
) => void;
type SidebarStateUpdateHandler = (state: SidebarStatePayload) => void;
type SettingsUpdatedHandler = () => void;
type RunStatusHandler = (chatId: string, startedAt: number | null) => void;

/** Detached next-turn settings projected by attach/session updates. */
export interface NextTurnSettingsSnapshot {
  workspaceScope?: WorkspaceScopePayload | null;
  modelPreset?: string | null;
  reasoningEffort?: string | null;
  revision?: number | null;
}

/** Next-turn facts carried by one ``session_updated`` notification. */
export interface NextTurnSessionUpdate {
  workspaceScope?: WorkspaceScopePayload | null;
  revision?: number | null;
}

type NextTurnSettingsHandler = (settings: NextTurnSettingsSnapshot) => void;

interface PendingNextTurnScope {
  chatId: string;
  socket: WebSocket;
  /** Last revision observed when the request started; older acks are stale. */
  baselineRevision: number | null;
  workspaceScope: WorkspaceScopePayload;
  resolve: (ack: NextTurnSettingsAcknowledgement) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/** Terminal run event consumed by the global task-notification hook. */
export type TaskNotificationEvent =
  | {
      kind: "turn_end";
      chatId: string;
      turnId?: string;
      outcome?: string;
      failureKind?: string;
      failureMessage?: string;
      failureErrorKind?: string;
      failureAttempts?: number;
    }
  | {
      kind: "recovery";
      chatId: string;
      status: "awaiting_user" | "failed";
    };

type TaskNotificationHandler = (event: TaskNotificationEvent) => void;

/** Structured errors surfaced to the UI.
 *
 * Most entries are transport-level or protocol-level faults. Runtime and
 * workspace failures are promoted here when they must remain visible outside
 * the message stream.
 */
export type StreamError =
  /** Server rejected the inbound frame as too large (WS close code 1009).
   * This is the transport fallback after text and attachment policies have
   * already been checked independently. */
  | { kind: "message_too_big"; chatId?: string; turnId?: string }
  | {
      kind: "workspace_scope_rejected";
      reason?: string;
      chatId?: string;
      turnId?: string;
    }
  | {
      kind: "turn_rejected";
      detail?: string;
      reason?: string;
      chatId: string;
      turnId: string;
    }
  | {
      kind: "model_request_failed";
      chatId: string;
      turnId?: string;
      errorKind?: string;
      attempts?: number;
    }
  | {
      /** The socket closed before the gateway acknowledged this send. */
      kind: "connection_closed";
      chatId: string;
      turnId: string;
    }
  | {
      /** Recovery blocks the chat while the input still lacks admission proof. */
      kind: "session_recovery_required";
      chatId: string;
      turnId: string;
    };

export function messageDeliveryErrorKind(error: StreamError): MessageDeliveryErrorKind | undefined {
  if (error.kind === "model_request_failed") return undefined;
  if (
    error.kind === "turn_rejected"
    && error.detail === "not_admitted"
    && error.reason === "session_recovery_required"
  ) return "session_recovery_required";
  return error.kind;
}

type ErrorHandler = (error: StreamError) => void;

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface PendingWebUIRequest extends PendingRequest<unknown> {
  serializedFrame: string;
}

class WebUIMutationError extends Error {
  status: number;
  workspaceModeConflict?: unknown;
  outcome?: WebUIMutationOutcome;
  requestId?: string;

  constructor(
    status: number,
    message: string,
    workspaceModeConflict?: unknown,
    options?: { outcome?: WebUIMutationOutcome; requestId?: string },
  ) {
    super(message);
    this.status = status;
    this.workspaceModeConflict = workspaceModeConflict;
    this.outcome = options?.outcome;
    this.requestId = options?.requestId;
    this.name = "WebUIMutationError";
  }
}

interface PendingChatRequest extends PendingRequest<string> {
  temporary: boolean;
}

const SYSTEM_COMMAND_TURN_PREFIX = "webui-system:";
const TURN_REJECTION_DETAILS = new Set([
  "access_denied",
  "attachment_rejected",
  "message_rejected",
  "missing content",
  "workspace_scope_rejected",
]);

export function isSystemCommandTurnId(value: string | null | undefined): value is string {
  return typeof value === "string" && value.startsWith(SYSTEM_COMMAND_TURN_PREFIX);
}

export interface KohakuClientOptions {
  url: string;
  reconnect?: boolean;
  /** Maximum UTF-8 bytes accepted for one websocket message. */
  maxFrameBytes?: number;
  /** Called when a connection drops so the app can refresh its token. */
  onReauth?: () => Promise<string | null>;
  /** Inject a custom WebSocket factory (used by unit tests). */
  socketFactory?: (url: string) => WebSocket;
  /** Delay-cap for reconnect backoff (ms). */
  maxBackoffMs?: number;
}

export interface CanonicalRunSnapshot {
  /** Explicit, well-typed live root activity fields from the matching HTTP response. */
  activitySnapshotAvailable: boolean;
  /** User turn ids present in the canonical transcript page. */
  observedTurnIds: readonly string[];
  /** Whether the server still considers the transcript tail active. */
  hasPendingToolCalls: boolean;
  /** Exact active turn when supplied by a current gateway. */
  activeTurnId?: string | null;
}

export interface ApprovalDecisionIdentity {
  sessionKey?: string;
  sessionIncarnation?: string;
  runId?: string;
  turnId?: string;
  effectId?: string;
  generation?: number;
}

type PendingMessageState = "queued" | "sent" | "unknown" | "accepted";

export interface QueuedAttemptBinding {
  queueKey: string | null;
  promptId: string;
  turnId: string;
}

export type QueuedAttemptOutcome =
  | { kind: "admitted" }
  | { kind: "not_admitted"; reason: string };

interface QueuedAttemptObservation {
  binding: QueuedAttemptBinding;
  outcome?: QueuedAttemptOutcome;
  promise: Promise<QueuedAttemptOutcome>;
  resolve: (outcome: QueuedAttemptOutcome) => void;
  consumed: boolean;
}

interface PendingMessageSend {
  chatId: string;
  turnId: string;
  startsNewRun: boolean;
  state: PendingMessageState;
  transportSettled?: boolean;
  queuedAttempt?: QueuedAttemptObservation;
  recovery?: {
    capability: InputAttemptCapability;
    binding: InputAttemptBinding | null;
    proven: boolean;
  };
}

/**
 * Singleton WebSocket client that multiplexes chat streams.
 *
 * One socket carries many chat_ids: the server tags every outbound event with
 * ``chat_id``, and this class fans those events out to handlers registered
 * per chat. Reconnects are transparent and re-attach every known chat_id.
 */
export class KohakuClient {
  private socket: WebSocket | null = null;
  private statusHandlers = new Set<StatusHandler>();
  private runtimeModelHandlers = new Set<RuntimeModelHandler>();
  private sessionUpdateHandlers = new Set<SessionUpdateHandler>();
  private sidebarStateUpdateHandlers = new Set<SidebarStateUpdateHandler>();
  private settingsUpdatedHandlers = new Set<SettingsUpdatedHandler>();
  private runStatusHandlers = new Set<RunStatusHandler>();
  private taskNotificationHandlers = new Set<TaskNotificationHandler>();
  private errorHandlers = new Set<ErrorHandler>();
  /** Runs submitted by this client, keyed by chat id so any ending turn can notify. */
  private notifiedTurnKeys = new Set<string>();
  private ownedRunTurnIdsByChatId = new Map<string, Set<string>>();
  /** Side-channel turns stay quiet even when their chat has an owned run. */
  private sideChannelTurnIdsByChatId = new Map<string, Set<string>>();
  // chat_id -> handlers listening on it
  private chatHandlers = new Map<string, Set<EventHandler>>();
  /** Inbound frames received while no subscriber is registered (e.g. user switched away). */
  private pendingInboundByChat = new Map<string, {
    event: InboundEvent;
    admission?: LocalInputAdmission;
  }[]>();
  private static readonly PENDING_INBOUND_MAX = 2000;
  // chat_ids we've attached to since connect; re-attached after reconnects
  private knownChats = new Set<string>();
  /** Live socket whose attach handshake for a chat is still valid. */
  private attachedSockets = new Map<string, WebSocket>();
  /** Chats whose source incarnation changed; they need a fresh attach. */
  private sourceReplacedChats = new Set<string>();
  /** Temporary chats are connection-owned and intentionally not reattached. */
  private temporaryChatIds = new Set<string>();
  /** Wall-clock run strip: updated from ``goal_status`` even with no ``onChat`` subscriber. */
  private runStartedAtByChatId = new Map<string, number>();
  /** Per-turn clocks let a rejected newer turn fall back without borrowing its timer. */
  private runStartedAtByTurnKey = new Map<string, number>();
  /** Monotonic per-chat generation for local sends and observed backend runs. */
  private runGenerationByChatId = new Map<string, number>();
  /** Turn associated with the latest generation, retained after idle for reconciliation. */
  private latestRunTurnIdByChatId = new Map<string, string>();
  /** Submitted or running turns not yet closed by lifecycle or canonical state. */
  private unsettledRunTurnIdsByChatId = new Map<string, Set<string>>();
  /** Correlated WebUI sends retained until protocol/canonical disposition. */
  private pendingMessageSends = new Map<string, PendingMessageSend>();
  private inputAttemptSources = new Map<string, InputAttemptCapability>();
  private inputAttemptQueries = new Map<string, { socket: WebSocket; pending: PendingMessageSend }>();
  private preparationGeneration = 0;
  /** Message sends written to the current socket but not yet acknowledged. */
  private socketPendingMessageSendKeys = new Set<string>();
  /** Last application frame written, used only for conservative 1009 attribution. */
  private lastSocketMessageSendKey: string | null = null;
  /** Canonically completed turns whose delayed websocket frames must be ignored. */
  private canonicalCompletedTurnIdsByChatId = new Map<string, Set<string>>();
  private static readonly COMPLETED_TURN_FENCE_MAX = 256;
  /** Latest ``goal_state`` snapshot per ``chat_id`` (multi-session isolation). */
  private goalStateByChatId = new Map<string, GoalStateWsPayload>();
  private pendingNewChat: PendingChatRequest | null = null;
  private pendingTranscriptions = new Map<string, PendingRequest<string>>();
  private pendingAttachmentUploads = new Map<string, PendingRequest<string>>();
  private pendingSystemCommands = new Map<string, PendingRequest<void>>();
  private pendingWebUIRequests = new Map<string, PendingWebUIRequest>();
  /** Detached next-turn facts keyed by chat id; never active-turn authority. */
  private nextTurnWorkspaceByChatId = new Map<string, WorkspaceScopePayload>();
  private nextModelPresetByChatId = new Map<string, string | null>();
  private nextReasoningEffortByChatId = new Map<string, string | null>();
  private nextTurnRevisionByChatId = new Map<string, number>();
  private pendingNextTurnScopes = new Map<string, PendingNextTurnScope>();
  private nextTurnSettingsHandlers = new Set<NextTurnSettingsHandler>();
  // Frames queued while the socket is not yet OPEN
  private sendQueue: Outbound[] = [];
  private reconnectAttempts = 0;
  private authRetryFailures = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly shouldReconnect: boolean;
  private readonly maxBackoffMs: number;
  private maxFrameBytes: number | undefined;
  private socketFactory: (url: string) => WebSocket;
  private currentUrl: string;
  private status_: ConnectionStatus = "idle";
  private readyChatId: string | null = null;
  private claudeAvailable = false;
  private harnessAvailabilityHandlers = new Set<(available: boolean) => void>();
  private liveTurnControls = new Set<string>();
  private externalApprovals = new Map<string, Map<string, Extract<InboundEvent, { event: "external_approval" }>>>();
  private harnessBindings = new Map<string, HarnessBinding>();
  private attachedModelPresets = new Map<string, string | null>();
  private attachedReasoningEfforts = new Map<string, string | null>();

  // Set by ``close()`` so the onclose handler knows the drop was intentional
  // and must not schedule a reconnect or flip status back to "reconnecting".
  private intentionallyClosed = false;

  constructor(private options: KohakuClientOptions) {
    this.shouldReconnect = options.reconnect ?? true;
    this.maxBackoffMs = options.maxBackoffMs ?? 15_000;
    this.maxFrameBytes = this.normalizeMaxFrameBytes(options.maxFrameBytes);
    this.socketFactory = options.socketFactory ?? createDefaultSocket;
    this.currentUrl = options.url;
  }

  get status(): ConnectionStatus {
    return this.status_;
  }

  get defaultChatId(): string | null {
    return this.readyChatId;
  }

  /** Swap the URL (e.g. after fetching a fresh token) then reconnect. */
  updateUrl(url: string, socketFactory?: (url: string) => WebSocket): void {
    this.currentUrl = url;
    if (socketFactory) {
      this.socketFactory = socketFactory;
    }
  }

  onStatus(handler: StatusHandler): Unsubscribe {
    this.statusHandlers.add(handler);
    handler(this.status_);
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  onHarnessAvailability(handler: (available: boolean) => void): Unsubscribe {
    this.harnessAvailabilityHandlers.add(handler);
    handler(this.claudeAvailable);
    return () => { this.harnessAvailabilityHandlers.delete(handler); };
  }

  getHarnessBinding(chatId: string): HarnessBinding | undefined {
    return this.harnessBindings.get(chatId);
  }

  getAttachedModelPreset(chatId: string): string | null | undefined {
    return this.attachedModelPresets.get(chatId);
  }

  getAttachedReasoningEffort(chatId: string): string | null | undefined {
    return this.attachedReasoningEfforts.get(chatId);
  }

  onRuntimeModelUpdate(handler: RuntimeModelHandler): Unsubscribe {
    this.runtimeModelHandlers.add(handler);
    return () => {
      this.runtimeModelHandlers.delete(handler);
    };
  }

  onSessionUpdate(handler: SessionUpdateHandler): Unsubscribe {
    this.sessionUpdateHandlers.add(handler);
    return () => {
      this.sessionUpdateHandlers.delete(handler);
    };
  }

  onSidebarStateUpdate(handler: SidebarStateUpdateHandler): Unsubscribe {
    this.sidebarStateUpdateHandlers.add(handler);
    return () => {
      this.sidebarStateUpdateHandlers.delete(handler);
    };
  }

  /** Another client changed app-level settings (presets, providers, skills). */
  onSettingsUpdated(handler: SettingsUpdatedHandler): Unsubscribe {
    this.settingsUpdatedHandlers.add(handler);
    return () => {
      this.settingsUpdatedHandlers.delete(handler);
    };
  }

  onRunStatus(handler: RunStatusHandler): Unsubscribe {
    this.runStatusHandlers.add(handler);
    for (const [chatId, startedAt] of this.runStartedAtByChatId) {
      handler(chatId, startedAt);
    }
    return () => {
      this.runStatusHandlers.delete(handler);
    };
  }

  /**
   * Subscribe to terminal run events independent of the mounted chat view.
   * The app-level notification hook uses this so leaving a chat does not lose
   * a completion notification.
   */
  onTaskNotification(handler: TaskNotificationHandler): Unsubscribe {
    this.taskNotificationHandlers.add(handler);
    return () => {
      this.taskNotificationHandlers.delete(handler);
    };
  }

  /** Subscribe to transport-level faults (see :type:`StreamError`). */
  onError(handler: ErrorHandler): Unsubscribe {
    this.errorHandlers.add(handler);
    return () => {
      this.errorHandlers.delete(handler);
    };
  }

  /** Last ``goal_status`` ``started_at`` (unix sec) for *chatId*, if the turn is running. */
  getRunStartedAt(chatId: string): number | null {
    const v = this.runStartedAtByChatId.get(chatId);
    return v === undefined ? null : v;
  }

  /** Canonical lifecycle turn currently owning the run for *chatId*, if known. */
  getRunTurnId(chatId: string): string | null {
    return this.latestRunTurnIdByChatId.get(chatId) ?? null;
  }

  /** Clear the optimistic run state immediately after the user stops a turn. */
  finishRunLocally(chatId: string): void {
    const unsettled = [...(this.unsettledRunTurnIdsByChatId.get(chatId) ?? [])];
    for (const turnId of unsettled) this.settleRunTurn(chatId, turnId);
    this.latestRunTurnIdByChatId.delete(chatId);
    this.clearOwnedRuns(chatId);
    this.clearSideChannelTurns(chatId);
    stopMobileRunKeepAlive(chatId);
    if (this.runStartedAtByChatId.delete(chatId)) {
      this.emitRunStatus(chatId, null);
    }
  }

  /** Refresh transport policy after bootstrap token renewal. */
  updateMaxFrameBytes(maxFrameBytes?: number): void {
    this.maxFrameBytes = this.normalizeMaxFrameBytes(maxFrameBytes);
  }

  /** Generation captured when an HTTP thread reconciliation starts. */
  getRunGeneration(chatId: string): number {
    return this.runGenerationByChatId.get(chatId) ?? 0;
  }

  /** Whether a locally submitted lifecycle turn still lacks a terminal disposition. */
  hasUnsettledRun(chatId: string): boolean {
    return (this.unsettledRunTurnIdsByChatId.get(chatId)?.size ?? 0) > 0;
  }

  private normalizeMaxFrameBytes(value: number | undefined): number | undefined {
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      return undefined;
    }
    return Math.floor(value);
  }

  private canonicalTurnWillSettle(
    chatId: string,
    turnId: string,
    completed: ReadonlySet<string>,
    observed: ReadonlySet<string>,
    snapshot?: CanonicalRunSnapshot,
  ): boolean {
    const pending = this.pendingMessageSends.get(this.runSendKey(chatId, turnId));
    // Activity and attempt disposition are different facts. An authorized idle
    // reset may retire optimistic activity even when its input is not on this
    // page, but cannot decide whether the UNKNOWN input was admitted.
    if (snapshot && (
      !snapshot.activitySnapshotAvailable
      || snapshot.activeTurnId !== null
      || snapshot.hasPendingToolCalls
    )) return false;
    if (pending?.state === "unknown" || (pending?.recovery && !pending.recovery.proven)) return snapshot !== undefined;
    if (completed.has(turnId)) return true;
    if (!snapshot) return false;
    if (observed.has(turnId)) return true;
    return pending?.state === "accepted";
  }

  private settleNonLifecycleCanonicalSends(
    chatId: string,
    completed: ReadonlySet<string>,
    observed: ReadonlySet<string>,
    snapshot?: CanonicalRunSnapshot,
  ): void {
    for (const pending of [...this.pendingMessageSends.values()]) {
      if (pending.chatId !== chatId || pending.startsNewRun
        || this.isUnknownPendingMessageSend(chatId, pending.turnId)) continue;
      if (!this.canonicalTurnWillSettle(
        chatId,
        pending.turnId,
        completed,
        observed,
        snapshot,
      )) continue;
      this.clearPendingMessageSend(chatId, pending.turnId);
    }
  }

  private prunePendingInboundTurn(chatId: string, turnId: string): void {
    const pending = this.pendingInboundByChat.get(chatId);
    if (!pending) return;
    const remaining = pending.filter(({ event }) => (
      !("turn_id" in event)
      || event.turn_id !== turnId
    ));
    if (remaining.length > 0) this.pendingInboundByChat.set(chatId, remaining);
    else this.pendingInboundByChat.delete(chatId);
  }

  /**
   * Pure preflight for canonical reconciliation.
   *
   * Unlike ``reconcileCanonicalCompletion``, this does not add completion
   * fences, prune queued frames, settle turns, or emit run-status updates.
   */
  canReconcileCanonicalCompletion(
    chatId: string,
    expectedRunGeneration: number,
    completedTurnIds: readonly string[],
    snapshot?: CanonicalRunSnapshot,
  ): boolean {
    if (snapshot && (
      !snapshot.activitySnapshotAvailable
      || snapshot.activeTurnId !== null
      || snapshot.hasPendingToolCalls
    )) return false;
    const completed = new Set(this.canonicalCompletedTurnIdsByChatId.get(chatId));
    for (const turnId of completedTurnIds) {
      if (turnId) completed.add(turnId);
    }
    const observed = new Set(
      snapshot?.observedTurnIds.filter((turnId) => turnId.length > 0) ?? [],
    );
    const willSettle = (turnId: string): boolean => this.canonicalTurnWillSettle(
      chatId,
      turnId,
      completed,
      observed,
      snapshot,
    );
    const latestRunTurnId = this.latestRunTurnIdByChatId.get(chatId);
    const latestRunIsRepresented = (
      typeof latestRunTurnId === "string"
      && (
        completed.has(latestRunTurnId)
        || (
          observed.has(latestRunTurnId)
          && willSettle(latestRunTurnId)
        )
      )
    );
    const unsettledTurnIds = this.unsettledRunTurnIdsByChatId.get(chatId);
    const hasUnrepresentedTurn = (
      unsettledTurnIds !== undefined
      && Array.from(unsettledTurnIds).some((turnId) => !willSettle(turnId))
    );
    const hasUnidentifiedActiveRun = (
      this.runStartedAtByChatId.has(chatId)
      && latestRunTurnId === undefined
      && (snapshot === undefined || snapshot.hasPendingToolCalls)
    );
    if (hasUnrepresentedTurn || hasUnidentifiedActiveRun) return false;
    return (
      this.getRunGeneration(chatId) === expectedRunGeneration
      || latestRunIsRepresented
    );
  }

  /**
   * Atomically accept an HTTP snapshot as completed if no unrepresented run
   * started while the request was in flight.
   *
   * Completed turn ids are fenced even when the snapshot loses the generation
   * race: delayed websocket frames for older turns must never mutate newer UI.
   */
  reconcileCanonicalCompletion(
    chatId: string,
    expectedRunGeneration: number,
    completedTurnIds: readonly string[],
    snapshot?: CanonicalRunSnapshot,
  ): boolean {
    const fences = this.canonicalCompletedTurnIdsByChatId.get(chatId) ?? new Set<string>();
    for (const turnId of completedTurnIds) {
      if (!turnId) continue;
      fences.add(turnId);
    }
    while (fences.size > KohakuClient.COMPLETED_TURN_FENCE_MAX) {
      const oldest = fences.values().next().value;
      if (typeof oldest !== "string") break;
      fences.delete(oldest);
    }
    if (fences.size > 0) this.canonicalCompletedTurnIdsByChatId.set(chatId, fences);
    const pendingInbound = this.pendingInboundByChat.get(chatId);
    if (pendingInbound) {
      const remaining = pendingInbound.filter(({ event }) => {
        const turnId = "turn_id" in event && typeof event.turn_id === "string"
          ? event.turn_id
          : null;
        return turnId === null || !fences.has(turnId);
      });
      if (remaining.length > 0) this.pendingInboundByChat.set(chatId, remaining);
      else this.pendingInboundByChat.delete(chatId);
    }

    if (!this.canReconcileCanonicalCompletion(
      chatId,
      expectedRunGeneration,
      [],
      snapshot,
    )) {
      return false;
    }

    const completed = new Set(fences);
    const observed = new Set(
      snapshot?.observedTurnIds.filter((turnId) => turnId.length > 0) ?? [],
    );
    const unsettledTurnIds = this.unsettledRunTurnIdsByChatId.get(chatId);
    if (unsettledTurnIds) {
      for (const turnId of [...unsettledTurnIds]) {
        if (!this.canonicalTurnWillSettle(
          chatId,
          turnId,
          completed,
          observed,
          snapshot,
        )) continue;
        unsettledTurnIds.delete(turnId);
        // Fence omitted optimistic turns too: a later output frame must not
        // silently consume their uncertainty or revive the retired cursor.
        fences.add(turnId);
        this.prunePendingInboundTurn(chatId, turnId);
        if (!this.isUnknownPendingMessageSend(chatId, turnId)) {
          this.clearPendingMessageSend(chatId, turnId);
        }
        this.runStartedAtByTurnKey.delete(this.runSendKey(chatId, turnId));
      }
      if (unsettledTurnIds.size === 0) this.unsettledRunTurnIdsByChatId.delete(chatId);
    }
    this.settleNonLifecycleCanonicalSends(chatId, completed, observed, snapshot);
    while (fences.size > KohakuClient.COMPLETED_TURN_FENCE_MAX) {
      const oldest = fences.values().next().value;
      if (typeof oldest !== "string") break;
      fences.delete(oldest);
    }
    if (fences.size > 0) this.canonicalCompletedTurnIdsByChatId.set(chatId, fences);
    const ownedTurns = this.ownedRunTurnIdsByChatId.get(chatId);
    if (ownedTurns) {
      // An idle snapshot can also retire an unacknowledged send. Only
      // announce completion when the gateway has evidence of admission.
      for (const turnId of ownedTurns) {
        if (this.isUnknownPendingMessageSend(chatId, turnId)
          && !completed.has(turnId) && !observed.has(turnId)) continue;
        this.emitTaskNotificationForEvent(chatId, {
          event: "turn_end", chat_id: chatId, turn_id: turnId, outcome: "completed",
        });
      }
    }
    this.clearOwnedRuns(chatId);
    this.clearSideChannelTurns(chatId);
    stopMobileRunKeepAlive(chatId);
    this.runStartedAtByChatId.delete(chatId);
    // Other consumers can learn running from HTTP without seeing a local WS
    // clock. Every accepted current idle commit must settle that projection.
    this.emitRunStatus(chatId, null);
    return true;
  }

  /** Last ``goal_state`` payload for *chatId*, if any frame has arrived this connection. */
  getGoalState(chatId: string): GoalStateWsPayload | undefined {
    return this.goalStateByChatId.get(chatId);
  }

  private advanceRunGeneration(chatId: string, turnId?: string): void {
    this.runGenerationByChatId.set(chatId, this.getRunGeneration(chatId) + 1);
    if (turnId) {
      this.latestRunTurnIdByChatId.set(chatId, turnId);
      const unsettled = this.unsettledRunTurnIdsByChatId.get(chatId) ?? new Set<string>();
      unsettled.add(turnId);
      this.unsettledRunTurnIdsByChatId.set(chatId, unsettled);
    } else {
      this.latestRunTurnIdByChatId.delete(chatId);
    }
  }

  private settleRunTurn(chatId: string, turnId?: string): void {
    if (!turnId) return;
    if (!this.isUnknownPendingMessageSend(chatId, turnId)) {
      this.clearPendingMessageSend(chatId, turnId);
    }
    this.runStartedAtByTurnKey.delete(this.runSendKey(chatId, turnId));
    const unsettled = this.unsettledRunTurnIdsByChatId.get(chatId);
    if (!unsettled) return;
    unsettled.delete(turnId);
    if (unsettled.size === 0) this.unsettledRunTurnIdsByChatId.delete(chatId);
  }

  private runSendKey(chatId: string, turnId: string): string {
    return `${chatId}\u0000${turnId}`;
  }

  private ownRun(chatId: string, turnId: string): void {
    const turns = this.ownedRunTurnIdsByChatId.get(chatId) ?? new Set<string>();
    turns.add(turnId);
    this.ownedRunTurnIdsByChatId.set(chatId, turns);
  }

  private disownRun(chatId: string, turnId: string): void {
    const turns = this.ownedRunTurnIdsByChatId.get(chatId);
    if (!turns) return;
    turns.delete(turnId);
    if (turns.size === 0) this.ownedRunTurnIdsByChatId.delete(chatId);
  }

  private moveOwnedRun(chatId: string, fromTurnId: string, toTurnId: string): void {
    const turns = this.ownedRunTurnIdsByChatId.get(chatId);
    if (!turns?.delete(fromTurnId)) return;
    turns.add(toTurnId);
  }

  private clearOwnedRuns(chatId: string): void {
    this.ownedRunTurnIdsByChatId.delete(chatId);
  }

  private hasOwnedRun(chatId: string): boolean {
    return (this.ownedRunTurnIdsByChatId.get(chatId)?.size ?? 0) > 0;
  }

  private isOwnedRun(chatId: string, turnId: string): boolean {
    return this.ownedRunTurnIdsByChatId.get(chatId)?.has(turnId) ?? false;
  }

  private addSideChannelTurn(chatId: string, turnId: string): void {
    const turns = this.sideChannelTurnIdsByChatId.get(chatId) ?? new Set<string>();
    turns.add(turnId);
    this.sideChannelTurnIdsByChatId.set(chatId, turns);
  }

  private hasSideChannelTurn(chatId: string, turnId: string): boolean {
    return this.sideChannelTurnIdsByChatId.get(chatId)?.has(turnId) ?? false;
  }

  private consumeSideChannelTurn(chatId: string, turnId: string): boolean {
    const turns = this.sideChannelTurnIdsByChatId.get(chatId);
    if (!turns) return false;
    const existed = turns.delete(turnId);
    if (turns.size === 0) this.sideChannelTurnIdsByChatId.delete(chatId);
    return existed;
  }

  private clearSideChannelTurns(chatId: string): void {
    this.sideChannelTurnIdsByChatId.delete(chatId);
  }

  private emitTaskNotification(event: TaskNotificationEvent): void {
    for (const handler of this.taskNotificationHandlers) {
      try {
        handler(event);
      } catch {
        // A notification subscriber must not stall transport bookkeeping.
      }
    }
  }

  /**
   * Emit before dispatching the event to chat subscribers. The recovery
   * handler calls finishRunLocally(), which clears ownership; notifying first
   * preserves the interruption/completion alert.
   */
  private emitTaskNotificationForEvent(chatId: string, ev: InboundEvent): void {
    if (!this.hasOwnedRun(chatId)) return;
    if (ev.event === "turn_end") {
      const turnId = typeof ev.turn_id === "string" ? ev.turn_id : undefined;
      if (turnId && (!this.isOwnedRun(chatId, turnId) || this.hasSideChannelTurn(chatId, turnId))) {
        return;
      }
      if (turnId) {
        const key = this.runSendKey(chatId, turnId);
        if (this.notifiedTurnKeys.has(key)) return;
        this.notifiedTurnKeys.add(key);
        if (this.notifiedTurnKeys.size > KohakuClient.COMPLETED_TURN_FENCE_MAX) {
          const oldest = this.notifiedTurnKeys.values().next().value;
          if (oldest !== undefined) this.notifiedTurnKeys.delete(oldest);
        }
      }
      const attempts = typeof ev.failure_attempts === "number"
        && Number.isInteger(ev.failure_attempts)
        && ev.failure_attempts > 0
        ? ev.failure_attempts
        : undefined;
      this.emitTaskNotification({
        kind: "turn_end",
        chatId,
        ...(turnId ? { turnId } : {}),
        ...(typeof ev.outcome === "string" ? { outcome: ev.outcome } : {}),
        ...(typeof ev.failure_kind === "string" ? { failureKind: ev.failure_kind } : {}),
        ...(typeof ev.failure_message === "string" ? { failureMessage: ev.failure_message } : {}),
        ...(typeof ev.failure_error_kind === "string"
          ? { failureErrorKind: ev.failure_error_kind }
          : {}),
        ...(attempts !== undefined ? { failureAttempts: attempts } : {}),
      });
      return;
    }
    if (ev.event === "recovery_state") {
      if (ev.status !== "awaiting_user" && ev.status !== "failed") return;
      if (ev.turn_id && !this.isOwnedRun(chatId, ev.turn_id)) return;
      this.emitTaskNotification({ kind: "recovery", chatId, status: ev.status });
    }
  }

  /**
   * Release the Android keepalive after the chat hook has seen the event.
   * Side-channel turns must not end an owned chat's keepalive.
   */
  private settleKeepAliveForEvent(chatId: string, ev: InboundEvent): void {
    if (!this.hasOwnedRun(chatId)) return;
    if (ev.event === "turn_end") {
      const turnId = typeof ev.turn_id === "string" ? ev.turn_id : undefined;
      if (turnId && (this.consumeSideChannelTurn(chatId, turnId) || !this.isOwnedRun(chatId, turnId))) {
        return;
      }
      this.clearOwnedRuns(chatId);
      stopMobileRunKeepAlive(chatId);
      return;
    }
    if (
      ev.event === "recovery_state"
      && (!ev.turn_id || this.isOwnedRun(chatId, ev.turn_id))
      && (
        ev.status === "awaiting_user"
        || ev.status === "failed"
        || ev.status === "recovered"
      )
    ) {
      this.clearOwnedRuns(chatId);
      this.clearSideChannelTurns(chatId);
      stopMobileRunKeepAlive(chatId);
    }
  }

  private trackPendingMessageSend(
    chatId: string,
    turnId: string,
    startsNewRun: boolean,
  ): void {
    const key = this.runSendKey(chatId, turnId);
    const existing = this.pendingMessageSends.get(key);
    if (existing?.queuedAttempt) {
      existing.startsNewRun = startsNewRun;
      existing.state = "queued";
      existing.transportSettled = false;
      return;
    }
    this.pendingMessageSends.set(key, { chatId, turnId, startsNewRun, state: "queued" });
  }

  private sameQueuedAttempt(
    left: QueuedAttemptBinding,
    right: QueuedAttemptBinding,
  ): boolean {
    return left.queueKey === right.queueKey
      && left.promptId === right.promptId
      && left.turnId === right.turnId;
  }

  /** Reserve one exact queued binding before upload or transport can begin. */
  prepareQueuedAttempt(
    chatId: string,
    binding: QueuedAttemptBinding,
    startsNewRun: boolean,
  ): boolean {
    if (
      !chatId
      || (binding.queueKey !== null && binding.queueKey !== chatId)
      || !binding.promptId.trim()
      || !binding.turnId.trim()
    ) return false;
    const key = this.runSendKey(chatId, binding.turnId);
    if (this.pendingMessageSends.has(key)) return false;
    let resolve!: (outcome: QueuedAttemptOutcome) => void;
    const promise = new Promise<QueuedAttemptOutcome>((resolvePromise) => {
      resolve = resolvePromise;
    });
    this.pendingMessageSends.set(key, {
      chatId,
      turnId: binding.turnId,
      startsNewRun,
      state: "queued",
      queuedAttempt: {
        binding: { ...binding },
        promise,
        resolve,
        consumed: false,
      },
    });
    return true;
  }

  /** Read-join only; this never creates a pending send or submits a frame. */
  waitForQueuedAttempt(
    chatId: string,
    binding: QueuedAttemptBinding,
  ): Promise<QueuedAttemptOutcome> | null {
    if (binding.queueKey !== null && binding.queueKey !== chatId) return null;
    const pending = this.pendingMessageSends.get(this.runSendKey(chatId, binding.turnId));
    const observation = pending?.queuedAttempt;
    if (
      !observation
      || observation.consumed
      || !this.sameQueuedAttempt(observation.binding, binding)
    ) return null;
    return observation.promise;
  }

  /** Consume only after the composer durably projected this exact outcome. */
  consumeQueuedAttempt(chatId: string, binding: QueuedAttemptBinding): boolean {
    const key = this.runSendKey(chatId, binding.turnId);
    const pending = this.pendingMessageSends.get(key);
    const observation = pending?.queuedAttempt;
    if (
      !observation
      || observation.consumed
      || !observation.outcome
      || !this.sameQueuedAttempt(observation.binding, binding)
    ) return false;
    observation.consumed = true;
    pending.queuedAttempt = undefined;
    if (pending.transportSettled) this.pendingMessageSends.delete(key);
    return true;
  }

  private recordQueuedAttemptOutcome(
    chatId: string,
    turnId: string,
    outcome: QueuedAttemptOutcome,
  ): void {
    const pending = this.pendingMessageSends.get(this.runSendKey(chatId, turnId));
    if (pending?.recovery && !pending.recovery.proven) return;
    const observation = pending?.queuedAttempt;
    if (!observation || observation.outcome) return;
    observation.outcome = outcome;
    observation.resolve(outcome);
  }

  private clearPendingMessageSend(chatId: string, turnId: string): void {
    const key = this.runSendKey(chatId, turnId);
    const pending = this.pendingMessageSends.get(key);
    if (pending?.queuedAttempt && !pending.queuedAttempt.consumed) {
      pending.transportSettled = true;
      // It no longer participates in transport rejection attribution or run
      // activity, but its typed queue receipt remains joinable until consumed.
      pending.state = "queued";
    } else {
      this.pendingMessageSends.delete(key);
    }
    this.socketPendingMessageSendKeys.delete(key);
    this.sendQueue = this.sendQueue.filter((frame) => !(
      frame.type === "message"
      && frame.chat_id === chatId
      && frame.turn_id === turnId
    ));
  }

  /** Admission uncertainty is owned by attempts, not the currently displayed page. */
  hasUnknownMessageSend(chatId: string): boolean {
    return [...this.pendingMessageSends.values()].some((pending) => (
      pending.chatId === chatId && pending.state === "unknown"
    ));
  }

  private isUnknownPendingMessageSend(chatId: string, turnId: string): boolean {
    const pending = this.pendingMessageSends.get(this.runSendKey(chatId, turnId));
    return pending?.state === "unknown" || !!(pending?.recovery && !pending.recovery.proven);
  }

  private recordRunAcceptance(
    chatId: string,
    turnId?: string,
    ownerReceipt = false,
  ): void {
    if (!turnId) return;
    const key = this.runSendKey(chatId, turnId);
    const pending = this.pendingMessageSends.get(key);
    if (!pending || (pending.recovery && !pending.recovery.proven)) return;
    // A route ACK or display echo cannot turn a lost-ACK send into a known
    // admission. Keep the attempt held until the owner receipt or live output.
    if (pending.state === "unknown" && !ownerReceipt) return;
    this.socketPendingMessageSendKeys.delete(key);
    if (
      !pending.startsNewRun
      || !this.unsettledRunTurnIdsByChatId.get(chatId)?.has(turnId)
      || this.canonicalCompletedTurnIdsByChatId.get(chatId)?.has(turnId)
    ) {
      this.clearPendingMessageSend(chatId, turnId);
      return;
    }
    pending.state = "accepted";
  }

  private recordCanonicalTurnOwnership(
    ev: Extract<
      InboundEvent,
      { event: "message_accepted" | "message_admitted" | "user_message" }
    >,
  ): void {
    const activeTurnId = ev.active_turn_id;
    if (!activeTurnId) return;

    // Two clients can optimistically submit while the chat still looks idle.
    // The gateway admits exactly one owner and classifies the other message as
    // steering. Replace the local guess before its ACK can preserve the wrong
    // run identity.
    const pendingAttempt = ev.turn_id
      ? this.pendingMessageSends.get(this.runSendKey(ev.chat_id, ev.turn_id))
      : undefined;
    if (pendingAttempt?.recovery && !pendingAttempt.recovery.proven) return;
    if (pendingAttempt?.state === "unknown" && ev.event !== "message_admitted") return;
    if (ev.turn_id && ev.turn_id !== activeTurnId) {
      const pending = pendingAttempt;
      if (pending?.startsNewRun) {
        this.settleRunTurn(ev.chat_id, ev.turn_id);
        if (this.isOwnedRun(ev.chat_id, ev.turn_id)) {
          if (ev.starts_turn === true) {
            // The gateway assigned a canonical turn id to this client's run.
            this.moveOwnedRun(ev.chat_id, ev.turn_id, activeTurnId);
          } else {
            // Another client won the run; this message became a steer.
            this.disownRun(ev.chat_id, ev.turn_id);
            if (!this.hasOwnedRun(ev.chat_id)) {
              stopMobileRunKeepAlive(ev.chat_id);
            }
          }
        }
      }
    }
    if (this.latestRunTurnIdByChatId.get(ev.chat_id) !== activeTurnId) {
      this.advanceRunGeneration(ev.chat_id, activeTurnId);
    }
    if (typeof ev.started_at === "number") {
      this.runStartedAtByTurnKey.set(
        this.runSendKey(ev.chat_id, activeTurnId),
        ev.started_at,
      );
      const previous = this.runStartedAtByChatId.get(ev.chat_id);
      this.runStartedAtByChatId.set(ev.chat_id, ev.started_at);
      if (previous !== ev.started_at) this.emitRunStatus(ev.chat_id, ev.started_at);
    }
  }

  private recordRunRejection(chatId: string, turnId?: string): void {
    if (!turnId) return;
    const pending = this.pendingMessageSends.get(this.runSendKey(chatId, turnId));
    if (pending?.recovery && !pending.recovery.proven) return;
    this.clearPendingMessageSend(chatId, turnId);
    this.prunePendingInboundTurn(chatId, turnId);
    this.retireRejectedRun(chatId, turnId);
  }

  private retireRejectedRun(chatId: string, turnId: string): void {
    const rejectedOwned = this.isOwnedRun(chatId, turnId);
    const rejectedLatest = this.latestRunTurnIdByChatId.get(chatId) === turnId;
    this.settleRunTurn(chatId, turnId);
    this.consumeSideChannelTurn(chatId, turnId);
    if (rejectedOwned) {
      this.disownRun(chatId, turnId);
      if (!this.hasOwnedRun(chatId)) stopMobileRunKeepAlive(chatId);
    }
    if (!rejectedLatest) return;

    const unsettled = this.unsettledRunTurnIdsByChatId.get(chatId);
    const previousTurnId = unsettled ? Array.from(unsettled).at(-1) : undefined;
    if (previousTurnId) {
      this.latestRunTurnIdByChatId.set(chatId, previousTurnId);
      const previousStartedAt = this.runStartedAtByTurnKey.get(
        this.runSendKey(chatId, previousTurnId),
      );
      const currentStartedAt = this.runStartedAtByChatId.get(chatId);
      if (previousStartedAt === undefined) {
        if (this.runStartedAtByChatId.delete(chatId)) {
          this.emitRunStatus(chatId, null);
        }
      } else {
        this.runStartedAtByChatId.set(chatId, previousStartedAt);
        if (currentStartedAt !== previousStartedAt) {
          this.emitRunStatus(chatId, previousStartedAt);
        }
      }
      return;
    }
    this.latestRunTurnIdByChatId.delete(chatId);
    if (this.runStartedAtByChatId.delete(chatId)) {
      this.emitRunStatus(chatId, null);
    }
  }

  private legacyRejectionTarget(ev: Extract<InboundEvent, { event: "error" }>): {
    chatId: string;
    turnId: string;
  } | null {
    if (!ev.detail || !TURN_REJECTION_DETAILS.has(ev.detail)) return null;
    if (
      ev.detail === "workspace_scope_rejected"
      && ev.chat_id === undefined
      && this.pendingNewChat
    ) return null;
    const candidates = [...this.pendingMessageSends.values()].filter((pending) => (
      // A legacy error can only reject a frame currently awaiting its first
      // server disposition. Accepted or prior-connection unknown sends are
      // not safe candidates for an uncorrelated frame.
      pending.state === "sent"
      && (ev.chat_id === undefined || pending.chatId === ev.chat_id)
    ));
    if (candidates.length !== 1) return null;
    const [candidate] = candidates;
    if (
      this.lastSocketMessageSendKey
      !== this.runSendKey(candidate.chatId, candidate.turnId)
    ) return null;
    return { chatId: candidate.chatId, turnId: candidate.turnId };
  }

  private uniqueUnsettledTurnId(chatId: string): string | null {
    const unsettled = this.unsettledRunTurnIdsByChatId.get(chatId);
    if (!unsettled || unsettled.size !== 1) return null;
    return unsettled.values().next().value ?? null;
  }

  private isCanonicalCompletedTurnEvent(chatId: string, ev: InboundEvent): boolean {
    const turnId = "turn_id" in ev && typeof ev.turn_id === "string" ? ev.turn_id : null;
    return (
      turnId !== null
      && this.canonicalCompletedTurnIdsByChatId.get(chatId)?.has(turnId) === true
    );
  }

  private isSupersededRunCompletion(chatId: string, ev: InboundEvent): boolean {
    if (
      ev.event !== "turn_end"
      && !(ev.event === "goal_status" && ev.status === "idle")
    ) {
      return false;
    }
    const turnId = "turn_id" in ev && typeof ev.turn_id === "string" ? ev.turn_id : undefined;
    const latestRunTurnId = this.latestRunTurnIdByChatId.get(chatId);
    if (turnId === undefined && latestRunTurnId !== undefined) return true;
    return (
      turnId !== undefined
      && latestRunTurnId !== undefined
      && turnId !== latestRunTurnId
    );
  }

  private recordRunCompletion(chatId: string, turnId?: string): void {
    this.settleRunTurn(chatId, turnId);
    const latestRunTurnId = this.latestRunTurnIdByChatId.get(chatId);
    const closesCurrentRun = latestRunTurnId === undefined || turnId === latestRunTurnId;
    if (closesCurrentRun && this.runStartedAtByChatId.delete(chatId)) {
      this.emitRunStatus(chatId, null);
    }
  }

  private recordGoalStatusForRunStrip(chatId: string, ev: InboundEvent): void {
    if (ev.event === "turn_end") {
      this.recordRunCompletion(chatId, ev.turn_id);
      return;
    }
    if (ev.event !== "goal_status") return;
    if (ev.status === "running" && typeof ev.started_at === "number") {
      this.advanceRunGeneration(chatId, ev.turn_id);
      if (ev.turn_id) {
        this.runStartedAtByTurnKey.set(
          this.runSendKey(chatId, ev.turn_id),
          ev.started_at,
        );
      }
      const previous = this.runStartedAtByChatId.get(chatId);
      this.runStartedAtByChatId.set(chatId, ev.started_at);
      if (previous !== ev.started_at) this.emitRunStatus(chatId, ev.started_at);
    } else {
      this.recordRunCompletion(chatId, ev.turn_id);
    }
  }

  private recordGoalStateSnapshot(chatId: string, ev: InboundEvent): void {
    if (ev.event === "goal_state") {
      this.goalStateByChatId.set(chatId, ev.goal_state);
      return;
    }
    if (ev.event === "turn_end" && ev.goal_state != null && typeof ev.goal_state === "object") {
      this.goalStateByChatId.set(chatId, ev.goal_state);
    }
  }

  /** Subscribe to events for a given chat_id. Auto-attaches on the next open. */
  onChat(chatId: string, handler: EventHandler): Unsubscribe {
    let handlers = this.chatHandlers.get(chatId);
    const firstListener = handlers === undefined;
    if (!handlers) {
      handlers = new Set();
      this.chatHandlers.set(chatId, handlers);
    }
    handlers.add(handler);
    const pending = this.pendingInboundByChat.get(chatId);
    if (pending !== undefined && pending.length > 0) {
      const flushed = pending.splice(0);
      this.pendingInboundByChat.delete(chatId);
      for (const { event, admission } of flushed) {
        if (admission) handler(event, admission);
        else handler(event);
      }
    }
    for (const approval of this.externalApprovals.get(chatId)?.values() ?? []) {
      handler(approval);
    }
    // The socket subscribes to the chat, not to each local listener.
    if (firstListener) this.attach(chatId);
    return () => {
      const current = this.chatHandlers.get(chatId);
      if (!current) return;
      current.delete(handler);
      if (current.size === 0) this.chatHandlers.delete(chatId);
    };
  }

  connect(): void {
    if (this.socket && this.socket.readyState < WS_CLOSING) return;
    this.intentionallyClosed = false;
    this.setStatus("connecting");
    const sock = this.socketFactory(this.currentUrl);
    this.socket = sock;
    sock.onopen = () => this.handleOpen(sock);
    sock.onmessage = (ev) => this.handleMessage(sock, ev);
    sock.onerror = () => {
      if (this.socket === sock) this.setStatus("error");
    };
    sock.onclose = (ev) => this.handleClose(sock, ev);
  }

  async retry(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.intentionallyClosed = false;
    this.reconnectAttempts = 0;
    this.authRetryFailures = 0;
    // Force-close an existing half-open or wedged socket so manual/resume retry
    // always establishes a fresh connection instead of returning early in connect().
    if (this.socket && this.socket.readyState < WS_CLOSING) {
      const stale = this.socket;
      this.socket = null;
      try {
        stale.close();
      } catch {
        // ignore close errors on already-broken sockets
      }
    }
    if (this.options.onReauth) {
      try {
        const refreshed = await this.options.onReauth();
        if (refreshed) this.currentUrl = refreshed;
      } catch {
        // Keep the current URL; the notice remains actionable.
      }
    }
    this.connect();
  }

  close(): void {
    this.intentionallyClosed = true;
    this.preparationGeneration += 1;
    this.inputAttemptSources.clear();
    this.inputAttemptQueries.clear();
    this.attachedSockets.clear();
    this.sourceReplacedChats.clear();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const sock = this.socket;
    this.socket = null;
    try {
      sock?.close();
    } catch {
      // ignore
    }
    this.clearTemporaryChats();
    this.ownedRunTurnIdsByChatId.clear();
    this.sideChannelTurnIdsByChatId.clear();
    stopMobileRunKeepAlive();
    this.setStatus("closed");
  }

  discardTemporaryChat(chatId: string): void {
    if (!this.temporaryChatIds.has(chatId)) return;
    if (this.socket?.readyState === WS_OPEN) {
      this.rawSend({ type: "discard_temporary_chat", chat_id: chatId });
    }
    this.forgetTemporaryChat(chatId);
  }

  /** Ask the server to provision a new chat_id; resolves with the assigned id. */
  newChat(
    timeoutMs: number = 5_000,
    workspaceScope?: WorkspaceScopePayload | null,
    harness?: ChatHarness,
    selection?: { modelPreset?: string | null; reasoningEffort?: string | null },
  ): Promise<string> {
    if (this.pendingNewChat) {
      return Promise.reject(new Error("newChat already in flight"));
    }
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingNewChat = null;
        reject(new Error("newChat timed out"));
      }, timeoutMs);
      this.pendingNewChat = { resolve, reject, timer, temporary: false };
      this.queueSend({
        type: "new_chat",
        ...(harness === "claude_sdk" ? { harness } : {}),
        ...(workspaceScope ? { workspace_scope: workspaceScope } : {}),
        ...(selection?.modelPreset ? { model_preset: selection.modelPreset } : {}),
        ...(selection?.reasoningEffort ? { reasoning_effort: selection.reasoningEffort } : {}),
      });
    });
  }

  /** Ask the WebUI gateway to create a connection-owned non-persistent chat. */
  newTemporaryChat(timeoutMs: number = 5_000): Promise<string> {
    if (this.pendingNewChat) {
      return Promise.reject(new Error("newChat already in flight"));
    }
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingNewChat = null;
        reject(new Error("newTemporaryChat timed out"));
      }, timeoutMs);
      this.pendingNewChat = { resolve, reject, timer, temporary: true };
      this.queueSend({ type: "new_temporary_chat" });
    });
  }

  transcribeAudio(
    dataUrl: string,
    options?: { durationMs?: number; timeoutMs?: number },
  ): Promise<string> {
    const requestId = crypto.randomUUID();
    const timeoutMs = options?.timeoutMs ?? 120_000;
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingTranscriptions.delete(requestId);
        reject(new Error("transcription timed out"));
      }, timeoutMs);
      this.pendingTranscriptions.set(requestId, { resolve, reject, timer });
      this.queueSend({
        type: "transcribe_audio",
        request_id: requestId,
        data_url: dataUrl,
        ...(options?.durationMs !== undefined ? { duration_ms: options.durationMs } : {}),
      });
    });
  }

  uploadAttachment(
    chatId: string,
    media: OutboundMedia,
    timeoutMs: number = 120_000,
  ): Promise<string> {
    if (media.id) return Promise.resolve(media.id);
    if (!media.data_url) {
      return Promise.reject(new Error("attachment has no payload"));
    }
    const requestId = crypto.randomUUID();
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingAttachmentUploads.delete(requestId);
        reject(new Error("attachment upload timed out"));
      }, timeoutMs);
      this.pendingAttachmentUploads.set(requestId, { resolve, reject, timer });
      this.queueSend({
        type: "attachment",
        request_id: requestId,
        chat_id: chatId,
        data_url: media.data_url,
        ...(media.name ? { name: media.name } : {}),
      });
    });
  }

  /**
   * Send one WebUI mutation over the authenticated socket. Pending requests are
   * replayed with the same request_id after reconnect so the gateway can join or
   * replay the original operation. A client-side timeout still ends all retries.
   */
  requestMutation<T>(
    action: string,
    payload: Record<string, unknown> = {},
    timeoutMs: number = 20_000,
    requestId: string = crypto.randomUUID(),
  ): Promise<T> {
    const socket = this.socket;
    if (!socket || socket.readyState !== WS_OPEN) {
      return Promise.reject(
        new WebUIMutationError(503, "WebUI connection is not open", undefined, {
          outcome: "uncertain",
          requestId,
        }),
      );
    }
    const frame: Outbound = {
      type: "webui_request",
      request_id: requestId,
      action,
      payload,
    };
    if (!this.frameFitsTransport(frame)) {
      return Promise.reject(
        new WebUIMutationError(413, "WebUI mutation payload is too large", undefined, {
          outcome: "not_applied",
          requestId,
        }),
      );
    }

    return new Promise<T>((resolve, reject) => {
      let serializedFrame: string;
      try {
        serializedFrame = JSON.stringify(frame);
      } catch {
        reject(new WebUIMutationError(503, "Could not encode WebUI request", undefined, {
          outcome: "uncertain",
          requestId,
        }));
        return;
      }
      const timer = setTimeout(() => {
        this.pendingWebUIRequests.delete(requestId);
        reject(
          new WebUIMutationError(
            504,
            `WebUI request timed out after ${timeoutMs}ms`,
            undefined,
            { outcome: "uncertain", requestId },
          ),
        );
      }, timeoutMs);
      this.pendingWebUIRequests.set(requestId, {
        resolve: (value) => resolve(value as T),
        reject,
        timer,
        serializedFrame,
      });
      try {
        socket.send(serializedFrame);
      } catch {
        clearTimeout(timer);
        this.pendingWebUIRequests.delete(requestId);
        reject(new WebUIMutationError(503, "Could not send WebUI request", undefined, {
          outcome: "uncertain",
          requestId,
        }));
      }
    });
  }

  /** Ask the server to create a non-destructive fork before a user-message index. */
  forkChat(
    sourceChatId: string,
    beforeUserIndex: number,
    title?: string,
    timeoutMs: number = 5_000,
  ): Promise<string> {
    if (this.pendingNewChat) {
      return Promise.reject(new Error("newChat already in flight"));
    }
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingNewChat = null;
        reject(new Error("forkChat timed out"));
      }, timeoutMs);
      this.pendingNewChat = { resolve, reject, timer, temporary: false };
      this.queueSend({
        type: "fork_chat",
        source_chat_id: sourceChatId,
        before_user_index: beforeUserIndex,
        ...(title?.trim() ? { title: title.trim() } : {}),
      });
    });
  }

  attach(chatId: string): void {
    if (this.temporaryChatIds.has(chatId)) return;
    this.knownChats.add(chatId);
    if (this.socket?.readyState !== WS_OPEN) return;
    const socket = this.socket;
    if (this.canReuseWarmAttach(chatId, socket)) return;
    this.queueSend(this.inputAttemptAttach(chatId, socket));
  }

  /**
   * Warm revisit: the same live socket still holds this chat's subscription.
   *
   * A -> B -> A over one still-open socket reuses the live projection, so no
   * duplicate full attach or stale recovery/Stop replay is sent. A reconnect,
   * a source replacement, an unsettled run, or unresolved input-attempt proof
   * always requires a fresh attach.
   */
  private canReuseWarmAttach(chatId: string, socket: WebSocket): boolean {
    if (this.attachedSockets.get(chatId) !== socket) return false;
    if (this.sourceReplacedChats.has(chatId)) return false;
    if (this.hasUnsettledRun(chatId)) return false;
    const unsettledProof = [...this.pendingMessageSends.values()].some((pending) => (
      pending.chatId === chatId
      && pending.recovery !== undefined
      && !pending.recovery.proven
    ));
    return !unsettledProof;
  }

  /**
   * Mark a chat's cached attachment stale after an observed source replacement.
   *
   * The next subscription must perform a fresh attach so recovery, Stop and
   * canonical admission are reconciled against the new incarnation.
   */
  invalidateAttachment(chatId: string): void {
    this.attachedSockets.delete(chatId);
    this.sourceReplacedChats.add(chatId);
  }

  private inputAttemptAttach(chatId: string, socket: WebSocket): Extract<Outbound, { type: "attach" }> {
    this.inputAttemptQueries.delete(chatId);
    const candidates = [...this.pendingMessageSends.values()].filter((pending) => (
      pending.chatId === chatId && pending.recovery?.binding && !pending.recovery.proven
      && (pending.state === "sent" || pending.state === "unknown")
    ));
    const pending = candidates.length === 1 ? candidates[0] : undefined;
    const binding = pending?.recovery?.binding;
    if (!pending || !binding) return { type: "attach", chat_id: chatId };
    this.inputAttemptQueries.set(chatId, { socket, pending });
    return { type: "attach", chat_id: chatId, input_attempt_query: {
      version: 1, turn_id: binding.turn_id, source: binding.source,
      source_token: binding.source_token, fingerprint: binding.fingerprint,
    } };
  }

  private acceptInputAttemptProof(
    chatId: string, proof: InputAttemptProof, pending: PendingMessageSend,
  ): LocalInputAdmission | null {
    if (this.pendingMessageSends.get(this.runSendKey(chatId, pending.turnId)) !== pending
      || !pending.recovery?.binding
      || !matchingAttemptProof(proof, pending.recovery.binding, chatId)) return null;
    const admission: LocalInputAdmission = { turnId: pending.turnId, disposition: "admitted" };
    pending.recovery.proven = true;
    this.recordQueuedAttemptOutcome(chatId, pending.turnId, { kind: "admitted" });
    this.recordRunAcceptance(chatId, pending.turnId, true);
    return admission;
  }

  sendMessage(
    chatId: string,
    content: string,
    media?: OutboundMedia[],
    options?: {
      cliApps?: OutboundCliAppMention[];
      mcpPresets?: OutboundMcpPresetMention[];
      sessionMentions?: SessionMention[];
      quotedContext?: string;
      requestIntent?: "new" | "steer";
      targetTurnId?: string;
      turnId?: string;
      approvalDecision?: ApprovalDecisionPayload;
      stopExpectation?: StopExpectation;
      /** False for side-channel or injected messages that do not own a lifecycle. */
      startsNewRun?: boolean;
      /** Internal/side-channel turns never raise completion notifications. */
      sideChannel?: boolean;
    },
  ): void {
    const temporary = this.temporaryChatIds.has(chatId);
    const capability = temporary ? undefined : this.inputAttemptSources.get(chatId);
    if (!temporary) this.knownChats.add(chatId);
    const systemCommand = options?.turnId
      ? isSystemCommandTurnId(options.turnId)
      : false;
    const frame: Outbound = {
      type: "message",
      chat_id: chatId,
      content,
      ...(media && media.length > 0 ? { media } : {}),
      ...(options?.cliApps?.length ? { cli_apps: options.cliApps } : {}),
      ...(options?.mcpPresets?.length ? { mcp_presets: options.mcpPresets } : {}),
      ...(options?.sessionMentions?.length
        ? { session_mentions: options.sessionMentions }
        : {}),
      ...(options?.quotedContext?.trim() ? { quoted_context: options.quotedContext.trim() } : {}),
      ...(options?.turnId ? { turn_id: options.turnId } : {}),
      ...(options?.approvalDecision
        ? { approval_decision: options.approvalDecision }
        : {}),
      ...(options?.stopExpectation
        ? { stop_expectation: options.stopExpectation }
        : {}),
      // System commands are internal chat controls, not workspace requests.
      ...(systemCommand
        ? {}
        : {
            request_settings_version: 1,
            request_intent: options?.requestIntent ?? "new",
            ...(options?.targetTurnId ? { target_turn_id: options.targetTurnId } : {}),
          }),
      ...(this.liveTurnControls.has(chatId) && options?.startsNewRun !== false
        ? { live_turn_controls: true as const }
        : {}),
      webui: true,
    };
    if (!this.frameFitsTransport(frame)) {
      if (options?.turnId && isSystemCommandTurnId(options.turnId)) {
        this.rejectSystemCommand(options.turnId, "message_too_big");
      }
      this.emitError({
        kind: "message_too_big",
        chatId,
        ...(options?.turnId ? { turnId: options.turnId } : {}),
      });
      return;
    }
    if (options?.turnId && !isSystemCommandTurnId(options.turnId)) {
      if (this.pendingMessageSends.get(this.runSendKey(chatId, options.turnId))?.recovery) return;
      const startsNewRun = options.startsNewRun !== false;
      if (startsNewRun) {
        this.advanceRunGeneration(chatId, options.turnId);
        this.ownRun(chatId, options.turnId);
        startMobileRunKeepAlive(chatId);
      }
      if (options.sideChannel) this.addSideChannelTurn(chatId, options.turnId);
      this.trackPendingMessageSend(chatId, options.turnId, startsNewRun);
    }
    const turnId = options?.turnId;
    const ordinary = turnId && validAttemptTurnId(turnId) && !systemCommand
      && content.trim().length > 0 && !content.trimStart().startsWith("/")
      && !content.trimStart().startsWith("!") && !media?.length
      && !options?.cliApps?.length && !options?.mcpPresets?.length
      && !options?.sessionMentions?.length && !options?.quotedContext
      && !options?.targetTurnId && !options?.approvalDecision && !options?.stopExpectation
      && !options?.sideChannel && options?.startsNewRun !== false
      && (options?.requestIntent ?? "new") === "new";
    const pending = turnId ? this.pendingMessageSends.get(this.runSendKey(chatId, turnId)) : undefined;
    if (capability && ordinary && pending && frame.type === "message") {
      const generation = this.preparationGeneration;
      pending.recovery = { capability, binding: null, proven: false };
      void inputAttemptFingerprint(content).then((fingerprint) => {
        if (this.preparationGeneration !== generation || this.intentionallyClosed
          || this.pendingMessageSends.get(this.runSendKey(chatId, turnId)) !== pending) return;
        const binding: InputAttemptBinding = Object.freeze({ ...capability, turn_id: turnId, fingerprint });
        pending.recovery = { capability, binding, proven: false };
        frame.input_attempt = { version: 1, source: binding.source,
          source_token: binding.source_token, fingerprint };
        if (!this.frameFitsTransport(frame)) throw new Error("Input attempt frame too large");
        this.queueSend(frame);
      }).catch(() => {
        if (this.preparationGeneration !== generation || this.intentionallyClosed
          || this.pendingMessageSends.get(this.runSendKey(chatId, turnId)) !== pending) return;
        // This owner has never sent a frame. Preserve text and the durable row.
        pending.recovery = undefined;
        this.recordQueuedAttemptOutcome(chatId, turnId, {
          kind: "not_admitted", reason: "input_attempt_preparation_failed",
        });
        this.recordRunRejection(chatId, turnId);
        this.emitError({ kind: "turn_rejected", chatId, turnId,
          detail: "not_admitted", reason: "input_attempt_preparation_failed" });
      });
      return;
    }
    this.queueSend(frame);
  }

  async sendLiveApprovalDecision(
    chatId: string,
    approval: ApprovalDecisionPayload,
  ): Promise<string | null> {
    try {
      await this.requestMutation("run.approve", { chat_id: chatId, approval });
      return null;
    } catch (error) {
      if (!(error instanceof WebUIMutationError)
        || error.status !== 404
        || error.outcome !== "not_applied"
        || error.message !== "unknown WebUI mutation action") throw error;
    }
    return this.sendApprovalDecision(chatId, approval.approval_id, approval.decision, {
      sessionKey: approval.session_key,
      sessionIncarnation: approval.session_incarnation,
      runId: approval.run_id,
      generation: approval.generation,
      effectId: approval.effect_id,
    });
  }

  sendApprovalDecision(
    chatId: string,
    approvalId: string,
    decision: ApprovalDecisionPayload["decision"],
    identity?: string | ApprovalDecisionIdentity,
  ): string {
    const sourceIdentity = typeof identity === "string"
      ? { sessionKey: identity }
      : identity;
    const turnId = crypto.randomUUID();
    this.sendMessage(chatId, "approval_decision", undefined, {
      turnId,
      approvalDecision: {
        approval_id: approvalId,
        decision,
        ...(sourceIdentity?.sessionKey ? { session_key: sourceIdentity.sessionKey } : {}),
        ...(sourceIdentity?.sessionIncarnation
          ? { session_incarnation: sourceIdentity.sessionIncarnation }
          : {}),
        ...(sourceIdentity?.runId ? { run_id: sourceIdentity.runId } : {}),
        ...(sourceIdentity?.turnId ? { turn_id: sourceIdentity.turnId } : {}),
        ...(sourceIdentity?.effectId ? { effect_id: sourceIdentity.effectId } : {}),
        ...(sourceIdentity?.generation !== undefined
          ? { generation: sourceIdentity.generation }
          : {}),
      },
      startsNewRun: false,
      sideChannel: true,
    });
    return turnId;
  }

  sendSystemCommand(chatId: string, command: string, timeoutMs = 15_000): Promise<void> {
    const normalized = command.trim();
    const turnId = `${SYSTEM_COMMAND_TURN_PREFIX}${crypto.randomUUID()}`;
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingSystemCommands.delete(turnId);
        reject(new Error("system command timed out"));
      }, timeoutMs);
      this.pendingSystemCommands.set(turnId, { resolve, reject, timer });
      this.sendMessage(chatId, normalized, undefined, { turnId });
    });
  }

  setWorkspaceScope(chatId: string, workspaceScope: WorkspaceScopePayload): void {
    if (this.temporaryChatIds.has(chatId)) return;
    this.knownChats.add(chatId);
    this.queueSend({
      type: "set_workspace_scope",
      chat_id: chatId,
      workspace_scope: workspaceScope,
    });
  }

  /**
   * Persist a working-folder choice for the *next* turn only.
   *
   * The active turn keeps its admitted scope, Stop target and folder-recovery
   * request. The correlated `session_updated` acknowledgement resolves the
   * returned promise; a refusal or timeout rejects it so a pending send can
   * stay unadmitted instead of running under the previous folder.
   */
  setNextTurnWorkspaceScope(
    chatId: string,
    workspaceScope: WorkspaceScopePayload,
    timeoutMs = 10_000,
  ): Promise<NextTurnSettingsAcknowledgement> {
    if (this.temporaryChatIds.has(chatId)) {
      return Promise.reject(new Error("temporary_chat_scope_is_connection_owned"));
    }
    this.knownChats.add(chatId);
    const socket = this.socket;
    if (!socket || socket.readyState !== WS_OPEN) {
      return Promise.reject(new Error("workspace settings require an open connection"));
    }
    return new Promise<NextTurnSettingsAcknowledgement>((resolve, reject) => {
      const pending: PendingNextTurnScope = {
        chatId,
        socket,
        baselineRevision: this.nextTurnRevisionByChatId.get(chatId) ?? null,
        workspaceScope,
        resolve,
        reject,
        timer: setTimeout(() => {
          if (this.pendingNextTurnScopes.get(chatId) !== pending) return;
          this.pendingNextTurnScopes.delete(chatId);
          reject(new Error("workspace settings timed out"));
        }, timeoutMs),
      };
      this.pendingNextTurnScopes.set(chatId, pending);
      this.queueSend({
        type: "set_workspace_scope",
        chat_id: chatId,
        next_turn: true,
        next_workspace_scope: workspaceScope,
      });
    });
  }

  /** Latest detached next-turn facts this client has observed for a chat. */
  getNextTurnSettings(chatId: string): NextTurnSettingsSnapshot {
    return {
      workspaceScope: this.nextTurnWorkspaceByChatId.get(chatId) ?? null,
      modelPreset: this.nextModelPresetByChatId.get(chatId) ?? null,
      reasoningEffort: this.nextReasoningEffortByChatId.get(chatId) ?? null,
      revision: this.nextTurnRevisionByChatId.get(chatId) ?? null,
    };
  }

  /** Subscribe to detached next-turn settings projected by the gateway. */
  onNextTurnSettings(handler: NextTurnSettingsHandler): Unsubscribe {
    this.nextTurnSettingsHandlers.add(handler);
    return () => {
      this.nextTurnSettingsHandlers.delete(handler);
    };
  }

  controlDelegatedChild(
    parentKey: string,
    childId: string,
    runtimeGeneration: number,
    action: "wait" | "steer" | "stop",
    content?: string,
  ): Promise<Record<string, unknown>> {
    return this.requestMutation<Record<string, unknown>>(
      "delegation.child.control",
      {
        parent_key: parentKey,
        child_id: childId,
        runtime_generation: runtimeGeneration,
        action,
        ...(content !== undefined ? { content } : {}),
      },
    );
  }

  setSidebarState(
    state: SidebarStatePayload,
    requestId?: string,
  ): Promise<SidebarStatePayload> {
    return this.requestMutation<SidebarStatePayload>(
      "sidebar.update",
      { state },
      20_000,
      requestId,
    );
  }

  // -- internals ---------------------------------------------------------

  private setStatus(status: ConnectionStatus): void {
    if (this.status_ === status) return;
    this.status_ = status;
    for (const handler of this.statusHandlers) handler(status);
  }

  private clearRunStatusesForReconnect(): void {
    if (this.runStartedAtByChatId.size === 0) return;
    const chatIds = [...this.runStartedAtByChatId.keys()];
    this.runStartedAtByChatId.clear();
    this.runStartedAtByTurnKey.clear();
    for (const chatId of chatIds) this.emitRunStatus(chatId, null);
  }

  private handleOpen(sock: WebSocket): void {
    if (this.socket !== sock) return;
    this.clearRunStatusesForReconnect();
    this.setStatus("open");
    this.reconnectAttempts = 0;
    this.authRetryFailures = 0;
    // Re-attach every known chat_id so deliveries continue routing after a drop.
    for (const chatId of this.knownChats) {
      this.rawSend(this.inputAttemptAttach(chatId, sock));
    }
    for (const pending of this.pendingWebUIRequests.values()) {
      this.rawSendSerialized(pending.serializedFrame);
    }
    // Flush anything queued during reconnect.
    const queued = this.sendQueue.splice(0);
    for (const frame of queued) this.rawSend(frame);
  }

  private handleMessage(sock: WebSocket, ev: MessageEvent): void {
    if (this.socket !== sock) return;
    let parsed: InboundEvent;
    try {
      parsed = JSON.parse(typeof ev.data === "string" ? ev.data : "") as InboundEvent;
      if (decodeNotification(parsed) === null) return;
    } catch {
      if (wsInboundDebugEnabled()) {
        const raw = typeof ev.data === "string" ? ev.data : String(ev.data);
        console.warn(
          "[nanobot ws inbound] invalid JSON",
          raw.length > 400 ? `${raw.slice(0, 400)}… (${raw.length} chars)` : raw,
        );
      }
      return;
    }

    if (wsInboundDebugEnabled()) {
      console.log("[nanobot ws inbound]", summarizeInboundWsPayload(parsed));
    }

    const receivedTurnId = "turn_id" in parsed && typeof parsed.turn_id === "string"
      ? parsed.turn_id
      : null;

    if (parsed.event === "error" && parsed.request_id) {
      const pending = this.pendingAttachmentUploads.get(parsed.request_id);
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingAttachmentUploads.delete(parsed.request_id);
        pending.reject(new Error(parsed.reason || parsed.detail || "attachment rejected"));
        return;
      }
    }
    if (parsed.event === "attachment_stored") {
      const pending = this.pendingAttachmentUploads.get(parsed.request_id ?? "");
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingAttachmentUploads.delete(parsed.request_id ?? "");
        pending.resolve(parsed.id);
      }
      return;
    }
    if (parsed.event === "webui_response") {
      const pending = this.pendingWebUIRequests.get(parsed.request_id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pendingWebUIRequests.delete(parsed.request_id);
      if (parsed.ok) {
        pending.resolve(parsed.result);
      } else {
        const status = Number.isFinite(parsed.error?.status)
          ? parsed.error.status
          : 500;
        const message = parsed.error?.message || "WebUI mutation failed";
        const workspaceModeConflict = (
          status === 409
          && parsed.error?.workspace_scope
          && parsed.error?.workspace_permissions
        ) ? {
          error: { status, message },
          workspace_scope: parsed.error.workspace_scope,
          workspace_permissions: parsed.error.workspace_permissions,
        } : undefined;
        pending.reject(new WebUIMutationError(status, message, workspaceModeConflict, {
          outcome: parsed.error?.outcome,
          requestId: parsed.request_id,
        }));
      }
      return;
    }

    if (parsed.event === "error" && !parsed.turn_id) {
      const fallback = this.legacyRejectionTarget(parsed);
      if (fallback) {
        parsed = {
          ...parsed,
          chat_id: parsed.chat_id ?? fallback.chatId,
          turn_id: fallback.turnId,
        };
      }
    }
    if (
      (parsed.event === "goal_status" || parsed.event === "turn_end")
      && !parsed.turn_id
    ) {
      const fallbackTurnId = this.uniqueUnsettledTurnId(parsed.chat_id);
      if (
        fallbackTurnId
        && !this.isUnknownPendingMessageSend(parsed.chat_id, fallbackTurnId)
      ) {
        parsed = { ...parsed, turn_id: fallbackTurnId };
      }
    }

    const turnId = "turn_id" in parsed && typeof parsed.turn_id === "string"
      ? parsed.turn_id
      : null;
    const correlatedChatId = (parsed as { chat_id?: string }).chat_id;
    let admission: LocalInputAdmission | undefined;
    const completedTurnEvent = correlatedChatId !== undefined
      && this.isCanonicalCompletedTurnEvent(correlatedChatId, parsed);
    if (parsed.event === "message_admitted" && correlatedChatId && receivedTurnId) {
      const pending = this.pendingMessageSends.get(this.runSendKey(correlatedChatId, receivedTurnId));
      if (pending?.recovery) {
        const binding = pending.recovery.binding;
        if (!binding || !matchingAttemptProof(parsed.input_attempt_proof, binding, correlatedChatId)) return;
        admission = this.acceptInputAttemptProof(correlatedChatId, parsed.input_attempt_proof, pending) ?? undefined;
        if (!admission) return;
      }
      // Exact owner receipts remain valid after canonical activity completion;
      // this repairs the attempt without reviving the retired run below.
      this.recordQueuedAttemptOutcome(correlatedChatId, receivedTurnId, { kind: "admitted" });
    }
    if (parsed.event === "message_admitted") {
      // An exact late receipt still repairs its own attempt/queue after the
      // activity fence, but must not reinstall that old run as the live owner.
      const latestTurnId = this.latestRunTurnIdByChatId.get(parsed.chat_id);
      const retiredActiveTurn = parsed.active_turn_id
        ? this.canonicalCompletedTurnIdsByChatId.get(parsed.chat_id)?.has(parsed.active_turn_id)
        : false;
      const pendingAttempt = parsed.turn_id
        ? this.pendingMessageSends.get(this.runSendKey(parsed.chat_id, parsed.turn_id))
        : undefined;
      const supersededAttempt = latestTurnId !== undefined
        && latestTurnId !== parsed.turn_id
        && (pendingAttempt?.startsNewRun === true || latestTurnId !== parsed.active_turn_id);
      this.recordRunAcceptance(parsed.chat_id, parsed.turn_id, true);
      if (!completedTurnEvent && !retiredActiveTurn && !supersededAttempt) {
        this.recordCanonicalTurnOwnership(parsed);
      }
      if (!isSystemCommandTurnId(turnId)) {
        this.dispatch(parsed.chat_id, parsed, admission);
      }
      return;
    }
    // Output proof is useful only while the live activity still accepts it.
    // Filtering later would consume UNKNOWN invisibly behind the UI fence.
    if (completedTurnEvent && parsed.event !== "error") return;
    if (correlatedChatId && receivedTurnId) {
      if (
        parsed.event === "goal_status"
        && parsed.status === "running"
        && typeof parsed.started_at === "number"
      ) {
        // Use the exact running proof accepted by the stream hook, and only
        // after the canonical completion fence above has permitted the event.
        this.recordQueuedAttemptOutcome(correlatedChatId, receivedTurnId, { kind: "admitted" });
      } else if (
        parsed.event === "error"
        && parsed.detail === "not_admitted"
        && typeof parsed.reason === "string"
        && parsed.reason.trim()
      ) {
        this.recordQueuedAttemptOutcome(correlatedChatId, receivedTurnId, {
          kind: "not_admitted",
          reason: parsed.reason.trim(),
        });
      }
    }
    if (parsed.event === "message_accepted" || parsed.event === "user_message") {
      this.recordCanonicalTurnOwnership(parsed);
    }
    if (parsed.event === "message_accepted") {
      if (!isSystemCommandTurnId(turnId)) {
        this.dispatch(parsed.chat_id, parsed);
      }
      return;
    }
    if (isSystemCommandTurnId(turnId)) {
      if (parsed.event === "error") {
        this.rejectSystemCommand(
          turnId,
          [parsed.detail, parsed.reason].filter(Boolean).join(":") || "server error",
        );
      } else if (parsed.event === "message" || parsed.event === "turn_end") {
        this.resolveSystemCommand(turnId);
      }
      return;
    }

    if (parsed.event === "error" && correlatedChatId && turnId) {
      const key = this.runSendKey(correlatedChatId, turnId);
      const pending = this.pendingMessageSends.get(key);
      if (pending?.recovery && !pending.recovery.proven) {
        pending.state = "unknown";
        if (parsed.detail === "input_attempt_unknown" && parsed.reason === "session_recovery_required") {
          this.socketPendingMessageSendKeys.delete(key);
          this.retireRejectedRun(correlatedChatId, turnId);
          this.emitError({ kind: "session_recovery_required", chatId: correlatedChatId, turnId });
        }
        return;
      }
      this.recordRunRejection(correlatedChatId, turnId);
      if (parsed.detail !== "workspace_scope_rejected") {
        this.emitError({
          kind: "turn_rejected",
          detail: parsed.detail,
          reason: parsed.reason,
          chatId: correlatedChatId,
          turnId,
        });
      }
    } else if (parsed.event !== "error" && correlatedChatId && turnId) {
      // Lifecycle traffic is also an implicit acceptance signal for clients
      // connected to an older gateway that doesn't emit message_accepted.
      this.recordRunAcceptance(
        correlatedChatId,
        turnId,
        parsed.event !== "user_message"
          && parsed.event !== "turn_end"
          && (parsed.event !== "goal_status" || parsed.status === "running"),
      );
    }

    if (parsed.event === "ready") {
      this.claudeAvailable = parsed.harness_capabilities?.claude_sdk?.available === true;
      for (const handler of this.harnessAvailabilityHandlers) handler(this.claudeAvailable);
      this.readyChatId = parsed.chat_id;
      this.knownChats.add(parsed.chat_id);
      return;
    }

    if (parsed.event === "attached") {
      this.externalApprovals.delete(parsed.chat_id);
      if (parsed.live_turn_controls === true) this.liveTurnControls.add(parsed.chat_id);
      else this.liveTurnControls.delete(parsed.chat_id);
      if (parsed.harness_binding) this.harnessBindings.set(parsed.chat_id, parsed.harness_binding);
      if ("model_preset" in parsed) {
        this.attachedModelPresets.set(parsed.chat_id, parsed.model_preset ?? null);
      }
      if ("reasoning_effort" in parsed) {
        this.attachedReasoningEfforts.set(parsed.chat_id, parsed.reasoning_effort ?? null);
      }
      const capability = parseAttemptCapability(parsed.input_attempt_source, parsed.chat_id);
      if (capability && parsed.temporary !== true) this.inputAttemptSources.set(parsed.chat_id, capability);
      else this.inputAttemptSources.delete(parsed.chat_id);
      if (parsed.temporary === true) {
        this.attachedSockets.delete(parsed.chat_id);
        this.sourceReplacedChats.delete(parsed.chat_id);
      } else {
        this.attachedSockets.set(parsed.chat_id, sock);
        this.sourceReplacedChats.delete(parsed.chat_id);
      }
      this.applyAttachedNextTurn(parsed);
      const query = this.inputAttemptQueries.get(parsed.chat_id);
      this.inputAttemptQueries.delete(parsed.chat_id);
      const binding = query?.pending.recovery?.binding;
      if (query?.socket === sock && binding
        && matchingAttemptResult(parsed.input_attempt_result, binding, parsed.chat_id)) {
        admission = this.acceptInputAttemptProof(parsed.chat_id, parsed.input_attempt_result.proof, query.pending) ?? undefined;
      }
      if (!admission && query?.socket === sock
        && this.pendingMessageSends.get(this.runSendKey(parsed.chat_id, query.pending.turnId)) === query.pending
        && query.pending.recovery && !query.pending.recovery.proven) {
        admission = { turnId: query.pending.turnId, disposition: "unresolved" };
      }
      if (parsed.temporary === true) {
        this.temporaryChatIds.add(parsed.chat_id);
      } else {
        this.knownChats.add(parsed.chat_id);
      }
      if (
        this.pendingNewChat
        && this.pendingNewChat.temporary === (parsed.temporary === true)
      ) {
        clearTimeout(this.pendingNewChat.timer);
        this.pendingNewChat.resolve(parsed.chat_id);
        this.pendingNewChat = null;
      }
      this.dispatch(parsed.chat_id, parsed, admission);
      return;
    }

    if (parsed.event === "runtime_model_updated") {
      this.emitRuntimeModelUpdate(parsed.model_name || null, parsed.model_preset ?? null);
      return;
    }

    if (parsed.event === "transcription_result") {
      this.resolveTranscription(parsed.request_id, parsed.text);
      return;
    }

    if (parsed.event === "transcription_error") {
      this.rejectTranscription(parsed.request_id, parsed.detail || "error");
      return;
    }

    if (parsed.event === "session_updated") {
      if (parsed.harness_binding) this.harnessBindings.set(parsed.chat_id, parsed.harness_binding);
      const nextTurn = this.applyNextTurnSessionUpdate(parsed);
      this.emitSessionUpdate(
        parsed.chat_id,
        parsed.scope,
        parsed.workspace_scope,
        parsed.workspace_access_request,
        parsed.workspace_permissions,
        nextTurn,
      );
      return;
    }

    if (parsed.event === "settings_updated") {
      this.emitSettingsUpdated();
      return;
    }
    if (parsed.event === "sidebar_state_updated") {
      this.emitSidebarStateUpdate(parsed.state);
      return;
    }

    if (parsed.event === "error" && parsed.detail === "workspace_scope_rejected") {
      const pendingScope = parsed.chat_id
        ? this.pendingNextTurnScopes.get(parsed.chat_id)
        : undefined;
      if (pendingScope && pendingScope.socket === this.socket) {
        // A refused next-turn choice rejects its own send ordering; it never
        // leaves the send silently running under the previous folder.
        this.pendingNextTurnScopes.delete(pendingScope.chatId);
        clearTimeout(pendingScope.timer);
        pendingScope.reject(new Error(parsed.reason || "workspace_scope_rejected"));
      }
      this.emitError({
        kind: "workspace_scope_rejected",
        reason: parsed.reason,
        chatId: parsed.chat_id,
        turnId: parsed.turn_id,
      });
      if (this.pendingNewChat) {
        clearTimeout(this.pendingNewChat.timer);
        this.pendingNewChat.reject(new Error(`workspace_scope_rejected:${parsed.reason || ""}`));
        this.pendingNewChat = null;
      }
    }

    if (parsed.event === "error" && this.pendingNewChat) {
      clearTimeout(this.pendingNewChat.timer);
      const detail = typeof parsed.detail === "string" ? parsed.detail : "server error";
      const reason = typeof parsed.reason === "string" && parsed.reason ? `:${parsed.reason}` : "";
      this.pendingNewChat.reject(new Error(`${detail}${reason}`));
      this.pendingNewChat = null;
    }

    const chatId = (parsed as { chat_id?: string }).chat_id;
    if (chatId) {
      if (parsed.event === "external_approval") {
        if (parsed.approval) {
          const approvals = this.externalApprovals.get(chatId) ?? new Map();
          approvals.set(parsed.approval_id, parsed);
          this.externalApprovals.set(chatId, approvals);
        } else {
          const approvals = this.externalApprovals.get(chatId);
          approvals?.delete(parsed.approval_id);
          if (approvals?.size === 0) this.externalApprovals.delete(chatId);
        }
      } else if (parsed.event === "turn_end" && !this.isSupersededRunCompletion(chatId, parsed)) {
        const approvals = this.externalApprovals.get(chatId);
        for (const [id, event] of approvals ?? []) {
          if (!event.child_id) approvals?.delete(id);
        }
        if (approvals?.size === 0) this.externalApprovals.delete(chatId);
      }
      if (this.isCanonicalCompletedTurnEvent(chatId, parsed)) return;
      const supersededRunCompletion = this.isSupersededRunCompletion(chatId, parsed);
      // Check before consuming the running state. A legacy synthesized ID,
      // duplicate terminal frame, or reconnect's idle snapshot is display
      // evidence only and cannot release a new input from the queue.
      const liveNaturalCompletion = parsed.event === "turn_end"
        && this.status_ === "open"
        && receivedTurnId !== null
        && receivedTurnId === this.getRunTurnId(chatId)
        && this.runStartedAtByChatId.has(chatId)
        && this.runStartedAtByTurnKey.has(this.runSendKey(chatId, receivedTurnId))
        && (parsed.outcome === undefined || parsed.outcome === "completed");
      this.recordGoalStatusForRunStrip(chatId, parsed);
      if (supersededRunCompletion) {
        if (parsed.event === "turn_end") {
          const turnId = typeof parsed.turn_id === "string" ? parsed.turn_id : undefined;
          if (turnId) {
            this.disownRun(chatId, turnId);
            this.consumeSideChannelTurn(chatId, turnId);
          }
          if (!this.hasOwnedRun(chatId)) stopMobileRunKeepAlive(chatId);
        }
        return;
      }
      this.recordGoalStateSnapshot(chatId, parsed);
      // Notify before dispatch: the recovery handler calls finishRunLocally(),
      // which clears ownership. Keepalive release waits until after dispatch.
      this.emitTaskNotificationForEvent(chatId, parsed);
      this.dispatch(chatId, parsed, undefined, liveNaturalCompletion || undefined);
      this.settleKeepAliveForEvent(chatId, parsed);
    }
  }

  private emitRuntimeModelUpdate(modelName: string | null, modelPreset?: string | null): void {
    for (const handler of this.runtimeModelHandlers) {
      handler(modelName, modelPreset);
    }
  }

  private emitSessionUpdate(
    chatId: string,
    scope?: SessionUpdateScope,
    workspaceScope?: WorkspaceScopePayload,
    workspaceAccessRequest?: FolderAccessRequest,
    workspacePermissions?: WorkspacePermissionProjection,
    nextTurn?: NextTurnSessionUpdate,
  ): void {
    for (const handler of this.sessionUpdateHandlers) {
      if (
        workspaceAccessRequest === undefined
        && workspacePermissions === undefined
        && nextTurn === undefined
      ) {
        handler(chatId, scope, workspaceScope);
      } else if (workspacePermissions === undefined && nextTurn === undefined) {
        handler(chatId, scope, workspaceScope, workspaceAccessRequest);
      } else if (workspaceAccessRequest === undefined && nextTurn === undefined) {
        handler(chatId, scope, workspaceScope, undefined, workspacePermissions);
      } else {
        handler(
          chatId,
          scope,
          workspaceScope,
          workspaceAccessRequest,
          workspacePermissions,
          nextTurn,
        );
      }
    }
  }

  /**
   * Apply detached next-turn facts from an attach handshake.
   *
   * The attach is a fresh authoritative read of the source, but a delayed
   * handshake must not erase a newer local choice made after it was sent. So a
   * missing revision only clears the snapshot while no local change is pending.
   */
  private applyAttachedNextTurn(
    parsed: Extract<InboundEvent, { event: "attached" }>,
  ): void {
    const chatId = parsed.chat_id;
    if (parsed.temporary === true) return;
    const revision = typeof parsed.next_turn_revision === "number"
      ? parsed.next_turn_revision
      : null;
    const current = this.nextTurnRevisionByChatId.get(chatId);
    if (revision !== null && current !== undefined && revision < current) return;
    if (revision === null && this.pendingNextTurnScopes.has(chatId)) return;
    if (revision !== null) this.nextTurnRevisionByChatId.set(chatId, revision);
    else this.nextTurnRevisionByChatId.delete(chatId);
    const scope = parsed.next_workspace_scope;
    if (scope === null || scope === undefined) this.nextTurnWorkspaceByChatId.delete(chatId);
    else this.nextTurnWorkspaceByChatId.set(chatId, scope);
    const preset = parsed.next_model_preset;
    if (preset === null || preset === undefined) this.nextModelPresetByChatId.delete(chatId);
    else this.nextModelPresetByChatId.set(chatId, preset);
    const effort = parsed.next_reasoning_effort;
    if (effort === null || effort === undefined) this.nextReasoningEffortByChatId.delete(chatId);
    else this.nextReasoningEffortByChatId.set(chatId, effort);
    for (const handler of this.nextTurnSettingsHandlers) handler(this.getNextTurnSettings(chatId));
  }

  /**
   * Record detached next-turn facts from one `session_updated` frame.
   *
   * A reply for a superseded request is ignored: only a strictly newer
   * revision, or the answer to this client's own pending request, may replace
   * the latest local selection. Older replies never overwrite a newer one.
   */
  private applyNextTurnSessionUpdate(
    parsed: Extract<InboundEvent, { event: "session_updated" }>,
  ): NextTurnSessionUpdate | undefined {
    const chatId = parsed.chat_id;
    if (!("next_workspace_scope" in parsed) && !("next_turn_revision" in parsed)) {
      return undefined;
    }
    const revision = typeof parsed.next_turn_revision === "number"
      ? parsed.next_turn_revision
      : null;
    const pending = this.pendingNextTurnScopes.get(chatId);
    const nextScope = parsed.next_workspace_scope;
    const matchesPending = !!(
      pending
      && pending.socket === this.socket
      && (revision === null
        || pending.baselineRevision === null
        || revision >= pending.baselineRevision)
      && nextScope !== undefined
      && nextScope !== null
      && nextScope.project_path === pending.workspaceScope.project_path
      && nextScope.access_mode === pending.workspaceScope.access_mode
    );
    const current = this.nextTurnRevisionByChatId.get(chatId);
    const advances = revision !== null && (current === undefined || revision > current);
    if (!advances && !matchesPending) {
      // A stale or replayed acknowledgement cannot replace a newer selection.
      return undefined;
    }
    if (revision !== null) this.nextTurnRevisionByChatId.set(chatId, revision);
    if (nextScope === null) this.nextTurnWorkspaceByChatId.delete(chatId);
    else if (nextScope !== undefined) this.nextTurnWorkspaceByChatId.set(chatId, nextScope);
    const update: NextTurnSessionUpdate = {
      workspaceScope: nextScope === undefined ? undefined : nextScope,
      revision,
    };
    if (pending) {
      this.pendingNextTurnScopes.delete(chatId);
      clearTimeout(pending.timer);
      pending.resolve({
        workspace_scope: parsed.workspace_scope,
        next_workspace_scope: nextScope ?? null,
        next_turn_revision: revision,
      });
    }
    for (const handler of this.nextTurnSettingsHandlers) handler(this.getNextTurnSettings(chatId));
    return update;
  }

  private emitSidebarStateUpdate(state: SidebarStatePayload): void {
    for (const handler of this.sidebarStateUpdateHandlers) {
      handler(state);
    }
  }

  private emitSettingsUpdated(): void {
    for (const handler of this.settingsUpdatedHandlers) {
      try {
        handler();
      } catch {
        // best-effort: a subscriber fault must not stall transport bookkeeping
      }
    }
  }

  private emitRunStatus(chatId: string, startedAt: number | null): void {
    for (const handler of this.runStatusHandlers) {
      handler(chatId, startedAt);
    }
  }

  private dispatch(
    chatId: string,
    ev: InboundEvent,
    admission?: LocalInputAdmission,
    liveNaturalCompletion?: true,
  ): void {
    if (!admission && "turn_id" in ev && typeof ev.turn_id === "string") {
      const pending = this.pendingMessageSends.get(this.runSendKey(chatId, ev.turn_id));
      if (pending?.recovery && !pending.recovery.proven) {
        admission = { turnId: pending.turnId, disposition: "unresolved" };
      }
    }
    const handlers = this.chatHandlers.get(chatId);
    if (handlers !== undefined && handlers.size > 0) {
      for (const h of handlers) {
        if (liveNaturalCompletion) h(ev, admission, liveNaturalCompletion);
        else if (admission) h(ev, admission);
        else h(ev);
      }
      return;
    }
    let q = this.pendingInboundByChat.get(chatId);
    if (!q) {
      q = [];
      this.pendingInboundByChat.set(chatId, q);
    }
    // Replay restores the display, never a deferred queue-send trigger.
    q.push({ event: ev, admission });
    const over = q.length - KohakuClient.PENDING_INBOUND_MAX;
    if (over > 0) {
      q.splice(0, over);
    }
  }

  private handleClose(sock: WebSocket, event?: { code?: number }): void {
    if (this.socket !== sock) return;
    this.socket = null;
    this.inputAttemptSources.clear();
    this.inputAttemptQueries.clear();
    // A reconnect gap always needs a fresh attach for every known chat.
    this.attachedSockets.clear();
    this.clearTemporaryChats();
    const willReconnect = !this.intentionallyClosed && this.shouldReconnect;
    if (this.pendingNewChat) {
      clearTimeout(this.pendingNewChat.timer);
      this.pendingNewChat.reject(new Error("socket closed"));
      this.pendingNewChat = null;
    }
    this.rejectAllTranscriptions("socket closed");
    for (const pending of this.pendingAttachmentUploads.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error("socket closed"));
    }
    this.pendingAttachmentUploads.clear();
    if (!willReconnect) {
      for (const pending of this.pendingWebUIRequests.values()) {
        clearTimeout(pending.timer);
        pending.reject(
          new WebUIMutationError(503, "Socket closed before WebUI response"),
        );
      }
      this.pendingWebUIRequests.clear();
    }
    for (const pending of this.pendingSystemCommands.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error("socket closed"));
    }
    this.pendingSystemCommands.clear();
    for (const pending of this.pendingNextTurnScopes.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error("socket closed"));
    }
    this.pendingNextTurnScopes.clear();
    // Surface structured reasons *before* reconnect logic so the UI can
    // display the error even while the client transparently reconnects.
    // Browsers populate ``CloseEvent.code`` with the wire-level close code;
    // 1009 = Message Too Big (server's max frame guard).
    const unacknowledged = Array.from(this.socketPendingMessageSendKeys)
      .map((key) => this.pendingMessageSends.get(key))
      .filter((pending): pending is PendingMessageSend => pending !== undefined);
    if (event?.code === 1009) {
      const soleKey = unacknowledged.length === 1
        ? this.runSendKey(unacknowledged[0].chatId, unacknowledged[0].turnId)
        : null;
      if (
        unacknowledged.length === 1
        && this.lastSocketMessageSendKey === soleKey
      ) {
        const [rejected] = unacknowledged;
        this.recordRunRejection(rejected.chatId, rejected.turnId);
        this.emitError({
          kind: "message_too_big",
          chatId: rejected.chatId,
          turnId: rejected.turnId,
        });
        this.dispatch(rejected.chatId, {
          event: "error",
          detail: "message_too_big",
          chat_id: rejected.chatId,
          turn_id: rejected.turnId,
        });
      } else {
        // A close frame identifies no offending application message. Never
        // roll back multiple chats merely because they shared one socket.
        this.emitError({ kind: "message_too_big" });
      }
    }
    for (const pending of unacknowledged) {
      const current = this.pendingMessageSends.get(
        this.runSendKey(pending.chatId, pending.turnId),
      );
      if (current?.state === "sent") {
        current.state = "unknown";
        this.canonicalCompletedTurnIdsByChatId.get(pending.chatId)?.delete(pending.turnId);
      }
      if (willReconnect) {
        // The frame left the socket but never got an acknowledgement. Keep the
        // attempt unresolved until owner proof arrives.
        this.emitError({
          kind: "connection_closed",
          chatId: pending.chatId,
          turnId: pending.turnId,
        });
      }
    }
    this.socketPendingMessageSendKeys.clear();
    this.lastSocketMessageSendKey = null;
    if (!willReconnect) {
      this.setStatus("closed");
      return;
    }
    this.scheduleReconnect();
  }

  private emitError(error: StreamError): void {
    // Isolate subscribers so a throwing handler cannot abort the surrounding
    // ``handleClose`` flow (which still owes us a reconnect decision + status
    // update). We deliberately swallow here: error reporting is best-effort
    // and must never be allowed to compound the failure it's reporting.
    for (const handler of this.errorHandlers) {
      try {
        handler(error);
      } catch {
        // best-effort: subscriber fault must not stall transport bookkeeping
      }
    }
  }

  private resolveTranscription(requestId: string, text: string): void {
    const pending = this.pendingTranscriptions.get(requestId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingTranscriptions.delete(requestId);
    pending.resolve(text);
  }

  private rejectTranscription(requestId: string | undefined, detail: string): void {
    if (!requestId) {
      this.rejectAllTranscriptions(detail);
      return;
    }
    const pending = this.pendingTranscriptions.get(requestId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingTranscriptions.delete(requestId);
    pending.reject(new Error(detail));
  }

  private rejectAllTranscriptions(detail: string): void {
    for (const [requestId, pending] of this.pendingTranscriptions) {
      clearTimeout(pending.timer);
      pending.reject(new Error(detail));
      this.pendingTranscriptions.delete(requestId);
    }
  }

  private resolveSystemCommand(turnId: string): void {
    const pending = this.pendingSystemCommands.get(turnId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingSystemCommands.delete(turnId);
    pending.resolve();
  }

  private rejectSystemCommand(turnId: string, detail: string): void {
    const pending = this.pendingSystemCommands.get(turnId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingSystemCommands.delete(turnId);
    pending.reject(new Error(detail));
  }

  private scheduleReconnect(): void {
    this.clearRunStatusesForReconnect();
    if (this.reconnectAttempts >= 8) {
      this.setStatus("error");
      return;
    }
    this.setStatus("reconnecting");
    const attempt = this.reconnectAttempts++;
    // Exponential backoff: 0.5s, 1s, 2s, 4s, capped.
    const delay = Math.min(500 * 2 ** attempt, this.maxBackoffMs);
    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      if (this.options.onReauth) {
        try {
          const refreshed = await this.options.onReauth();
          if (refreshed) {
            this.currentUrl = refreshed;
            this.authRetryFailures = 0;
          } else {
            this.authRetryFailures += 1;
            if (this.authRetryFailures >= 3) {
              this.setStatus("error");
              return;
            }
          }
        } catch {
          // A thrown error signals a transient network/transport failure rather
          // than an auth rejection; continue the normal exponential backoff
          // instead of aborting permanently after 3.5s.
          if (!this.intentionallyClosed) {
            this.scheduleReconnect();
          }
          return;
        }
      }
      this.connect();
    }, delay);
  }

  private queueSend(frame: Outbound): void {
    if (this.socket?.readyState === WS_OPEN) {
      this.rawSend(frame);
    } else {
      this.sendQueue.push(frame);
    }
  }

  private clearTemporaryChats(): void {
    for (const chatId of [...this.temporaryChatIds]) {
      this.forgetTemporaryChat(chatId);
    }
  }

  private forgetTemporaryChat(chatId: string): void {
    this.temporaryChatIds.delete(chatId);
    this.knownChats.delete(chatId);
    this.chatHandlers.delete(chatId);
    this.externalApprovals.delete(chatId);
    this.harnessBindings.delete(chatId);
    this.attachedModelPresets.delete(chatId);
    this.attachedReasoningEfforts.delete(chatId);
    this.pendingInboundByChat.delete(chatId);
    const wasRunning = this.runStartedAtByChatId.delete(chatId);
    const hadOwnedRun = this.hasOwnedRun(chatId);
    this.clearOwnedRuns(chatId);
    this.clearSideChannelTurns(chatId);
    if (wasRunning || hadOwnedRun) stopMobileRunKeepAlive(chatId);
    this.runGenerationByChatId.delete(chatId);
    this.latestRunTurnIdByChatId.delete(chatId);
    this.unsettledRunTurnIdsByChatId.delete(chatId);
    this.canonicalCompletedTurnIdsByChatId.delete(chatId);
    this.goalStateByChatId.delete(chatId);
    this.nextTurnWorkspaceByChatId.delete(chatId);
    this.nextModelPresetByChatId.delete(chatId);
    this.nextReasoningEffortByChatId.delete(chatId);
    this.nextTurnRevisionByChatId.delete(chatId);
    const pendingScope = this.pendingNextTurnScopes.get(chatId);
    if (pendingScope) {
      this.pendingNextTurnScopes.delete(chatId);
      clearTimeout(pendingScope.timer);
      pendingScope.reject(new Error("temporary chat discarded"));
    }
    for (const key of [...this.runStartedAtByTurnKey.keys()]) {
      if (key.startsWith(`${chatId}\u0000`)) this.runStartedAtByTurnKey.delete(key);
    }
    for (const [key, pending] of [...this.pendingMessageSends]) {
      if (pending.chatId !== chatId) continue;
      this.pendingMessageSends.delete(key);
      this.socketPendingMessageSendKeys.delete(key);
    }
    this.sendQueue = this.sendQueue.filter((frame) => (
      !("chat_id" in frame) || frame.chat_id !== chatId
    ));
    if (this.lastSocketMessageSendKey?.startsWith(`${chatId}\u0000`)) {
      this.lastSocketMessageSendKey = null;
    }
    if (wasRunning) this.emitRunStatus(chatId, null);
  }

  private frameFitsTransport(frame: Outbound): boolean {
    if (this.maxFrameBytes === undefined) return true;
    return new TextEncoder().encode(JSON.stringify(frame)).byteLength <= this.maxFrameBytes;
  }

  private rawSend(frame: Outbound): void {
    if (!this.socket) return;
    try {
      this.socket.send(JSON.stringify(frame));
      this.lastSocketMessageSendKey = null;
      if (frame.type === "message" && frame.turn_id) {
        const key = this.runSendKey(frame.chat_id, frame.turn_id);
        const pending = this.pendingMessageSends.get(key);
        if (pending) {
          pending.state = "sent";
          this.socketPendingMessageSendKeys.add(key);
          this.lastSocketMessageSendKey = key;
        }
      }
    } catch {
      // Send failure will materialize as a close; queue the frame for retry.
      this.sendQueue.push(frame);
    }
  }

  private rawSendSerialized(serializedFrame: string): void {
    if (!this.socket) return;
    try {
      this.socket.send(serializedFrame);
    } catch {
      // The pending request remains available for the next successful reconnect.
    }
  }
}
