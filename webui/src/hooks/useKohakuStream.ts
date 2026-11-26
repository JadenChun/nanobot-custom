import { acceptsCompactionPhase } from "../../../packages/client-events/notifications";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useClient } from "@/providers/ClientProvider";
import { toMediaAttachment } from "@/lib/media";
import {
  applyToolProgressPatches,
  mergeToolProgressEvents,
  mergeToolProgressTraceLines,
  normalizeToolProgressEvents,
  toolTraceLinesFromEvents,
} from "@/lib/tool-traces";
import { hasPendingAgentActivity } from "@/lib/activity-timeline";
import { parseRunState } from "@/lib/run-state";
import {
  delegatedChildMatchesRoot,
  mergeDelegatedChildRows,
  parseDelegatedChildPayload,
  parseStopExpectation,
} from "@/lib/delegated-child-projection";
import {
  messageDeliveryErrorKind,
  type LocalInputAdmission,
  type QueuedAttemptBinding,
  type QueuedAttemptOutcome,
  type StreamError,
} from "@/lib/kohaku-client";
import {
  closeReasoningStream,
  contextUsageFromEvent,
  filterCoveredFileEditToolEvents,
  finalizeStreamedTurn,
  findActiveAssistantPlaceholderIndex,
  findFileEditTraceIndex,
  findStreamingAssistantIndex,
  matchesTurn,
  mergeFileEdits,
  replaceMessageAt,
  stampLastAssistantCompletion,
  stripCoveredFileEditToolHintsFromMessages,
  turnFieldsFromEvent,
} from "@/lib/thread-event-projection";
import type { UIMessageTurnFields } from "@/lib/thread-event-projection";
import { formatQuotedUserMessage } from "@/lib/user-message-quote";
import type {
  StopExpectation,
  InboundEvent,
  OutboundCliAppMention,
  OutboundMcpPresetMention,
  OutboundMedia,
  SessionMention,
  GoalStateWsPayload,
  MessageDeliveryStatus,
  RecoveryState,
  RunState,
  RetryStatus,
  ToolProgressEvent,
  UIMediaAttachment,
  UIMessage,
} from "@/lib/types";

interface StreamBuffer {
  /** ID of the assistant message currently receiving deltas (cleared when its segment closes). */
  messageId: string;
}

interface ActiveAssistantCursor {
  id: string;
  index: number;
}

type PendingStreamEvent =
  | { kind: "delta"; text: string; turn: UIMessageTurnFields; source?: UIMessage["source"] }
  | { kind: "reasoning"; text: string; turn: UIMessageTurnFields };

const BACKGROUND_STREAM_FLUSH_INTERVAL_MS = 1_000;
/** Cap streamed UI updates at ~30 fps so display refresh (60–240 Hz) cannot
 * multiply markdown parsing and layout work. Final/terminal flushes bypass
 * this interval and remain immediate. */
const STREAM_FLUSH_INTERVAL_MS = 33;

function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

/**
 * Append a reasoning chunk to the last open reasoning stream in ``prev``.
 *
 * Lookup rule: reasoning can only extend the current reasoning placeholder.
 * Once ordinary answer text has appeared, the next reasoning chunk starts a
 * fresh activity surface so streamed output stays in arrival order while the
 * final answer remains the only visible answer bubble.
 */
function attachReasoningChunk(
  prev: UIMessage[],
  chunk: string,
  segments?: {
    ensure: () => string;
  },
  turn: UIMessageTurnFields = {},
): UIMessage[] {
  for (let i = prev.length - 1; i >= 0; i -= 1) {
    const candidate = prev[i];
    // A user turn is a hard boundary: reasoning after it belongs to the new
    // assistant turn, never to an earlier assistant reply.
    if (candidate.role === "user") break;
    // A trace row (e.g. Used tools) is also a phase boundary. Reasoning after
    // tools belongs to the next assistant iteration, not the assistant turn
    // that produced those tool calls.
    if (candidate.kind === "trace") break;
    if (candidate.role !== "assistant") continue;
    if (!matchesTurn(candidate, turn)) break;
    const activitySegmentId = candidate.activitySegmentId ?? segments?.ensure();
    const hasAnswer = candidate.content.length > 0;
    if (hasAnswer) break;
    // ``reasoning_end`` closes this row even though the assistant placeholder
    // stays streaming for the rest of the turn. The next reasoning stream must
    // get its own row when an intervening tool trace is delayed or unavailable.
    if (
      candidate.reasoningStreaming
      || (candidate.isStreaming && candidate.reasoning === undefined)
    ) {
      const merged: UIMessage = {
        ...candidate,
        reasoning: (candidate.reasoning ?? "") + chunk,
        reasoningStreaming: true,
        ...(activitySegmentId ? { activitySegmentId } : {}),
        ...turn,
      };
      return [...prev.slice(0, i), merged, ...prev.slice(i + 1)];
    }
    break;
  }
  const activitySegmentId = segments?.ensure();
  return [
    ...prev,
    {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      isStreaming: true,
      reasoning: chunk,
      reasoningStreaming: true,
      ...(activitySegmentId ? { activitySegmentId } : {}),
      ...turn,
      createdAt: Date.now(),
    },
  ];
}

function absorbCompleteAssistantMessage(
  prev: UIMessage[],
  message: Omit<UIMessage, "id" | "role" | "createdAt">,
): UIMessage[] {
  const supersedes = (candidate: UIMessage): boolean => {
    const current = candidate.content.trim();
    const completed = message.content.trim();
    return current.length === 0
      || completed.startsWith(current)
      || current === completed;
  };
  for (let index = prev.length - 1; index >= 0; index -= 1) {
    const candidate = prev[index];
    if (candidate.role === "user") break;
    if (candidate.kind === "trace") continue;
    if (
      candidate.role !== "assistant"
      || !candidate.isStreaming
      || !matchesTurn(candidate, message)
      || !supersedes(candidate)
    ) continue;
    const merged: UIMessage = {
      ...candidate,
      ...message,
      id: candidate.id,
      isStreaming: false,
      reasoningStreaming: false,
      ...(candidate.activitySegmentId
        ? { activitySegmentId: candidate.activitySegmentId }
        : {}),
    };
    return prev.flatMap((item, itemIndex) => {
      if (itemIndex === index) return [merged];
      if (
        item.role === "assistant"
        && item.kind !== "trace"
        && item.isStreaming
        && item.content.trim().length > 0
        && matchesTurn(item, message)
        && supersedes(item)
      ) return [];
      return [item];
    });
  }
  return [
    ...prev,
    {
      id: crypto.randomUUID(),
      role: "assistant",
      createdAt: Date.now(),
      ...message,
    },
  ];
}

/**
 * Subscribe to a chat by ID. Returns the in-memory message list for the chat,
 * a streaming flag, and a ``send`` function. Initial history must be seeded
 * separately (e.g. via ``fetchWebuiThread``) since the server only replays
 * live events.
 */
/** Payload passed to ``send`` when the user attaches one or more files.
 *
 * ``media`` is handed to the wire client verbatim; ``preview`` powers the
 * optimistic user bubble. Keeping the two separate lets the bubble re-use the
 * local data URL even after the server persists the file under a different
 * name. */
export interface SendAttachment {
  media: OutboundMedia;
  preview: UIMediaAttachment;
}

export interface SendOptions {
  cliApps?: OutboundCliAppMention[];
  mcpPresets?: OutboundMcpPresetMention[];
  sessionMentions?: SessionMention[];
  quotedContext?: string;
  sideChannel?: boolean;
  finalizeActiveTurn?: boolean;
  /** Append guidance to the running turn without detaching its active answer segment. */
  continueActiveTurn?: boolean;
  /** The turn presented by the Steer action, not whatever runs after upload. */
  steerTargetTurnId?: string;
  /** Content-free client binding for one persisted queued submission. */
  queuedAttempt?: QueuedAttemptBinding;
}

export interface SubmittedTurn {
  turnId: string;
  userMessageId: string;
  sideChannel: boolean;
  /** Owner admission for this attempt. Resolves true only after the session
   * owner commits the input; a route-level ``message_accepted`` never resolves
   * it. Undefined for surfaces that do not need to hold queue state. */
  admission?: Promise<boolean>;
}

function findLiveProgressRow(
  messages: UIMessage[],
  patches: ToolProgressEvent[],
): number {
  const callIds = new Set<string>();
  for (const patch of patches) {
    if (patch.call_id) callIds.add(patch.call_id);
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.kind !== "trace" || !message.toolEvents?.length) continue;
    if (message.toolEvents.some((event) => event.call_id && callIds.has(event.call_id))) {
      return index;
    }
  }
  return -1;
}

function eventTurnId(ev: InboundEvent): string | undefined {
  return "turn_id" in ev && typeof ev.turn_id === "string" ? ev.turn_id : undefined;
}

function runIdentity(state: RunState | null | undefined): string | null {
  return state ? JSON.stringify([
    state.source_incarnation,
    state.turn_id,
    state.run_id,
  ]) : null;
}

function sameStopTarget(a: StopExpectation, b: StopExpectation): boolean {
  const identity = (target: StopExpectation) => JSON.stringify([
    target.source.store_identity,
    target.source.session_key,
    target.source.incarnation,
    target.source.generation,
    target.control_epoch,
    target.active_root_present,
    target.active_turn_id,
    target.active_root_run_ids,
    target.children.map((child) => [
      child.child_id,
      child.source.store_identity,
      child.source.session_key,
      child.source.incarnation,
      child.source.generation,
      child.child_run_id,
      child.run_sequence,
      child.control_epoch,
    ]),
    target.pending_work_digest,
  ]);
  return identity(a) === identity(b);
}

function isMutationError(
  error: unknown,
  expected: { status: number; message: string; outcome?: string },
): boolean {
  if (typeof error !== "object" || error === null) return false;
  const row = error as Record<string, unknown>;
  return row.status === expected.status
    && row.message === expected.message
    && (expected.outcome === undefined || row.outcome === expected.outcome);
}

function transitionTurnDelivery(
  messages: UIMessage[],
  turnId: string,
  status: MessageDeliveryStatus,
  repairUncertain = false,
): UIMessage[] {
  let changed = false;
  const next = messages.map((message) => {
    if (
      message.role !== "user"
      || message.turnId !== turnId
      || message.deliveryStatus === status
      || (status === "accepted"
        && message.deliveryStatus !== "sending"
        && !(
          repairUncertain
          && (
            message.deliveryStatus === "unknown"
            || (
              message.deliveryStatus === "failed"
              && message.deliveryErrorKind === "connection_closed"
            )
          )
        ))
    ) {
      return message;
    }
    changed = true;
    return {
      ...message,
      deliveryStatus: status,
      ...(status === "accepted" ? { deliveryErrorKind: undefined } : {}),
    };
  });
  return changed ? next : messages;
}

function appendProjectedSessionInput(
  messages: UIMessage[],
  event: Extract<InboundEvent, { event: "user_message" }>,
): UIMessage[] {
  const sessionMessage = event.provenance?.session_message;
  const messageId = sessionMessage?.message_id?.trim();
  if (!sessionMessage || !messageId) return messages;
  if (messages.some((message) => message.sessionMessage?.message_id === messageId)) return messages;

  const row: UIMessage = {
    id: `session-message:${messageId}`,
    role: "user",
    content: event.text,
    createdAt: typeof event.created_at_ms === "number"
      && Number.isFinite(event.created_at_ms)
      ? event.created_at_ms
      : Date.now(),
    sessionMessage,
    ...turnFieldsFromEvent(event, "user"),
  };
  return [...messages, row];
}

export function useKohakuStream(
  chatId: string | null,
  initialMessages: UIMessage[] = [],
  hasPendingToolCalls = false,
  onTurnEnd?: () => void,
  expectedRootSessionKey: string | null = null,
  initialStopTarget: StopExpectation | null = null,
  onProjectionRefreshNeeded?: () => void,
  initialStopTargetSnapshotAvailable = initialStopTarget !== null,
  requireVerifiedStopTarget = false,
  initialRunStateInput: RunState | null | undefined = undefined,
): {
  messages: UIMessage[];
  /** Whether ``messages`` belongs to the current ``chatId`` after a session switch. */
  messagesReady: boolean;
  isStreaming: boolean;
  /** Live, natural settlement of this chat's exact run; never inferred from idle. */
  queueCompletionTurnId: string | null;
  hasOwnedWork: boolean;
  stopCanBeSent: boolean;
  stopPending: boolean;
  stopNotice: "stale_target" | "unavailable" | null;
  stopFeedback: string | null;
  dismissStopNotice: () => void;
  stopTarget: StopExpectation | null;
  runState: RunState | null | undefined;
  connectionOpen: boolean;
  /** At least one input still awaits exact owner disposition, independent of activity. */
  hasUnknownMessageSend: boolean;
  /** Unix epoch seconds when the current user turn started (WebSocket ``goal_status``). */
  runStartedAt: number | null;
  /** Transient model retry state for the active turn. */
  retryStatus: RetryStatus | null;
  /** Latest sustained goal for this ``chatId`` (``goal_state`` WS events). */
  goalState: GoalStateWsPayload | undefined;
  recoveryState: RecoveryState | null;
  continueRecovery: () => Promise<void>;
  dismissRecovery: () => Promise<void>;
  send: (
    content: string,
    images?: SendAttachment[],
    options?: SendOptions,
  ) => SubmittedTurn | null;
  joinQueuedAttempt: (
    binding: QueuedAttemptBinding,
  ) => Promise<QueuedAttemptOutcome> | null;
  consumeQueuedAttempt: (binding: QueuedAttemptBinding) => boolean;
  transcribeAudio: (dataUrl: string, options?: { durationMs?: number }) => Promise<string>;
  stop: () => void;
  /** Mark an accepted canonical snapshot as the definitive end of the active turn. */
  reconcileTurnComplete: () => void;
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  /** Latest transport-level fault raised since the last ``dismissStreamError``.
   * ``null`` when there is nothing to show. */
  streamError: StreamError | null;
  /** Clear the current ``streamError`` (e.g. after the user dismisses the
   * notification or starts a fresh action). */
  dismissStreamError: () => void;
} {
  const { client } = useClient();
  const joinQueuedAttempt = useCallback((binding: QueuedAttemptBinding) => (
    chatId ? client.waitForQueuedAttempt(chatId, binding) : null
  ), [chatId, client]);
  const consumeQueuedAttempt = useCallback((binding: QueuedAttemptBinding) => (
    chatId ? client.consumeQueuedAttempt(chatId, binding) : false
  ), [chatId, client]);
  const initialRunStartedAt = initialRunStateInput !== undefined
    ? initialRunStateInput?.started_at ?? null
    : (chatId ? client.getRunStartedAt(chatId) : null);
  const [messages, setMessages] = useState<UIMessage[]>(initialMessages);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const [messageOwnerChatId, setMessageOwnerChatId] = useState(chatId);
  /** If history ends in unfinished agent activity, keep the loading spinner alive. */
  const initialStreaming = hasPendingAgentActivity(initialMessages);
  const [isStreaming, setIsStreaming] = useState(
    initialStreaming
    || hasPendingToolCalls
    || initialRunStartedAt !== null
    || (initialRunStateInput !== undefined && initialRunStateInput !== null),
  );
  /** Unix epoch seconds when the current user turn started; cleared on ``idle``. */
  const [runStartedAt, setRunStartedAt] = useState<number | null>(initialRunStartedAt);
  const [retryStatus, setRetryStatus] = useState<RetryStatus | null>(null);
  const [goalState, setGoalState] = useState<GoalStateWsPayload | undefined>(undefined);
  const [runState, setRunState] = useState<RunState | null | undefined>(initialRunStateInput);
  const [recoveryState, setRecoveryState] = useState<RecoveryState | null>(null);
  const [streamError, setStreamError] = useState<StreamError | null>(null);
  const [stopTarget, setStopTarget] = useState<StopExpectation | null>(initialStopTarget);
  const [stopTargetSnapshotAvailable, setStopTargetSnapshotAvailable] = useState(
    initialStopTargetSnapshotAvailable,
  );
  const [stopPending, setStopPending] = useState(false);
  const stopPendingRef = useRef(stopPending);
  const [stopNotice, setStopNotice] = useState<"stale_target" | "unavailable" | null>(null);
  const [stopFeedback, setStopFeedback] = useState<{ text: string; turnId?: string } | null>(null);
  const [connectionStatus, setConnectionStatus] = useState(client.status);
  const [unknownMessageSend, setUnknownMessageSend] = useState(() => ({
    chatId,
    value: chatId ? client.hasUnknownMessageSend(chatId) : false,
  }));
  const refreshUnknownMessageSend = useCallback(() => {
    const value = chatId ? client.hasUnknownMessageSend(chatId) : false;
    setUnknownMessageSend((current) => current.chatId === chatId && current.value === value
      ? current
      : { chatId, value });
  }, [chatId, client]);
  const buffer = useRef<StreamBuffer | null>(null);
  const activeAssistantRef = useRef<ActiveAssistantCursor | null>(null);
  const closedAssistantStreamIdsRef = useRef<Set<string>>(new Set());
  const activitySegmentRef = useRef<string | null>(null);
  const fileEditSegmentRef = useRef<string | null>(null);
  const activitySegmentCounterRef = useRef(0);
  const pendingStreamEventsRef = useRef<PendingStreamEvent[]>([]);
  const streamFrameRef = useRef<number | null>(null);
  const streamTimerRef = useRef<number | null>(null);
  const lastStreamFlushAtRef = useRef(Number.NEGATIVE_INFINITY);
  const mediaReplyTextsRef = useRef(new Map<string | undefined, Set<string>>());
  const deferredAnswersRef = useRef(new Map<string | undefined, string>());
  const sideChannelTurnIdsRef = useRef<Set<string>>(new Set());
  const liveStopTargetSeenRef = useRef(false);
  const liveRunStateSeenRef = useRef(false);
  const stopTargetRef = useRef(stopTarget);
  const runStateRef = useRef(runState);
  const stoppedTurnIdRef = useRef<string | null>(null);
  const [queueCompletion, setQueueCompletion] =
    useState<{ chatId: string; turnId: string } | null>(null);
  const admissionWaitersRef = useRef(new Map<string, (admitted: boolean) => void>());
  const trustedStopSourceRef = useRef(initialStopTarget?.source ?? null);
  const retiredRunIdentitiesRef = useRef(new Set<string>());
  const staleStopTargetRef = useRef<StopExpectation | null>(null);
  const staleStopTargetHistoryBaselineRef = useRef<StopExpectation | null>(null);
  const pendingStopRunIdentityRef = useRef<string | null>(null);
  const staleRunStateRefreshPendingRef = useRef(false);
  const childRowsRef = useRef<UIMessage[]>(
    initialMessages.filter((message) => message.kind === "delegated_child"),
  );
  const projectionRefreshRef = useRef(onProjectionRefreshNeeded);
  const childOwnedWorkRef = useRef(false);
  stopTargetRef.current = stopTarget;
  projectionRefreshRef.current = onProjectionRefreshNeeded;
  stopPendingRef.current = stopPending;

  const childOwnedWork = useMemo(() => {
    const rootSource = stopTarget?.source ?? null;
    return messages.some((message) => {
      const child = message.delegatedChild;
      if (
        !child
        || child.history_kind === "summary"
        || (child.owned_work !== true && child.holds_capacity !== true)
      ) return false;
      return !rootSource || delegatedChildMatchesRoot(child, rootSource);
    });
  }, [messages, stopTarget]);
  childOwnedWorkRef.current = childOwnedWork;
  runStateRef.current = runState;
  const hasActiveRun = runState !== undefined && runState !== null;
  const hasOwnedWork = hasActiveRun || stopTarget !== null || childOwnedWork || stopPending;
  const connectionOpen = connectionStatus === "open";
  const hasModernStopTargetSnapshot = stopTargetSnapshotAvailable || liveStopTargetSeenRef.current;
  const legacyTargetTurnId = chatId ? client.getRunTurnId(chatId) : null;
  const runStateTargetTurnId = runState && typeof runState.turn_id === "string"
    && runState.turn_id.trim().length > 0
    ? runState.turn_id
    : null;
  const hasLegacyTargetTurnId = typeof legacyTargetTurnId === "string"
    && legacyTargetTurnId.trim().length > 0;
  const canSendRunStateStop = runState !== undefined
    && runState !== null
    && runState.state !== "stopping"
    && runState.can_stop
    && runStateTargetTurnId !== null;
  const stopTargetCanStop = stopTarget !== null && (
    !stopTarget.active_root_present
    || runState === undefined
    || (
      runState !== null
      && runState.state !== "stopping"
      && runState.can_stop
    )
  );
  const stopCanBeSent = connectionOpen
    && !stopPending
    && (
      canSendRunStateStop
      || stopTargetCanStop
      || (
        runState === undefined
        && !liveRunStateSeenRef.current
        && !staleRunStateRefreshPendingRef.current
        && !requireVerifiedStopTarget
        && !hasModernStopTargetSnapshot
        && isStreaming
        && !childOwnedWork
        && hasLegacyTargetTurnId
      )
    );

  const dismissStreamError = useCallback(() => setStreamError(null), []);
  const dismissStopNotice = useCallback(() => setStopNotice(null), []);
  const acceptRunState = useCallback((value: unknown, source: "history" | "attached" | "live") => {
    const candidate = parseRunState(value);
    if (candidate === undefined) return;
    const current = runStateRef.current;
    const currentIdentity = runIdentity(current);
    const candidateIdentity = runIdentity(candidate);
    if (candidateIdentity && retiredRunIdentitiesRef.current.has(candidateIdentity)) return;
    if (
      candidate
      && current?.state === "stopping"
      && candidateIdentity === currentIdentity
      && candidate.state !== "stopping"
      && source !== "attached"
    ) return;
    if (
      candidate
      && current?.state === "stopping"
      && candidateIdentity === currentIdentity
      && candidate.state !== "stopping"
      && source === "attached"
      && pendingStopRunIdentityRef.current === currentIdentity
    ) pendingStopRunIdentityRef.current = null;
    if (candidate && current && candidateIdentity !== currentIdentity) {
      if (
        current.started_at !== null
        && candidate.started_at !== null
        && candidate.started_at < current.started_at
      ) return;
      if (currentIdentity) retiredRunIdentitiesRef.current.add(currentIdentity);
      if (pendingStopRunIdentityRef.current === currentIdentity) {
        pendingStopRunIdentityRef.current = null;
        setStopPending(false);
      }
    } else if (candidate === null && currentIdentity) {
      retiredRunIdentitiesRef.current.add(currentIdentity);
      if (current?.state !== "stopping"
        && pendingStopRunIdentityRef.current === currentIdentity) {
        pendingStopRunIdentityRef.current = null;
      }
    }
    if (source !== "history") liveRunStateSeenRef.current = true;
    staleRunStateRefreshPendingRef.current = false;
    runStateRef.current = candidate;
    setRunState(candidate);
    if (candidate === null) {
      setRunStartedAt(null);
      const wasStopping = current?.state === "stopping";
      setIsStreaming(wasStopping);
      if (!wasStopping && !childOwnedWorkRef.current) setStopPending(false);
      return;
    }
    setRunStartedAt(candidate.started_at);
    setIsStreaming(true);
    stoppedTurnIdRef.current = candidate.state === "stopping" ? candidate.turn_id : null;
    setStopPending(candidate.state === "stopping");
  }, []);

  useEffect(() => {
    if (!chatId || liveRunStateSeenRef.current || initialRunStateInput === undefined) return;
    client.hydrateActiveRun(chatId, initialRunStateInput);
    acceptRunState(initialRunStateInput, "history");
  }, [acceptRunState, chatId, client, initialRunStateInput]);

  const resolveAdmission = useCallback((turnId: string | null | undefined, admitted: boolean) => {
    if (!turnId) return;
    const resolve = admissionWaitersRef.current.get(turnId);
    if (!resolve) return;
    admissionWaitersRef.current.delete(turnId);
    resolve(admitted);
  }, []);

  useEffect(() => client.onStatus((status) => {
    if (status !== "open") setQueueCompletion(null);
    setConnectionStatus(status);
    refreshUnknownMessageSend();
  }), [client, refreshUnknownMessageSend]);

  useEffect(() => {
    if (!chatId) return;
    const afterStaleTarget = staleStopTargetRef.current !== null
      && initialStopTarget !== staleStopTargetHistoryBaselineRef.current;
    if ((liveStopTargetSeenRef.current || liveRunStateSeenRef.current) && !afterStaleTarget) return;
    setStopTargetSnapshotAvailable((current) => current || initialStopTargetSnapshotAvailable);
    if (
      initialStopTarget
      && expectedRootSessionKey
      && initialStopTarget.source.session_key !== expectedRootSessionKey
    ) return;
    let target = initialStopTarget;
    const currentRun = runStateRef.current;
    if (target && currentRun !== undefined && (
      (currentRun === null && target.active_root_present)
      || (
        currentRun !== null
        && currentRun.source_incarnation !== null
        && currentRun.source_incarnation !== target.source.incarnation
      )
      || (
        currentRun !== null
        && target.active_root_present
        && currentRun.turn_id !== null
        && currentRun.turn_id !== target.active_turn_id
      )
    )) target = null;
    trustedStopSourceRef.current = target?.source ?? null;
    stopTargetRef.current = target;
    setStopTarget(target);
    if (afterStaleTarget) {
      staleStopTargetRef.current = null;
      staleStopTargetHistoryBaselineRef.current = null;
    }
    if (target) setStopNotice(null);
  }, [
    chatId,
    expectedRootSessionKey,
    initialStopTarget,
    initialStopTargetSnapshotAvailable,
  ]);

  useEffect(() => {
    childRowsRef.current = messages.filter((message) => message.kind === "delegated_child");
  }, [messages]);

  const clearPendingStreamWork = useCallback(() => {
    if (streamFrameRef.current !== null) {
      window.cancelAnimationFrame(streamFrameRef.current);
      streamFrameRef.current = null;
    }
    if (streamTimerRef.current !== null) {
      window.clearTimeout(streamTimerRef.current);
      streamTimerRef.current = null;
    }
    pendingStreamEventsRef.current = [];
    deferredAnswersRef.current.clear();
    lastStreamFlushAtRef.current = Number.NEGATIVE_INFINITY;
  }, []);

  const isSideChannelEvent = useCallback((ev: InboundEvent) => {
    const turnId = eventTurnId(ev);
    return turnId !== undefined && sideChannelTurnIdsRef.current.has(turnId);
  }, []);

  const createActivitySegmentId = useCallback((activate = true) => {
    activitySegmentCounterRef.current += 1;
    const id = `activity-${activitySegmentCounterRef.current}`;
    if (activate) activitySegmentRef.current = id;
    return id;
  }, []);

  const freshActivitySegmentId = useCallback(
    () => createActivitySegmentId(true),
    [createActivitySegmentId],
  );

  const detachedActivitySegmentId = useCallback(
    () => createActivitySegmentId(false),
    [createActivitySegmentId],
  );

  const ensureActivitySegmentId = useCallback(() => {
    if (activitySegmentRef.current) return activitySegmentRef.current;
    return freshActivitySegmentId();
  }, [freshActivitySegmentId]);

  const clearActivitySegment = useCallback(() => {
    activitySegmentRef.current = null;
    fileEditSegmentRef.current = null;
  }, []);

  const closeActiveAssistantStream = useCallback(() => {
    const closedStreamId = buffer.current?.messageId ?? activeAssistantRef.current?.id;
    if (closedStreamId) closedAssistantStreamIdsRef.current.add(closedStreamId);
    buffer.current = null;
    activeAssistantRef.current = null;
    return !!closedStreamId;
  }, []);

  const applyStreamError = useCallback((err: StreamError) => {
    // One multiplexed client serves every thread. A correlated send fault
    // belongs only to its target chat. An uncorrelated transport close can
    // still be shown in the mounted thread, but cannot roll back any turn.
    if (!chatId || (err.chatId && err.chatId !== chatId)) return;
    if (err.kind === "connection_closed" || err.kind === "session_recovery_required") {
      const currentRow = messagesRef.current.find((message) => (
        message.role === "user" && message.turnId === err.turnId
      ));
      const alreadyUncertain = currentRow?.deliveryStatus === "unknown" && (
        err.kind === "session_recovery_required" || currentRow.deliveryErrorKind === "connection_closed"
      );
      const legacyUncertain = (
        currentRow?.deliveryStatus === "failed"
        && currentRow.deliveryErrorKind === "connection_closed"
      );
      if (
        currentRow
        && currentRow.deliveryStatus !== "sending"
        && !alreadyUncertain
        && !legacyUncertain
      ) return;
      setStreamError(err);
      setMessages((prev) => prev.map((message) => (
        message.role === "user"
        && message.turnId === err.turnId
        && (message.deliveryStatus === "sending"
          || (err.kind === "session_recovery_required" && message.deliveryStatus === "unknown"))
          ? {
              ...message,
              deliveryStatus: "unknown",
              deliveryErrorKind: err.kind,
            }
          : message
      )));
      if (err.kind === "session_recovery_required") {
        const startedAt = client.getRunStartedAt(chatId);
        setRunStartedAt(startedAt);
        setIsStreaming(startedAt !== null || client.hasUnsettledRun(chatId));
      }
      return;
    }
    setStreamError(err);
    if (err.kind === "model_request_failed") return;
    if (!err.turnId) return;

    resolveAdmission(err.turnId, false);
    const rejectedTurnId = err.turnId;
    deferredAnswersRef.current.delete(rejectedTurnId);
    pendingStreamEventsRef.current = pendingStreamEventsRef.current.filter(
      (event) => event.turn.turnId !== rejectedTurnId,
    );
    sideChannelTurnIdsRef.current.delete(rejectedTurnId);
    setMessages((prev) => {
      const rejectedRows = prev.filter((message) => message.turnId === rejectedTurnId);
      if (rejectedRows.length === 0) return prev;
      const rejectedIds = new Set(rejectedRows.map((message) => message.id));
      const rejectedSegments = new Set(
        rejectedRows
          .map((message) => message.activitySegmentId)
          .filter((segmentId): segmentId is string => typeof segmentId === "string"),
      );
      if (
        activeAssistantRef.current
        && rejectedIds.has(activeAssistantRef.current.id)
      ) {
        activeAssistantRef.current = null;
      }
      if (buffer.current && rejectedIds.has(buffer.current.messageId)) {
        buffer.current = null;
      }
      for (const id of rejectedIds) closedAssistantStreamIdsRef.current.delete(id);
      if (
        activitySegmentRef.current
        && rejectedSegments.has(activitySegmentRef.current)
      ) {
        activitySegmentRef.current = null;
      }
      if (
        fileEditSegmentRef.current
        && rejectedSegments.has(fileEditSegmentRef.current)
      ) {
        fileEditSegmentRef.current = null;
      }
      return prev.flatMap((message) => {
        if (message.turnId !== rejectedTurnId) return [message];
        if (message.role !== "user") return [];
        return [{
          ...message,
          deliveryStatus: "failed",
          deliveryErrorKind: messageDeliveryErrorKind(err),
        }];
      });
    });

    const remainingStartedAt = client.getRunStartedAt(chatId);
    const hasRemainingRun = (
      remainingStartedAt !== null
      || client.hasUnsettledRun(chatId)
    );
    setRunStartedAt(remainingStartedAt);
    setIsStreaming(hasRemainingRun);
    if (!hasRemainingRun) mediaReplyTextsRef.current.clear();
  }, [chatId, client, resolveAdmission]);

  useEffect(() => client.onError((error) => {
    refreshUnknownMessageSend();
    applyStreamError(error);
  }), [applyStreamError, client, refreshUnknownMessageSend]);

  const resolveActiveAssistantIndex = useCallback((
    prev: UIMessage[],
    turn: UIMessageTurnFields = {},
  ): number | null => {
    const cursor = activeAssistantRef.current;
    if (!cursor) return null;
    const indexed = prev[cursor.index];
    if (
      indexed?.id === cursor.id
      && indexed.role === "assistant"
      && indexed.kind !== "trace"
      && indexed.isStreaming
      && matchesTurn(indexed, turn)
    ) {
      return cursor.index;
    }
    const idx = prev.findIndex((m) => m.id === cursor.id);
    if (idx === -1) {
      activeAssistantRef.current = null;
      return null;
    }
    const found = prev[idx];
    if (
      found.role !== "assistant"
      || found.kind === "trace"
      || !found.isStreaming
      || !matchesTurn(found, turn)
    ) {
      activeAssistantRef.current = null;
      return null;
    }
    activeAssistantRef.current = { id: cursor.id, index: idx };
    return idx;
  }, []);

  const appendAnswerChunk = useCallback(
    (
      prev: UIMessage[],
      chunk: string,
      turn: UIMessageTurnFields = {},
      source?: UIMessage["source"],
    ): UIMessage[] => {
      let next = prev;
      let targetIndex = resolveActiveAssistantIndex(next, turn);

      if (targetIndex === null) {
        targetIndex = findActiveAssistantPlaceholderIndex(next, turn);
      }
      if (targetIndex === null) {
        targetIndex = findStreamingAssistantIndex(next, closedAssistantStreamIdsRef.current, turn);
      }
      if (targetIndex === null) {
        const id = crypto.randomUUID();
        next = [
          ...next,
          {
            id,
            role: "assistant",
            content: "",
            isStreaming: true,
            createdAt: Date.now(),
          },
        ];
        targetIndex = next.length - 1;
      }

      const target = next[targetIndex];
      const merged: UIMessage = {
        ...target,
        content: target.content + chunk,
        isStreaming: true,
        ...turn,
        ...(source ? { source } : {}),
      };
      closedAssistantStreamIdsRef.current.delete(merged.id);
      activeAssistantRef.current = { id: merged.id, index: targetIndex };
      buffer.current = { messageId: merged.id };
      return replaceMessageAt(next, targetIndex, merged);
    },
    [resolveActiveAssistantIndex],
  );

  const applyPendingStreamEvents = useCallback(
    (prev: UIMessage[], events: PendingStreamEvent[]): UIMessage[] => {
      let next = prev;
      for (const event of events) {
        if (event.kind === "delta") {
          next = appendAnswerChunk(next, event.text, event.turn, event.source);
        } else {
          if (closeActiveAssistantStream()) clearActivitySegment();
          next = attachReasoningChunk(
            next,
            event.text,
            { ensure: ensureActivitySegmentId },
            event.turn,
          );
        }
      }
      return next;
    },
    [appendAnswerChunk, clearActivitySegment, closeActiveAssistantStream, ensureActivitySegmentId],
  );

  const flushPendingStreamEvents = useCallback((options?: {
    closeAnswerSegment?: boolean;
    finalAnswerText?: string;
    turn?: UIMessageTurnFields;
    source?: UIMessage["source"];
  }) => {
    if (streamFrameRef.current !== null) {
      window.cancelAnimationFrame(streamFrameRef.current);
      streamFrameRef.current = null;
    }
    if (streamTimerRef.current !== null) {
      window.clearTimeout(streamTimerRef.current);
      streamTimerRef.current = null;
    }
    const events = pendingStreamEventsRef.current;
    const finalAnswerText = options?.finalAnswerText;
    const turn = options?.turn ?? {};
    const source = options?.source;
    if (events.length === 0 && finalAnswerText === undefined && source === undefined && !options?.closeAnswerSegment) {
      return;
    }
    pendingStreamEventsRef.current = [];
    lastStreamFlushAtRef.current = nowMs();
    setMessages((prev) => {
      let next = events.length > 0 ? applyPendingStreamEvents(prev, events) : prev;
      if (finalAnswerText !== undefined) {
        const targetIndex =
          resolveActiveAssistantIndex(next, turn)
          ?? findStreamingAssistantIndex(next, closedAssistantStreamIdsRef.current, turn);
        if (targetIndex !== null) {
          const target = next[targetIndex];
          const merged = {
            ...target,
            content: finalAnswerText,
            isStreaming: true,
            ...turn,
            ...(source ? { source } : {}),
          };
          next = replaceMessageAt(next, targetIndex, merged);
          if (!options?.closeAnswerSegment) {
            closedAssistantStreamIdsRef.current.delete(merged.id);
            activeAssistantRef.current = { id: merged.id, index: targetIndex };
            buffer.current = { messageId: merged.id };
          }
        } else {
          const id = crypto.randomUUID();
          next = [
            ...next,
            {
              id,
              role: "assistant",
              content: finalAnswerText,
              isStreaming: true,
              ...turn,
              ...(source ? { source } : {}),
              createdAt: Date.now(),
            },
          ];
          if (options?.closeAnswerSegment) {
            closedAssistantStreamIdsRef.current.add(id);
          } else {
            activeAssistantRef.current = { id, index: next.length - 1 };
            buffer.current = { messageId: id };
          }
        }
      } else if (source) {
        const targetIndex =
          resolveActiveAssistantIndex(next, turn)
          ?? findStreamingAssistantIndex(next, closedAssistantStreamIdsRef.current, turn);
        if (targetIndex !== null) {
          const target = next[targetIndex];
          next = replaceMessageAt(next, targetIndex, {
            ...target,
            ...turn,
            source,
          });
        }
      }
      if (options?.closeAnswerSegment) closeActiveAssistantStream();
      return next;
    });
  }, [applyPendingStreamEvents, closeActiveAssistantStream, resolveActiveAssistantIndex]);

  const flushScheduledStreamEvents = useCallback(() => {
    const events = pendingStreamEventsRef.current;
    if (events.length === 0) return;
    pendingStreamEventsRef.current = [];
    lastStreamFlushAtRef.current = nowMs();
    setMessages((prev) => applyPendingStreamEvents(prev, events));
  }, [applyPendingStreamEvents]);

  const schedulePendingStreamFlush = useCallback(() => {
    if (streamFrameRef.current !== null || streamTimerRef.current !== null) return;
    if (document.visibilityState === "hidden") {
      streamTimerRef.current = window.setTimeout(() => {
        streamTimerRef.current = null;
        flushScheduledStreamEvents();
      }, BACKGROUND_STREAM_FLUSH_INTERVAL_MS);
      return;
    }

    const elapsed = nowMs() - lastStreamFlushAtRef.current;
    const remaining = STREAM_FLUSH_INTERVAL_MS - elapsed;
    if (remaining <= 0) {
      streamFrameRef.current = window.requestAnimationFrame(() => {
        streamFrameRef.current = null;
        flushScheduledStreamEvents();
      });
      return;
    }
    streamTimerRef.current = window.setTimeout(() => {
      streamTimerRef.current = null;
      flushScheduledStreamEvents();
    }, remaining);
  }, [flushScheduledStreamEvents]);

  useEffect(() => {
    const flushOnReturn = () => {
      if (document.visibilityState !== "visible") return;
      if (pendingStreamEventsRef.current.length === 0) return;
      flushPendingStreamEvents();
    };
    document.addEventListener("visibilitychange", flushOnReturn);
    return () => document.removeEventListener("visibilitychange", flushOnReturn);
  }, [flushPendingStreamEvents]);

  useEffect(() => {
    if (!chatId) return;
    return client.onRunStatus((runChatId, startedAt) => {
      if (runChatId !== chatId) return;
      if (startedAt !== null) {
        setRunStartedAt(startedAt);
        setIsStreaming(true);
        return;
      }
      if (
        (runStateRef.current !== undefined && runStateRef.current !== null)
        || pendingStopRunIdentityRef.current !== null
      ) return;
      flushPendingStreamEvents();
      buffer.current = null;
      activeAssistantRef.current = null;
      closedAssistantStreamIdsRef.current.clear();
      clearActivitySegment();
      setMessages((prev) => prev.map((message) => (
        message.isStreaming ? { ...message, isStreaming: false } : message
      )));
      setRunStartedAt(null);
      setIsStreaming(false);
    });
  }, [chatId, client, clearActivitySegment, flushPendingStreamEvents]);

  // Reset local state when switching chats. Do not reset on every
  // ``initialMessages`` update: a brand-new chat can receive an empty/404
  // history response after the optimistic first message has already rendered.
  useEffect(() => {
    const restoredRunState = initialRunStateInput;
    const restoredRunStartedAt = restoredRunState !== undefined
      ? restoredRunState?.started_at ?? null
      : (chatId ? client.getRunStartedAt(chatId) : null);
    setMessages(initialMessages);
    setMessageOwnerChatId(chatId);
    setIsStreaming(
      hasPendingAgentActivity(initialMessages)
      || hasPendingToolCalls
      || restoredRunStartedAt !== null
      || (restoredRunState !== undefined && restoredRunState !== null),
    );
    setStreamError(null);
    refreshUnknownMessageSend();
    setRunStartedAt(restoredRunStartedAt);
    setRetryStatus(null);
    setGoalState(chatId ? client.getGoalState(chatId) : undefined);
    setRunState(restoredRunState);
    runStateRef.current = restoredRunState;
    setRecoveryState(null);
    setStopTarget(initialStopTarget);
    setStopTargetSnapshotAvailable(initialStopTargetSnapshotAvailable);
    stopTargetRef.current = initialStopTarget;
    trustedStopSourceRef.current = initialStopTarget?.source ?? null;
    liveStopTargetSeenRef.current = false;
    liveRunStateSeenRef.current = false;
    retiredRunIdentitiesRef.current.clear();
    staleStopTargetRef.current = null;
    staleStopTargetHistoryBaselineRef.current = null;
    pendingStopRunIdentityRef.current = null;
    staleRunStateRefreshPendingRef.current = false;
    stoppedTurnIdRef.current = null;
    setQueueCompletion(null);
    setStopPending(false);
    setStopNotice(null);
    setStopFeedback(null);
    admissionWaitersRef.current.clear();
    childRowsRef.current = initialMessages.filter((message) => message.kind === "delegated_child");
    buffer.current = null;
    activeAssistantRef.current = null;
    closedAssistantStreamIdsRef.current.clear();
    clearActivitySegment();
    clearPendingStreamWork();
    sideChannelTurnIdsRef.current.clear();
    mediaReplyTextsRef.current.clear();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, client, clearActivitySegment, clearPendingStreamWork]);

  useEffect(() => {
    if (hasPendingToolCalls) setIsStreaming(true);
  }, [hasPendingToolCalls]);

  useEffect(() => {
    if (!chatId) return;

    const acceptStopTarget = (value: unknown) => {
      liveStopTargetSeenRef.current = true;
      setStopTargetSnapshotAvailable(true);
      const candidate = value === null ? null : parseStopExpectation(value);
      const target = candidate
        && (!expectedRootSessionKey || candidate.source.session_key === expectedRootSessionKey)
        ? candidate
        : null;
      if (value !== null && target === null) {
        stopTargetRef.current = null;
        setStopTarget(null);
        setStopNotice("unavailable");
        setStopPending(false);
        projectionRefreshRef.current?.();
        return;
      }
      const currentRun = runStateRef.current;
      if (target && currentRun !== undefined && (
        (currentRun === null && target.active_root_present)
        || (
          currentRun !== null
          && currentRun.source_incarnation !== null
          && currentRun.source_incarnation !== target.source.incarnation
        )
        || (
          currentRun !== null
          && target.active_root_present
          && currentRun.turn_id !== null
          && currentRun.turn_id !== target.active_turn_id
        )
      )) {
        stopTargetRef.current = null;
        setStopTarget(null);
        setStopPending(false);
        projectionRefreshRef.current?.();
        return;
      }
      if (
        target
        && staleStopTargetRef.current
        && sameStopTarget(target, staleStopTargetRef.current)
      ) {
        projectionRefreshRef.current?.();
        return;
      }
      const trustedSource = trustedStopSourceRef.current;
      const sourceChanged = trustedSource !== null
        && target !== null
        && (
          trustedSource.store_identity !== target.source.store_identity
          || trustedSource.session_key !== target.source.session_key
          || trustedSource.incarnation !== target.source.incarnation
          || trustedSource.generation !== target.source.generation
        );
      if (sourceChanged && target && currentRun === undefined) {
        // A current run state validates a live source replacement. Older
        // peers need a fresh history snapshot because they have no identity.
        staleStopTargetRef.current = target;
        staleStopTargetHistoryBaselineRef.current = initialStopTarget;
        stopTargetRef.current = null;
        setStopTarget(null);
        setStopNotice("unavailable");
        setStopPending(false);
        projectionRefreshRef.current?.();
        return;
      }
      if (target) trustedStopSourceRef.current = target.source;
      if (target && staleStopTargetRef.current) staleStopTargetRef.current = null;
      stopTargetRef.current = target;
      setStopTarget(target);
      if (target) {
        setStopNotice(null);
        return;
      }
      if (target === null) {
        if (
          runStateRef.current?.state !== "stopping"
          && pendingStopRunIdentityRef.current === null
          && stopPendingRef.current
        ) setStopPending(false);
        if (childOwnedWorkRef.current) {
          projectionRefreshRef.current?.();
        }
      }
    };

    const handle = (ev: InboundEvent, admission?: LocalInputAdmission, liveNaturalCompletion?: true) => {
      // Read the attempt owner after every existing callback, including receipt
      // early returns; transcript hydration never supplies this safety fact.
      refreshUnknownMessageSend();
      if (admission?.disposition === "admitted") {
        // The client captured exact proof before its pending-owner cleanup.
        // This repairs disposition only, including attached after completion.
        resolveAdmission(admission.turnId, true);
        setMessages((prev) => transitionTurnDelivery(prev, admission.turnId, "accepted", true));
        setStreamError((current) => (
          (current?.kind === "connection_closed" || current?.kind === "session_recovery_required")
          && current.turnId === admission.turnId
            ? null
            : current
        ));
      }
      if (ev.event === "error") {
        resolveAdmission(ev.turn_id, false);
        if (
          ev.detail === "stale_stop_target"
          || ev.reason === "stale_stop_target"
          || ev.detail === "invalid_stop_expectation"
          || ev.reason === "invalid_stop_expectation"
        ) {
          staleStopTargetRef.current = stopTargetRef.current;
          staleStopTargetHistoryBaselineRef.current = initialStopTarget;
          stopTargetRef.current = null;
          setStopTarget(null);
          const attemptedRunIdentity = pendingStopRunIdentityRef.current;
          if (attemptedRunIdentity && runIdentity(runStateRef.current) === attemptedRunIdentity) {
            retiredRunIdentitiesRef.current.add(attemptedRunIdentity);
            runStateRef.current = undefined;
            setRunState(undefined);
            liveRunStateSeenRef.current = false;
            staleRunStateRefreshPendingRef.current = true;
            pendingStopRunIdentityRef.current = null;
          }
          setStopPending(false);
          setStopNotice("stale_target");
          projectionRefreshRef.current?.();
          return;
        }
        if (ev.detail === "message_too_big") {
          applyStreamError({
            kind: "message_too_big",
            chatId,
            turnId: ev.turn_id,
          });
        } else if (ev.detail === "workspace_scope_rejected") {
          applyStreamError({
            kind: "workspace_scope_rejected",
            reason: ev.reason,
            chatId,
            turnId: ev.turn_id,
          });
        } else if (ev.turn_id) {
          applyStreamError({
            kind: "turn_rejected",
            detail: ev.detail,
            reason: ev.reason,
            chatId,
            turnId: ev.turn_id,
          });
        }
        return;
      }
      if (ev.event === "message" && ev.kind === "stop_status") {
        setStopFeedback({ text: ev.text, turnId: ev.stop_turn_id });
        return;
      }
      const turnId = eventTurnId(ev);
      const unresolvedAdmission = admission?.disposition === "unresolved"
        && admission.turnId === turnId;
      // Route-level `message_accepted` confirms the gateway received the
      // submission and clears the transient "Sending…" indicator immediately,
      // while durable input-attempt admission remains gated on `message_admitted`.
      if (turnId && (!unresolvedAdmission || ev.event === "message_accepted")) {
        const repairsUncertain = (
          ev.event !== "message_accepted"
          && ev.event !== "user_message"
          && ev.event !== "turn_end"
          && (ev.event !== "goal_status" || ev.status === "running")
        );
        setMessages((prev) => transitionTurnDelivery(
          prev,
          turnId,
          "accepted",
          repairsUncertain,
        ));
        if (repairsUncertain) {
          setStreamError((current) => (
            current?.kind === "connection_closed" && current.turnId === turnId
              ? null
              : current
          ));
        }
      }
      if (ev.event === "message_accepted") {
        if (ev.starts_turn && typeof ev.started_at === "number" && !unresolvedAdmission) {
          setRunStartedAt(ev.started_at);
        }
        return;
      }
      if (ev.event === "message_admitted") {
        // Owner receipt: only this event commits a queued attempt.
        if (!unresolvedAdmission) resolveAdmission(turnId, true);
        return;
      }
      if (ev.event === "user_message") {
        if (ev.provenance?.session_message) {
          flushPendingStreamEvents({ closeAnswerSegment: true });
          clearActivitySegment();
          setIsStreaming(true);
          setMessages((prev) => appendProjectedSessionInput(prev, ev));
          return;
        }
        setMessages((prev) => {
          if (ev.turn_id && prev.some((message) => (
            message.role === "user" && message.turnId === ev.turn_id
          ))) return prev;
          return [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "user",
              content: ev.text,
              ...(ev.turn_id ? { turnId: ev.turn_id } : {}),
              turnPhase: "user",
              turnSeq: 0,
              deliveryStatus: unresolvedAdmission ? "unknown" : "accepted",
              createdAt: typeof ev.created_at_ms === "number"
                && Number.isFinite(ev.created_at_ms)
                ? ev.created_at_ms
                : Date.now(),
              ...(ev.media_urls?.length ? { media: ev.media_urls } : {}),
              ...(ev.cli_apps?.length ? { cliApps: ev.cli_apps } : {}),
              ...(ev.mcp_presets?.length ? { mcpPresets: ev.mcp_presets } : {}),
              ...(ev.session_mentions?.length
                ? { sessionMentions: ev.session_mentions }
                : {}),
            },
          ];
        });
        if (ev.active_turn_id || ev.starts_turn) {
          setIsStreaming(true);
          if (typeof ev.started_at === "number") setRunStartedAt(ev.started_at);
        }
        return;
      }
      const sideChannelEvent = isSideChannelEvent(ev);
      if (ev.event === "context_compaction") {
        // Compaction is activity, not an answer-stream boundary. Length recovery
        // can keep this answer open for cumulative stream_end replacements.
        flushPendingStreamEvents();
        clearActivitySegment();
        const compaction = {
          id: ev.compaction_id,
          phase: ev.phase,
          announce: true,
        };
        setMessages((prev) => {
          const id = `compaction-${ev.compaction_id}`;
          const existing = prev.findIndex((message) => message.id === id);
          if (!acceptsCompactionPhase(prev[existing]?.compaction?.phase, ev.phase)) return prev;
          const next = {
            id,
            role: "assistant" as const,
            content: "",
            kind: "compaction" as const,
            createdAt: Date.now(),
            compaction,
            ...turnFieldsFromEvent(ev, "activity"),
          };
          if (existing < 0) return [...prev, next];
          return prev.map((message, index) => (
            index === existing
              ? { ...next, createdAt: message.createdAt }
              : message
          ));
        });
        return;
      }
      if (
        ev.event === "delta"
        || ev.event === "message"
        || ev.event === "file_edit"
        || ev.event === "reasoning_delta"
        || ev.event === "stream_end"
      ) setRetryStatus(null);
      if (ev.event === "delta") {
        let chunk = typeof ev.text === "string" ? ev.text : "";
        if (!chunk) return;
        const turnId = eventTurnId(ev);
        const captions = mediaReplyTextsRef.current.get(turnId);
        if (captions && captions.size > 0) {
          // Hold correlated text only while it could still be an exact repeat
          // of an already-sent media caption; diverge to live streaming immediately.
          const deferred = deferredAnswersRef.current;
          const combined = (deferred.get(turnId) ?? "") + chunk;
          const trimmed = combined.trim();
          const couldMatchCaption =
            trimmed.length === 0
            || Array.from(captions).some((caption) => caption.startsWith(trimmed));
          if (couldMatchCaption) {
            deferred.set(turnId, combined);
            return;
          }
          deferred.delete(turnId);
          chunk = combined;
        }
        clearActivitySegment();
        setIsStreaming(true);
        pendingStreamEventsRef.current.push({
          kind: "delta",
          text: chunk,
          turn: turnFieldsFromEvent(ev, "answer"),
          source: ev.source,
        });
        schedulePendingStreamFlush();
        return;
      }

      if (ev.event === "reasoning_delta") {
        const chunk = ev.text;
        if (!chunk) return;
        if (fileEditSegmentRef.current) clearActivitySegment();
        setIsStreaming(true);
        pendingStreamEventsRef.current.push({
          kind: "reasoning",
          text: chunk,
          turn: turnFieldsFromEvent(ev, "reasoning"),
        });
        schedulePendingStreamFlush();
        return;
      }

      if (ev.event === "stream_end") {
        const turn = turnFieldsFromEvent(ev, "answer");
        const captions = mediaReplyTextsRef.current.get(turn.turnId);
        const hasTurnMediaCaptions = Boolean(captions && captions.size > 0);
        const mergeNext = ev.resuming === true && ev.merge_next === true;
        const deferred = deferredAnswersRef.current;
        let finalAnswerText = typeof ev.text === "string" ? ev.text : undefined;
        if (deferred.has(turn.turnId)) {
          finalAnswerText ??= deferred.get(turn.turnId);
          deferred.delete(turn.turnId);
          if (mergeNext && finalAnswerText) deferred.set(turn.turnId, finalAnswerText);
        }
        const finalText = finalAnswerText?.trim();
        const hasLiveStreamedAnswer =
          buffer.current !== null
          || activeAssistantRef.current !== null
          || pendingStreamEventsRef.current.some((item) => item.kind === "delta");
        if (hasTurnMediaCaptions && (
          finalText !== undefined
            ? (!finalText || captions!.has(finalText))
            : !hasLiveStreamedAnswer
        )) {
          setIsStreaming(true);
          return;
        }
        flushPendingStreamEvents({
          closeAnswerSegment: !mergeNext,
          ...(finalAnswerText !== undefined ? { finalAnswerText } : {}),
          turn,
          source: ev.source,
        });
        if (ev.resuming) {
          setIsStreaming(true);
          return;
        }
        // ``stream_end`` closes the current answer segment, not the turn.
        // Tools and follow-up model segments may still arrive before the
        // definitive ``turn_end`` event.
        setIsStreaming(true);
        return;
      }

      const shouldCloseAnswerBeforeEvent =
        ev.event === "file_edit"
        || (
          ev.event === "message"
          && (ev.kind === "tool_hint" || ev.kind === "progress")
        );
      flushPendingStreamEvents({ closeAnswerSegment: shouldCloseAnswerBeforeEvent });

      if (ev.event === "reasoning_end") {
        setMessages((prev) => closeReasoningStream(prev, Date.now()));
        return;
      }

      if (ev.event === "stop_target") {
        acceptStopTarget(ev.stop_target);
        return;
      }

      if (ev.event === "run_state") {
        acceptRunState(ev.run_state, "live");
        return;
      }

      if (ev.event === "delegated_child") {
        const child = parseDelegatedChildPayload(ev);
        if (!child || !expectedRootSessionKey) {
          setStopNotice(child ? "unavailable" : null);
          projectionRefreshRef.current?.();
          return;
        }
        const context = {
          expectedRootKey: expectedRootSessionKey,
          ...(trustedStopSourceRef.current
            ? { expectedRoot: trustedStopSourceRef.current }
            : {}),
          mode: "live" as const,
        };
        const preview = mergeDelegatedChildRows(childRowsRef.current, [child], context);
        if (preview.refreshRequired) {
          projectionRefreshRef.current?.();
          return;
        }
        childRowsRef.current = preview.messages;
        setMessages((prev) => mergeDelegatedChildRows(prev, [child], context).messages);
        return;
      }

      if (ev.event === "goal_state") {
        setGoalState(ev.goal_state);
        return;
      }

      if (ev.event === "goal_status") {
        const activeRun = runStateRef.current;
        if (pendingStopRunIdentityRef.current !== null && ev.status === "idle") return;
        if (
          activeRun?.turn_id
          && (ev.turn_id === undefined || ev.turn_id !== activeRun.turn_id)
        ) return;
        if (ev.status === "running" && typeof ev.started_at === "number") {
          if (!unresolvedAdmission) setStreamError(null);
          setRunStartedAt(ev.started_at);
          setIsStreaming(true);
          // Legacy activity remains implicit admission; negotiated input needs
          // the client's exact proof independently of this live activity.
          if (ev.turn_id && !unresolvedAdmission) resolveAdmission(ev.turn_id, true);
        } else if (ev.status === "stopping") {
          // Stop was accepted; keep the turn visible until settlement.
          stoppedTurnIdRef.current = ev.turn_id ?? client.getRunTurnId(chatId);
          setQueueCompletion(null);
          setStopPending(true);
          setIsStreaming(true);
        } else {
          setRunStartedAt(null);
          setIsStreaming(false);
          setRetryStatus(null);
          if (!childOwnedWorkRef.current && stopTargetRef.current === null) {
            setStopPending(false);
          }
        }
        return;
      }

      if (ev.event === "retry_status") {
        const activeTurnId = client.getRunTurnId(chatId);
        if (ev.turn_id && activeTurnId && ev.turn_id !== activeTurnId) return;
        if (ev.state === "recovered" || ev.state === "cleared") {
          setRetryStatus(null);
        } else {
          const retryAfterSeconds =
            typeof ev.retry_after_s === "number" &&
            Number.isFinite(ev.retry_after_s) &&
            ev.retry_after_s >= 0
              ? ev.retry_after_s
              : undefined;
          setRetryStatus({
            state: ev.state,
            attempt: ev.attempt,
            error_kind: ev.error_kind,
            ...(typeof ev.max_attempts === "number"
              ? { max_attempts: ev.max_attempts }
              : {}),
            ...(retryAfterSeconds !== undefined
              ? { next_retry_at: Date.now() / 1000 + retryAfterSeconds }
              : {}),
            ...(ev.turn_id ? { turn_id: ev.turn_id } : {}),
          });
          setIsStreaming(true);
        }
        return;
      }

      if (ev.event === "turn_end") {
        const completedTurnId = typeof ev.turn_id === "string" ? ev.turn_id : null;
        const activeRun = runStateRef.current;
        const wasStopping = activeRun?.state === "stopping"
          || pendingStopRunIdentityRef.current !== null;
        if (
          completedTurnId
          && activeRun?.turn_id
          && completedTurnId !== activeRun.turn_id
        ) return;
        if (activeRun !== undefined) acceptRunState(null, "live");
        pendingStopRunIdentityRef.current = null;
        if (
          liveNaturalCompletion
          && !wasStopping
          && completedTurnId
          && completedTurnId !== stoppedTurnIdRef.current
          && !stopPendingRef.current
        ) {
          setQueueCompletion({ chatId, turnId: completedTurnId });
        }
        // A turn that ends without an owner receipt cannot be treated as
        // admitted; keep the queued item unresolved and held.
        if (completedTurnId) sideChannelTurnIdsRef.current.delete(completedTurnId);
        if ("goal_state" in ev && ev.goal_state != null && typeof ev.goal_state === "object") {
          setGoalState(ev.goal_state);
        }
        setRunStartedAt(null);
        setRetryStatus(null);
        setRunState(null);
        runStateRef.current = null;
        // Definitive signal that the turn is fully complete, so stop the
        // loading indicator immediately.
        setIsStreaming(false);
        if (wasStopping || (!childOwnedWorkRef.current && stopTargetRef.current === null)) {
          setStopPending(false);
        }
        const modelRequestFailed = ev.outcome === "failed" && ev.failure_kind === "model";
        const failureAttempts =
          typeof ev.failure_attempts === "number"
          && Number.isInteger(ev.failure_attempts)
          && ev.failure_attempts > 0
            ? ev.failure_attempts
            : undefined;
        const modelFailure = modelRequestFailed
          ? {
              kind: "model_request_failed" as const,
              chatId,
              ...(ev.turn_id ? { turnId: ev.turn_id } : {}),
              ...(typeof ev.failure_error_kind === "string"
                ? { errorKind: ev.failure_error_kind }
                : {}),
              ...(failureAttempts !== undefined ? { attempts: failureAttempts } : {}),
            }
          : null;
        if (modelFailure) setStreamError(modelFailure);
        const completedAt = Date.now();
        const contextUsage = contextUsageFromEvent(ev);
        setMessages((prev) => {
          let finalized = prev.map((m) => (
            m.isStreaming || m.reasoningStreaming
              ? { ...m, isStreaming: false, reasoningStreaming: false }
              : m
          ));
          const latencyMs =
            typeof ev.latency_ms === "number" && ev.latency_ms >= 0
              ? Math.round(ev.latency_ms)
              : undefined;
          finalized = stampLastAssistantCompletion(
            finalized,
            {
              ...(latencyMs !== undefined ? { latencyMs } : {}),
              ...(ev.usage ? { usage: ev.usage } : {}),
              ...(ev.round_usages?.length ? { roundUsages: ev.round_usages } : {}),
              ...(contextUsage !== undefined ? { contextUsage } : {}),
              ...(typeof ev.context_window_tokens === "number"
                ? { contextWindowTokens: ev.context_window_tokens }
                : {}),
              completedAt,
            },
            ev.turn_id,
          );
          buffer.current = null;
          activeAssistantRef.current = null;
          clearActivitySegment();
          closedAssistantStreamIdsRef.current.clear();
          if (ev.outcome === "cancelled" && !finalized.some((message) => (
            message.kind === "run_stopped" && Boolean(ev.turn_id) && message.turnId === ev.turn_id
          ))) {
            finalized.push({
              id: ev.turn_id ? `run-stopped-${ev.turn_id}` : crypto.randomUUID(),
              role: "system",
              kind: "run_stopped",
              content: "",
              createdAt: completedAt,
              ...turnFieldsFromEvent(ev, "complete"),
            });
          }
          return finalized;
        });
        mediaReplyTextsRef.current.clear();
        deferredAnswersRef.current.clear();
        onTurnEnd?.();
        return;
      }

      if (ev.event === "recovery_state") {
        const next: RecoveryState = {
          status: ev.status,
          recovery_id: ev.recovery_id,
          ...(ev.reason ? { reason: ev.reason } : {}),
          ...(typeof ev.attempts === "number" ? { attempts: ev.attempts } : {}),
          ...(typeof ev.can_continue === "boolean"
            ? { can_continue: ev.can_continue }
            : {}),
        };
        setRecoveryState(next);
        if (ev.status === "resuming") {
          setRunStartedAt((current) => current ?? Date.now() / 1000);
          setIsStreaming(true);
        }
        if (
          ev.status === "awaiting_user"
          || ev.status === "recovered"
          || ev.status === "failed"
        ) {
          setRetryStatus(null);
          // Recovery is an explicit boundary. The interrupted turn is no
          // longer running, so do not let the stale start time keep the
          // activity clock (or composer stop state) alive underneath the
          // recovery notice.
          setRunStartedAt(null);
          setIsStreaming(false);
          client.finishRunLocally(chatId);
          clearPendingStreamWork();
          closeActiveAssistantStream();
          clearActivitySegment();
        }
        return;
      }

      if (ev.event === "attached") {
        if ("run_state" in ev) acceptRunState(ev.run_state, "attached");
        if ("stop_target" in ev) acceptStopTarget(ev.stop_target);
        setRecoveryState(ev.recovery_state ?? null);
        if (ev.recovery_state?.status === "resuming") {
          setRunStartedAt((current) => current ?? Date.now() / 1000);
          setIsStreaming(true);
        } else if (
          ev.recovery_state?.status === "awaiting_user"
          || ev.recovery_state?.status === "failed"
        ) {
          setRunStartedAt(null);
          setIsStreaming(false);
          client.finishRunLocally(chatId);
          clearPendingStreamWork();
          closeActiveAssistantStream();
          clearActivitySegment();
        }
        return;
      }

      if (ev.event === "message") {
        // Back-compat: a legacy ``kind: "reasoning"`` message (no streaming
        // partner) is treated as one complete delta + immediate end so the
        // bubble renders identically to the streaming path.
        if (ev.kind === "reasoning") {
          const line = ev.text;
          if (!line) return;
          if (fileEditSegmentRef.current) clearActivitySegment();
          setMessages((prev) => closeReasoningStream(
            attachReasoningChunk(
              prev,
              line,
              { ensure: ensureActivitySegmentId },
              turnFieldsFromEvent(ev, "reasoning"),
            ),
            Date.now(),
          ));
          return;
        }
        // Intermediate agent breadcrumbs (tool-call hints, raw progress).
        // Attach them to the last trace row if it was the last emitted item
        // so a sequence of calls collapses into one compact trace group.
        if (ev.kind === "tool_hint" || ev.kind === "progress") {
          const incomingEvents = normalizeToolProgressEvents(ev.tool_events);
          const progressPatches = incomingEvents.filter((event) => event.phase === "progress");
          const structuredEvents = incomingEvents.filter((event) => event.phase !== "progress");
          if (structuredEvents.length === 0 && progressPatches.length > 0) {
            // A live progress frame refreshes its running tool row in place.
            // It never adds a row, so long GPU handoffs cannot duplicate traces.
            setMessages((prev) => {
              const index = findLiveProgressRow(prev, progressPatches);
              if (index < 0) return prev;
              const row = prev[index];
              const toolEvents = applyToolProgressPatches(row.toolEvents, progressPatches);
              if (toolEvents === row.toolEvents) return prev;
              return replaceMessageAt(prev, index, { ...row, toolEvents });
            });
            return;
          }
          const turn = turnFieldsFromEvent(ev, "activity");
          setMessages((prev) => {
            const segmentId = ensureActivitySegmentId();
            const base = prev;
            const visibleStructuredEvents = filterCoveredFileEditToolEvents(base, structuredEvents);
            const structuredLines = toolTraceLinesFromEvents(visibleStructuredEvents);
            const lines = structuredLines.length > 0
              ? structuredLines
              : structuredEvents.length > 0
                ? []
                : ev.text
                  ? [ev.text]
                  : [];
            if (lines.length === 0) return base;
            const last = base[base.length - 1];
            if (
              last
              && last.kind === "trace"
              && !last.isStreaming
              && (!last.activitySegmentId || last.activitySegmentId === segmentId)
            ) {
              const previousTraces = last.traces?.length
                ? last.traces
                : last.content
                  ? [last.content]
                  : [];
              const mergedEvents = visibleStructuredEvents.length > 0
                ? mergeToolProgressEvents(last.toolEvents, visibleStructuredEvents)
                : last.toolEvents;
              const mergedLines = visibleStructuredEvents.length > 0
                ? mergeToolProgressTraceLines(
                    previousTraces,
                    last.toolEvents,
                    structuredLines,
                    visibleStructuredEvents,
                  )
                : null;
              const merged: UIMessage = {
                ...last,
                traces: mergedLines ?? [...previousTraces, ...lines],
                content: mergedLines
                  ? mergedLines[mergedLines.length - 1]
                  : lines[lines.length - 1],
                toolEvents: applyToolProgressPatches(mergedEvents, progressPatches),
                activitySegmentId: last.activitySegmentId ?? segmentId,
                ...turn,
              };
              return [...base.slice(0, -1), merged];
            }
            return [
              ...base,
              {
                id: crypto.randomUUID(),
                role: "tool",
                kind: "trace",
                content: lines[lines.length - 1],
                traces: lines,
                ...(visibleStructuredEvents.length ? {
                  toolEvents: applyToolProgressPatches(
                    visibleStructuredEvents,
                    progressPatches,
                  ),
                } : {}),
                activitySegmentId: segmentId,
                ...turn,
                createdAt: Date.now(),
              },
            ];
          });
          return;
        }

        const media = ev.media_urls?.length
          ? ev.media_urls.map((m) => toMediaAttachment(m))
          : ev.media?.map((url) => toMediaAttachment({ url }));
        const hasMedia = !!media && media.length > 0;
        if (sideChannelEvent) {
          setMessages((prev) => absorbCompleteAssistantMessage(prev, {
            content: ev.text,
            ...(hasMedia ? { media } : {}),
            ...(ev.source ? { source: ev.source } : {}),
            ...turnFieldsFromEvent(ev, "answer"),
          }));
          if (typeof ev.turn_id === "string") sideChannelTurnIdsRef.current.delete(ev.turn_id);
          return;
        }

        // A complete (non-streamed) assistant message. If a stream was in
        // flight, drop the placeholder so we don't render the text twice.
        // ``turn_end`` is the turn boundary. ``stream_end`` only closes the
        // current text segment so a following tool/reasoning segment remains
        // part of the same live activity surface.
        clearActivitySegment();
        const completedTurn = turnFieldsFromEvent(ev, "answer");
        if (completedTurn.turnId) deferredAnswersRef.current.delete(completedTurn.turnId);
        setMessages((prev) => {
          const activeId = buffer.current?.messageId;
          buffer.current = null;
          activeAssistantRef.current = null;
          const filtered = activeId ? prev.filter((m) => m.id !== activeId) : prev;
          const content = ev.text;
          const lat =
            typeof ev.latency_ms === "number" && ev.latency_ms >= 0
              ? Math.round(ev.latency_ms)
              : undefined;
          return absorbCompleteAssistantMessage(filtered, {
            content,
            ...(hasMedia ? { media } : {}),
            ...(lat !== undefined ? { latencyMs: lat } : {}),
            ...(ev.source ? { source: ev.source } : {}),
            ...completedTurn,
          });
        });
        if (hasMedia) {
          const caption = ev.text.trim();
          if (caption) {
            const mediaReplyTexts = mediaReplyTextsRef.current;
            const texts = mediaReplyTexts.get(completedTurn.turnId) ?? new Set<string>();
            texts.add(caption);
            mediaReplyTexts.set(completedTurn.turnId, texts);
          }
        }
        return;
      }
      if (ev.event === "file_edit") {
        const edits = Array.isArray(ev.edits) ? ev.edits : [];
        if (edits.length === 0) return;
        const normalized = mergeFileEdits(undefined, edits);
        if (normalized.length === 0) return;
        const turn = turnFieldsFromEvent(ev, "activity");
        const opensFileEditPhase = normalized.some(
          (edit) => edit.status === "editing" || edit.phase === "start",
        );
        let eventSegmentId = fileEditSegmentRef.current;
        if (!eventSegmentId && opensFileEditPhase) {
          eventSegmentId = detachedActivitySegmentId();
          fileEditSegmentRef.current = eventSegmentId;
        }
        setMessages((prev) => {
          let segmentId = eventSegmentId;
          const base = stripCoveredFileEditToolHintsFromMessages(prev, normalized, turn);
          const targetIndex = findFileEditTraceIndex(base, segmentId, normalized);
          if (targetIndex !== null) {
            const target = base[targetIndex];
            segmentId = target.activitySegmentId ?? segmentId ?? detachedActivitySegmentId();
            if (opensFileEditPhase) fileEditSegmentRef.current = segmentId;
            const merged: UIMessage = {
              ...target,
              fileEdits: mergeFileEdits(target.fileEdits, normalized),
              activitySegmentId: segmentId,
              ...turn,
            };
            return replaceMessageAt(base, targetIndex, merged);
          }
          segmentId = segmentId ?? detachedActivitySegmentId();
          if (opensFileEditPhase) fileEditSegmentRef.current = segmentId;
          return [
            ...base,
            {
              id: crypto.randomUUID(),
              role: "tool",
              kind: "trace",
              content: "",
              traces: [],
              fileEdits: normalized,
              activitySegmentId: segmentId,
              ...turn,
              createdAt: Date.now(),
            },
          ];
        });
        return;
      }
    };

    const unsub = client.onChat(chatId, handle);
    return () => {
      unsub();
      buffer.current = null;
      activeAssistantRef.current = null;
      closedAssistantStreamIdsRef.current.clear();
      clearActivitySegment();
      clearPendingStreamWork();
    };
  }, [
    applyStreamError,
    acceptRunState,
    chatId,
    closeActiveAssistantStream,
    client,
    clearActivitySegment,
    clearPendingStreamWork,
    detachedActivitySegmentId,
    ensureActivitySegmentId,
    flushPendingStreamEvents,
    isSideChannelEvent,
    onTurnEnd,
    resolveAdmission,
    refreshUnknownMessageSend,
    schedulePendingStreamFlush,
    expectedRootSessionKey,
    initialStopTarget,
  ]);

  const send = useCallback(
    (content: string, images?: SendAttachment[], options?: SendOptions) => {
      if (!chatId) return null;
      const hasAttachments = !!images && images.length > 0;
      // Text is optional when files are attached — the agent will still see
      // them via ``media`` paths.
      if (!hasAttachments && !content.trim()) return null;
      const continueActiveTurn = options?.continueActiveTurn === true;
      const targetTurnId = continueActiveTurn
        ? (options?.steerTargetTurnId ?? client.getRunTurnId(chatId))
        : null;
      // Keep the selected target even if it has since changed. The gateway
      // validates that exact turn and can settle a queued attempt as refused.
      if (continueActiveTurn && !targetTurnId) return null;

      const queuedAttempt = options?.queuedAttempt;
      const startsNewRun = !options?.sideChannel && !options?.continueActiveTurn;
      if (
        queuedAttempt
        && !client.prepareQueuedAttempt(chatId, queuedAttempt, startsNewRun)
      ) return null;
      if (startsNewRun) {
        setQueueCompletion(null);
        setStopFeedback(null);
      }

      setStreamError(null);
      const sideChannel = options?.sideChannel === true;
      const finalizeActiveTurn = options?.finalizeActiveTurn === true;
      const outboundContent = options?.quotedContext
        ? formatQuotedUserMessage(content, options.quotedContext)
        : content;
      flushPendingStreamEvents();
      if (finalizeActiveTurn) {
        setIsStreaming(false);
        setRetryStatus(null);
      }
      const turnId = queuedAttempt?.turnId ?? crypto.randomUUID();
      const userMessageId = crypto.randomUUID();
      const admission = sideChannel || queuedAttempt
        ? undefined
        : new Promise<boolean>((resolve) => {
            admissionWaitersRef.current.set(turnId, resolve);
          });
      if (sideChannel) sideChannelTurnIdsRef.current.add(turnId);
      const previews = hasAttachments ? images!.map((i) => i.preview) : undefined;
      setMessages((prev) => {
        if ((!sideChannel && !continueActiveTurn) || finalizeActiveTurn) {
          buffer.current = null;
          activeAssistantRef.current = null;
          closedAssistantStreamIdsRef.current.clear();
          clearActivitySegment();
          mediaReplyTextsRef.current.clear();
        } else if (continueActiveTurn) {
          // Guidance belongs to the active backend turn. Preserve the answer
          // cursor so its resuming stream_end can finalize the text already
          // shown before the new user row, while starting fresh activity after it.
          clearActivitySegment();
        }
        const base = finalizeActiveTurn ? finalizeStreamedTurn(prev) : prev;
        return [
          ...base,
          {
            id: userMessageId,
            role: "user",
            content: outboundContent,
            turnId,
            turnPhase: "user",
            turnSeq: 0,
            deliveryStatus: "sending",
            createdAt: Date.now(),
            ...(previews ? { media: previews } : {}),
            ...(options?.cliApps?.length ? { cliApps: options.cliApps } : {}),
            ...(options?.mcpPresets?.length ? { mcpPresets: options.mcpPresets } : {}),
            ...(options?.sessionMentions?.length
              ? { sessionMentions: options.sessionMentions }
              : {}),
          },
        ];
      });
      if (!sideChannel) setIsStreaming(true);
      const wireMedia = hasAttachments ? images!.map((i) => i.media) : undefined;
      const clientOptions = {
        ...options,
        turnId,
        requestIntent: continueActiveTurn ? "steer" as const : "new" as const,
        ...(targetTurnId ? { targetTurnId } : {}),
        ...((sideChannel || continueActiveTurn) ? { startsNewRun: false } : {}),
      };
      delete clientOptions.queuedAttempt;
      delete clientOptions.quotedContext;
      delete clientOptions.finalizeActiveTurn;
      delete clientOptions.continueActiveTurn;
      delete clientOptions.steerTargetTurnId;
      const needsUpload = !!wireMedia?.some(
        (media) => !!media.data_url && !media.id,
      );
      if (
        needsUpload
        && wireMedia
        && typeof client.uploadAttachment === "function"
      ) {
        void (async () => {
          try {
            const uploaded = await Promise.all(
              wireMedia.map(async (media) => {
                if (media.id || !media.data_url) {
                  return media;
                }
                const id = await client.uploadAttachment(chatId, media);
                return { id, ...(media.name ? { name: media.name } : {}) };
              }),
            );
            client.sendMessage(chatId, outboundContent, uploaded, clientOptions);
          } catch {
            // Upload-first is an optimization; the inline envelope remains valid.
            client.sendMessage(chatId, outboundContent, wireMedia, clientOptions);
          }
        })();
      } else {
        client.sendMessage(chatId, outboundContent, wireMedia, clientOptions);
      }
      return { turnId, userMessageId, sideChannel, ...(admission ? { admission } : {}) };
    },
    [chatId, clearActivitySegment, client, flushPendingStreamEvents],
  );

  const stop = useCallback(() => {
    if (!chatId) return;
    if (!connectionOpen) {
      setStopNotice("unavailable");
      return;
    }
    const target = stopTargetRef.current;
    const currentRun = runStateRef.current;
    const canStopRun = currentRun !== undefined
      && currentRun !== null
      && currentRun.state !== "stopping"
      && currentRun.can_stop;
    const targetCanStop = target !== null && (
      !target.active_root_present
      || currentRun === undefined
      || (
        currentRun !== null
        && currentRun.state !== "stopping"
        && currentRun.can_stop
      )
    );
    const completeRunIdentity = canStopRun
      && typeof currentRun.turn_id === "string"
      && currentRun.turn_id.trim().length > 0
      && typeof currentRun.run_id === "string"
      && currentRun.run_id.trim().length > 0
      && typeof currentRun.source_incarnation === "string"
      && currentRun.source_incarnation.trim().length > 0;
    const currentRunTurnId = currentRun && typeof currentRun.turn_id === "string"
      && currentRun.turn_id.trim().length > 0
      ? currentRun.turn_id
      : null;
    const legacyTargetTurnId = canStopRun
      ? currentRunTurnId
      : currentRun === undefined
        ? client.getRunTurnId(chatId)
        : null;
    const hasValidLegacyTargetTurnId = typeof legacyTargetTurnId === "string"
      && legacyTargetTurnId.trim().length > 0;
    const hasModernSnapshot = stopTargetSnapshotAvailable || liveStopTargetSeenRef.current;
    const legacyFallbackAllowed = currentRun === undefined
      && !liveRunStateSeenRef.current
      && !staleRunStateRefreshPendingRef.current
      && !requireVerifiedStopTarget
      && !hasModernSnapshot
      && isStreaming
      && !childOwnedWorkRef.current
      && hasValidLegacyTargetTurnId;
    const legacyTarget = targetCanStop ? target : null;
    if (!completeRunIdentity && !legacyTarget && !(
      canStopRun && hasValidLegacyTargetTurnId
    ) && !legacyFallbackAllowed) {
      setStopNotice("unavailable");
      projectionRefreshRef.current?.();
      return;
    }
    if (legacyTarget && expectedRootSessionKey
      && legacyTarget.source.session_key !== expectedRootSessionKey) {
      stopTargetRef.current = null;
      setStopTarget(null);
      setStopNotice("unavailable");
      projectionRefreshRef.current?.();
      return;
    }
    flushPendingStreamEvents();
    const capturedRunIdentity = canStopRun ? runIdentity(currentRun) : null;
    stoppedTurnIdRef.current = canStopRun
      ? currentRun.turn_id
      : legacyTarget?.active_turn_id ?? legacyTargetTurnId;
    setQueueCompletion(null);
    setStopPending(true);
    setStopNotice(null);
    setStopFeedback(null);
    if (canStopRun && currentRun) {
      const stoppingRun = { ...currentRun, state: "stopping" as const };
      runStateRef.current = stoppingRun;
      setRunState(stoppingRun);
      setRunStartedAt(stoppingRun.started_at);
      setIsStreaming(true);
      pendingStopRunIdentityRef.current = capturedRunIdentity;
    }
    if (completeRunIdentity && currentRun) {
      const captured = currentRun;
      void client.requestMutation("run.stop", {
        chat_id: chatId,
        source_incarnation: captured.source_incarnation,
        turn_id: captured.turn_id,
        run_id: captured.run_id,
      }).catch((error: unknown) => {
        if (runIdentity(runStateRef.current) !== capturedRunIdentity) return;
        if (isMutationError(error, {
          status: 404,
          outcome: "not_applied",
          message: "unknown WebUI mutation action",
        })) {
          client.sendMessage(chatId, "/stop", undefined, { targetTurnId: captured.turn_id! });
          return;
        }
        setStopPending(false);
        if (isMutationError(error, { status: 409, message: "stale_stop_target" })) {
          staleStopTargetHistoryBaselineRef.current = initialStopTarget;
          staleStopTargetRef.current = stopTargetRef.current;
          retiredRunIdentitiesRef.current.add(capturedRunIdentity!);
          runStateRef.current = undefined;
          setRunState(undefined);
          liveRunStateSeenRef.current = false;
          staleRunStateRefreshPendingRef.current = true;
          stopTargetRef.current = null;
          setStopTarget(null);
          pendingStopRunIdentityRef.current = null;
          setStopNotice("stale_target");
          projectionRefreshRef.current?.();
          return;
        }
        setStopNotice("unavailable");
      });
      return;
    }
    if (legacyTarget) {
      client.sendMessage(chatId, "/stop", undefined, { stopExpectation: legacyTarget });
    } else if (hasValidLegacyTargetTurnId) {
      client.sendMessage(chatId, "/stop", undefined, { targetTurnId: legacyTargetTurnId! });
    }
  }, [
    chatId,
    client,
    connectionOpen,
    expectedRootSessionKey,
    flushPendingStreamEvents,
    isStreaming,
    initialStopTarget,
    requireVerifiedStopTarget,
    stopTargetSnapshotAvailable,
  ]);

  const reconcileTurnComplete = useCallback(() => {
    clearPendingStreamWork();
    buffer.current = null;
    activeAssistantRef.current = null;
    closedAssistantStreamIdsRef.current.clear();
    clearActivitySegment();
    mediaReplyTextsRef.current.clear();
    setRunStartedAt(null);
    setRetryStatus(null);
    setIsStreaming(false);
  }, [clearActivitySegment, clearPendingStreamWork]);

  const transcribeAudio = useCallback(
    (dataUrl: string, options?: { durationMs?: number }) =>
      client.transcribeAudio(dataUrl, options),
    [client],
  );

  const recoveryAction = useCallback(async (action: "continue" | "dismiss") => {
    if (!chatId || !recoveryState) return;
    await client.requestMutation(`recovery.${action}`, {
      chat_id: chatId,
      recovery_id: recoveryState.recovery_id,
    });
  }, [chatId, client, recoveryState]);

  const continueRecovery = useCallback(
    () => recoveryAction("continue"),
    [recoveryAction],
  );
  const dismissRecovery = useCallback(
    () => recoveryAction("dismiss"),
    [recoveryAction],
  );

  return {
    messages,
    messagesReady: messageOwnerChatId === chatId,
    isStreaming,
    queueCompletionTurnId: queueCompletion?.chatId === chatId
      && queueCompletion.turnId === client.getRunTurnId(chatId)
      ? queueCompletion.turnId
      : null,
    hasOwnedWork,
    stopCanBeSent,
    stopPending,
    stopNotice,
    stopFeedback: stopFeedback && !(
      stopFeedback.turnId && messages.some((message) => (
        message.kind === "run_stopped" && message.turnId === stopFeedback.turnId
      ))
    ) ? stopFeedback.text : null,
    dismissStopNotice,
    stopTarget,
    runState,
    connectionOpen,
    hasUnknownMessageSend: unknownMessageSend.chatId === chatId
      ? unknownMessageSend.value
      : chatId ? client.hasUnknownMessageSend(chatId) : false,
    runStartedAt,
    retryStatus,
    goalState,
    recoveryState,
    continueRecovery,
    dismissRecovery,
    send,
    joinQueuedAttempt,
    consumeQueuedAttempt,
    transcribeAudio,
    stop,
    reconcileTurnComplete,
    setMessages,
    streamError,
    dismissStreamError,
  };
}
