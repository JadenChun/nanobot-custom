import { useCallback, useEffect, useRef, useState } from "react";

import { useClient } from "@/providers/ClientProvider";
import i18n from "@/i18n";
import {
  ApiError,
  deleteSession as apiDeleteSession,
  fetchDelegatedChildren,
  fetchSessionAutomations,
  fetchSessionTurns,
  fetchWebuiThread,
  listSessions,
  normalizeWorkspacePermissionProjection,
} from "@/lib/api";
import { hasPendingAgentActivity } from "@/lib/activity-timeline";
import {
  delegatedChildReferences,
  mergeDelegatedChildRows,
  parseStopExpectation,
} from "@/lib/delegated-child-projection";
import { deriveTitle } from "@/lib/format";
import {
  readCachedSessionPage,
  writeCachedSessionPage,
} from "@/lib/session-page-cache";
import type {
  ChatSummary,
  DelegatedChildReference,
  FolderAccessRequest,
  SessionAutomationJob,
  SessionDeleteResult,
  SessionTurnSummary,
  UIMessage,
  StopExpectation,
  WorkspacePermissionProjection,
  WorkspaceScopePayload,
} from "@/lib/types";

const EMPTY_MESSAGES: UIMessage[] = [];
const INITIAL_HISTORY_PAGE_LIMIT = 40;
const TURN_INDEX_PAGE_LIMIT = 400;
const OLDER_HISTORY_PAGE_LIMIT = 120;
const DELEGATED_CHILD_PAGE_LIMIT = 120;
const CHAT_CREATE_TIMEOUT_MS = 60_000;

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object"
    && error !== null
    && "name" in error
    && error.name === "AbortError"
  );
}

export type SessionHistoryContinuity = "initial" | "overlap" | "reset";

function mergePermissionProjection(
  current: WorkspacePermissionProjection | null,
  incoming: WorkspacePermissionProjection | null,
): WorkspacePermissionProjection | null {
  if (
    current
    && incoming
    && current.runtime_id === incoming.runtime_id
    && incoming.revision < current.revision
  ) return current;
  return incoming;
}

function persistedMessagesToUi(messages: UIMessage[]): UIMessage[] {
  return messages.map((m, idx) => ({
    ...m,
    id: m.id ?? `hist-${idx}`,
    createdAt: typeof m.createdAt === "number" ? m.createdAt : Date.now(),
  }));
}

function hydrateDelegatedChildren(
  messages: UIMessage[],
  children: DelegatedChildReference[],
  parentKey: string,
  rootSource: StopExpectation["source"] | null,
): { children: DelegatedChildReference[]; refreshRequired: boolean } {
  const options = {
    expectedRootKey: parentKey,
    expectedRoot: rootSource,
    mode: "hydration" as const,
  };
  const persisted = mergeDelegatedChildRows([], delegatedChildReferences(messages), options);
  const hydrated = mergeDelegatedChildRows(persisted.messages, children, options);
  return {
    children: delegatedChildReferences(hydrated.messages),
    refreshRequired: persisted.refreshRequired || hydrated.refreshRequired,
  };
}

function sameSemanticMessage(a: UIMessage, b: UIMessage): boolean {
  return (
    a.role === b.role
    && (a.kind ?? "") === (b.kind ?? "")
    && a.content === b.content
    && (!a.turnId || !b.turnId || a.turnId === b.turnId)
  );
}

function longestSemanticOverlap(previous: UIMessage[], latest: UIMessage[]): number {
  const maxOverlap = Math.min(previous.length, latest.length);
  for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
    const previousStart = previous.length - overlap;
    let matches = true;
    for (let index = 0; index < overlap; index += 1) {
      if (!sameSemanticMessage(previous[previousStart + index], latest[index])) {
        matches = false;
        break;
      }
    }
    if (matches) return overlap;
  }
  return 0;
}

function mergeLatestHistory(
  previous: UIMessage[],
  latest: UIMessage[],
  initial: boolean,
): {
  continuity: SessionHistoryContinuity;
  messages: UIMessage[];
  retainedPrefixLength: number;
} {
  if (initial) {
    return {
      continuity: "initial",
      messages: latest,
      retainedPrefixLength: 0,
    };
  }
  const overlapLength = longestSemanticOverlap(previous, latest);
  if (overlapLength === 0) {
    return {
      continuity: "reset",
      messages: latest,
      retainedPrefixLength: 0,
    };
  }
  const retainedPrefixLength = previous.length - overlapLength;
  return {
    continuity: "overlap",
    messages: [...previous.slice(0, retainedPrefixLength), ...latest],
    retainedPrefixLength,
  };
}

function hasPendingToolCallsFromThread(
  body: Awaited<ReturnType<typeof fetchWebuiThread>>,
  messages: UIMessage[],
): boolean {
  if (typeof body?.has_pending_tool_calls === "boolean") {
    return body.has_pending_tool_calls;
  }
  return hasPendingAgentActivity(messages);
}

function completedTurnIdsFromThread(
  body: Awaited<ReturnType<typeof fetchWebuiThread>>,
): string[] {
  if (!Array.isArray(body?.completed_turn_ids)) return [];
  return Array.from(new Set(
    body.completed_turn_ids.filter(
      (turnId): turnId is string => typeof turnId === "string" && turnId.length > 0,
    ),
  ));
}

async function fetchAllDelegatedChildren(
  getToken: () => string,
  key: string,
  signal: AbortSignal,
): Promise<{
  children: DelegatedChildReference[];
  stopTarget: StopExpectation | null;
  stopTargetSnapshotAvailable: boolean;
  incomplete: boolean;
}> {
  const children: DelegatedChildReference[] = [];
  const seenCursors = new Set<string>();
  let before: string | null = null;
  let stopTarget: StopExpectation | null = null;
  let stopTargetSnapshotAvailable = false;
  let incomplete = false;
  do {
    const page = await fetchDelegatedChildren(getToken(), key, {
      limit: DELEGATED_CHILD_PAGE_LIMIT,
      before,
      signal,
    });
    if (!page) return { children, stopTarget, stopTargetSnapshotAvailable, incomplete };
    if (page.parentSessionKey && page.parentSessionKey !== key) {
      return {
        children: [],
        stopTarget: null,
        stopTargetSnapshotAvailable: "stop_target" in page,
        incomplete: true,
      };
    }
    if (before === null) {
      stopTargetSnapshotAvailable = "stop_target" in page;
      stopTarget = parseStopExpectation(page.stop_target);
    }
    children.push(...(Array.isArray(page.children) ? page.children : []));
    if (page.page?.has_more_before !== true) break;
    const next = page.page.before_cursor;
    if (!next || seenCursors.has(next)) {
      incomplete = true;
      break;
    }
    seenCursors.add(next);
    before = next;
  } while (!signal.aborted);
  return { children, stopTarget, stopTargetSnapshotAvailable, incomplete };
}

/** Sidebar state: fetches the full session list and exposes create / delete actions. */
export function useSessions(): {
  sessions: ChatSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  createChat: (
    workspaceScope?: WorkspaceScopePayload | null,
    modelPreset?: string | null,
  ) => Promise<string>;
  forkChat: (sourceChatId: string, beforeUserIndex: number, title?: string) => Promise<string>;
  deleteChat: (
    key: string,
    options?: { deleteAutomations?: boolean },
  ) => Promise<SessionDeleteResult>;
  getSessionAutomations: (key: string) => Promise<SessionAutomationJob[]>;
} {
  const { client, token } = useClient();
  const [sessions, setSessions] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const tokenRef = useRef(token);
  const optimisticKeysRef = useRef<Set<string>>(new Set());
  const refreshPendingRef = useRef(false);
  const refreshInFlightRef = useRef<Promise<void> | null>(null);
  tokenRef.current = token;

  const refresh = useCallback((): Promise<void> => {
    refreshPendingRef.current = true;
    if (refreshInFlightRef.current) return refreshInFlightRef.current;
    const request = (async () => {
      setLoading(true);
      try {
        while (refreshPendingRef.current) {
          refreshPendingRef.current = false;
          try {
            const rows = await listSessions(tokenRef.current);
            const serverKeys = new Set(rows.map((row) => row.key));
            setSessions((prev) => [
              ...rows,
              ...prev.filter(
                (session) =>
                  optimisticKeysRef.current.has(session.key)
                  && !serverKeys.has(session.key),
              ),
            ]);
            for (const key of Array.from(optimisticKeysRef.current)) {
              if (serverKeys.has(key)) optimisticKeysRef.current.delete(key);
            }
            setError(null);
          } catch (e) {
            const msg =
              e instanceof ApiError ? `HTTP ${e.status}` : (e as Error).message;
            setError(msg);
          }
        }
      } finally {
        refreshInFlightRef.current = null;
        setLoading(false);
      }
    })();
    refreshInFlightRef.current = request;
    return request;
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let disposed = false;
    let refreshQueued = false;
    const unsubscribe = client.onSessionUpdate(() => {
      if (refreshQueued) return;
      refreshQueued = true;
      queueMicrotask(() => {
        refreshQueued = false;
        if (!disposed) void refresh();
      });
    });
    return () => {
      disposed = true;
      unsubscribe();
    };
  }, [client, refresh]);

  const createChat = useCallback(async (
    workspaceScope?: WorkspaceScopePayload | null,
    modelPreset?: string | null,
  ): Promise<string> => {
    const chatId = await client.newChat(CHAT_CREATE_TIMEOUT_MS, workspaceScope);
    const key = `websocket:${chatId}`;
    optimisticKeysRef.current.add(key);
    // Optimistic insert; a subsequent refresh will replace it with the
    // authoritative row once the server persists the session.
    setSessions((prev) => [
      {
        key,
        channel: "websocket",
        chatId,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        title: "",
        preview: "",
        modelPreset: modelPreset ?? null,
        workspaceScope: workspaceScope ?? null,
      },
      ...prev.filter((s) => s.key !== key),
    ]);
    return chatId;
  }, [client]);

  const forkChat = useCallback(async (
    sourceChatId: string,
    beforeUserIndex: number,
    title?: string,
  ): Promise<string> => {
    const chatId = await client.forkChat(
      sourceChatId,
      beforeUserIndex,
      title,
      CHAT_CREATE_TIMEOUT_MS,
    );
    const key = `websocket:${chatId}`;
    optimisticKeysRef.current.add(key);
    setSessions((prev) => [
      {
        key,
        channel: "websocket",
        chatId,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        title: title ?? "",
        preview: "",
        workspaceScope: null,
      },
      ...prev.filter((s) => s.key !== key),
    ]);
    return chatId;
  }, [client]);

  const deleteChat = useCallback(
    async (key: string, options?: { deleteAutomations?: boolean }) => {
      const optimistic = optimisticKeysRef.current.has(key);
      const result = await apiDeleteSession(client, key, options);
      if (result.blocked_by_automations || (!result.deleted && !optimistic)) return result;
      optimisticKeysRef.current.delete(key);
      setSessions((prev) => prev.filter((s) => s.key !== key));
      // The gateway may have restarted and forgotten an unpersisted chat's
      // draft scope, but removing that optimistic session is still a deletion.
      return result.deleted ? result : { ...result, deleted: true };
    },
    [client],
  );

  const getSessionAutomations = useCallback(async (key: string) => {
    const result = await fetchSessionAutomations(tokenRef.current, key);
    return result.jobs;
  }, []);

  return {
    sessions,
    loading,
    error,
    refresh,
    createChat,
    forkChat,
    deleteChat,
    getSessionAutomations,
  };
}

/** Lazy-load a session's on-disk messages the first time the UI displays it. */
export function useSessionHistory(key: string | null): {
  messages: UIMessage[];
  loading: boolean;
  loadingOlder: boolean;
  error: string | null;
  refresh: () => void;
  loadOlder: () => Promise<void>;
  hasMoreBefore: boolean;
  userMessageOffset: number;
  version: number;
  forkBoundaryMessageCount: number | null;
  /** ``true`` when the server reports that the turn is still in flight. */
  hasPendingToolCalls: boolean;
  /** Turn identities backed by explicit persisted completion events. */
  completedTurnIds: string[];
  /** Relationship between the latest canonical page and its predecessor. */
  continuity: SessionHistoryContinuity;
  /** Stable across overlapping latest pages; changes on initial load or reset. */
  lineage: number;
  /** Exact active turn when supplied by a current gateway. */
  activeTurnId: string | null;
  /** The matching root response explicitly supplied both live activity fields. */
  activitySnapshotAvailable: boolean;
  /** Live root-scoped snapshot used only for an explicitly requested Stop. */
  stopTarget: StopExpectation | null;
  /** True when authenticated history explicitly included the live Stop field, even when null. */
  stopTargetSnapshotAvailable: boolean;
  /** Child activity is hydrated separately; it cannot delay root history or permissions. */
  delegatedChildren: DelegatedChildReference[];
  delegatedChildrenLoading: boolean;
  /** True when a bounded child-history page failed or conflicted. */
  delegatedChildrenError: boolean;
  workspaceAccessRequest: FolderAccessRequest | null;
  workspaceAccessRequestLoading: boolean;
  workspaceAccessRequestError: boolean;
  markWorkspaceAccessRequestStale: () => void;
  updateWorkspaceAccessRequest: (request: FolderAccessRequest | null) => void;
  refreshWorkspaceAccessRequest: () => Promise<FolderAccessRequest | null>;
  workspacePermissions: WorkspacePermissionProjection | null;
  workspacePermissionsLoading: boolean;
  workspacePermissionsError: boolean;
  markWorkspacePermissionsStale: () => void;
  updateWorkspacePermissions: (permissions: WorkspacePermissionProjection | null) => void;
  refreshWorkspacePermissions: () => Promise<WorkspacePermissionProjection | null>;
  /** Every turn in the session, newest page last; loaded on demand. */
  turnIndex: SessionTurnSummary[];
  totalTurnCount: number;
  loadTurnIndex: () => Promise<SessionTurnSummary[]>;
  /** Replace the window with the page ending at this turn cursor. */
  loadTurnAt: (cursor: string) => Promise<void>;
} {
  const { getToken } = useClient();
  const loadingOlderRef = useRef(false);
  const olderRequestAbortRef = useRef<AbortController | null>(null);
  const historyVersionRef = useRef(0);
  const accessRequestReadVersionRef = useRef(0);
  const permissionReadVersionRef = useRef(0);
  const [refreshSeq, setRefreshSeq] = useState(0);
  const [turnIndex, setTurnIndex] = useState<SessionTurnSummary[]>([]);
  const [totalTurnCount, setTotalTurnCount] = useState(0);
  const refresh = useCallback(() => {
    setRefreshSeq((value) => value + 1);
  }, []);
  const [state, setState] = useState<{
    key: string | null;
    messages: UIMessage[];
    loading: boolean;
    loadingOlder: boolean;
    error: string | null;
    hasPendingToolCalls: boolean;
    completedTurnIds: string[];
    forkBoundaryMessageCount: number | null;
    beforeCursor: string | null;
    hasMoreBefore: boolean;
    userMessageOffset: number;
    version: number;
    continuity: SessionHistoryContinuity;
    lineage: number;
    activeTurnId: string | null;
    activitySnapshotAvailable: boolean;
    stopTarget: StopExpectation | null;
    stopTargetSnapshotAvailable: boolean;
    delegatedChildren: DelegatedChildReference[];
    delegatedChildrenLoading: boolean;
    delegatedChildrenError: boolean;
    workspaceAccessRequest: FolderAccessRequest | null;
    workspaceAccessRequestLoading: boolean;
    workspaceAccessRequestError: boolean;
    workspacePermissions: WorkspacePermissionProjection | null;
    workspacePermissionsLoading: boolean;
    workspacePermissionsError: boolean;
  }>({
    key: null,
    messages: [],
    loading: false,
    loadingOlder: false,
    error: null,
    hasPendingToolCalls: false,
    completedTurnIds: [],
    forkBoundaryMessageCount: null,
    beforeCursor: null,
    hasMoreBefore: false,
    userMessageOffset: 0,
    version: 0,
    continuity: "initial",
    lineage: 0,
    activeTurnId: null,
    activitySnapshotAvailable: false,
    stopTarget: null,
    stopTargetSnapshotAvailable: false,
    delegatedChildren: [],
    delegatedChildrenLoading: false,
    delegatedChildrenError: false,
    workspaceAccessRequest: null,
    workspaceAccessRequestLoading: false,
    workspaceAccessRequestError: false,
    workspacePermissions: null,
    workspacePermissionsLoading: false,
    workspacePermissionsError: false,
  });

  useEffect(() => () => {
    olderRequestAbortRef.current?.abort();
    olderRequestAbortRef.current = null;
    loadingOlderRef.current = false;
  }, []);

  useEffect(() => {
    if (!key) {
      olderRequestAbortRef.current?.abort();
      olderRequestAbortRef.current = null;
      loadingOlderRef.current = false;
      setState({
        key: null,
        messages: [],
        loading: false,
        loadingOlder: false,
        error: null,
        hasPendingToolCalls: false,
        completedTurnIds: [],
        forkBoundaryMessageCount: null,
        beforeCursor: null,
        hasMoreBefore: false,
        userMessageOffset: 0,
        version: 0,
        continuity: "initial",
        lineage: 0,
        activeTurnId: null,
        activitySnapshotAvailable: false,
        stopTarget: null,
        stopTargetSnapshotAvailable: false,
        delegatedChildren: [],
        delegatedChildrenLoading: false,
        delegatedChildrenError: false,
        workspaceAccessRequest: null,
        workspaceAccessRequestLoading: false,
        workspaceAccessRequestError: false,
        workspacePermissions: null,
        workspacePermissionsLoading: false,
        workspacePermissionsError: false,
      });
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    const accessRequestReadVersion = ++accessRequestReadVersionRef.current;
    const permissionReadVersion = ++permissionReadVersionRef.current;
    olderRequestAbortRef.current?.abort();
    olderRequestAbortRef.current = null;
    loadingOlderRef.current = false;
    setTurnIndex((current) => (current.length === 0 ? current : []));
    // Paint the previously loaded window immediately on a revisit. The network
    // refresh below still runs and replaces it, so this only removes the blank
    // wait; it never decides what the UI believes after the fetch settles.
    const cachedPage = readCachedSessionPage(key);
    if (cachedPage) {
      historyVersionRef.current += 1;
      setState((prev) => prev.key === key
        ? { ...prev, messages: cachedPage.messages, loading: false }
        : {
            key,
            messages: cachedPage.messages,
            loading: false,
            loadingOlder: false,
            error: null,
            hasPendingToolCalls: false,
            completedTurnIds: cachedPage.completedTurnIds,
            forkBoundaryMessageCount: cachedPage.forkBoundaryMessageCount,
            beforeCursor: cachedPage.beforeCursor,
            hasMoreBefore: cachedPage.hasMoreBefore,
            userMessageOffset: cachedPage.userMessageOffset,
            version: cachedPage.version,
            continuity: "initial",
            // Seed lineage at 0 (not the cached version) so the refresh below
            // is still treated as the initial page for this mount. The cached
            // window is a display stand-in only; treating it as an already
            // merged lineage would make the refresh discard the prefix.
            lineage: 0,
            activeTurnId: null,
            activitySnapshotAvailable: false,
            stopTarget: null,
            stopTargetSnapshotAvailable: false,
            delegatedChildren: [],
            delegatedChildrenLoading: false,
            delegatedChildrenError: false,
            workspaceAccessRequest: null,
            workspaceAccessRequestLoading: true,
            workspaceAccessRequestError: false,
            workspacePermissions: null,
            workspacePermissionsLoading: true,
            workspacePermissionsError: false,
          });
    }
    // Mark the new key as loading immediately so callers never see stale
    // messages from the previous session during the render right after a switch.
    setState((prev) => prev.key === key
      ? {
          ...prev,
          loading: true,
          loadingOlder: false,
          error: null,
          activitySnapshotAvailable: false,
          delegatedChildrenLoading: true,
          workspaceAccessRequestLoading: true,
          workspaceAccessRequestError: false,
          workspacePermissionsLoading: true,
          workspacePermissionsError: false,
        }
      : {
          key,
          messages: [],
          loading: true,
          loadingOlder: false,
          error: null,
          hasPendingToolCalls: false,
          completedTurnIds: [],
          forkBoundaryMessageCount: null,
          beforeCursor: null,
          hasMoreBefore: false,
          userMessageOffset: 0,
          version: 0,
          continuity: "initial",
          lineage: 0,
          activeTurnId: null,
          activitySnapshotAvailable: false,
          stopTarget: null,
          stopTargetSnapshotAvailable: false,
          delegatedChildren: [],
          delegatedChildrenLoading: true,
          delegatedChildrenError: false,
          workspaceAccessRequest: null,
          workspaceAccessRequestLoading: true,
          workspaceAccessRequestError: false,
          workspacePermissions: null,
          workspacePermissionsLoading: true,
          workspacePermissionsError: false,
        });
    (async () => {
      try {
        const body = await fetchWebuiThread(getToken(), key, {
          limit: INITIAL_HISTORY_PAGE_LIMIT,
          direction: "latest",
          signal: controller.signal,
        });
        if (cancelled) return;
        const persisted = persistedMessagesToUi(body?.messages ?? []);
        const ui = persisted.filter((message) => message.kind !== "delegated_child");
        const userMessageOffset = Math.max(0, body?.page?.user_message_offset ?? 0);
        historyVersionRef.current += 1;
        const responseVersion = historyVersionRef.current;
        const completedTurnIds = completedTurnIdsFromThread(body);
        const permissionProjection = normalizeWorkspacePermissionProjection(body?.workspace_permissions);
        const threadStopTargetSnapshotAvailable = body !== null && "stop_target" in body;
        const threadStopTarget = parseStopExpectation(body?.stop_target);
        const hasPending = hasPendingToolCallsFromThread(body, ui);
        const forkBoundary = typeof body?.fork_boundary_message_count === "number"
          ? Math.max(0, Math.min(body.fork_boundary_message_count, ui.length))
          : null;
        if (typeof body?.page?.total_turn_count === "number") {
          setTotalTurnCount(body.page.total_turn_count);
        }
        setState((prev) => {
          const merged = prev.key === key
            ? mergeLatestHistory(prev.messages, ui, prev.lineage === 0)
            : mergeLatestHistory([], ui, true);
          const retainedPrefix = merged.retainedPrefixLength > 0;
          const retainedForkBoundary = (
            retainedPrefix
            && prev.forkBoundaryMessageCount !== null
            && prev.forkBoundaryMessageCount <= merged.retainedPrefixLength
          )
            ? prev.forkBoundaryMessageCount
            : null;
          return {
            key,
            messages: merged.messages,
            loading: false,
            loadingOlder: false,
            error: null,
            hasPendingToolCalls: hasPending,
            completedTurnIds,
            forkBoundaryMessageCount: forkBoundary === null
              ? retainedForkBoundary
              : forkBoundary + merged.retainedPrefixLength,
            beforeCursor: retainedPrefix
              ? prev.beforeCursor
              : body?.page?.before_cursor ?? null,
            hasMoreBefore: retainedPrefix
              ? prev.hasMoreBefore
              : body?.page?.has_more_before === true,
            userMessageOffset: retainedPrefix
              ? prev.userMessageOffset
              : userMessageOffset,
            version: responseVersion,
            continuity: merged.continuity,
            lineage: merged.continuity === "overlap"
              ? prev.lineage
              : responseVersion,
            activeTurnId: typeof body?.active_turn_id === "string"
              ? body.active_turn_id
              : null,
            activitySnapshotAvailable: body !== null
              && (body.sessionKey === undefined || body.sessionKey === key)
              && (body.source === undefined || body.source === null || body.source.session_key === key)
              && body.history_kind !== "summary"
              && body.parent_session_key === undefined
              && (body.root_key === undefined || body.root_key === key)
              && (body.active_turn_id === null || typeof body.active_turn_id === "string")
              && typeof body.has_pending_tool_calls === "boolean",
            stopTarget: threadStopTarget,
            stopTargetSnapshotAvailable: threadStopTargetSnapshotAvailable,
            delegatedChildren: [],
            delegatedChildrenLoading: body !== null,
            delegatedChildrenError: false,
            workspaceAccessRequest: accessRequestReadVersion === accessRequestReadVersionRef.current
              ? body?.workspace_access_request ?? null
              : prev.workspaceAccessRequest,
            workspaceAccessRequestLoading: accessRequestReadVersion === accessRequestReadVersionRef.current
              ? false
              : prev.workspaceAccessRequestLoading,
            workspaceAccessRequestError: accessRequestReadVersion === accessRequestReadVersionRef.current
              ? false
              : prev.workspaceAccessRequestError,
            workspacePermissions: permissionReadVersion === permissionReadVersionRef.current
              ? mergePermissionProjection(prev.key === key ? prev.workspacePermissions : null, permissionProjection)
              : prev.workspacePermissions,
            workspacePermissionsLoading: permissionReadVersion === permissionReadVersionRef.current
              ? false
              : prev.workspacePermissionsLoading,
            workspacePermissionsError: permissionReadVersion === permissionReadVersionRef.current
              ? false
              : prev.workspacePermissionsError,
          };
        });
        if (body === null) return;

        // Fetch delegated children only when this page actually shows a child
        // row. Codex loads children per-parent on demand rather than scanning
        // every session on open, and the route is the most expensive read in
        // the gateway (list_sessions plus a full snapshot per child). A
        // conversation with no delegated activity has nothing to hydrate, so
        // the request cannot return anything. The thread payload already
        // carries `stop_target` when this gateway can derive it; the child page
        // is only the fallback for sessions that render child rows.
        const hasChildRows = persisted.some((message) => message.kind === "delegated_child");
        const childPages = hasChildRows
          ? await fetchAllDelegatedChildren(getToken, key, controller.signal)
            .catch(() => ({
              children: [],
              stopTarget: null,
              stopTargetSnapshotAvailable: false,
              incomplete: true,
            }))
          : {
            children: [],
            stopTarget: null,
            stopTargetSnapshotAvailable: false,
            incomplete: false,
          };
        if (cancelled) return;
        // Explicit null is a modern idle snapshot, not a request to fall back.
        const stopTarget = threadStopTargetSnapshotAvailable ? threadStopTarget : childPages.stopTarget;
        const hydrated = hydrateDelegatedChildren(persisted, childPages.children, key, stopTarget?.source ?? null);
        setState((prev) => prev.key === key && prev.version === responseVersion
          ? {
              ...prev,
              stopTarget,
              stopTargetSnapshotAvailable: threadStopTargetSnapshotAvailable
                || childPages.stopTargetSnapshotAvailable,
              delegatedChildren: hydrated.children,
              delegatedChildrenLoading: false,
              delegatedChildrenError: childPages.incomplete || hydrated.refreshRequired,
            }
          : prev);
      } catch (e) {
        if (cancelled || isAbortError(e)) return;
        controller.abort();
        if (e instanceof ApiError && e.status === 404) {
          historyVersionRef.current += 1;
          const responseVersion = historyVersionRef.current;
          setState((prev) => {
            const continuity = prev.key === key && prev.lineage > 0
              ? "reset"
              : "initial";
            return {
              key,
              messages: [],
              loading: false,
              loadingOlder: false,
              error: null,
              hasPendingToolCalls: false,
              completedTurnIds: [],
              forkBoundaryMessageCount: null,
              beforeCursor: null,
              hasMoreBefore: false,
              userMessageOffset: 0,
              version: responseVersion,
              continuity,
              lineage: responseVersion,
              activeTurnId: null,
              activitySnapshotAvailable: false,
              stopTarget: null,
              stopTargetSnapshotAvailable: false,
              delegatedChildren: [],
              delegatedChildrenLoading: false,
              delegatedChildrenError: false,
              workspaceAccessRequest: null,
              workspaceAccessRequestLoading: false,
              workspaceAccessRequestError: false,
              workspacePermissions: null,
              workspacePermissionsLoading: false,
              workspacePermissionsError: false,
            };
          });
        } else {
          setState((prev) => ({
            key,
            messages: [],
            loading: false,
            loadingOlder: false,
            error: (e as Error).message,
            hasPendingToolCalls: false,
            completedTurnIds: [],
            forkBoundaryMessageCount: null,
            beforeCursor: null,
            hasMoreBefore: false,
            userMessageOffset: 0,
            version: prev.key === key ? prev.version : 0,
            continuity: prev.key === key ? prev.continuity : "initial",
            lineage: prev.key === key ? prev.lineage : 0,
            activeTurnId: prev.key === key ? prev.activeTurnId : null,
            activitySnapshotAvailable: false,
            stopTarget: prev.key === key ? prev.stopTarget : null,
            stopTargetSnapshotAvailable: prev.key === key
              ? prev.stopTargetSnapshotAvailable
              : false,
            delegatedChildren: [],
            delegatedChildrenLoading: false,
            delegatedChildrenError: prev.key === key
              ? prev.delegatedChildrenError
              : false,
            workspaceAccessRequest: prev.key === key ? prev.workspaceAccessRequest : null,
            workspaceAccessRequestLoading: false,
            workspaceAccessRequestError: prev.key === key ? true : false,
            workspacePermissions: prev.key === key ? prev.workspacePermissions : null,
            workspacePermissionsLoading: false,
            workspacePermissionsError: prev.key === key ? true : false,
          }));
        }
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [getToken, key, refreshSeq]);

  // Keep the renderer cache in step with the settled window. Writing it here,
  // rather than mid-fetch, means a scroll-back prefix and any merged overlap are
  // part of what a revisit paints. Loading and error states are skipped so a
  // half-built window never becomes the cached view.
  useEffect(() => {
    if (!key || state.key !== key) return;
    if (state.loading || state.loadingOlder || state.error) return;
    writeCachedSessionPage(key, {
      messages: state.messages,
      beforeCursor: state.beforeCursor,
      hasMoreBefore: state.hasMoreBefore,
      userMessageOffset: state.userMessageOffset,
      completedTurnIds: state.completedTurnIds,
      forkBoundaryMessageCount: state.forkBoundaryMessageCount,
      version: state.version,
    });
  }, [key, state]);

  const markWorkspaceAccessRequestStale = useCallback(() => {
    accessRequestReadVersionRef.current += 1;
    if (!key) return;
    setState((prev) => prev.key === key
      ? { ...prev, workspaceAccessRequestLoading: true, workspaceAccessRequestError: false }
      : prev);
  }, [key]);

  const updateWorkspaceAccessRequest = useCallback((request: FolderAccessRequest | null) => {
    accessRequestReadVersionRef.current += 1;
    if (!key) return;
    setState((prev) => {
      if (prev.key !== key) return prev;
      if (
        request
        && prev.workspaceAccessRequest?.access_request_id === request.access_request_id
        && request.revision < prev.workspaceAccessRequest.revision
      ) return prev;
      return { ...prev, workspaceAccessRequest: request, workspaceAccessRequestLoading: false, workspaceAccessRequestError: false };
    });
  }, [key]);

  const refreshWorkspaceAccessRequest = useCallback(async (): Promise<FolderAccessRequest | null> => {
    if (!key) return null;
    const readVersion = ++accessRequestReadVersionRef.current;
    setState((prev) => prev.key === key
      ? { ...prev, workspaceAccessRequestLoading: true, workspaceAccessRequestError: false }
      : prev);
    try {
      const body = await fetchWebuiThread(getToken(), key, { limit: 1, direction: "latest" });
      const request = body?.workspace_access_request ?? null;
      if (readVersion === accessRequestReadVersionRef.current) {
        setState((prev) => prev.key === key
          ? { ...prev, workspaceAccessRequest: request, workspaceAccessRequestLoading: false, workspaceAccessRequestError: false }
          : prev);
      }
      return request;
    } catch (error) {
      if (readVersion === accessRequestReadVersionRef.current) {
        setState((prev) => prev.key === key
          ? { ...prev, workspaceAccessRequestLoading: false, workspaceAccessRequestError: true }
          : prev);
      }
      throw error;
    }
  }, [getToken, key]);

  const refreshWorkspacePermissions = useCallback(async (): Promise<WorkspacePermissionProjection | null> => {
    if (!key) return null;
    const readVersion = ++permissionReadVersionRef.current;
    const requestedKey = key;
    setState((prev) => prev.key === requestedKey
      ? { ...prev, workspacePermissionsLoading: true, workspacePermissionsError: false }
      : prev);
    try {
      const body = await fetchWebuiThread(getToken(), requestedKey, { limit: 1, direction: "latest" });
      const permissions = normalizeWorkspacePermissionProjection(body?.workspace_permissions);
      if (readVersion === permissionReadVersionRef.current) {
        setState((prev) => prev.key === requestedKey
          ? {
              ...prev,
              workspacePermissions: mergePermissionProjection(prev.workspacePermissions, permissions),
              workspacePermissionsLoading: false,
              workspacePermissionsError: false,
            }
          : prev);
      }
      return permissions;
    } catch (error) {
      if (readVersion === permissionReadVersionRef.current) {
        setState((prev) => prev.key === requestedKey
          ? { ...prev, workspacePermissionsLoading: false, workspacePermissionsError: true }
          : prev);
      }
      throw error;
    }
  }, [getToken, key]);

  const markWorkspacePermissionsStale = useCallback(() => {
    permissionReadVersionRef.current += 1;
    if (!key) return;
    setState((prev) => prev.key === key
      ? { ...prev, workspacePermissionsLoading: true, workspacePermissionsError: false }
      : prev);
  }, [key]);

  const updateWorkspacePermissions = useCallback((permissions: WorkspacePermissionProjection | null) => {
    if (!key) return;
    const normalized = normalizeWorkspacePermissionProjection(permissions);
    if (!normalized) {
      void refreshWorkspacePermissions();
      return;
    }
    const current = state.key === key ? state.workspacePermissions : null;
    if (current && current.runtime_id !== normalized.runtime_id) {
      void refreshWorkspacePermissions();
      return;
    }
    setState((prev) => prev.key === key
      ? {
          ...prev,
          workspacePermissions: mergePermissionProjection(prev.workspacePermissions, normalized),
          workspacePermissionsLoading: false,
          workspacePermissionsError: false,
        }
      : prev);
  }, [key, refreshWorkspacePermissions, state.key, state.workspacePermissions]);

  const loadOlder = useCallback(async () => {
    if (!key || loadingOlderRef.current) return;
    const requestKey = key;
    const requestLineage = state.key === requestKey ? state.lineage : 0;
    const beforeCursor = state.key === requestKey ? state.beforeCursor : null;
    if (!beforeCursor || !state.hasMoreBefore || requestLineage === 0) return;
    const matchesRequest = (candidate: typeof state) => (
      candidate.key === requestKey
      && candidate.lineage === requestLineage
      && candidate.beforeCursor === beforeCursor
    );
    loadingOlderRef.current = true;
    const controller = new AbortController();
    olderRequestAbortRef.current = controller;
    setState((prev) => matchesRequest(prev)
      ? { ...prev, loadingOlder: true, error: null }
      : prev);
    try {
      const body = await fetchWebuiThread(getToken(), requestKey, {
        limit: OLDER_HISTORY_PAGE_LIMIT,
        before: beforeCursor,
        signal: controller.signal,
      });
      setState((prev) => {
        if (!matchesRequest(prev)) return prev;
        if (!body?.messages?.length) {
          return {
            ...prev,
            loadingOlder: false,
            hasMoreBefore: false,
            beforeCursor: null,
          };
        }
        const older = persistedMessagesToUi(body.messages);
        const olderBoundary = typeof body.fork_boundary_message_count === "number"
          ? Math.max(0, Math.min(body.fork_boundary_message_count, older.length))
          : null;
        const shiftedBoundary = prev.forkBoundaryMessageCount === null
          ? null
          : prev.forkBoundaryMessageCount + older.length;
        const nextMessages = [...older, ...prev.messages];
        // An older page cannot change the authoritative latest-turn lifecycle
        // state or masquerade as a completed latest-page refresh.
        return {
          ...prev,
          messages: nextMessages,
          loadingOlder: false,
          error: null,
          forkBoundaryMessageCount: olderBoundary ?? shiftedBoundary,
          beforeCursor: body.page?.before_cursor ?? null,
          hasMoreBefore: body.page?.has_more_before === true,
          userMessageOffset: Math.max(0, body.page?.user_message_offset ?? 0),
        };
      });
    } catch (e) {
      if (isAbortError(e)) return;
      setState((prev) => matchesRequest(prev)
        ? {
            ...prev,
            loadingOlder: false,
            error: (e as Error).message,
          }
        : prev);
    } finally {
      if (olderRequestAbortRef.current === controller) {
        olderRequestAbortRef.current = null;
        loadingOlderRef.current = false;
      }
    }
  }, [
    key,
    state.beforeCursor,
    state.hasMoreBefore,
    state.key,
    state.lineage,
    getToken,
  ]);

  /**
   * Load the whole-session turn index, newest page last.
   *
   * The navigator needs every turn, not just the loaded window, so this pages
   * the cheap user-row index until the server reports no older page.
   */
  const loadTurnIndex = useCallback(async (): Promise<SessionTurnSummary[]> => {
    if (!key) return [];
    const collected: SessionTurnSummary[] = [];
    let before: string | null = null;
    let total = 0;
    for (let page = 0; page < 20; page += 1) {
      const payload = await fetchSessionTurns(getToken(), key, {
        limit: TURN_INDEX_PAGE_LIMIT,
        before,
      });
      if (!payload) break;
      total = payload.total_turn_count ?? total;
      collected.unshift(...(Array.isArray(payload.turns) ? payload.turns : []));
      if (payload.page?.has_more_before !== true) break;
      const next = payload.page?.before_cursor ?? null;
      if (!next || next === before) break;
      before = next;
    }
    setTotalTurnCount(total || collected.length);
    setTurnIndex(collected);
    return collected;
  }, [getToken, key]);

  /**
   * Replace the rendered window with the page that ends at one turn.
   *
   * This is the navigator's jump. It intentionally starts a fresh lineage: the
   * target window may be far from the current one, so merging the two would
   * invent continuity that does not exist.
   */
  const loadTurnAt = useCallback(async (cursor: string): Promise<void> => {
    if (!key || !cursor) return;
    olderRequestAbortRef.current?.abort();
    const controller = new AbortController();
    olderRequestAbortRef.current = controller;
    loadingOlderRef.current = true;
    setState((prev) => (prev.key === key
      ? { ...prev, loading: true, error: null }
      : prev));
    try {
      const body = await fetchWebuiThread(getToken(), key, {
        limit: INITIAL_HISTORY_PAGE_LIMIT,
        direction: "latest",
        before: cursor,
        signal: controller.signal,
      });
      if (!body) return;
      const ui = persistedMessagesToUi(body.messages ?? [])
        .filter((message) => message.kind !== "delegated_child");
      historyVersionRef.current += 1;
      const responseVersion = historyVersionRef.current;
      const pending = hasPendingToolCallsFromThread(body, ui);
      setState((prev) => (prev.key === key
        ? {
            ...prev,
            messages: ui,
            loading: false,
            loadingOlder: false,
            error: null,
            beforeCursor: body.page?.before_cursor ?? null,
            hasMoreBefore: body.page?.has_more_before === true,
            userMessageOffset: Math.max(0, body.page?.user_message_offset ?? 0),
            version: responseVersion,
            continuity: "initial",
            lineage: responseVersion,
            hasPendingToolCalls: pending,
            completedTurnIds: completedTurnIdsFromThread(body),
            forkBoundaryMessageCount: typeof body.fork_boundary_message_count === "number"
              ? Math.max(0, Math.min(body.fork_boundary_message_count, ui.length))
              : null,
            activeTurnId: typeof body.active_turn_id === "string"
              ? body.active_turn_id
              : null,
          }
        : prev));
    } catch (e) {
      if (isAbortError(e)) return;
      setState((prev) => (prev.key === key
        ? { ...prev, loading: false, error: (e as Error).message }
        : prev));
    } finally {
      if (olderRequestAbortRef.current === controller) {
        olderRequestAbortRef.current = null;
        loadingOlderRef.current = false;
      }
    }
  }, [getToken, key]);

  if (!key) {
    return {
      messages: EMPTY_MESSAGES,
      loading: false,
      loadingOlder: false,
      error: null,
      refresh,
      loadOlder,
      hasMoreBefore: false,
      userMessageOffset: 0,
      version: 0,
      forkBoundaryMessageCount: null,
      hasPendingToolCalls: false,
      completedTurnIds: [],
      continuity: "initial",
      lineage: 0,
      activeTurnId: null,
      activitySnapshotAvailable: false,
      stopTarget: null,
      stopTargetSnapshotAvailable: false,
      delegatedChildren: [],
      delegatedChildrenLoading: false,
      delegatedChildrenError: false,
      workspaceAccessRequest: null,
      workspaceAccessRequestLoading: false,
      workspaceAccessRequestError: false,
      markWorkspaceAccessRequestStale,
      updateWorkspaceAccessRequest,
      refreshWorkspaceAccessRequest,
      workspacePermissions: null,
      workspacePermissionsLoading: false,
      workspacePermissionsError: false,
      markWorkspacePermissionsStale,
      updateWorkspacePermissions,
      refreshWorkspacePermissions,
      turnIndex,
      totalTurnCount,
      loadTurnIndex,
      loadTurnAt,
    };
  }

  // Even before the effect above commits its loading state, never surface the
  // previous session's payload for a brand-new key.
  if (state.key !== key) {
    return {
      messages: EMPTY_MESSAGES,
      loading: true,
      loadingOlder: false,
      error: null,
      refresh,
      loadOlder,
      hasMoreBefore: false,
      userMessageOffset: 0,
      version: 0,
      forkBoundaryMessageCount: null,
      hasPendingToolCalls: false,
      completedTurnIds: [],
      continuity: "initial",
      lineage: 0,
      activeTurnId: null,
      activitySnapshotAvailable: false,
      stopTarget: null,
      stopTargetSnapshotAvailable: false,
      delegatedChildren: [],
      delegatedChildrenLoading: true,
      delegatedChildrenError: false,
      workspaceAccessRequest: null,
      workspaceAccessRequestLoading: true,
      workspaceAccessRequestError: false,
      markWorkspaceAccessRequestStale,
      updateWorkspaceAccessRequest,
      refreshWorkspaceAccessRequest,
      workspacePermissions: null,
      workspacePermissionsLoading: true,
      workspacePermissionsError: false,
      markWorkspacePermissionsStale,
      updateWorkspacePermissions,
      refreshWorkspacePermissions,
      turnIndex,
      totalTurnCount,
        loadTurnIndex,
      loadTurnAt,
    };
  }

  return {
    messages: state.messages,
    loading: state.loading,
    loadingOlder: state.loadingOlder,
    error: state.error,
    refresh,
    loadOlder,
    hasMoreBefore: state.hasMoreBefore,
    userMessageOffset: state.userMessageOffset,
    version: state.version,
    forkBoundaryMessageCount: state.forkBoundaryMessageCount,
    hasPendingToolCalls: state.hasPendingToolCalls,
    completedTurnIds: state.completedTurnIds,
    continuity: state.continuity,
    lineage: state.lineage,
    activeTurnId: state.activeTurnId,
    activitySnapshotAvailable: state.activitySnapshotAvailable,
    stopTarget: state.stopTarget,
    stopTargetSnapshotAvailable: state.stopTargetSnapshotAvailable,
    delegatedChildren: state.delegatedChildren,
    delegatedChildrenLoading: state.delegatedChildrenLoading,
    delegatedChildrenError: state.delegatedChildrenError,
    workspaceAccessRequest: state.workspaceAccessRequest,
    workspaceAccessRequestLoading: state.workspaceAccessRequestLoading,
    workspaceAccessRequestError: state.workspaceAccessRequestError,
    markWorkspaceAccessRequestStale,
    updateWorkspaceAccessRequest,
    refreshWorkspaceAccessRequest,
    workspacePermissions: state.workspacePermissions,
    workspacePermissionsLoading: state.workspacePermissionsLoading,
    workspacePermissionsError: state.workspacePermissionsError,
    markWorkspacePermissionsStale,
    updateWorkspacePermissions,
    refreshWorkspacePermissions,
    turnIndex,
    totalTurnCount,
    loadTurnIndex,
    loadTurnAt,
  };
}

/** Produce a compact display title for a session. */
export function sessionTitle(
  session: ChatSummary,
  firstUserMessage?: string,
): string {
  return deriveTitle(
    session.title || firstUserMessage || session.preview,
    i18n.t("chat.newChat"),
  );
}
