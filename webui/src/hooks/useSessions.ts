import { useCallback, useEffect, useRef, useState } from "react";

import { useClient } from "@/providers/ClientProvider";
import i18n from "@/i18n";
import {
  ApiError,
  deleteSession as apiDeleteSession,
  fetchSessionAutomations,
  fetchWebuiThread,
  listSessions,
  normalizeWorkspacePermissionProjection,
} from "@/lib/api";
import { hasPendingAgentActivity } from "@/lib/activity-timeline";
import { deriveTitle } from "@/lib/format";
import type {
  ChatSummary,
  FolderAccessRequest,
  SessionAutomationJob,
  SessionDeleteResult,
  UIMessage,
  WorkspacePermissionProjection,
  WorkspaceScopePayload,
} from "@/lib/types";

const EMPTY_MESSAGES: UIMessage[] = [];
const INITIAL_HISTORY_PAGE_LIMIT = 80;
const OLDER_HISTORY_PAGE_LIMIT = 120;
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
} {
  const { getToken } = useClient();
  const loadingOlderRef = useRef(false);
  const olderRequestAbortRef = useRef<AbortController | null>(null);
  const historyVersionRef = useRef(0);
  const accessRequestReadVersionRef = useRef(0);
  const permissionReadVersionRef = useRef(0);
  const [refreshSeq, setRefreshSeq] = useState(0);
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
    // Mark the new key as loading immediately so callers never see stale
    // messages from the previous session during the render right after a switch.
    setState((prev) => prev.key === key
      ? {
          ...prev,
          loading: true,
          loadingOlder: false,
          error: null,
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
        historyVersionRef.current += 1;
        const responseVersion = historyVersionRef.current;
        const completedTurnIds = completedTurnIdsFromThread(body);
        const permissionProjection = normalizeWorkspacePermissionProjection(body?.workspace_permissions);
        const ui = persistedMessagesToUi(body?.messages ?? []);
        const hasPending = hasPendingToolCallsFromThread(body, ui);
        const forkBoundary = typeof body?.fork_boundary_message_count === "number"
          ? Math.max(0, Math.min(body.fork_boundary_message_count, ui.length))
          : null;
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
              : Math.max(0, body?.page?.user_message_offset ?? 0),
            version: responseVersion,
            continuity: merged.continuity,
            lineage: merged.continuity === "overlap"
              ? prev.lineage
              : responseVersion,
            activeTurnId: typeof body?.active_turn_id === "string"
              ? body.active_turn_id
              : null,
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
      } catch (e) {
        if (cancelled || isAbortError(e)) return;
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
