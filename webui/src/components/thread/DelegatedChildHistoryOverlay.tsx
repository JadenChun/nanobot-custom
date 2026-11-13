import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { ThreadMessages } from "@/components/thread/ThreadMessages";
import { Button } from "@/components/ui/button";
import { fetchWebuiThread } from "@/lib/api";
import {
  parseDelegatedChildPayload,
  parseDelegatedChildSummaryPayload,
  parseSourceOrigin,
  sameSourceOrigin,
} from "@/lib/delegated-child-projection";
import { projectWebuiThreadMessages } from "@/lib/thread-display-compat";
import { parseRiskyApprovalPrompt } from "@/lib/risky-approval";
import type {
  DelegatedChildPayload,
  DelegatedChildReference,
  DelegatedChildSummaryPayload,
  SourceOrigin,
  UIMessage,
  WebuiThreadPersistedPayload,
} from "@/lib/types";

const CHILD_HISTORY_PAGE_LIMIT = 120;
const MAX_APPROVAL_HISTORY_PAGES = 100;
const DISPLAY_UNIT_SELECTOR = "[data-thread-display-unit]";

export interface DelegatedChildHistoryOverlayProps {
  child: DelegatedChildReference;
  rootKey: string;
  rootRows: DelegatedChildReference[];
  expectedRoot?: SourceOrigin | null;
  provenanceComplete?: boolean;
  token: string;
  onClose: () => void;
}

interface ChildHistoryPage {
  messages: UIMessage[];
  historyKind: "canonical" | "summary";
  beforeCursor: string | null;
  hasMoreBefore: boolean;
}

interface PendingScrollAnchor {
  key: string | null;
  offsetTop: number | null;
  scrollHeight: number;
  scrollTop: number;
}

export function delegatedChildSource(child: DelegatedChildPayload): SourceOrigin | null {
  return parseSourceOrigin({
    store_identity: child.child_store_identity,
    session_key: child.child_session_key,
    incarnation: child.child_incarnation,
    generation: child.child_generation,
  });
}

export function delegatedChildHasValidAncestry(
  childValue: DelegatedChildPayload,
  rows: DelegatedChildPayload[],
  rootKey: string,
  expectedRoot?: SourceOrigin | null,
): boolean {
  const child = parseDelegatedChildPayload(childValue);
  if (!child || child.root_key !== rootKey) return false;
  const root = parseSourceOrigin({
    store_identity: child.root_store_identity,
    session_key: child.root_key,
    incarnation: child.root_incarnation,
    generation: child.root_generation,
  });
  if (!root || root.session_key !== rootKey) return false;
  if (expectedRoot && !sameSourceOrigin(root, expectedRoot)) return false;

  const known = new Map<string, DelegatedChildPayload>();
  for (const value of rows) {
    const row = parseDelegatedChildPayload(value);
    if (row?.root_key === rootKey) known.set(row.child_session_key, row);
  }
  let current: DelegatedChildPayload = child;
  const visited = new Set<string>();
  while (true) {
    if (visited.has(current.child_session_key)) return false;
    visited.add(current.child_session_key);
    const parentSessionKey = current.parent_session_key;
    const parent = parseSourceOrigin({
      store_identity: current.parent_store_identity,
      session_key: parentSessionKey,
      incarnation: current.parent_incarnation,
      generation: current.parent_generation,
    });
    const currentRoot = parseSourceOrigin({
      store_identity: current.root_store_identity,
      session_key: current.root_key,
      incarnation: current.root_incarnation,
      generation: current.root_generation,
    });
    if (!parentSessionKey || !parent || !currentRoot || !sameSourceOrigin(root, currentRoot)) return false;
    if (parentSessionKey === rootKey) {
      return sameSourceOrigin(parent, root);
    }
    const parentRow = known.get(parentSessionKey);
    const parentRowSource = parentRow ? delegatedChildSource(parentRow) : null;
    if (!parentRow || !parentRowSource || !sameSourceOrigin(parent, parentRowSource)) return false;
    current = parentRow;
  }
}

export function delegatedChildSummaryIsAuthorized(
  childValue: DelegatedChildSummaryPayload,
  rootKey: string,
): boolean {
  const child = parseDelegatedChildSummaryPayload(childValue);
  return child !== null
    && child.history_kind === "summary"
    && child.root_key === rootKey
    && child.child_store_identity === null
    && child.child_incarnation === null
    && child.child_generation === null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseHistoryMessage(value: unknown): UIMessage | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.id !== "string"
    || value.id.length === 0
    || value.id.length > 1024
    || !(value.role === "user" || value.role === "assistant" || value.role === "tool" || value.role === "system")
    || typeof value.content !== "string"
    || typeof value.createdAt !== "number"
    || !Number.isFinite(value.createdAt)
    || (value.timestampAvailable !== undefined
      && typeof value.timestampAvailable !== "boolean")
  ) return null;
  return value as unknown as UIMessage;
}

function stableValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableValue).join(",")}]`;
  if (isRecord(value)) {
    const entries = Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableValue(value[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value) ?? "undefined";
}

function parseHistoryPage(
  payload: WebuiThreadPersistedPayload | null,
  child: DelegatedChildReference,
  rootKey: string,
): ChildHistoryPage {
  const isSummaryChild = child.history_kind === "summary";
  const childSource = isSummaryChild ? null : delegatedChildSource(child);
  const page = payload?.page;
  const payloadSource = parseSourceOrigin(payload?.source);
  const summarySourceIsExplicitlyUnavailable = payload?.history_kind === "summary"
    && payload.source === null;
  if (
    !payload
    || payload.read_only !== true
    || (payload.history_kind !== "canonical" && payload.history_kind !== "summary")
    || payload.parent_session_key !== child.parent_session_key
    || payload.root_key !== rootKey
    || payload.sessionKey !== undefined && payload.sessionKey !== child.child_session_key
    || (payload.history_kind === "canonical"
      && (!childSource || !payloadSource || !sameSourceOrigin(payloadSource, childSource)))
    || (payload.history_kind === "summary"
      && (!summarySourceIsExplicitlyUnavailable || !isSummaryChild))
    || (isSummaryChild && payload.history_kind !== "summary")
    || !page
    || typeof page.has_more_before !== "boolean"
    || !(page.before_cursor === null || typeof page.before_cursor === "string")
    || typeof page.loaded_message_count !== "number"
    || !Number.isSafeInteger(page.loaded_message_count)
    || page.loaded_message_count < 0
    || !Array.isArray(payload.messages)
  ) {
    throw new Error("This delegated history could not be verified. Return to the parent chat and refresh.");
  }
  const messages = payload.messages.map(parseHistoryMessage);
  if (
    messages.some((message) => message === null)
    || messages.length !== page.loaded_message_count
    || (page.has_more_before && (!page.before_cursor || page.before_cursor.length > 2048))
  ) {
    throw new Error("This delegated history page is incomplete. Return to the parent chat and refresh.");
  }
  return {
    messages: messages.filter((message): message is UIMessage => message !== null),
    historyKind: payload.history_kind,
    beforeCursor: page.before_cursor,
    hasMoreBefore: page.has_more_before,
  };
}

export async function findDelegatedChildApprovalPrompt(
  token: string,
  child: DelegatedChildPayload,
  rootKey: string,
  signal: AbortSignal,
): Promise<{ summary: string; approvalId: string } | null> {
  const approvalId = child.pending_approval_id;
  if (
    child.state !== "waiting_approval"
    || !approvalId
    || !child.blocked_run_id
    || child.blocked_run_id !== child.child_run_id
  ) return null;

  const usedCursors = new Set<string>();
  let before: string | null = null;
  for (let pageIndex = 0; pageIndex < MAX_APPROVAL_HISTORY_PAGES; pageIndex += 1) {
    const payload = await fetchWebuiThread(token, child.child_session_key, {
      limit: CHILD_HISTORY_PAGE_LIMIT,
      direction: "latest",
      ...(before ? { before } : {}),
      signal,
    });
    const page = parseHistoryPage(payload, child, rootKey);
    if (page.historyKind !== "canonical") return null;
    for (let index = page.messages.length - 1; index >= 0; index -= 1) {
      const message = page.messages[index]!;
      if (message.role !== "assistant" || message.kind === "trace") continue;
      const prompt = parseRiskyApprovalPrompt(message.content);
      if (prompt?.approvalId === approvalId) return { summary: prompt.summary, approvalId };
    }
    if (!page.hasMoreBefore) return null;
    if (!page.beforeCursor || usedCursors.has(page.beforeCursor)) {
      throw new Error("Delegated approval history pagination could not be verified.");
    }
    usedCursors.add(page.beforeCursor);
    before = page.beforeCursor;
  }
  throw new Error("Delegated approval history is too large to verify completely.");
}

function mergeOlderMessages(older: UIMessage[], current: UIMessage[]): UIMessage[] {
  const currentById = new Map(current.map((message) => [message.id, message]));
  for (const message of older) {
    const existing = currentById.get(message.id);
    if (existing && stableValue(existing) !== stableValue(message)) {
      throw new Error("Delegated history pages disagree. Return to the parent chat and refresh.");
    }
  }
  const currentIds = new Set(currentById.keys());
  return [
    ...older.filter((message) => !currentIds.has(message.id)),
    ...current,
  ];
}

function captureScrollAnchor(scroller: HTMLElement, content: HTMLElement | null): PendingScrollAnchor {
  const bounds = scroller.getBoundingClientRect();
  const units = Array.from(content?.querySelectorAll<HTMLElement>(DISPLAY_UNIT_SELECTOR) ?? []);
  const anchor = units.find((unit) => {
    const rect = unit.getBoundingClientRect();
    return rect.bottom > bounds.top && rect.top < bounds.bottom;
  });
  return {
    key: anchor?.dataset.threadDisplayUnit ?? null,
    offsetTop: anchor ? anchor.getBoundingClientRect().top - bounds.top : null,
    scrollHeight: scroller.scrollHeight,
    scrollTop: scroller.scrollTop,
  };
}

export function DelegatedChildHistoryOverlay({
  child,
  rootKey,
  rootRows,
  expectedRoot = null,
  provenanceComplete = true,
  token,
  onClose,
}: DelegatedChildHistoryOverlayProps) {
  const { t } = useTranslation();
  const backButtonRef = useRef<HTMLButtonElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef(0);
  const olderRequestRef = useRef(false);
  const olderRequestAbortRef = useRef<AbortController | null>(null);
  const usedCursorsRef = useRef(new Set<string>());
  const pendingScrollAnchorRef = useRef<PendingScrollAnchor | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [historyKind, setHistoryKind] = useState<"canonical" | "summary" | null>(null);
  const [beforeCursor, setBeforeCursor] = useState<string | null>(null);
  const [hasMoreBefore, setHasMoreBefore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestPage = useCallback(async (before: string | null, signal: AbortSignal) => {
    const payload = await fetchWebuiThread(token, child.child_session_key, {
      limit: CHILD_HISTORY_PAGE_LIMIT,
      direction: "latest",
      ...(before ? { before } : {}),
      signal,
    });
    return parseHistoryPage(payload, child, rootKey);
  }, [child, rootKey, token]);

  useEffect(() => {
    const controller = new AbortController();
    const requestId = ++requestRef.current;
    setMessages([]);
    setHistoryKind(null);
    setBeforeCursor(null);
    setHasMoreBefore(false);
    setLoading(true);
    setLoadingOlder(false);
    setError(null);
    usedCursorsRef.current = new Set();
    const authorized = child.history_kind === "summary"
      ? delegatedChildSummaryIsAuthorized(child, rootKey)
      : delegatedChildHasValidAncestry(
          child,
          rootRows.flatMap((row) => {
            const canonical = parseDelegatedChildPayload(row);
            return canonical ? [canonical] : [];
          }),
          rootKey,
          expectedRoot,
        );
    if (!provenanceComplete || !authorized) {
      setLoading(false);
      setMessages([]);
      setError("This delegated history no longer matches a complete current parent source. Return to the parent chat and refresh.");
      return () => controller.abort();
    }
    void requestPage(null, controller.signal).then((page) => {
      if (controller.signal.aborted || requestRef.current !== requestId) return;
      setMessages(page.messages);
      setHistoryKind(page.historyKind);
      setBeforeCursor(page.beforeCursor);
      setHasMoreBefore(page.hasMoreBefore);
      setLoading(false);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted || requestRef.current !== requestId) return;
      setLoading(false);
      setError(reason instanceof Error ? reason.message : "Delegated history could not be loaded.");
    });
    return () => {
      requestRef.current += 1;
      controller.abort();
      olderRequestAbortRef.current?.abort();
      olderRequestAbortRef.current = null;
      olderRequestRef.current = false;
    };
  }, [child, expectedRoot, provenanceComplete, requestPage, rootKey]);

  useLayoutEffect(() => {
    backButtonRef.current?.focus();
  }, []);

  useLayoutEffect(() => {
    const anchor = pendingScrollAnchorRef.current;
    const scroller = scrollerRef.current;
    if (!anchor || !scroller) return;
    pendingScrollAnchorRef.current = null;
    const target = anchor.key && contentRef.current
      ? Array.from(contentRef.current.querySelectorAll<HTMLElement>(DISPLAY_UNIT_SELECTOR))
          .find((unit) => unit.dataset.threadDisplayUnit === anchor.key)
      : null;
    let nextTop = anchor.scrollTop;
    if (target && anchor.offsetTop !== null) {
      nextTop += target.getBoundingClientRect().top - scroller.getBoundingClientRect().top - anchor.offsetTop;
    } else {
      nextTop += scroller.scrollHeight - anchor.scrollHeight;
    }
    scroller.scrollTop = Math.min(
      Math.max(0, scroller.scrollHeight - scroller.clientHeight),
      Math.max(0, nextTop),
    );
  }, [messages]);

  const loadOlder = useCallback(async () => {
    if (loadingOlder || !hasMoreBefore || !beforeCursor || !scrollerRef.current) return;
    if (usedCursorsRef.current.has(beforeCursor)) {
      setError("Delegated history pagination repeated a cursor. Return to the parent chat and refresh.");
      setHasMoreBefore(false);
      return;
    }
    usedCursorsRef.current.add(beforeCursor);
    pendingScrollAnchorRef.current = captureScrollAnchor(scrollerRef.current, contentRef.current);
    const controller = new AbortController();
    const requestId = requestRef.current;
    olderRequestAbortRef.current = controller;
    olderRequestRef.current = true;
    setLoadingOlder(true);
    setError(null);
    try {
      const page = await requestPage(beforeCursor, controller.signal);
      if (requestRef.current !== requestId) return;
      if (page.historyKind !== historyKind) {
        throw new Error("Delegated history changed while paging. Return to the parent chat and refresh.");
      }
      setMessages((current) => mergeOlderMessages(page.messages, current));
      setBeforeCursor(page.beforeCursor);
      setHasMoreBefore(page.hasMoreBefore);
    } catch (reason) {
      if (requestRef.current === requestId) {
        pendingScrollAnchorRef.current = null;
        setError(reason instanceof Error ? reason.message : "Earlier delegated history could not be loaded.");
      }
    } finally {
      olderRequestRef.current = false;
      if (olderRequestAbortRef.current === controller) olderRequestAbortRef.current = null;
      if (requestRef.current === requestId) setLoadingOlder(false);
    }
  }, [beforeCursor, hasMoreBefore, historyKind, loadingOlder, requestPage]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const taskLabel = child.task_label?.trim() || child.profile || "Delegated task";
  const displayMessages = projectWebuiThreadMessages(messages);
  return (
    <section
      role="dialog"
      aria-modal="true"
      aria-labelledby="delegated-child-history-title"
      className="flex h-full min-h-0 flex-col bg-background"
      data-testid="delegated-child-history"
    >
      <header className="flex shrink-0 items-center gap-3 border-b border-border/70 px-3 py-2.5 sm:px-4">
        <Button
          ref={backButtonRef}
          type="button"
          variant="outline"
          size="sm"
          onClick={onClose}
          aria-label={t("thread.delegatedChild.backToParent", { defaultValue: "Back to parent chat" })}
        >
          {t("thread.delegatedChild.back", { defaultValue: "Back" })}
        </Button>
        <div className="min-w-0 flex-1">
          <h2 id="delegated-child-history-title" className="truncate text-sm font-semibold">
            {taskLabel}
          </h2>
          <p className="text-xs text-muted-foreground">
            {historyKind === "summary"
              ? t("thread.delegatedChild.summaryHistory", { defaultValue: "Summary-only history · read only" })
              : t("thread.delegatedChild.canonicalHistory", { defaultValue: "Canonical activity · read only" })}
          </p>
        </div>
      </header>
      {historyKind === "summary" ? (
        <p className="shrink-0 border-b border-border/60 bg-muted/20 px-4 py-2 text-xs text-muted-foreground">
          {t("thread.delegatedChild.summaryOnlyNotice", {
            defaultValue: "This source has a summary record. Full canonical activity is unavailable.",
          })}
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="shrink-0 border-b border-destructive/30 bg-destructive/5 px-4 py-2 text-xs text-destructive">
          {error}
        </p>
      ) : null}
      <div
        ref={scrollerRef}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-5 pt-3 sm:px-4"
        onScroll={(event) => {
          if (event.currentTarget.scrollTop < 80 && hasMoreBefore && !loadingOlder) {
            void loadOlder();
          }
        }}
      >
        {hasMoreBefore ? (
          <div className="mx-auto mb-3 flex w-full max-w-[49.5rem] justify-center">
            <Button type="button" variant="outline" size="sm" disabled={loadingOlder} onClick={() => void loadOlder()}>
              {loadingOlder
                ? t("thread.delegatedChild.loadingEarlier", { defaultValue: "Loading earlier activity…" })
                : t("thread.delegatedChild.loadEarlier", { defaultValue: "Load earlier activity" })}
            </Button>
          </div>
        ) : null}
        {loading ? (
          <p role="status" className="py-10 text-center text-sm text-muted-foreground">
            {t("thread.delegatedChild.loadingHistory", { defaultValue: "Loading delegated history…" })}
          </p>
        ) : displayMessages.length > 0 ? (
          <div ref={contentRef} className="mx-auto w-full max-w-[49.5rem]">
            <ThreadMessages messages={displayMessages} isStreaming={false} />
          </div>
        ) : !error ? (
          <p className="py-10 text-center text-sm text-muted-foreground">
            {t("thread.delegatedChild.emptyHistory", { defaultValue: "No canonical activity is available." })}
          </p>
        ) : null}
      </div>
      <p className="sr-only" aria-live="polite">
        {olderRequestRef.current ? t("thread.delegatedChild.loadingEarlier", { defaultValue: "Loading earlier activity…" }) : ""}
      </p>
    </section>
  );
}
