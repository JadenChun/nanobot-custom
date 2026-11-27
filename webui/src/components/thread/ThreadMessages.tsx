import { memo, type RefObject, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useTranslation } from "react-i18next";
import { MessageBubble } from "@/components/MessageBubble";
import { AgentActivityCluster } from "@/components/thread/AgentActivityCluster";
import { AssistantSelectionAction } from "@/components/thread/AssistantSelectionAction";
import { projectActivityTimeline, type TurnUnit } from "@/lib/activity-timeline";
import type {
  CliAppInfo,
  DelegatedChildReference,
  McpPresetInfo,
  RetryStatus,
  SlashCommand,
  UIMessage,
} from "@/lib/types";

interface ThreadMessagesProps {
  messages: UIMessage[];
  temporary?: boolean;
  /** When true, agent turn still in flight — keeps activity timeline expanded. */
  isStreaming?: boolean;
  /** Reveal the latest ordinary assistant body after a conversation handoff. */
  revealLatestHistoryMessage?: boolean;
  activeTurnId?: string | null;
  /** Optimistic or canonical active-turn start, in unix seconds. */
  runStartedAt?: number | null;
  retryStatus?: RetryStatus | null;
  hiddenUserMessageCount?: number;
  cliApps?: CliAppInfo[];
  mcpPresets?: McpPresetInfo[];
  slashCommands?: SlashCommand[];
  forkBoundaryMessageCount?: number | null;
  onOpenFilePreview?: (path: string) => void;
  onForkFromMessage?: (beforeUserIndex: number) => void;
  onQuoteSelection?: (text: string) => void;
  onOpenChildHistory?: (child: DelegatedChildReference, trigger: HTMLButtonElement) => void;
  delegatedStatusAvailable?: boolean;
  /** Prompt id (stable ``promptId`` or row ``id``) currently highlighted after a reveal. */
  highlightedPromptId?: string | null;
  childJump?: { childId: string } | null;
  onChildJumpTarget?: (row: HTMLElement) => void;
  /** Scrollport ref for virtualized infinite scrolling from the bottom. */
  scrollRef?: RefObject<HTMLDivElement | null>;
  /** Enable TanStack Virtual end-anchored virtualization (disabled in layoutless DOMs). */
  virtualize?: boolean;
  /** Register an index-based jump handler for virtualized off-screen prompts. */
  onRegisterScrollToPrompt?: (handler: ((promptId: string) => boolean) | null) => void;
}

export type DisplayUnit = TurnUnit;

export function buildDisplayUnits(messages: UIMessage[]): DisplayUnit[] {
  return projectActivityTimeline(messages);
}

export function assistantForkFlags(units: DisplayUnit[]): boolean[] {
  const flags = new Array<boolean>(units.length).fill(true);
  let hasLaterUnitBeforeUser = false;
  for (let i = units.length - 1; i >= 0; i -= 1) {
    const unit = units[i];
    if (unit.type === "message" && unit.message.role === "user") {
      hasLaterUnitBeforeUser = false;
      continue;
    }
    if (
      unit.type === "message"
      && (unit.message.kind === "compaction" || unit.message.kind === "run_stopped")
    ) {
      // Lifecycle notices must neither expose nor displace the answer-level fork action.
      flags[i] = false;
      continue;
    }
    if (unit.type === "message" && unit.message.role === "assistant") {
      flags[i] = !hasLaterUnitBeforeUser;
    }
    hasLaterUnitBeforeUser = true;
  }
  return flags;
}

export const ThreadMessages = memo(function ThreadMessages({
  messages,
  temporary = false,
  isStreaming = false,
  revealLatestHistoryMessage = false,
  activeTurnId = null,
  runStartedAt = null,
  retryStatus = null,
  hiddenUserMessageCount = 0,
  cliApps = [],
  mcpPresets = [],
  slashCommands = [],
  forkBoundaryMessageCount = null,
  onOpenFilePreview,
  onForkFromMessage,
  onQuoteSelection,
  onOpenChildHistory,
  delegatedStatusAvailable = true,
  highlightedPromptId = null,
  childJump = null,
  onChildJumpTarget,
  scrollRef,
  virtualize = false,
  onRegisterScrollToPrompt,
}: ThreadMessagesProps) {
  const { t } = useTranslation();
  const messageListRef = useRef<HTMLDivElement>(null);
  const units = useMemo(
    () => buildDisplayUnits(messages),
    [messages],
  );
  const forkBoundaryAfterUnitIndex = useMemo(
    () => unitIndexAfterMessageCount(units, forkBoundaryMessageCount),
    [forkBoundaryMessageCount, units],
  );
  const forkFlags = useMemo(() => assistantForkFlags(units), [units]);
  const liveActivityClusterIndices = useMemo(
    () => isStreaming
      ? currentActivityClusterIndices(units, activeTurnId)
      : new Set<number>(),
    [activeTurnId, isStreaming, units],
  );
  const currentTurnStartIndex = isStreaming
    ? activeTurnStartIndex(units, activeTurnId)
    : units.length;
  const promptUnit = units[currentTurnStartIndex];
  const activePrompt = promptUnit?.type === "message"
    && (activeTurnId === null || promptUnit.message.turnId === activeTurnId)
    ? promptUnit.message
    : null;
  // The elapsed clock must measure model work, not the client's send path.
  // Before the server reports runStartedAt, show pulsing "Working" without a
  // ticking timer so the clock does not start at send time and reset once the
  // gateway begins the turn.
  const promptStartedAtMs = activePrompt
    && activePrompt.timestampAvailable !== false
    && Number.isFinite(activePrompt.createdAt)
    ? activePrompt.createdAt
    : undefined;
  const pendingStartedAtMs = runStartedAt != null
    ? runStartedAt * 1000
    : undefined;
  const activeStartedAtMs = runStartedAt != null
    ? runStartedAt * 1000
    : promptStartedAtMs;
  const showPendingActivity = isStreaming
    && liveActivityClusterIndices.size === 0
    && (activePrompt !== null || activeStartedAtMs !== undefined);
  const lastUnit = units[units.length - 1];
  const pendingActivityMarginTop =
    units.length === 0
      ? undefined
      : lastUnit?.type === "message" && lastUnit.message.role === "assistant"
        ? "mt-2"
        : "mt-5";
  const unitKeys = useMemo(() => unitKeysForDisplay(units), [units]);
  let revealMessageId: string | null = null;
  if (revealLatestHistoryMessage && !isStreaming) {
    for (let index = units.length - 1; index >= 0; index -= 1) {
      const unit = units[index];
      if (unit.type !== "message") continue;
      const message = unit.message;
      if (
        message.role !== "assistant"
        || (message.kind !== undefined && message.kind !== "message")
        || message.compactReply !== undefined
        || message.sessionMessage !== undefined
        || message.activityKind === "model"
      ) continue;
      revealMessageId = message.id;
      break;
    }
  }
  const forkIndices = useMemo(() => {
    const result = new Array<number | undefined>(units.length);
    let nextUserIndex = hiddenUserMessageCount;
    for (let index = 0; index < units.length; index += 1) {
      const unit = units[index];
      if (unit.type === "message" && unit.message.role === "assistant" && forkFlags[index]) {
        result[index] = nextUserIndex;
      }
      if (
        unit.type === "message"
        && unit.message.role === "user"
        && unit.message.deliveryStatus !== "failed"
      ) {
        nextUserIndex += 1;
      }
    }
    return result;
  }, [forkFlags, hiddenUserMessageCount, units]);

  const shouldVirtualize = Boolean(virtualize && scrollRef && units.length > 0);

  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? units.length : 0,
    getScrollElement: () => scrollRef?.current ?? null,
    estimateSize: (index) =>
      estimateDisplayUnitHeight(units[index], liveActivityClusterIndices.has(index)),
    getItemKey: (index) => unitKeys[index] ?? index,
    measureElement: (element, entry, instance) => {
      const index = instance.indexFromElement(element);
      const key = instance.options.getItemKey(index);
      const cachedSize = instance.itemSizeCache.get(key);
      // When the thread container is hidden (e.g. display: none while Settings is open),
      // ResizeObserver reports 0px. Preserve cached/estimated heights instead of
      // overwriting itemSizeCache with 0 while a session streams in the background.
      const scrollEl = instance.scrollElement;
      if (scrollEl && scrollEl.clientHeight === 0 && scrollEl.clientWidth === 0) {
        return cachedSize ?? instance.options.estimateSize(index);
      }
      if (entry?.borderBoxSize?.[0]) {
        const size = Math.round(entry.borderBoxSize[0].blockSize);
        if (size > 0) return size;
        if (cachedSize !== undefined) return cachedSize;
      }
      const measured = (element as HTMLElement).offsetHeight;
      if (measured > 0) return measured;
      return cachedSize ?? instance.options.estimateSize(index);
    },
    overscan: 6,
    anchorTo: "end",
    followOnAppend: false,
    scrollEndThreshold: 48,
    enabled: shouldVirtualize,
    directDomUpdates: true,
    directDomUpdatesMode: "transform",
  });

  const measureVirtualRow = useCallback(
    (node: HTMLDivElement | null) => {
      virtualizer.measureElement(node);
      if (!node || !node.isConnected || !virtualizer.isScrolling) return;
      const index = virtualizer.indexFromElement(node);
      const key = unitKeys[index] ?? index;
      if (virtualizer.itemSizeCache.has(key)) return;
      const measured =
        node.offsetHeight || Math.round(node.getBoundingClientRect().height);
      if (measured > 0) {
        virtualizer.resizeItem(index, measured);
      }
    },
    [unitKeys, virtualizer],
  );

  useEffect(() => {
    if (!onRegisterScrollToPrompt) return undefined;
    if (!shouldVirtualize) {
      onRegisterScrollToPrompt(null);
      return undefined;
    }
    onRegisterScrollToPrompt((promptId: string) => {
      const index = units.findIndex(
        (unit) =>
          unit.type === "message"
          && unit.message.role === "user"
          && (unit.message.id === promptId || unit.message.promptId === promptId),
      );
      if (index < 0) return false;
      virtualizer.scrollToIndex(index, { align: "start", behavior: "auto" });
      return true;
    });
    return () => onRegisterScrollToPrompt(null);
  }, [onRegisterScrollToPrompt, shouldVirtualize, units, virtualizer]);

  const childJumpIndex = useMemo(() => !childJump ? -1 : units.findIndex((unit) => (
    unit.type === "activity"
      ? unit.messages.some((message) => message.delegatedChild?.child_id === childJump.childId)
      : unit.message.delegatedChild?.child_id === childJump.childId
  )), [childJump, units]);
  const virtualItems = virtualizer.getVirtualItems();
  const childJumpMounted = !shouldVirtualize || virtualItems.some((item) => item.index === childJumpIndex);
  useLayoutEffect(() => {
    if (!childJump || childJumpIndex === -1 || !onChildJumpTarget) return;
    if (!childJumpMounted) {
      virtualizer.scrollToIndex(childJumpIndex, { align: "center", behavior: "auto" });
      return;
    }
    // An activity fold opens in its layout effect before we measure the row.
    const frame = window.requestAnimationFrame(() => {
      const rows = messageListRef.current?.querySelectorAll<HTMLElement>("[data-delegated-child]");
      const row = rows && Array.from(rows).find((element) => element.dataset.delegatedChild === childJump.childId);
      if (row) onChildJumpTarget(row);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [childJump, childJumpIndex, childJumpMounted, onChildJumpTarget, virtualizer]);

  const renderUnit = (unit: DisplayUnit, index: number, usePaddingSpacing: boolean) => {
    const prev = units[index - 1];
    const marginTop =
      index > 0
        ? (usePaddingSpacing ? paddingAfterPrevUnit(prev, unit) : marginAfterPrevUnit(prev, unit))
        : "";
    const next = units[index + 1];
    const hasBodyBelow =
      unit.type === "activity"
      && next?.type === "message"
      && next.message.role === "assistant";
    const stoppedActivity = unit.type === "activity"
      && next?.type === "message"
      && next.message.kind === "run_stopped"
      && next.message.turnId === unit.turnId;
    const isIntermediateAssistant =
      unit.type === "message"
      && unit.message.role === "assistant"
      && next !== undefined
      && (next.type === "activity"
        || (next.message.role !== "user" && next.message.kind !== "run_stopped"));
    const userPromptId =
      unit.type === "message" && unit.message.role === "user"
        ? unit.message.id
        : undefined;
    const userPromptKey =
      unit.type === "message" && unit.message.role === "user"
        ? (unit.message.promptId ?? unit.message.id)
        : undefined;
    const isPromptHighlighted = Boolean(
      highlightedPromptId
      && unit.type === "message"
      && unit.message.role === "user"
      && (
        unit.message.id === highlightedPromptId
        || unit.message.promptId === highlightedPromptId
      ),
    );
    const forkIndex = forkIndices[index];

    const isUnitInActiveTurn = isStreaming && (
      unit.type === "activity"
        ? (
            unit.turnId && activeTurnId !== null
              ? unit.turnId === activeTurnId
              : index > currentTurnStartIndex
          )
        : (
            unit.message.turnId && activeTurnId !== null
              ? unit.message.turnId === activeTurnId
              : index > currentTurnStartIndex
          )
    );
    const isActivityLive = unit.type === "activity" && liveActivityClusterIndices.has(index);
    const showTurnDuration = unit.type === "activity"
      ? (isActivityLive || (unit.showTurnDuration === true && !isUnitInActiveTurn))
      : false;

    return (
      <ThreadDisplayUnit
        key={unitKeys[index]}
        unitKey={unitKeys[index]}
        unit={unit}
        marginTop={marginTop}
        userPromptId={userPromptId}
        userPromptKey={userPromptKey}
        isPromptHighlighted={isPromptHighlighted}
        revealOnHistoryEntry={
          unit.type === "message" && unit.message.id === revealMessageId
        }
        hasBodyBelow={hasBodyBelow}
        stoppedActivity={stoppedActivity}
        isTurnStreaming={
          unit.type === "activity"
            ? isActivityLive
            : isUnitInActiveTurn
        }
        suppressReservedFooter={isIntermediateAssistant}
        showTurnDuration={showTurnDuration}
        retryStatus={
          isActivityLive
            ? retryStatus
            : null
        }
        activeStartedAtMs={isActivityLive ? activeStartedAtMs : undefined}
        forkIndex={forkIndex}
        showForkBoundary={index === forkBoundaryAfterUnitIndex}
        forkBoundaryLabel={t("thread.forkedFromHistory")}
        temporary={temporary}
        cliApps={cliApps}
        mcpPresets={mcpPresets}
        slashCommands={slashCommands}
        onOpenFilePreview={onOpenFilePreview}
        onForkFromMessage={onForkFromMessage}
        onOpenChildHistory={onOpenChildHistory}
        delegatedStatusAvailable={delegatedStatusAvailable}
        childJump={index === childJumpIndex ? childJump : null}
      />
    );
  };

  if (shouldVirtualize) {
    return (
      <div ref={messageListRef} className="flex w-full flex-col">
        <AssistantSelectionAction
          containerRef={messageListRef}
          onQuoteSelection={onQuoteSelection}
        />
        <div
          ref={virtualizer.containerRef}
          data-testid="thread-virtual-list"
          className="relative w-full"
        >
          {virtualItems.map((virtualRow) => {
            const unit = units[virtualRow.index];
            if (!unit) return null;
            return (
              <div
                key={virtualRow.key}
                data-index={virtualRow.index}
                ref={measureVirtualRow}
                className="w-full"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                }}
              >
                {renderUnit(unit, virtualRow.index, true)}
              </div>
            );
          })}
        </div>
        {showPendingActivity ? (
          <div className={pendingActivityMarginTop}>
            <AgentActivityCluster
              messages={[]}
              isTurnStreaming
              hasBodyBelow={false}
              retryStatus={retryStatus}
              startedAtMs={pendingStartedAtMs}
            />
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div ref={messageListRef} className="flex w-full flex-col">
      <AssistantSelectionAction
        containerRef={messageListRef}
        onQuoteSelection={onQuoteSelection}
      />
      {units.map((unit, index) => renderUnit(unit, index, false))}
      {showPendingActivity ? (
        <div className={pendingActivityMarginTop}>
          <AgentActivityCluster
            messages={[]}
            isTurnStreaming
            hasBodyBelow={false}
            retryStatus={retryStatus}
            startedAtMs={pendingStartedAtMs}
          />
        </div>
      ) : null}
    </div>
  );
});

interface ThreadDisplayUnitProps {
  unitKey: string;
  unit: DisplayUnit;
  marginTop: string;
  userPromptId?: string;
  userPromptKey?: string;
  isPromptHighlighted?: boolean;
  revealOnHistoryEntry: boolean;
  hasBodyBelow: boolean;
  stoppedActivity: boolean;
  isTurnStreaming: boolean;
  suppressReservedFooter: boolean;
  showTurnDuration: boolean;
  retryStatus: RetryStatus | null;
  activeStartedAtMs?: number;
  forkIndex?: number;
  showForkBoundary: boolean;
  forkBoundaryLabel: string;
  temporary: boolean;
  cliApps: CliAppInfo[];
  mcpPresets: McpPresetInfo[];
  slashCommands: SlashCommand[];
  onOpenFilePreview?: (path: string) => void;
  onForkFromMessage?: (beforeUserIndex: number) => void;
  onOpenChildHistory?: (child: DelegatedChildReference, trigger: HTMLButtonElement) => void;
  delegatedStatusAvailable: boolean;
  childJump: { childId: string } | null;
}

const ThreadDisplayUnit = memo(function ThreadDisplayUnit({
  unitKey,
  unit,
  marginTop,
  userPromptId,
  userPromptKey,
  isPromptHighlighted = false,
  revealOnHistoryEntry,
  hasBodyBelow,
  stoppedActivity,
  isTurnStreaming,
  suppressReservedFooter,
  showTurnDuration,
  retryStatus,
  activeStartedAtMs,
  forkIndex,
  showForkBoundary,
  forkBoundaryLabel,
  temporary,
  cliApps,
  mcpPresets,
  slashCommands,
  onOpenFilePreview,
  onForkFromMessage,
  onOpenChildHistory,
  delegatedStatusAvailable,
  childJump,
}: ThreadDisplayUnitProps) {
  const onForkFromHere = useCallback(() => {
    if (forkIndex !== undefined) onForkFromMessage?.(forkIndex);
  }, [forkIndex, onForkFromMessage]);
  return (
    <>
      <div
        className={
          isPromptHighlighted
            ? `${marginTop} rounded-2xl ring-2 ring-primary/40 bg-primary/5 transition-colors duration-500`.trim()
            : marginTop
        }
        data-thread-display-unit={unitKey}
        data-user-prompt-id={userPromptId}
        data-prompt-key={userPromptKey}
        data-prompt-highlighted={isPromptHighlighted ? "true" : undefined}
      >
        {unit.type === "activity" ? (
          <AgentActivityCluster
            messages={unit.messages}
            isTurnStreaming={isTurnStreaming}
            retryStatus={retryStatus}
            hasBodyBelow={hasBodyBelow}
            stopped={stoppedActivity}
            showTurnDuration={showTurnDuration}
            turnLatencyMs={unit.turnLatencyMs}
            startedAtMs={activeStartedAtMs ?? unit.startedAtMs}
            cliApps={cliApps}
            mcpPresets={mcpPresets}
            onOpenFilePreview={onOpenFilePreview}
            onOpenChildHistory={onOpenChildHistory}
            delegatedStatusAvailable={delegatedStatusAvailable}
            childJump={childJump}
          />
        ) : (
          <MessageBubble
            message={unit.message}
            isTurnStreaming={isTurnStreaming}
            suppressReservedFooter={suppressReservedFooter}
            revealOnHistoryEntry={revealOnHistoryEntry}
            temporary={temporary}
            cliApps={cliApps}
            mcpPresets={mcpPresets}
            slashCommands={slashCommands}
            onOpenFilePreview={onOpenFilePreview}
            onForkFromHere={
              forkIndex !== undefined && onForkFromMessage ? onForkFromHere : undefined
            }
            onOpenChildHistory={onOpenChildHistory}
            delegatedStatusAvailable={delegatedStatusAvailable}
          />
        )}
      </div>
      {showForkBoundary ? <ForkBoundaryDivider label={forkBoundaryLabel} /> : null}
    </>
  );
}, threadDisplayUnitPropsEqual);

function threadDisplayUnitPropsEqual(
  previous: ThreadDisplayUnitProps,
  next: ThreadDisplayUnitProps,
): boolean {
  return (
    displayUnitsEqual(previous.unit, next.unit)
    && previous.marginTop === next.marginTop
    && previous.userPromptId === next.userPromptId
    && previous.userPromptKey === next.userPromptKey
    && previous.isPromptHighlighted === next.isPromptHighlighted
    && previous.revealOnHistoryEntry === next.revealOnHistoryEntry
    && previous.hasBodyBelow === next.hasBodyBelow
    && previous.stoppedActivity === next.stoppedActivity
    && previous.isTurnStreaming === next.isTurnStreaming
    && previous.suppressReservedFooter === next.suppressReservedFooter
    && previous.showTurnDuration === next.showTurnDuration
    && previous.retryStatus === next.retryStatus
    && previous.activeStartedAtMs === next.activeStartedAtMs
    && previous.forkIndex === next.forkIndex
    && previous.showForkBoundary === next.showForkBoundary
    && previous.forkBoundaryLabel === next.forkBoundaryLabel
    && previous.temporary === next.temporary
    && previous.cliApps === next.cliApps
    && previous.mcpPresets === next.mcpPresets
    && previous.slashCommands === next.slashCommands
    && previous.onOpenFilePreview === next.onOpenFilePreview
    && previous.onOpenChildHistory === next.onOpenChildHistory
    && previous.delegatedStatusAvailable === next.delegatedStatusAvailable
    && previous.childJump === next.childJump
    && previous.onForkFromMessage === next.onForkFromMessage
  );
}

function activeTurnStartIndex(units: DisplayUnit[], activeTurnId: string | null): number {
  if (activeTurnId) {
    const index = units.findIndex((unit) => (
      unit.type === "message"
      && unit.message.role === "user"
      && unit.message.deliveryStatus !== "failed"
      && unit.message.turnId === activeTurnId
    ));
    if (index >= 0) return index;
  }
  for (let i = units.length - 1; i >= 0; i -= 1) {
    const unit = units[i];
    if (
      unit.type === "message"
      && unit.message.role === "user"
      && unit.message.deliveryStatus !== "failed"
    ) return i;
  }
  return -1;
}

function displayUnitsEqual(previous: DisplayUnit, next: DisplayUnit): boolean {
  if (previous.type !== next.type) return false;
  if (previous.type === "message" && next.type === "message") {
    return (
      previous.sourceMessageCount === next.sourceMessageCount
      && shallowMessageEqual(previous.message, next.message)
    );
  }
  if (previous.type !== "activity" || next.type !== "activity") return false;
  return (
    previous.sourceMessageCount === next.sourceMessageCount
    && previous.turnId === next.turnId
    && previous.showTurnDuration === next.showTurnDuration
    && previous.turnLatencyMs === next.turnLatencyMs
    && previous.startedAtMs === next.startedAtMs
    && previous.messages.length === next.messages.length
    && previous.messages.every((message, index) =>
      shallowMessageEqual(message, next.messages[index]))
  );
}

function shallowMessageEqual(previous: UIMessage, next: UIMessage): boolean {
  if (previous === next) return true;
  const previousKeys = Object.keys(previous) as Array<keyof UIMessage>;
  const nextKeys = Object.keys(next) as Array<keyof UIMessage>;
  return previousKeys.length === nextKeys.length
    && previousKeys.every((key) => previous[key] === next[key]);
}

function unitIndexAfterMessageCount(
  units: DisplayUnit[],
  messageCount: number | null | undefined,
): number | null {
  if (messageCount == null || messageCount <= 0) return null;
  let seen = 0;
  for (let i = 0; i < units.length; i += 1) {
    const unit = units[i];
    seen += unit.sourceMessageCount;
    if (seen >= messageCount) return i;
  }
  return null;
}

function ForkBoundaryDivider({ label }: { label: string }) {
  return (
    <div className="my-5 flex items-center gap-3 text-[11px] text-muted-foreground/80">
      <span aria-hidden className="h-px flex-1 bg-border/70" />
      <span className="shrink-0">{label}</span>
      <span aria-hidden className="h-px flex-1 bg-border/70" />
    </div>
  );
}

/**
 * Only a trailing activity fold at the bottom of the active turn owns the live
 * "Working" surface. Earlier activity folds before an intermediate assistant
 * response stay collapsed above that response, and if an intermediate response
 * is currently the last unit while the turn is still running, the pending
 * "Working" indicator appears below it at the bottom of the thread.
 */
function currentActivityClusterIndices(
  units: DisplayUnit[],
  activeTurnId: string | null,
): Set<number> {
  const indices = new Set<number>();
  for (let i = units.length - 1; i >= 0; i -= 1) {
    const unit = units[i];
    if (
      unit.type === "message"
      && unit.message.role === "user"
      && unit.message.deliveryStatus === "failed"
    ) {
      continue;
    }
    if (unit.type === "activity") {
      if (!activeTurnId || !unit.turnId || unit.turnId === activeTurnId) {
        indices.add(i);
      }
    }
    break;
  }
  return indices;
}

export function unitKeysForDisplay(units: DisplayUnit[]): string[] {
  const occurrences = new Map<string, number>();
  return units.map((unit, index) => {
    const base = unitKeyBase(unit, index);
    if (!base.startsWith("turn-") || base.endsWith("-user")) return base;
    const next = (occurrences.get(base) ?? 0) + 1;
    occurrences.set(base, next);
    return `${base}-${next}`;
  });
}

function unitKeyBase(unit: DisplayUnit, index: number): string {
  if (unit.type === "activity") {
    const anchor = unit.messages[0];
    const turnId = unit.turnId ?? anchor?.turnId;
    if (turnId) return `turn-${turnId}-activity`;
    const anchorId = anchor?.id;
    return anchorId != null ? `activity-${anchorId}` : `activity-idx-${index}`;
  }
  const turnKey = stableTurnMessageKey(unit.message);
  if (turnKey) return turnKey;
  return unit.message.id;
}

function stableTurnMessageKey(message: UIMessage | undefined, fallbackPhase?: string): string | null {
  if (!message?.turnId) return null;
  const phase = message.turnPhase ?? fallbackPhase ?? message.kind ?? message.role;
  if (message.role === "user") return `turn-${message.turnId}-user`;
  if (message.kind === "trace") {
    return `turn-${message.turnId}-${phase}-${message.activitySegmentId ?? "activity"}`;
  }
  return `turn-${message.turnId}-${phase}`;
}

function marginAfterPrevUnit(prev: DisplayUnit, current: DisplayUnit): string {
  if (prev.type === "activity") {
    return "mt-4";
  }
  const p = prev.message;
  const denseP =
    p.kind === "trace"
    || (
      p.role === "assistant"
      && p.content.trim().length === 0
      && (!!p.reasoning || !!p.reasoningStreaming)
    );
  if (denseP) {
    return "mt-2";
  }
  const sameAssistantTurn =
    p.role === "assistant"
    && (current.type === "activity" || current.message.role !== "user");
  if (sameAssistantTurn) {
    return "mt-3";
  }
  return "mt-5";
}

function paddingAfterPrevUnit(prev: DisplayUnit, current: DisplayUnit): string {
  if (prev.type === "activity") {
    return "pt-4";
  }
  const p = prev.message;
  const denseP =
    p.kind === "trace"
    || (
      p.role === "assistant"
      && p.content.trim().length === 0
      && (!!p.reasoning || !!p.reasoningStreaming)
    );
  if (denseP) {
    return "pt-2";
  }
  const sameAssistantTurn =
    p.role === "assistant"
    && (current.type === "activity" || current.message.role !== "user");
  if (sameAssistantTurn) {
    return "pt-3";
  }
  return "pt-5";
}

function estimateDisplayUnitHeight(
  unit: DisplayUnit | undefined,
  isLiveActivity: boolean,
): number {
  if (!unit) return 80;
  if (unit.type === "activity") {
    return isLiveActivity ? 180 : 48;
  }
  const content = unit.message.content;
  const contentLen = content.length;
  const hasMedia = Boolean(unit.message.media?.length || unit.message.images?.length);
  const newlineCount = content.split("\n").length - 1;
  if (unit.message.role === "user") {
    const lines = Math.max(1, Math.ceil(contentLen / 60), newlineCount + 1);
    return 44 + Math.min(280, lines * 24) + (hasMedia ? 160 : 0);
  }
  const codeBlocks = Math.floor((content.split("```").length - 1) / 2);
  const lines = Math.max(1, Math.ceil(contentLen / 68), newlineCount + 1);
  return 60 + Math.min(900, lines * 24 + codeBlocks * 44) + (hasMedia ? 260 : 0);
}
