import { memo, useCallback, useMemo, useRef } from "react";
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
      && unit.message.role === "assistant"
      && unit.message.kind === "compaction"
    ) {
      // Compaction notices are session lifecycle markers, not assistant answers.
      // They must neither expose nor displace the answer-level fork action.
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

export function ThreadMessages({
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
  const activeStartedAtMs = activePrompt
    && activePrompt.timestampAvailable !== false
    && Number.isFinite(activePrompt.createdAt)
    ? activePrompt.createdAt
    : runStartedAt != null ? runStartedAt * 1000 : undefined;
  const showPendingActivity = isStreaming
    && liveActivityClusterIndices.size === 0
    && (activePrompt !== null || activeStartedAtMs !== undefined);
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
  let nextUserIndex = hiddenUserMessageCount;

  // Render historical units eagerly. `content-visibility: auto` skipped their
  // layout until they entered the viewport, and the browser's estimated height
  // changed the scroll extent mid-scroll, which made the transcript bounce.
  return (
    <div ref={messageListRef} className="flex w-full flex-col">
      <AssistantSelectionAction
        containerRef={messageListRef}
        onQuoteSelection={onQuoteSelection}
      />
      {units.map((unit, index) => {
        const prev = units[index - 1];
        const marginTop =
          index > 0
            ? marginAfterPrevUnit(prev)
            : "";
        const next = units[index + 1];        const hasBodyBelow =
          unit.type === "activity"
          && next?.type === "message"
          && next.message.role === "assistant";
        const userPromptId =
          unit.type === "message" && unit.message.role === "user"
            ? unit.message.id
            : undefined;
        const forkIndex =
          unit.type === "message" && unit.message.role === "assistant" && forkFlags[index]
            ? nextUserIndex
            : undefined;
        if (
          unit.type === "message"
          && unit.message.role === "user"
          && unit.message.deliveryStatus !== "failed"
        ) nextUserIndex += 1;

        return (
          <ThreadDisplayUnit
            key={unitKeys[index]}
            unitKey={unitKeys[index]}
            unit={unit}
            marginTop={marginTop}
            userPromptId={userPromptId}
            revealOnHistoryEntry={
              unit.type === "message" && unit.message.id === revealMessageId
            }
            hasBodyBelow={hasBodyBelow}
            isTurnStreaming={
              unit.type === "activity"
                ? liveActivityClusterIndices.has(index)
                : isStreaming && (
                    unit.message.turnId && activeTurnId !== null
                      ? unit.message.turnId === activeTurnId
                      : index > currentTurnStartIndex
                  )
            }
            retryStatus={
              unit.type === "activity" && liveActivityClusterIndices.has(index)
                ? retryStatus
                : null
            }
            activeStartedAtMs={liveActivityClusterIndices.has(index) ? activeStartedAtMs : undefined}
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
          />
        );
      })}
      {showPendingActivity ? (
        <div className={units.length > 0 ? "mt-5" : undefined}>
          <AgentActivityCluster
            messages={[]}
            isTurnStreaming
            hasBodyBelow={false}
            retryStatus={retryStatus}
            startedAtMs={activeStartedAtMs}
          />
        </div>
      ) : null}
    </div>
  );
}

interface ThreadDisplayUnitProps {
  unitKey: string;
  unit: DisplayUnit;
  marginTop: string;
  userPromptId?: string;
  revealOnHistoryEntry: boolean;
  hasBodyBelow: boolean;
  isTurnStreaming: boolean;
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
}

const ThreadDisplayUnit = memo(function ThreadDisplayUnit({
  unitKey,
  unit,
  marginTop,
  userPromptId,
  revealOnHistoryEntry,
  hasBodyBelow,
  isTurnStreaming,
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
}: ThreadDisplayUnitProps) {
  const onForkFromHere = useCallback(() => {
    if (forkIndex !== undefined) onForkFromMessage?.(forkIndex);
  }, [forkIndex, onForkFromMessage]);
  return (
    <>
      <div
        className={marginTop}
        data-thread-display-unit={unitKey}
        data-user-prompt-id={userPromptId}
      >
        {unit.type === "activity" ? (
          <AgentActivityCluster
            messages={unit.messages}
            isTurnStreaming={isTurnStreaming}
            retryStatus={retryStatus}
            hasBodyBelow={hasBodyBelow}
            showTurnDuration={isTurnStreaming || unit.showTurnDuration === true}
            turnLatencyMs={unit.turnLatencyMs}
            startedAtMs={activeStartedAtMs ?? unit.startedAtMs}
            cliApps={cliApps}
            mcpPresets={mcpPresets}
            onOpenFilePreview={onOpenFilePreview}
          />
        ) : (
          <MessageBubble
            message={unit.message}
            isTurnStreaming={isTurnStreaming}
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
    && previous.revealOnHistoryEntry === next.revealOnHistoryEntry
    && previous.hasBodyBelow === next.hasBodyBelow
    && previous.isTurnStreaming === next.isTurnStreaming
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

function currentActivityClusterIndices(
  units: DisplayUnit[],
  activeTurnId: string | null,
): Set<number> {
  const indices = new Set<number>();
  if (activeTurnId) {
    for (let i = units.length - 1; i >= 0; i -= 1) {
      const unit = units[i];
      if (
        unit.type === "activity"
        && unit.turnId === activeTurnId
      ) {
        indices.add(i);
        return indices;
      }
    }
  }

  let markedCurrentActivity = false;
  for (let i = units.length - 1; i >= 0; i -= 1) {
    const unit = units[i];
    if (unit.type === "activity") {
      if (!markedCurrentActivity) {
        indices.add(i);
        markedCurrentActivity = true;
      }
      continue;
    }
    if (unit.message.role === "assistant" && unit.message.isStreaming) continue;
    if (unit.message.role === "user") break;
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
    const turnKey = stableTurnMessageKey(anchor, "activity");
    if (turnKey) return turnKey;
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

function marginAfterPrevUnit(prev: DisplayUnit): string {
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
  return "mt-5";
}
