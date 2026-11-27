import { anchorDelegatedChildRows } from "@/lib/delegated-child-projection";
import type { UIMessage } from "@/lib/types";

/** A turn is projected into ordered answer messages and collapsible activity
 * runs. Folding changes presentation only; it never changes semantic order. */
export type TurnUnit =
  | {
      type: "activity";
      messages: UIMessage[];
      /** Number of raw UI messages represented by this display unit. */
      sourceMessageCount: number;
      turnId?: string;
      showTurnDuration?: boolean;
      turnLatencyMs?: number;
      startedAtMs?: number;
    }
  | {
      type: "message";
      message: UIMessage;
      /** Number of raw UI messages represented by this display unit. */
      sourceMessageCount: number;
    };

interface TurnTiming {
  promptAtMs?: number;
  firstAtMs?: number;
  lastAtMs?: number;
  latencyMs?: number;
  lastActivityIndex?: number;
}

export function isReasoningOnlyAssistant(message: UIMessage): boolean {
  if (message.role !== "assistant" || message.kind === "trace") return false;
  if (
    message.activityKind === "model"
    || message.content.trim().length > 0
    || !!message.media?.length
    || !!message.images?.length
  ) return false;
  return !!(message.reasoning?.length || message.reasoningStreaming || message.isStreaming);
}

export function isAgentActivityMember(message: UIMessage): boolean {
  return isReasoningOnlyAssistant(message)
    || message.kind === "trace"
    || message.kind === "delegated_child"
    || message.activityKind === "model";
}

export function hasPendingAgentActivity(allMessages: UIMessage[]): boolean {
  // A child reports its own lifecycle; it says nothing about the parent turn.
  const messages = allMessages.filter((message) => message.kind !== "delegated_child");
  const last = messages.at(-1);
  if (!last || !isAgentActivityMember(last)) return false;
  if (last.isStreaming || last.reasoningStreaming) return true;

  const lastTurnId = last.turnId;
  const previous = messages.at(-2);
  // A trace without a visible answer is an unfinished turn on replay. Once a
  // final assistant answer exists after it, the activity is simply history.
  return !previous
    || previous.role !== "assistant"
    || isAgentActivityMember(previous)
    || previous.turnId !== lastTurnId;
}

/** Project gateway rows without changing their causal order.
 *
 * Messages use ``turnSeq`` when every row in the turn provides it and fall
 * back to stable arrival order otherwise. Only contiguous rows of the same
 * display class are combined: activity rows (including pre-tool prose) share
 * a collapsible surface, and adjacent final answer slices share an answer
 * bubble. Completed empty transport frames create no display boundary.
 */
export function normalizeActivityTimeline(
  messages: UIMessage[],
): TurnUnit[] {
  const units: TurnUnit[] = [];
  const timings = new Map<string | number, TurnTiming>();
  let turnMessages: UIMessage[] = [];
  let activeTurnId: string | undefined;
  let activeTurnStartedAtMs: number | undefined;

  const timingFor = (turnId: string | undefined): TurnTiming => {
    const key = turnId ?? timings.size;
    const timing = timings.get(key) ?? {};
    timings.set(key, timing);
    return timing;
  };

  const flushTurn = () => {
    if (!turnMessages.length) {
      activeTurnId = undefined;
      activeTurnStartedAtMs = undefined;
      return;
    }

    const ordered = orderMessagesByTurnSeq(turnMessages);
    const projected = projectOrderedTurn(ordered, activeTurnId);
    const timing = timingFor(activeTurnId);
    timing.promptAtMs ??= activeTurnStartedAtMs;
    for (const message of ordered) {
      // A child row's creation time is delivery or placeholder time, not turn work.
      // Completion facts stay: turn_end can stamp them on the turn's last row.
      const timestamp = message.kind === "delegated_child"
        ? undefined
        : availableCreatedAtMs(message);
      if (timestamp !== undefined) {
        timing.firstAtMs = Math.min(timing.firstAtMs ?? timestamp, timestamp);
        timing.lastAtMs = Math.max(timing.lastAtMs ?? timestamp, timestamp);
      }
      if (typeof message.completedAt === "number" && Number.isFinite(message.completedAt)) {
        timing.lastAtMs = Math.max(timing.lastAtMs ?? message.completedAt, message.completedAt);
      }
      if (isValidLatency(message.latencyMs)) timing.latencyMs = message.latencyMs;
    }
    if (projected.length) {
      for (let index = projected.length - 1; index >= 0; index -= 1) {
        const unit = projected[index]!;
        if (
          unit.type !== "activity"
          || unit.messages.every((message) => message.kind === "delegated_child")
        ) continue;
        timing.lastActivityIndex = units.length + index;
        break;
      }
      units.push(...projected);
    } else if (units.length) {
      // A turn containing only completed transport placeholders has no
      // display surface. Keep its source count on the preceding prompt so
      // persisted fork-boundary offsets still map to a visible unit.
      const lastIndex = units.length - 1;
      const last = units[lastIndex];
      units[lastIndex] = {
        ...last,
        sourceMessageCount: last.sourceMessageCount + turnMessages.length,
      };
    }

    turnMessages = [];
    activeTurnId = undefined;
    activeTurnStartedAtMs = undefined;
  };

  for (const message of anchorDelegatedChildRows(messages)) {
    if (message.role === "user") {
      flushTurn();
      units.push({ type: "message", message, sourceMessageCount: 1 });
      activeTurnId = message.turnId;
      activeTurnStartedAtMs = availableCreatedAtMs(message);
      if (activeTurnId) timingFor(activeTurnId).promptAtMs ??= activeTurnStartedAtMs;
      continue;
    }
    if (message.turnId && activeTurnId && message.turnId !== activeTurnId) flushTurn();
    if (message.turnId) activeTurnId = message.turnId;
    turnMessages.push(message);
  }

  flushTurn();
  for (const timing of timings.values()) {
    const index = timing.lastActivityIndex;
    if (index === undefined) continue;
    const unit = units[index];
    if (unit.type !== "activity") continue;
    const startedAtMs = timing.promptAtMs ?? timing.firstAtMs;
    const elapsedMs = startedAtMs !== undefined && timing.lastAtMs !== undefined
      ? Math.max(0, timing.lastAtMs - startedAtMs)
      : undefined;
    units[index] = {
      ...unit,
      showTurnDuration: true,
      startedAtMs,
      turnLatencyMs: timing.latencyMs ?? elapsedMs,
    };
  }
  return units;
}

export function projectActivityTimeline(
  messages: UIMessage[],
): TurnUnit[] {
  return normalizeActivityTimeline(messages);
}

/** Use the first occurrence of a tool call, not a late update to an older call,
 * as evidence that the model continued working after a reply. */
function toolEvidence(message: UIMessage, seenCalls: Set<string>): "start" | "tool" | null {
  if (message.kind !== "trace") return null;
  let evidence: "start" | "tool" | null = null;
  for (const event of message.toolEvents ?? []) {
    if (event.call_id && seenCalls.has(event.call_id)) continue;
    if (event.call_id) seenCalls.add(event.call_id);
    // Live progress patches update the start row in place, including its phase.
    const live = event.phase === "start" || event.phase === "progress";
    evidence = live ? "start" : evidence ?? "tool";
  }
  if (message.fileEdits?.some((edit) => edit.phase === "start" || edit.status === "editing")) return "start";
  if (message.toolEvents?.length) return evidence;
  return message.fileEdits?.length || message.content.trim() || message.traces?.some((line) => line.trim())
    ? "tool"
    : null;
}

function isStandaloneMessage(message: UIMessage): boolean {
  return message.kind === "compaction" || message.kind === "run_stopped"
    || !!message.sessionMessage || message.compactReply !== undefined;
}

function interimReplies(messages: UIMessage[]): Set<UIMessage> {
  const seenCalls = new Set<string>();
  const evidence = messages.map((message) => toolEvidence(message, seenCalls));
  const interim = new Set<UIMessage>();
  let answerFollows = false;
  let toolBeforeAnswer = Infinity;
  let toolStart = Infinity;
  let identifiedTurnTool = Infinity;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!;
    if (isStandaloneMessage(message)) {
      answerFollows = false;
      toolBeforeAnswer = toolStart = identifiedTurnTool = Infinity;
    } else if (evidence[index]) {
      if (answerFollows) toolBeforeAnswer = index;
      if (evidence[index] === "start") toolStart = index;
      if (message.turnId) identifiedTurnTool = index;
    } else if (!isRawActivity(message) && isAssistantAnswer(message) && !isCompletedDisplayNoop(message)) {
      answerFollows = true;
      // A legacy trace that only arrives after the last reply is ambiguous.
      // Require a later answer, a tool start, or identified same-turn work.
      const nextTool = messages[Math.min(
        toolBeforeAnswer, toolStart, message.turnId ? identifiedTurnTool : Infinity,
      )];
      if (!nextTool || !message.content.trim() || hasAttachedMedia(message)) continue;
      if (message.completedAt !== undefined && nextTool.timestampAvailable !== false
        && message.completedAt < nextTool.createdAt) continue;
      interim.add(message);
    }
  }
  return interim;
}

function projectOrderedTurn(
  messages: UIMessage[],
  turnId?: string,
): TurnUnit[] {
  const units: TurnUnit[] = [];
  const interim = interimReplies(messages);
  let activity: UIMessage[] = [];
  let activitySourceMessageCount = 0;
  let answers: UIMessage[] = [];
  let answerSourceMessageCount = 0;
  let leadingNoopSourceMessageCount = 0;

  const flushActivity = () => {
    if (!activity.length) return;
    units.push({
      type: "activity",
      messages: activity,
      sourceMessageCount: activitySourceMessageCount,
      turnId,
    });
    activity = [];
    activitySourceMessageCount = 0;
  };

  const flushAnswers = () => {
    if (!answers.length) return;
    units.push({
      type: "message",
      message: mergeAssistantAnswers(answers),
      sourceMessageCount: answerSourceMessageCount,
    });
    answers = [];
    answerSourceMessageCount = 0;
  };

  const claimLeadingNoops = () => {
    const count = leadingNoopSourceMessageCount;
    leadingNoopSourceMessageCount = 0;
    return count;
  };

  const appendActivity = (message: UIMessage, sourceMessageCount: number) => {
    flushAnswers();
    activity.push(message);
    activitySourceMessageCount += sourceMessageCount + claimLeadingNoops();
  };

  const absorbDisplayNoop = () => {
    if (activity.length) {
      activitySourceMessageCount += 1;
    } else if (answers.length) {
      answerSourceMessageCount += 1;
    } else {
      leadingNoopSourceMessageCount += 1;
    }
  };

  for (const message of messages) {
    if (message.kind === "delegated_child") {
      appendActivity(message, 1);
      continue;
    }
    if (message.kind === "compaction" || message.kind === "run_stopped") {
      flushActivity();
      flushAnswers();
      units.push({
        type: "message",
        message,
        sourceMessageCount: 1 + claimLeadingNoops(),
      });
      continue;
    }
    if (isCompletedDisplayNoop(message)) {
      absorbDisplayNoop();
      continue;
    }
    if (isRawActivity(message)) {
      appendActivity(message, 1);
      continue;
    }
    if (isAssistantAnswer(message)) {
      if (message.reasoning?.trim() || message.reasoningStreaming) {
        // The synthetic reasoning row carries no additional source count.
        appendActivity(reasoningOnlyMessageFromAnswer(message), 0);
      }
      if (interim.has(message)) {
        appendActivity({
          ...stripInlineReasoning(message), activityKind: "model", isStreaming: false,
        }, 1);
        continue;
      }
      flushActivity();
      // Keep media inline where the agent emitted it: once a buffered answer
      // already carries media/images, later text starts a new bubble below the
      // media instead of pulling the media down to the end of the turn.
      if (answers.some(hasAttachedMedia)) {
        flushAnswers();
      }
      answerSourceMessageCount += claimLeadingNoops();
      answers.push(stripInlineReasoning(message));
      answerSourceMessageCount += 1;
      continue;
    }
    appendActivity(message, 1);
  }

  flushActivity();
  flushAnswers();

  return units;
}

function isRawActivity(message: UIMessage): boolean {
  return isAgentActivityMember(message);
}

/** A completed transport placeholder carries ordering/accounting metadata but
 * no user-visible semantics, so it cannot define a display boundary. */
function isCompletedDisplayNoop(message: UIMessage): boolean {
  return (
    message.role === "assistant"
    && message.kind !== "trace"
    && message.activityKind !== "model"
    && !message.isStreaming
    && !message.reasoningStreaming
    && message.content.trim().length === 0
    && !message.reasoning?.trim()
    && !message.media?.length
    && !message.images?.length
    && !message.traces?.some((line) => line.trim().length > 0)
    && !message.toolEvents?.length
    && !message.fileEdits?.length
    && !message.sessionMessage
  );
}

function isAssistantAnswer(message: UIMessage): boolean {
  if (message.role !== "assistant" || message.kind === "trace" || message.activityKind === "model") {
    return false;
  }
  if (message.turnPhase === "reasoning" || message.turnPhase === "activity") return false;
  return (
    message.turnPhase === "answer"
    || message.content.trim().length > 0
    || hasAttachedMedia(message)
  );
}

function hasAttachedMedia(message: UIMessage): boolean {
  return Boolean(message.media?.length || message.images?.length);
}

function orderMessagesByTurnSeq(messages: UIMessage[]): UIMessage[] {
  if (messages.length < 2 || !messages.every((message) => Number.isFinite(message.turnSeq))) {
    return messages;
  }
  return messages
    .map((message, index) => ({ message, index }))
    .sort((left, right) => (left.message.turnSeq! - right.message.turnSeq!) || (left.index - right.index))
    .map(({ message }) => message);
}

function mergeAssistantAnswers(answers: UIMessage[]): UIMessage {
  const first = answers[0];
  const last = answers.at(-1)!;
  const media = answers.flatMap((message) => message.media ?? []);
  const images = answers.flatMap((message) => message.images ?? []);
  const merged: UIMessage = {
    ...first,
    ...last,
    id: first.id,
    content: answers.map((message) => message.content.trim()).filter(Boolean).join("\n\n"),
    createdAt: first.createdAt,
    isStreaming: answers.some((message) => message.isStreaming),
  };
  if (media.length) merged.media = media;
  else delete merged.media;
  if (images.length) merged.images = images;
  else delete merged.images;
  return merged;
}

function reasoningOnlyMessageFromAnswer(message: UIMessage): UIMessage {
  return {
    id: `${message.id}-reasoning`,
    role: "assistant",
    content: "",
    createdAt: message.createdAt,
    reasoning: message.reasoning,
    reasoningStreaming: message.reasoningStreaming,
    isStreaming: message.reasoningStreaming,
    activitySegmentId: message.activitySegmentId,
    latencyMs: message.latencyMs,
    turnId: message.turnId,
    turnPhase: "reasoning",
    turnSeq: message.turnSeq,
  };
}

function stripInlineReasoning(message: UIMessage): UIMessage {
  const next = { ...message };
  delete next.reasoning;
  delete next.reasoningStreaming;
  return next;
}

function availableCreatedAtMs(message: UIMessage): number | undefined {
  return message.timestampAvailable !== false && Number.isFinite(message.createdAt)
    ? message.createdAt
    : undefined;
}

function isValidLatency(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
