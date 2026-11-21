import type { ToolProgressEvent } from "@/lib/types";

export function formatToolCallTrace(call: unknown): string | null {
  if (!call || typeof call !== "object") return null;
  const item = call as {
    name?: unknown;
    arguments?: unknown;
    function?: { name?: unknown; arguments?: unknown };
  };
  const name =
    typeof item.function?.name === "string"
      ? item.function.name
      : typeof item.name === "string"
        ? item.name
        : "";
  if (!name) return null;
  const args = item.function?.arguments ?? item.arguments;
  if (typeof args === "string" && args.trim()) return `${name}(${args})`;
  if (args && typeof args === "object") {
    const serialized = JSON.stringify(args);
    return serialized === "{}" || serialized === "[]" ? `${name}()` : `${name}(${serialized})`;
  }
  return `${name}()`;
}

export function canonicalToolTrace(line: string): string {
  const trimmed = line.trim();
  const match = /^([a-zA-Z0-9_.-]+)\((.*)\)$/.exec(trimmed);
  if (!match) return trimmed;
  const args = match[2].trim();
  if (!args) return `${match[1]}()`;
  try {
    return `${match[1]}(${JSON.stringify(JSON.parse(args))})`;
  } catch {
    return trimmed;
  }
}

const VALID_PHASES = new Set(["start", "end", "error", "progress"]);
const PHASE_RANK: Record<string, number> = { start: 1, progress: 1, end: 2, error: 3 };

export function normalizeToolProgressEvents(events: unknown): ToolProgressEvent[] {
  if (!Array.isArray(events)) return [];
  const out: ToolProgressEvent[] = [];
  for (const event of events) {
    if (!event || typeof event !== "object") continue;
    const record = event as ToolProgressEvent;
    const phase = record.phase;
    if (!(phase && typeof phase === "string" && VALID_PHASES.has(phase))) continue;
    const name = typeof record.name === "string" ? record.name : "";
    const functionName =
      typeof (record as { function?: { name?: unknown } }).function?.name === "string"
        ? String((record as { function?: { name?: unknown } }).function?.name)
        : "";
    if (!name && !functionName) continue;
    out.push(record);
  }
  return out;
}

function toolEventKey(event: ToolProgressEvent): string {
  if (event.call_id) return `call:${event.call_id}`;
  return formatToolCallTrace(event) ?? JSON.stringify(event);
}

function withoutLiveProgress(event: ToolProgressEvent): ToolProgressEvent {
  if (event.phase !== "end" && event.phase !== "error") return event;
  const rest = { ...event };
  delete rest.progress;
  delete rest.total;
  delete rest.message;
  return rest;
}

export function mergeToolProgressEvents(
  previous: ToolProgressEvent[] | undefined,
  incoming: ToolProgressEvent[],
): ToolProgressEvent[] {
  if (!previous?.length) return incoming.map(withoutLiveProgress);
  if (!incoming.length) return previous;
  const next = [...previous];
  const indexByKey = new Map(next.map((event, index) => [toolEventKey(event), index]));
  for (const event of incoming) {
    const key = toolEventKey(event);
    const existingIndex = indexByKey.get(key);
    if (existingIndex === undefined) {
      indexByKey.set(key, next.length);
      next.push(withoutLiveProgress(event));
      continue;
    }
    const existing = next[existingIndex];
    const incomingRank = PHASE_RANK[String(event.phase)] ?? 0;
    const existingRank = PHASE_RANK[String(existing.phase)] ?? 0;
    next[existingIndex] = incomingRank >= existingRank
      ? withoutLiveProgress({ ...existing, ...event })
      : existing;
  }
  return next;
}

/**
 * Apply live `phase: "progress"` frames to the in-flight tool rows.
 *
 * Progress frames carry no new row: they only refresh the live phase text and
 * percentage of the call they belong to, so the activity timeline never grows
 * duplicate rows while a long ComfyUI/MCP step is running.
 */
export function applyToolProgressPatches(
  events: ToolProgressEvent[] | undefined,
  patches: ToolProgressEvent[],
): ToolProgressEvent[] | undefined {
  if (!events?.length || !patches.length) return events;
  const patchByKey = new Map<string, ToolProgressEvent>();
  for (const patch of patches) patchByKey.set(toolEventKey(patch), patch);
  let changed = false;
  const next = events.map((event) => {
    const patch = patchByKey.get(toolEventKey(event));
    if (!patch) return event;
    changed = true;
    return { ...event, ...patch };
  });
  return changed ? next : events;
}

export function toolTraceLinesFromEvents(events: unknown): string[] {
  const seen = new Set<string>();
  const lines: string[] = [];
  for (const event of normalizeToolProgressEvents(events)) {
    // A live progress frame belongs to an existing call row; it never adds one.
    if (event.phase === "progress") continue;
    const callId = (event as { call_id?: unknown }).call_id;
    if (callId && typeof callId === "string") {
      if (seen.has(callId)) continue;
      seen.add(callId);
    }
    const line = formatToolCallTrace(event);
    if (!line) continue;
    lines.push(line);
  }
  return lines;
}

export function mergeUniqueToolTraceLines(
  previousTraces: string[],
  lines: string[],
): { traces: string[]; added: boolean } {
  const seen = new Set(previousTraces.map(canonicalToolTrace));
  const traces = [...previousTraces];
  let added = false;
  for (const line of lines) {
    const key = canonicalToolTrace(line);
    if (seen.has(key)) continue;
    seen.add(key);
    traces.push(line);
    added = true;
  }
  return { traces, added };
}

export function mergeToolProgressTraceLines(
  previousTraces: string[],
  previousEvents: ToolProgressEvent[] | undefined,
  incomingTraces: string[],
  incomingEvents: ToolProgressEvent[],
): string[] {
  const mergedEvents = mergeToolProgressEvents(previousEvents, incomingEvents);
  const candidates = mergeUniqueToolTraceLines(previousTraces, incomingTraces).traces;
  const eventTraceKeys = new Set([
    ...toolTraceLinesFromEvents(previousEvents),
    ...toolTraceLinesFromEvents(incomingEvents),
  ].map(canonicalToolTrace));
  const nonEventTraces = candidates.filter(
    (line) => !eventTraceKeys.has(canonicalToolTrace(line)),
  );
  return mergeUniqueToolTraceLines(
    nonEventTraces,
    toolTraceLinesFromEvents(mergedEvents),
  ).traces;
}
