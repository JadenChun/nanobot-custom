import { useEffect, useState } from "react";

import { describeMcpActivity } from "@/components/thread/activity/mcp-activity-model";
import { formatToolCallTrace } from "@/lib/tool-traces";
import type { McpPresetInfo, ToolProgressEvent, UIMessage } from "@/lib/types";

const MCP_TOOL_NAME_RE = /^mcp_([a-z0-9_-]+?)_(.+)$/i;
const WORKING_TICK_MS = 1_000;

const PRODUCT_NAME_OVERRIDES: Record<string, string> = {
  github: "GitHub",
  gitlab: "GitLab",
  openai: "OpenAI",
};

export interface WorkingSummary {
  /** Human-readable name of the running step, e.g. "ComfyUI · Generating video". */
  label: string;
  /** Live MCP phase text, e.g. "ComfyUI is generating prompt...". */
  message?: string;
  /** Rounded percentage when the running step reports progress. */
  percent?: number;
}

/** Turn a preset key into a readable product name. */
export function titleFromPresetName(name: string): string {
  const productName = PRODUCT_NAME_OVERRIDES[name.toLowerCase()];
  if (productName) return productName;
  return name
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ") || name;
}

/** Short, stable duration copy such as `12s` or `2m 14s`. */
export function formatWorkDuration(ms: number): string {
  const seconds = ms > 0 && ms < 1000 ? 1 : Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

/** One shared once-per-second clock for live work surfaces. */
export function useWorkingNow(active: boolean, pageVisible = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active || !pageVisible) return undefined;
    setNow(Date.now());
    const interval = window.setInterval(() => setNow(Date.now()), WORKING_TICK_MS);
    return () => window.clearInterval(interval);
  }, [active, pageVisible]);
  return now;
}

/** Convert unix seconds (server run start) into epoch milliseconds. */
export function runStartedAtMs(runStartedAt?: number | null): number | null {
  if (typeof runStartedAt !== "number" || !Number.isFinite(runStartedAt) || runStartedAt <= 0) {
    return null;
  }
  return Math.round(runStartedAt * 1000);
}

/** Earliest known timestamp for the visible turn, used when the start is unknown. */
export function earliestMessageMs(messages: UIMessage[]): number | null {
  let earliest: number | null = null;
  for (const message of messages) {
    if (message.timestampAvailable === false) continue;
    if (!Number.isFinite(message.createdAt)) continue;
    earliest = earliest === null ? message.createdAt : Math.min(earliest, message.createdAt);
  }
  return earliest;
}

export function progressPercent(
  progress?: number | null,
  total?: number | null,
): number | null {
  if (typeof progress !== "number" || !Number.isFinite(progress)) return null;
  if (typeof total === "number" && Number.isFinite(total) && total > 0) {
    return Math.min(100, Math.max(0, Math.round((progress / total) * 100)));
  }
  if (progress >= 0 && progress <= 100) return Math.round(progress);
  return null;
}

function toolEventKey(event: ToolProgressEvent): string | null {
  if (event.call_id) return `call:${event.call_id}`;
  return formatToolCallTrace(event);
}

function runningToolEvents(messages: UIMessage[]): ToolProgressEvent[] {
  const running = new Map<string, ToolProgressEvent>();
  for (const message of messages) {
    for (const event of message.toolEvents ?? []) {
      const key = toolEventKey(event);
      if (!key) continue;
      if (event.phase === "progress") {
        const previous = running.get(key);
        if (previous) running.set(key, { ...previous, ...event });
        continue;
      }
      if (event.phase === "start") {
        running.delete(key);
        running.set(key, event);
        continue;
      }
      running.delete(key);
    }
  }
  return [...running.values()];
}

/**
 * Summarize the step the agent is working on right now, so a persistent live
 * bar can name it without duplicating the activity timeline logic.
 */
export function describeRunningWork(
  messages: UIMessage[],
  mcpPresets: McpPresetInfo[] = [],
): WorkingSummary | null {
  const running = runningToolEvents(messages);
  const event = running[running.length - 1];
  if (!event) return null;
  const name = typeof event.name === "string" ? event.name : "";
  const percent = progressPercent(event.progress, event.total);
  const message = typeof event.message === "string" && event.message.trim()
    ? event.message.trim()
    : undefined;
  const mcpMatch = MCP_TOOL_NAME_RE.exec(name);
  if (mcpMatch) {
    const presetName = mcpMatch[1].toLowerCase();
    const preset = mcpPresets.find((candidate) => candidate.name.toLowerCase() === presetName);
    const displayName = preset?.display_name || titleFromPresetName(presetName);
    const activity = describeMcpActivity(mcpMatch[2], event.arguments, "running");
    return {
      label: [displayName, activity.action, activity.target].filter(Boolean).join(" · "),
      ...(message ? { message } : {}),
      ...(percent !== null ? { percent } : {}),
    };
  }
  if (!name) return null;
  return {
    label: titleFromPresetName(name),
    ...(message ? { message } : {}),
    ...(percent !== null ? { percent } : {}),
  };
}
