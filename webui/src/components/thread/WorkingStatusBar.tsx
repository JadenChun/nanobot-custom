import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import {
  describeRunningWork,
  earliestMessageMs,
  formatWorkDuration,
  runStartedAtMs,
  useWorkingNow,
} from "@/components/thread/working-status";
import { usePageVisibility } from "@/hooks/usePageVisibility";
import { cn } from "@/lib/utils";
import type { McpPresetInfo, UIMessage } from "@/lib/types";

interface WorkingStatusBarProps {
  messages: UIMessage[];
  /** Server or optimistic start of the active turn, in unix seconds. */
  runStartedAt?: number | null;
  mcpPresets?: McpPresetInfo[];
  className?: string;
}

/**
 * Persistent live working bar pinned right above the composer.
 *
 * The activity fold can scroll out of view while a long turn runs. This bar
 * always shows that the agent is working, which step it is on, and how long the
 * turn has been running. The parent renders it only while the turn streams, so
 * a settled chat stays clean.
 */
export function WorkingStatusBar({
  messages,
  runStartedAt = null,
  mcpPresets = [],
  className,
}: WorkingStatusBarProps) {
  const { t } = useTranslation();
  const pageVisible = usePageVisibility();
  const now = useWorkingNow(true, pageVisible);
  const summary = useMemo(
    () => describeRunningWork(messages, mcpPresets),
    [messages, mcpPresets],
  );
  const startedAt = useMemo(() => {
    // Mirror the activity fold: the server run start wins, then the current
    // turn's prompt row, then any visible timestamp.
    const explicit = runStartedAtMs(runStartedAt);
    if (explicit !== null) return explicit;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role !== "user" || message.timestampAvailable === false) continue;
      if (Number.isFinite(message.createdAt)) return message.createdAt;
    }
    return earliestMessageMs(messages);
  }, [messages, runStartedAt]);
  const elapsed = formatWorkDuration(
    startedAt === null ? 0 : Math.max(0, now - startedAt),
  );
  const stepLabel = summary?.label
    ?? t("thread.workingBar.defaultStep", { defaultValue: "Working" });
  const detail = summary?.message;

  return (
    <div
      data-testid="thread-working-bar"
      role="status"
      aria-live="polite"
      aria-label={[
        t("thread.workingBar.ariaLabel", { defaultValue: "Current step" }),
        stepLabel,
        detail,
        elapsed,
      ].filter(Boolean).join(" · ")}
      className={cn(
        "pointer-events-none mx-auto w-full max-w-[58rem] px-3 pb-1.5 sm:px-4",
        "animate-in fade-in duration-200 motion-reduce:animate-none",
        className,
      )}
    >
      <div className="relative overflow-hidden rounded-full border border-border/55 bg-background/85 shadow-sm backdrop-blur">
        <div className="flex min-w-0 items-center gap-2 px-3 py-1 text-[12px]">
          <span aria-hidden className="relative grid h-2 w-2 shrink-0 place-items-center">
            <span className="absolute inset-0 animate-ping rounded-full bg-primary/45 motion-reduce:animate-none" />
            <span className="relative h-2 w-2 rounded-full bg-primary" />
          </span>
          <span className="min-w-0 flex-1 truncate font-medium text-muted-foreground/85">
            {stepLabel}
            {detail ? (
              <span className="text-muted-foreground/60"> · {detail}</span>
            ) : null}
          </span>
          {summary?.percent !== undefined ? (
            <span className="shrink-0 tabular-nums text-[11px] text-muted-foreground/70">
              {summary.percent}%
            </span>
          ) : null}
          <span
            data-testid="thread-working-bar-timer"
            className="shrink-0 tabular-nums text-[11px] font-medium text-muted-foreground/70"
          >
            {elapsed}
          </span>
        </div>
        {summary?.percent !== undefined ? (
          <span aria-hidden className="absolute inset-x-0 bottom-0 h-px bg-primary/15">
            <span
              className="block h-full bg-primary/50 transition-[width] duration-500 ease-out motion-reduce:transition-none"
              style={{ width: `${summary.percent}%` }}
            />
          </span>
        ) : null}
      </div>
    </div>
  );
}
