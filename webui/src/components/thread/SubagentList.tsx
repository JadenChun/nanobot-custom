import { Bot } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ClaudeBadge } from "@/components/thread/ClaudeBadge";
import { SubagentStatus, subagentTaskLabel } from "@/components/thread/SubagentStatus";
import { formatWorkDuration } from "@/components/thread/working-status";
import { Button } from "@/components/ui/button";
import type { DelegatedChildReference } from "@/lib/types";

export interface SessionSubagents {
  entries: readonly { child: DelegatedChildReference; createdAt: number; canJump: boolean }[];
  statusAvailable: boolean;
  onJump: (child: DelegatedChildReference) => void;
  onOpenHistory: (child: DelegatedChildReference, trigger: HTMLButtonElement) => void;
}

export function SubagentList({ entries, statusAvailable, onJump, onOpenHistory }: SessionSubagents) {
  const { t } = useTranslation("common");
  if (!entries.length) return null;
  // Reverse the source order as a tie-breaker for legacy rows with no timestamp.
  const ordered = [...entries].reverse().sort((a, b) => b.createdAt - a.createdAt);
  const unavailableJump = t("thread.sessionInfo.subagentNotLoaded", "This activity is not in the current history window");
  return (
    <section
      className="space-y-1.5 border-t border-border/45 pt-2.5"
      aria-label={t("thread.sessionInfo.subagentsCount", {
        count: entries.length, defaultValue: "Subagents ({{count}})",
      })}
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="flex min-w-0 items-center gap-1.5 text-[12.5px] font-medium">
          <Bot className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
          {t("thread.sessionInfo.subagents", "Subagents")}
        </h3>
        <span className="rounded-full bg-muted/65 px-2 py-0.5 text-[10.5px] tabular-nums text-muted-foreground">
          {t("thread.sessionInfo.count", { count: entries.length })}
        </span>
      </div>
      <ul className="space-y-1">
        {ordered.map(({ child, canJump }) => {
          const title = subagentTaskLabel(child);
          const jumpLabel = t("thread.sessionInfo.jumpToSubagent", {
            task: title.short, defaultValue: "Jump to {{task}}",
          });
          return (
            <li key={child.child_id} className="min-w-0 rounded-control bg-muted/30 px-2.5 py-2">
              <div className="flex min-w-0 items-center gap-1.5">
                <span title={title.full} className="truncate text-[12.5px] font-medium">{title.short}</span>
                {child.harness === "claude_sdk" ? <ClaudeBadge /> : (
                  <span className="shrink-0 rounded-md border border-border/70 px-1.5 py-0.5 text-[10px] font-medium leading-none text-muted-foreground">
                    Kohaku
                  </span>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
                <SubagentStatus child={child} available={statusAvailable} />
                {child.duration_ms !== undefined ? (
                  <span className="tabular-nums">{formatWorkDuration(child.duration_ms)}</span>
                ) : null}
                {child.step_count !== undefined ? (
                  <span className="tabular-nums">
                    {t("thread.delegatedChild.stepCount", { count: child.step_count, defaultValue: "{{count}} steps" })}
                  </span>
                ) : null}
              </div>
              <div className="-ms-1 mt-1 flex flex-wrap items-center gap-x-1">
                <span title={canJump ? undefined : unavailableJump}>
                  <Button
                    type="button" variant="ghost" size="sm"
                    className="h-7 px-1.5 text-xs"
                    disabled={!canJump}
                    aria-label={jumpLabel}
                    aria-description={canJump ? undefined : unavailableJump}
                    onClick={() => onJump(child)}
                  >
                    {t("thread.sessionInfo.jumpTo", "Jump to")}
                  </Button>
                </span>
                <Button
                  type="button" variant="ghost" size="sm"
                  className="h-7 px-1.5 text-xs"
                  aria-label={t("thread.sessionInfo.openSubagentHistory", {
                    task: title.short, defaultValue: "Open history for {{task}}",
                  })}
                  onClick={(event) => onOpenHistory(child, event.currentTarget)}
                >
                  {t("thread.sessionInfo.openHistory", "Open history")}
                </Button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
