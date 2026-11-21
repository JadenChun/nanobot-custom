import { Check, CircleAlert, CircleDot, Clock3, Loader2, Square } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { DelegatedChildReference } from "@/lib/types";
import { cn } from "@/lib/utils";

export function SubagentStatus({
  child,
  available,
  summaryLabel = true,
}: {
  child: DelegatedChildReference;
  available: boolean;
  summaryLabel?: boolean;
}) {
  const { t } = useTranslation();
  const working = { label: t("thread.delegatedChild.working", "Working"), Icon: Loader2 };
  const states = {
    reserved: { label: t("thread.delegatedChild.starting", "Starting"), Icon: Clock3 },
    active: working,
    running: working,
    joining: { label: t("thread.delegatedChild.finishing", "Finishing"), Icon: Loader2 },
    waiting_approval: { label: t("thread.delegatedChild.waitingApproval", "Waiting for approval"), Icon: Clock3 },
    stopping: { label: t("thread.delegatedChild.stopping", "Stopping"), Icon: Loader2 },
    completed: { label: t("thread.delegatedChild.completed", "Completed"), Icon: Check },
    failed: { label: t("thread.delegatedChild.failed", "Failed"), Icon: CircleAlert },
    stopped: { label: t("thread.delegatedChild.stopped", "Stopped"), Icon: Square },
    refused: { label: t("thread.delegatedChild.refused", "Could not start"), Icon: CircleAlert },
    uncertain: { label: t("thread.delegatedChild.uncertain", "Outcome uncertain"), Icon: CircleAlert },
    interrupted: { label: t("thread.delegatedChild.interrupted", "Interrupted"), Icon: CircleAlert },
  };
  const state = Object.hasOwn(states, child.state)
    ? states[child.state as keyof typeof states]
    : { label: t("thread.delegatedChild.unknownStatus", "Unknown status"), Icon: CircleDot };
  const summary = child.history_kind === "summary";
  const label = summary && summaryLabel
    ? t("thread.delegatedChild.summaryState", {
        defaultValue: "Summary only · {{state}}",
        state: state.label,
      })
    : available || summary
      ? state.label
      : t("thread.delegatedChild.unavailable", {
          defaultValue: "Status unavailable (last known {{state}})",
          state: state.label,
        });
  const Icon = summary || !available ? Clock3 : state.Icon;
  return (
    <span className="flex min-w-0 items-start gap-1.5 text-xs leading-4 text-muted-foreground">
      <Icon
        aria-hidden
        className={cn("mt-px size-3.5 shrink-0", Icon === Loader2 && "motion-safe:animate-spin")}
      />
      <span>{label}</span>
    </span>
  );
}
