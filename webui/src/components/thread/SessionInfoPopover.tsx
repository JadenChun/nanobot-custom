import { useState } from "react";
import {
  CalendarClock,
  CircleAlert,
  ListTodo,
  RefreshCcw,
  X,
} from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useSessionAutomationJobs } from "@/hooks/useSessionAutomationJobs";
import { currentLocale } from "@/i18n";
import { fmtDateTime } from "@/lib/format";
import type { SessionAutomationJob } from "@/lib/types";
import { cn } from "@/lib/utils";

const RELATIVE_THRESHOLDS: [number, Intl.RelativeTimeFormatUnit][] = [
  [60, "second"],
  [60, "minute"],
  [24, "hour"],
  [7, "day"],
  [4.345, "week"],
  [12, "month"],
  [Number.POSITIVE_INFINITY, "year"],
];

export interface SessionSummaryPanelProps {
  sessionKey: string;
  token: string;
  title: string;
  active: boolean;
  modal?: boolean;
  onClose?: () => void;
}

export function SessionSummaryPanel({
  sessionKey,
  token,
  title,
  active,
  modal = false,
  onClose,
}: SessionSummaryPanelProps) {
  const { t } = useTranslation("common");
  const { jobs, loading, loadFailed, now } = useSessionAutomationJobs(active, token, sessionKey);
  const automationContent = loading ? (
    <div className="flex items-center gap-2 rounded-floating bg-muted/45 px-3 py-3 text-[12.5px] text-muted-foreground">
      <RefreshCcw className="h-3.5 w-3.5 animate-spin" />
      {t("thread.sessionInfo.loading")}
    </div>
  ) : loadFailed ? (
    <div className="flex items-center gap-2 rounded-floating bg-destructive/10 px-3 py-3 text-[12.5px] text-destructive">
      <CircleAlert className="h-3.5 w-3.5" />
      {t("thread.sessionInfo.loadFailed")}
    </div>
  ) : jobs.length ? (
    <div className="space-y-1.5">
      {jobs.map((job) => (
        <AutomationRow key={job.id} job={job} now={now} />
      ))}
    </div>
  ) : (
    <div className="rounded-floating bg-muted/35 px-3 py-3 text-[12.5px] leading-relaxed text-muted-foreground">
      {t("thread.sessionInfo.empty")}
    </div>
  );

  return (
    <div
      data-testid="session-summary-panel"
      className="flex h-full min-h-0 w-full flex-col overflow-y-auto px-4 py-4"
    >
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[12px] font-normal text-muted-foreground/75">
              {t("thread.sessionInfo.title")}
            </div>
            <div className="mt-0.5 truncate text-[14px] font-medium text-foreground">
              {title || t("thread.sessionInfo.untitled")}
            </div>
          </div>
          {modal && onClose ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={t("common.close", { defaultValue: "Close" })}
              onClick={onClose}
              className="-mr-1.5 -mt-1 h-7 w-7 shrink-0 rounded-full text-muted-foreground/80 hover:bg-accent/40 hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </Button>
          ) : null}
        </div>

        <div className="h-px bg-border/45" />

        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <CalendarClock className="h-3.5 w-3.5 shrink-0 text-muted-foreground/80" />
            <span className="truncate text-[13px] font-medium text-foreground">
              {t("thread.sessionInfo.automations")}
            </span>
          </div>
          <span className="rounded-full bg-muted/70 px-2 py-0.5 text-[11px] text-muted-foreground">
            {t("thread.sessionInfo.count", { count: jobs.length })}
          </span>
        </div>

        {automationContent}
      </div>
    </div>
  );
}

export interface SessionSummaryCardProps {
  sessionKey: string;
  token: string;
  title: string;
  active: boolean;
  onClose?: () => void;
  className?: string;
}

export function SessionSummaryCard({
  sessionKey,
  token,
  title,
  active,
  onClose,
  className,
}: SessionSummaryCardProps) {
  const { t } = useTranslation("common");
  const { jobs, loading, loadFailed, now } = useSessionAutomationJobs(active, token, sessionKey);
  const automationContent = loading ? (
    <div className="flex items-center gap-2 rounded-lg bg-muted/40 px-2.5 py-2 text-[12px] text-muted-foreground">
      <RefreshCcw className="h-3.5 w-3.5 animate-spin" />
      {t("thread.sessionInfo.loading")}
    </div>
  ) : loadFailed ? (
    <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-2.5 py-2 text-[12px] text-destructive">
      <CircleAlert className="h-3.5 w-3.5" />
      {t("thread.sessionInfo.loadFailed")}
    </div>
  ) : jobs.length ? (
    <div className="space-y-1">
      {jobs.map((job) => (
        <AutomationRow key={job.id} job={job} now={now} />
      ))}
    </div>
  ) : (
    <div className="rounded-lg bg-muted/30 px-2.5 py-2 text-[12px] leading-relaxed text-muted-foreground">
      {t("thread.sessionInfo.empty")}
    </div>
  );

  return (
    <section
      aria-label={t("thread.sessionInfo.title")}
      data-testid="session-summary-card"
      className={cn(
        "flex max-h-full w-full flex-col overflow-hidden rounded-xl border border-border/55 bg-card/90 shadow-sm",
        "dark:border-white/[0.08] dark:bg-card/85",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2 px-3.5 py-2.5">
        <div className="min-w-0">
          <div className="text-[11.5px] font-medium text-muted-foreground/75">
            {t("thread.sessionInfo.title")}
          </div>
          <div className="mt-0.5 truncate text-[13px] font-medium text-foreground">
            {title || t("thread.sessionInfo.untitled")}
          </div>
        </div>
        {onClose ? (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t("common.close", { defaultValue: "Close" })}
            onClick={onClose}
            className="-mr-1.5 -mt-0.5 h-6 w-6 shrink-0 rounded-full text-muted-foreground/75 hover:bg-accent/40 hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        ) : null}
      </div>

      <div className="h-px bg-border/45" />

      <div className="scrollbar-thin scrollbar-track-transparent min-h-0 flex-1 space-y-2.5 overflow-y-auto px-3.5 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-1.5">
            <CalendarClock className="h-3.5 w-3.5 shrink-0 text-muted-foreground/80" />
            <span className="truncate text-[12.5px] font-medium text-foreground">
              {t("thread.sessionInfo.automations")}
            </span>
          </div>
          <span className="rounded-full bg-muted/65 px-2 py-0.5 text-[10.5px] tabular-nums text-muted-foreground">
            {t("thread.sessionInfo.count", { count: jobs.length })}
          </span>
        </div>

        {automationContent}
      </div>
    </section>
  );
}

interface SessionInfoPopoverProps {
  sessionKey: string;
  token: string;
  title: string;
}

export function SessionInfoPopover({ sessionKey, token, title }: SessionInfoPopoverProps) {
  const { t } = useTranslation("common");
  const [open, setOpen] = useState(false);
  const { jobs, loading, loadFailed, now } = useSessionAutomationJobs(open, token, sessionKey);
  const automationContent = loading ? (
    <div className="flex items-center gap-2 rounded-floating bg-muted/45 px-3 py-3 text-[12.5px] text-muted-foreground">
      <RefreshCcw className="h-3.5 w-3.5 animate-spin" />
      {t("thread.sessionInfo.loading")}
    </div>
  ) : loadFailed ? (
    <div className="flex items-center gap-2 rounded-floating bg-destructive/10 px-3 py-3 text-[12.5px] text-destructive">
      <CircleAlert className="h-3.5 w-3.5" />
      {t("thread.sessionInfo.loadFailed")}
    </div>
  ) : jobs.length ? (
    <div className="space-y-1.5">
      {jobs.map((job) => (
        <AutomationRow key={job.id} job={job} now={now} />
      ))}
    </div>
  ) : (
    <div className="rounded-floating bg-muted/35 px-3 py-3 text-[12.5px] leading-relaxed text-muted-foreground">
      {t("thread.sessionInfo.empty")}
    </div>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("thread.header.sessionInfo")}
          aria-pressed={open}
          className={cn(
            "host-no-drag h-8 w-8 rounded-full text-muted-foreground/85",
            "hover:bg-accent/40 hover:text-foreground",
            open && "bg-accent/45 text-foreground",
          )}
        >
          <ListTodo className="h-4 w-4 stroke-[1.75]" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={8}
        data-testid="session-summary-panel"
        onPointerDownOutside={(event) => event.preventDefault()}
        onInteractOutside={(event) => event.preventDefault()}
        onFocusOutside={(event) => event.preventDefault()}
        className="w-[min(23rem,calc(100vw-1.5rem))] overflow-hidden rounded-2xl border border-border/75 bg-popover/98 p-0 shadow-[0_12px_36px_rgba(15,23,42,0.16)] dark:border-white/10 dark:shadow-[0_14px_40px_rgba(0,0,0,0.45)]"
      >
        <div className="flex items-start justify-between gap-2 border-b border-border/50 px-4 py-3">
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 text-[11.5px] font-medium text-muted-foreground/80">
              <ListTodo className="h-3.5 w-3.5 shrink-0" aria-hidden />
              <span>{t("thread.sessionInfo.title")}</span>
            </div>
            <div className="mt-0.5 truncate text-[13.5px] font-semibold text-foreground">
              {title || t("thread.sessionInfo.untitled")}
            </div>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t("common.close", { defaultValue: "Close" })}
            onClick={() => setOpen(false)}
            className="-mr-1.5 -mt-0.5 h-7 w-7 shrink-0 rounded-full text-muted-foreground/80 hover:bg-accent/40 hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        </div>

        <div className="scrollbar-thin scrollbar-track-transparent max-h-[min(65vh,28rem)] space-y-3 overflow-y-auto px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2">
              <CalendarClock className="h-3.5 w-3.5 shrink-0 text-muted-foreground/80" />
              <span className="truncate text-[13px] font-medium text-foreground">
                {t("thread.sessionInfo.automations")}
              </span>
            </div>
            <span className="rounded-full bg-muted/70 px-2 py-0.5 text-[11px] text-muted-foreground">
              {t("thread.sessionInfo.count", { count: jobs.length })}
            </span>
          </div>

          {automationContent}
        </div>
      </PopoverContent>
    </Popover>
  );
}

function AutomationRow({ job, now }: { job: SessionAutomationJob; now: number }) {
  const { t } = useTranslation("common");
  const schedule = formatSchedule(job, t);
  const nextRun = formatNextRun(job, t, now);
  const statusClass = job.enabled
    ? job.state.last_status === "error"
      ? "bg-destructive"
      : "bg-emerald-500"
    : "bg-muted-foreground/35";

  return (
    <div className="rounded-floating px-3 py-2.5 transition-colors hover:bg-muted/40">
      <div className="flex items-start gap-2.5">
        <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", statusClass)} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate text-[13px] font-medium text-foreground">{job.name}</span>
            {!job.enabled ? (
              <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-[10.5px] text-muted-foreground">
                {t("thread.sessionInfo.disabled")}
              </span>
            ) : null}
          </div>
          <div className="mt-1 line-clamp-2 text-[12px] leading-snug text-muted-foreground">
            {job.payload.message}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11.5px] text-muted-foreground/80">
            <span>{schedule}</span>
            <span aria-hidden>·</span>
            <span title={nextRun.title}>{nextRun.label}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatSchedule(job: SessionAutomationJob, t: TFunction) {
  const locale = currentLocale();
  if (isLocalTriggerAutomation(job)) {
    return t("thread.sessionInfo.schedule.local");
  }
  if (job.schedule.kind === "at" && job.schedule.at_ms) {
    return t("thread.sessionInfo.schedule.at", { time: fmtDateTime(job.schedule.at_ms, locale) });
  }
  if (job.schedule.kind === "every" && job.schedule.every_ms) {
    return t("thread.sessionInfo.schedule.every", {
      duration: formatDuration(job.schedule.every_ms, locale),
    });
  }
  if (job.schedule.kind === "cron" && job.schedule.expr) {
    return job.schedule.tz
      ? t("thread.sessionInfo.schedule.cronWithTz", {
          expr: job.schedule.expr,
          tz: job.schedule.tz,
        })
      : t("thread.sessionInfo.schedule.cron", { expr: job.schedule.expr });
  }
  return t("thread.sessionInfo.schedule.unknown");
}

function formatNextRun(job: SessionAutomationJob, t: TFunction, now: number) {
  const locale = currentLocale();
  if (!job.enabled) {
    return { label: t("thread.sessionInfo.next.disabled"), title: "" };
  }
  if (job.state.pending) {
    return { label: t("thread.sessionInfo.next.pending"), title: "" };
  }
  if (isLocalTriggerAutomation(job)) {
    return { label: t("thread.sessionInfo.next.local"), title: "" };
  }
  const next = job.state.next_run_at_ms;
  if (!next) {
    return { label: t("thread.sessionInfo.next.none"), title: "" };
  }
  return {
    label: t("thread.sessionInfo.next.label", { time: relativeTimeFrom(next, now, locale) }),
    title: fmtDateTime(next, locale),
  };
}

function isLocalTriggerAutomation(job: SessionAutomationJob): boolean {
  return job.kind === "local_trigger"
    || job.payload.kind === "local_trigger"
    || job.schedule.kind === "local";
}

function relativeTimeFrom(value: number, now: number, locale: string): string {
  let delta = (value - now) / 1000;
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  for (const [step, unit] of RELATIVE_THRESHOLDS) {
    if (Math.abs(delta) < step) {
      return formatter.format(Math.round(delta), unit);
    }
    delta /= step;
  }
  return formatter.format(Math.round(delta), "year");
}

function formatDuration(ms: number, locale: string): string {
  const units: Array<[Intl.NumberFormatOptions["unit"], number]> = [
    ["day", 86_400_000],
    ["hour", 3_600_000],
    ["minute", 60_000],
    ["second", 1000],
  ];
  for (const [unit, size] of units) {
    if (ms >= size && ms % size === 0) {
      return new Intl.NumberFormat(locale, {
        style: "unit",
        unit,
        unitDisplay: "long",
        maximumFractionDigits: 0,
      }).format(ms / size);
    }
  }
  return new Intl.NumberFormat(locale, {
    style: "unit",
    unit: "minute",
    unitDisplay: "long",
    maximumFractionDigits: 1,
  }).format(ms / 60_000);
}
