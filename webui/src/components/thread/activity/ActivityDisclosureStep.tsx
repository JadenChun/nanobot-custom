import { useId, useState, type CSSProperties, type ReactNode } from "react";
import { ChevronRight, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

import { ActivityStep, type ActivityStepTone } from "./ActivityStep";
import { compactActivityPath, redactActivityText, redactShellCommand } from "./activity-text";

const MAX_DETAIL_TEXT_CHARS = 2_400;
const MAX_DETAIL_ENTRIES = 12;

export interface ActivityDisclosureStepProps {
  icon?: LucideIcon;
  marker?: ReactNode;
  label: string;
  description?: ReactNode;
  ariaLabel?: string;
  active?: boolean;
  tone?: ActivityStepTone;
  sheen?: boolean;
  hasDetails: boolean;
  expanded?: boolean;
  defaultExpanded?: boolean;
  onToggle?: () => void;
  className?: string;
  stepClassName?: string;
  labelClassName?: string;
  markerClassName?: string;
  bodyClassName?: string;
  style?: CSSProperties;
  children?: ReactNode;
}

/**
 * Standardized expandable activity row:
 * - Header line: icon/marker + concise single-line action + rotating right chevron
 * - Detail panel: indented left-bordered region directly underneath, mounted only when open
 */
export function ActivityDisclosureStep({
  icon,
  marker,
  label,
  description,
  ariaLabel,
  active = false,
  tone,
  sheen = false,
  hasDetails,
  expanded: controlledExpanded,
  defaultExpanded = false,
  onToggle,
  className,
  stepClassName,
  labelClassName,
  markerClassName,
  bodyClassName,
  style,
  children,
}: ActivityDisclosureStepProps) {
  const id = useId();
  const [uncontrolledExpanded, setUncontrolledExpanded] = useState(defaultExpanded);
  const expanded = controlledExpanded ?? uncontrolledExpanded;
  const disclosureId = `${id}-disclosure`;
  const bodyId = `${id}-body`;

  const handleToggle = () => {
    if (onToggle) {
      onToggle();
      return;
    }
    setUncontrolledExpanded((current) => !current);
  };

  const renderedLabel = hasDetails ? (
    <button
      type="button"
      id={disclosureId}
      data-thread-disclosure=""
      aria-label={ariaLabel}
      aria-expanded={expanded}
      aria-controls={bodyId}
      title={label}
      onClick={handleToggle}
      className={cn(
        "group inline-flex min-w-0 max-w-full items-center gap-1 rounded-sm text-start",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      )}
    >
      <span
        data-sheen-text={sheen ? label : undefined}
        className={cn("min-w-0 truncate", sheen && "streaming-text-sheen")}
      >
        {label}
      </span>
      <ChevronRight
        aria-hidden
        className={cn(
          "h-3 w-3 shrink-0 text-muted-foreground/55 transition-transform duration-200 ease-out",
          "motion-reduce:transition-none group-hover:text-muted-foreground",
          expanded && "rotate-90",
        )}
        strokeWidth={1.8}
      />
    </button>
  ) : (
    label
  );

  return (
    <div className={cn("min-w-0", className)} style={style}>
      <ActivityStep
        icon={icon}
        marker={marker}
        active={active}
        tone={tone}
        label={renderedLabel}
        description={description}
        className={stepClassName}
        labelClassName={labelClassName}
        markerClassName={markerClassName}
        contentClassName="overflow-hidden"
      />
      {hasDetails ? (
        <div
          id={bodyId}
          role="region"
          aria-labelledby={disclosureId}
          aria-hidden={!expanded}
          hidden={!expanded}
          className={cn(
            "relative ml-[1.125rem] mt-1.5 mb-1 min-w-0 border-l border-border/45 pl-3.5 [overflow-wrap:anywhere]",
            bodyClassName,
          )}
        >
          {expanded ? children : null}
        </div>
      ) : null}
    </div>
  );
}

export interface ActivityDetailField {
  label: string;
  value: string;
  mono?: boolean;
}

export interface ActivityDetailItem {
  title: string;
  meta?: string;
  error?: string;
}

export interface ActivityDetailContent {
  command?: string;
  fields?: ActivityDetailField[];
  items?: ActivityDetailItem[];
  result?: string;
  error?: string;
}

export function hasActivityDetailContent(detail: ActivityDetailContent | null | undefined): boolean {
  if (!detail) return false;
  return Boolean(
    detail.command
    || detail.error
    || detail.result
    || (detail.fields && detail.fields.length > 0)
    || (detail.items && detail.items.length > 0),
  );
}

export function ActivityDetailPanel({
  detail,
  scrollable = true,
}: {
  detail: ActivityDetailContent;
  scrollable?: boolean;
}) {
  return (
    <div
      data-testid="activity-detail-panel"
      className={cn(
        "space-y-2 pr-1 text-[12px] leading-relaxed text-muted-foreground/85",
        scrollable && "scrollbar-thin scrollbar-track-transparent max-h-[min(14rem,32vh)] overflow-y-auto",
      )}
    >
      {detail.command ? (
        <pre className="overflow-x-auto rounded-md bg-muted/45 px-2.5 py-1.5 font-mono text-[11.5px] leading-5 text-foreground/85 whitespace-pre-wrap break-words">
          {clipDetailText(redactShellCommand(detail.command))}
        </pre>
      ) : null}
      {detail.fields && detail.fields.length > 0 ? (
        <dl className="space-y-1">
          {detail.fields.map((field, index) => (
            <div key={`${field.label}-${index}`} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
              <dt className="shrink-0 text-[11px] font-medium text-muted-foreground/70">
                {field.label}
              </dt>
              <dd
                className={cn(
                  "min-w-0 break-words text-foreground/80",
                  field.mono && "font-mono text-[11.5px]",
                )}
              >
                {field.value}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      {detail.items && detail.items.length > 0 ? (
        <ul className="space-y-1">
          {detail.items.map((item, index) => (
            <li key={`${item.title}-${index}`} className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono text-[11.5px] text-foreground/80">{item.title}</span>
                {item.meta ? (
                  <span className="text-[11px] text-muted-foreground/70">{item.meta}</span>
                ) : null}
              </div>
              {item.error ? (
                <p className="mt-0.5 text-[11.5px] text-destructive/85">
                  {clipDetailText(redactActivityText(item.error), 600)}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
      {detail.error ? (
        <pre className="overflow-x-auto rounded-md border border-destructive/25 bg-destructive/5 px-2.5 py-1.5 font-mono text-[11.5px] leading-5 text-destructive/90 whitespace-pre-wrap break-words">
          {clipDetailText(redactActivityText(detail.error), 1_200)}
        </pre>
      ) : null}
      {detail.result ? (
        <pre className="overflow-x-auto rounded-md bg-muted/35 px-2.5 py-1.5 font-mono text-[11.5px] leading-5 text-muted-foreground/90 whitespace-pre-wrap break-words">
          {clipDetailText(redactActivityText(detail.result), 1_200)}
        </pre>
      ) : null}
    </div>
  );
}

export function formatDetailFieldsFromArgs(args: unknown): ActivityDetailField[] {
  if (args === null || args === undefined) return [];
  if (typeof args === "string") {
    const trimmed = args.trim();
    if (!trimmed) return [];
    return [{ label: "args", value: clipDetailText(compactActivityPath(redactActivityText(trimmed)), 600), mono: true }];
  }
  if (Array.isArray(args)) {
    if (args.length === 0) return [];
    return [{ label: "args", value: clipDetailText(redactActivityText(safeStringify(args)), 600), mono: true }];
  }
  if (typeof args !== "object") return [];
  const entries = Object.entries(args as Record<string, unknown>)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, MAX_DETAIL_ENTRIES);
  return entries.map(([key, value]) => {
    const raw = typeof value === "string" ? value : safeStringify(value);
    const cleaned = compactActivityPath(redactActivityText(raw));
    const isPathOrCode = /path|file|dir|url|cmd|command|pattern|glob|script|workflow/i.test(key)
      || typeof value !== "string";
    return {
      label: key.replace(/_/g, " "),
      value: clipDetailText(cleaned, 400),
      mono: isPathOrCode,
    };
  });
}

export function formatDetailResult(result: unknown): string | undefined {
  if (result === null || result === undefined) return undefined;
  if (typeof result === "string") {
    const trimmed = result.trim();
    return trimmed ? clipDetailText(compactActivityPath(redactActivityText(trimmed)), 1_200) : undefined;
  }
  if (typeof result === "object") {
    const text = safeStringify(result);
    if (!text || text === "{}" || text === "[]") return undefined;
    return clipDetailText(compactActivityPath(redactActivityText(text)), 1_200);
  }
  return String(result);
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function clipDetailText(value: string, maxChars = MAX_DETAIL_TEXT_CHARS): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars).trimEnd()}…`;
}
