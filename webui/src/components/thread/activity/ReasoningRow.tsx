import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, CircleDashed } from "lucide-react";
import { useTranslation } from "react-i18next";

import { MarkdownText } from "@/components/MarkdownText";
import { cn } from "@/lib/utils";

import { ActivityStep } from "./ActivityStep";
import { compactReasoningPreview } from "./reasoning-preview";

export function ReasoningRow({
  text,
  streaming,
  className,
}: {
  text: string;
  streaming: boolean;
  className?: string;
}) {
  const { t } = useTranslation();
  const id = useId();
  const [expanded, setExpanded] = useState(false);
  const fallback = streaming
    ? t("message.reasoningStreaming", { defaultValue: "Thinking…" })
    : t("message.reasoning", { defaultValue: "Thinking" });
  const preview = compactReasoningPreview(text) || fallback;
  const hasDetails = text.trim().length > 0;
  const disclosureId = `${id}-disclosure`;
  const bodyId = `${id}-body`;
  const label = hasDetails ? (
    <button
      type="button"
      id={disclosureId}
      data-thread-disclosure=""
      aria-label={t("message.reasoningSummary", { defaultValue: "Reasoning" })}
      aria-expanded={expanded}
      aria-controls={bodyId}
      onClick={() => setExpanded((open) => !open)}
      className={cn(
        "group inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-sm text-start",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      )}
    >
      <span
        data-sheen-text={streaming ? preview : undefined}
        className={cn(
          "min-w-0 flex-1 truncate",
          streaming && "streaming-text-sheen",
        )}
      >
        {preview}
      </span>
      <ChevronDown
        aria-hidden
        className={cn(
          "h-3 w-3 shrink-0 text-muted-foreground/60 transition-transform duration-200 ease-out",
          "motion-reduce:transition-none group-hover:text-muted-foreground",
          expanded && "rotate-180",
        )}
        strokeWidth={1.8}
      />
    </button>
  ) : preview;

  return (
    <div className={cn("min-w-0", className)}>
      <ActivityStep
        marker={<ReasoningMarker streaming={streaming} />}
        active={streaming}
        tone={streaming ? "active" : "success"}
        label={label}
        labelClassName="italic text-muted-foreground/78"
        contentClassName="overflow-hidden"
      />
      {hasDetails ? (
        <div
          id={bodyId}
          role="region"
          aria-labelledby={disclosureId}
          aria-hidden={!expanded}
          hidden={!expanded}
          className="ml-[1.625rem] mt-1 min-w-0 [overflow-wrap:anywhere]"
        >
          {expanded ? (
            <MarkdownText
              className="min-w-0 text-[13px] leading-relaxed [&_pre]:whitespace-pre-wrap [&_pre_code]:whitespace-pre-wrap"
              streaming={streaming}
            >
              {text}
            </MarkdownText>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function ReasoningMarker({ streaming }: { streaming: boolean }) {
  const wasStreamingRef = useRef(streaming);
  const [justCompleted, setJustCompleted] = useState(false);

  useEffect(() => {
    if (wasStreamingRef.current && !streaming) {
      setJustCompleted(true);
      const timeout = window.setTimeout(() => setJustCompleted(false), 300);
      wasStreamingRef.current = streaming;
      return () => window.clearTimeout(timeout);
    }
    wasStreamingRef.current = streaming;
    return undefined;
  }, [streaming]);

  if (streaming) {
    return (
      <CircleDashed
        data-testid="activity-reasoning-marker"
        data-state="thinking"
        className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground/55"
        strokeWidth={1.8}
        aria-hidden
      />
    );
  }
  return (
    <span
      data-testid="activity-reasoning-marker"
      data-state="done"
      className={cn(
        "grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border border-emerald-500/28 text-emerald-500/78",
        "bg-emerald-500/[0.035] transition-[border-color,background-color,box-shadow,transform] duration-300 ease-out",
        justCompleted
          && "animate-in fade-in-0 zoom-in-75 shadow-[0_0_0_3px_rgba(16,185,129,0.10)] motion-reduce:animate-none",
      )}
      aria-hidden
    >
      <Check
        className={cn(
          "h-2.5 w-2.5 stroke-[2.4]",
          justCompleted && "animate-in fade-in-0 zoom-in-50 duration-300 motion-reduce:animate-none",
        )}
      />
    </span>
  );
}
