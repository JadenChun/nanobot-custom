import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Check, CircleDashed } from "lucide-react";
import { useTranslation } from "react-i18next";

import { MarkdownText } from "@/components/MarkdownText";
import { formatWorkDuration } from "@/components/thread/working-status";
import { cn } from "@/lib/utils";

import { ActivityDisclosureStep } from "./ActivityDisclosureStep";

const STREAM_AUTO_FOLLOW_THRESHOLD_PX = 24;

export function ReasoningRow({
  text,
  streaming,
  durationMs,
  nestedInScrollViewport = false,
  className,
}: {
  text: string;
  streaming: boolean;
  durationMs?: number;
  /** When inside the outer AgentActivityCluster scroll viewport, flow naturally
   * without a competing nested inner scrollbar. */
  nestedInScrollViewport?: boolean;
  className?: string;
}) {
  const { t } = useTranslation();
  // null = automatic (show live reasoning body while streaming, collapse to
  // a compact "Thought [for Xs]" row when done); boolean = explicit user toggle.
  const [userExpanded, setUserExpanded] = useState<boolean | null>(null);
  const streamViewportRef = useRef<HTMLDivElement | null>(null);
  const streamContentRef = useRef<HTMLDivElement | null>(null);
  const autoFollowRef = useRef(true);
  const lastScrollTopRef = useRef(0);
  const [fadeTop, setFadeTop] = useState(false);

  const trimmed = text.trim();
  const hasDetails = trimmed.length > 0;
  const shouldAutoExpand = streaming && hasDetails;
  const expanded = userExpanded ?? shouldAutoExpand;

  const hasDuration = typeof durationMs === "number" && Number.isFinite(durationMs) && durationMs > 0;
  const summaryLabel = streaming
    ? t("message.reasoningStreaming", { defaultValue: "Thinking…" })
    : hasDuration
      ? t("message.activityThoughtFor", {
          duration: formatWorkDuration(durationMs),
          defaultValue: "Thought for {{duration}}",
        })
      : t("message.activityThought", { defaultValue: "Thought" });

  const syncFadeTop = useCallback((el: HTMLDivElement) => {
    const nextFadeTop = el.scrollTop > 4;
    setFadeTop((current) => (current === nextFadeTop ? current : nextFadeTop));
  }, []);

  const scrollStreamToBottom = useCallback(() => {
    const el = streamViewportRef.current;
    if (!el) return;
    el.scrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
    lastScrollTopRef.current = el.scrollTop;
    syncFadeTop(el);
  }, [syncFadeTop]);

  const syncScrollState = useCallback(() => {
    const el = streamViewportRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    // Only disengage auto-follow when the user actually scrolls upward.
    // Content growth or a delayed scroll event after a programmatic scroll
    // increases scrollHeight without decreasing scrollTop and must not detach.
    const scrolledUp = el.scrollTop < lastScrollTopRef.current - 1;
    if (distanceFromBottom <= STREAM_AUTO_FOLLOW_THRESHOLD_PX) {
      autoFollowRef.current = true;
    } else if (scrolledUp) {
      autoFollowRef.current = false;
    }
    lastScrollTopRef.current = el.scrollTop;
    syncFadeTop(el);
  }, [syncFadeTop]);

  useEffect(() => {
    if (streaming && expanded) {
      autoFollowRef.current = true;
    } else if (!expanded) {
      lastScrollTopRef.current = 0;
      setFadeTop(false);
    }
  }, [expanded, streaming]);

  useLayoutEffect(() => {
    if (nestedInScrollViewport || !streaming || !expanded) return;
    const el = streamViewportRef.current;
    if (!el) return;
    if (autoFollowRef.current) {
      scrollStreamToBottom();
    } else {
      syncFadeTop(el);
    }
  }, [expanded, nestedInScrollViewport, scrollStreamToBottom, streaming, syncFadeTop, text]);

  useEffect(() => {
    if (nestedInScrollViewport || !streaming || !expanded) return undefined;
    const target = streamContentRef.current;
    if (!target || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      if (autoFollowRef.current) {
        scrollStreamToBottom();
      } else if (streamViewportRef.current) {
        syncFadeTop(streamViewportRef.current);
      }
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [expanded, nestedInScrollViewport, scrollStreamToBottom, streaming, syncFadeTop]);

  return (
    <ActivityDisclosureStep
      marker={<ReasoningMarker streaming={streaming} />}
      active={streaming}
      tone={streaming ? "active" : "success"}
      label={summaryLabel}
      ariaLabel={t("message.reasoningSummary", { defaultValue: "Reasoning" })}
      sheen={streaming}
      hasDetails={hasDetails}
      expanded={expanded}
      onToggle={() => setUserExpanded((current) => !(current ?? shouldAutoExpand))}
      className={className}
      labelClassName="text-muted-foreground/75"
    >
      <div
        ref={nestedInScrollViewport ? undefined : streamViewportRef}
        data-testid={streaming ? "reasoning-live-stream" : "reasoning-body-scroll"}
        onScroll={nestedInScrollViewport ? undefined : syncScrollState}
        className={cn(
          "pr-1",
          streaming && "text-muted-foreground/85",
          !nestedInScrollViewport && "scrollbar-thin scrollbar-track-transparent overflow-y-auto",
          !nestedInScrollViewport && (
            streaming
              ? "max-h-[min(14rem,32vh)]"
              : "max-h-[min(24rem,50vh)]"
          ),
        )}
      >
        <div ref={streamContentRef} className="min-w-0">
          <MarkdownText
            className="min-w-0 text-[12.5px] leading-relaxed [&_pre]:whitespace-pre-wrap [&_pre_code]:whitespace-pre-wrap"
            streaming={streaming}
          >
            {text}
          </MarkdownText>
        </div>
      </div>
      {!nestedInScrollViewport && fadeTop ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 z-10 ml-3 h-4 bg-gradient-to-b from-background to-transparent"
        />
      ) : null}
    </ActivityDisclosureStep>
  );
}

function ReasoningMarker({ streaming }: { streaming: boolean }) {
  return (
    <span
      data-testid="activity-reasoning-marker"
      data-state={streaming ? "thinking" : "done"}
      className={cn(
        "grid h-3.5 w-3.5 place-items-center rounded-full border bg-background transition-colors",
        streaming
          ? "border-muted-foreground/28 text-muted-foreground/72"
          : "animate-in fade-in-0 zoom-in-75 duration-200 motion-reduce:animate-none border-emerald-500/28 text-emerald-500/78",
      )}
    >
      {streaming ? (
        <CircleDashed className="h-2.5 w-2.5 animate-spin motion-reduce:animate-none" strokeWidth={2.2} />
      ) : (
        <Check className="h-2.5 w-2.5" strokeWidth={2.35} />
      )}
    </span>
  );
}
