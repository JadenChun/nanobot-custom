import {
  Fragment,
  memo,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";

import { MarkdownText } from "@/components/MarkdownText";
import { floatingSurfaceElevationClassName } from "@/components/ui/floating-surface";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";
import type {
  PendingPromptReveal,
  SessionPromptSummary,
  SessionTurnSummary,
  UIMessage,
} from "@/lib/types";
import type { RevealPromptTarget } from "@/hooks/useSessions";
import {
  type PromptAnchor,
  userPromptAnchors,
} from "@/components/thread/promptNavigation";

interface PromptRailProps {
  bottomOffset: number;
  messages: UIMessage[];
  onJumpToPrompt: (promptId: string) => void;
  scrollRef: RefObject<HTMLDivElement>;
  /**
   * Bounded visible user prompt summaries, when loaded. Preferred over legacy
   * turn summaries because a turn can contain multiple user inputs (e.g. Steer).
   */
  sessionPrompts?: SessionPromptSummary[];
  totalPromptCount?: number;
  hasMoreBefore?: boolean;
  loadingOlder?: boolean;
  onLoadOlder?: () => Promise<void> | void;
  hasMoreAfter?: boolean;
  hasMorePromptsBefore?: boolean;
  loadingMorePrompts?: boolean;
  onLoadMorePrompts?: () => void;
  onRevealPrompt?: (target: RevealPromptTarget) => void;
  pendingReveal?: PendingPromptReveal | null;
  /**
   * Legacy turn summaries retained for backwards compatibility when a gateway
   * does not emit ``prompts``.
   */
  sessionTurns?: SessionTurnSummary[];
  totalTurnCount?: number;
  onNeedSessionTurns?: () => void;
  onJumpToTurn?: (cursor: string) => void;
}

interface MeasuredPrompt extends PromptAnchor {
  top: number;
  topPercent: number;
}

interface PromptMarker {
  /** Stable unique React key for both loaded and unloaded markers. */
  key: string;
  answerPreview: string;
  count: number;
  /** Rendered row IDs covered by this marker (empty when unloaded). */
  ids: string[];
  /** Primary rendered row ID when the target prompt is already mounted. */
  renderedId?: string;
  /** Stable server prompt id for selective reveal. */
  promptId?: string;
  /** Opaque reveal locator bound to the session history origin. */
  locator?: string;
  /** Legacy page cursor for a turn outside the rendered window. */
  cursor?: string;
  label: string;
  preview: string;
  topPercent: number;
}

const MIN_PROMPTS_FOR_RAIL = 3;
const RAIL_MIN_SCROLL_RANGE_PX = 80;
const DENSE_PROMPT_THRESHOLD = 30;
const DENSE_BUCKET_HEIGHT_PX = 12;
const DENSE_BUCKET_FALLBACK_COUNT = 32;
const DENSE_BUCKET_MAX_COUNT = 42;
const MARKER_MIN_GAP_PX = 9;
const MARKER_BASE_WIDTH_PX = 9;
const MARKER_STACK_GAP_PX = 16;
const RAIL_FALLBACK_HEIGHT_PX = 300;
const RESIZE_MEASURE_THROTTLE_MS = 150;
const HOVER_MARKER_WIDTHS_PX = [28, 22, 16, 11];

export const PromptRail = memo(function PromptRail({
  bottomOffset,
  messages,
  onJumpToPrompt,
  scrollRef,
  sessionPrompts = [],
  totalPromptCount = 0,
  hasMoreBefore = false,
  loadingOlder = false,
  onLoadOlder,
  hasMoreAfter = false,
  hasMorePromptsBefore = false,
  loadingMorePrompts = false,
  onLoadMorePrompts,
  onRevealPrompt,
  pendingReveal = null,
  sessionTurns = [],
  totalTurnCount = 0,
  onNeedSessionTurns,
  onJumpToTurn,
}: PromptRailProps) {
  const { t } = useTranslation();
  const enabled = useMediaQuery("(min-width: 768px)");
  const railRef = useRef<HTMLDivElement>(null);
  const measuredPromptsRef = useRef<MeasuredPrompt[]>([]);
  // The rail only needs to re-anchor when the prompt set (or streaming phase)
  // changes. Assistant tokens stream every frame and must not trigger a full
  // prompt re-measurement.
  const promptRevision = useMemo(() => {
    const parts: string[] = [];
    let userCount = 0;
    let assistantCount = 0;
    for (const message of messages) {
      if (message.role === "user") {
        parts.push(`${message.id}:${message.promptId ?? ""}:${message.content.length}`);
        userCount += 1;
      } else if (message.role === "assistant" && message.kind !== "trace") {
        assistantCount += 1;
      }
    }
    parts.push(messages.some((message) => message.isStreaming) ? "streaming" : "idle");
    parts.push(`users:${userCount}`, `assistants:${assistantCount}`);
    return parts.join("|");
  }, [messages]);
  const promptAnchors = useMemo(
    () => userPromptAnchors(messages),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [promptRevision],
  );
  const effectiveTotalPrompts = Math.max(
    promptAnchors.length,
    totalPromptCount,
    totalTurnCount,
    sessionPrompts.length,
    sessionTurns.length,
  );
  const hasUnloadedHistory = (
    effectiveTotalPrompts > promptAnchors.length
    || hasMoreBefore
    || hasMoreAfter
  );
  const [markers, setMarkers] = useState<PromptMarker[]>([]);
  const [activePromptId, setActivePromptId] = useState<string | null>(null);
  const [focusedMarkerIndex, setFocusedMarkerIndex] = useState<number | null>(null);

  const updateMarkers = useCallback(() => {
    const scrollEl = scrollRef.current;
    const nextRailHeight = railRef.current?.clientHeight ?? 0;

    if (!enabled || !scrollEl || (effectiveTotalPrompts < MIN_PROMPTS_FOR_RAIL && !hasMoreBefore)) {
      measuredPromptsRef.current = [];
      setMarkers((current) => current.length === 0 ? current : []);
      setActivePromptId(null);
      return;
    }

    const scrollRange = scrollEl.scrollHeight - scrollEl.clientHeight;
    if (scrollRange < RAIL_MIN_SCROLL_RANGE_PX && !hasUnloadedHistory) {
      measuredPromptsRef.current = [];
      setMarkers((current) => current.length === 0 ? current : []);
      setActivePromptId(null);
      return;
    }

    const measured = measurePrompts(scrollEl, promptAnchors, Math.max(1, scrollRange));
    measuredPromptsRef.current = measured;

    const canLoadOlderViaIndex = hasMorePromptsBefore && Boolean(onLoadMorePrompts);
    const canLoadOlderDirectly =
      !canLoadOlderViaIndex
      && hasMoreBefore
      && sessionPrompts.length === 0
      && sessionTurns.length === 0
      && Boolean(onLoadOlder);
    const topInsetPx = canLoadOlderViaIndex || canLoadOlderDirectly ? 28 : 0;

    const nextActive = activePromptForScroll(measured, scrollEl.scrollTop, promptAnchors);

    // Prefer per-prompt summaries so multi-prompt turns (e.g. Steer) get
    // distinct, identity-bound markers. Bucket dense ranges so long histories
    // never render one DOM node per historical prompt.
    if (sessionPrompts.length > 0) {
      const rawMarkers = sessionPromptSummaryMarkers(sessionPrompts, promptAnchors);
      const bounded = bucketSummaryMarkers(rawMarkers, nextRailHeight);
      const nextMarkers = distributeMarkerPositions(bounded, nextRailHeight, topInsetPx);
      setMarkers((current) => promptMarkersEqual(current, nextMarkers) ? current : nextMarkers);
      setActivePromptId((current) => nextActive ?? current);
      return;
    }

    // Legacy fallback: turn summaries when a gateway does not return prompts.
    if (sessionTurns.length > 0) {
      const rendered = new Map<number, PromptAnchor>();
      for (const anchor of promptAnchors) {
        if (anchor.ordinal !== undefined) rendered.set(anchor.ordinal, anchor);
      }
      const rawMarkers = sessionPromptMarkers(sessionTurns, rendered);
      const bounded = bucketSummaryMarkers(rawMarkers, nextRailHeight);
      const nextMarkers = distributeMarkerPositions(bounded, nextRailHeight, topInsetPx);
      setMarkers((current) => promptMarkersEqual(current, nextMarkers) ? current : nextMarkers);
      setActivePromptId((current) => nextActive ?? current);
      return;
    }

    // When ThreadMessages is virtualized, only viewport+overscan prompts are
    // mounted in the DOM. Build rail markers from all loaded promptAnchors so
    // the rail never truncates to the visible slice or jumps during scroll.
    const isVirtualized =
      measured.length < promptAnchors.length
      && Boolean(scrollEl.querySelector("[data-testid='thread-virtual-list']"));
    const grouped = isVirtualized
      ? bucketSummaryMarkers(localPromptAnchorMarkers(promptAnchors), nextRailHeight)
      : groupPromptMarkers(measured, nextRailHeight);
    const nextMarkers = distributeMarkerPositions(grouped, nextRailHeight, topInsetPx);
    setMarkers((current) => promptMarkersEqual(current, nextMarkers) ? current : nextMarkers);
    setActivePromptId((current) => nextActive ?? current);
  }, [
    effectiveTotalPrompts,
    enabled,
    hasMoreBefore,
    hasMorePromptsBefore,
    hasUnloadedHistory,
    onLoadMorePrompts,
    onLoadOlder,
    promptAnchors,
    scrollRef,
    sessionPrompts,
    sessionTurns,
  ]);

  const updateMarkersRef = useRef(updateMarkers);
  updateMarkersRef.current = updateMarkers;
  const promptAnchorsRef = useRef(promptAnchors);
  promptAnchorsRef.current = promptAnchors;

  const updateActivePrompt = useCallback(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;
    const anchors = promptAnchorsRef.current;
    const isVirtualized = Boolean(scrollEl.querySelector("[data-testid='thread-virtual-list']"));
    const measured = isVirtualized
      ? measurePrompts(
        scrollEl,
        anchors,
        Math.max(1, scrollEl.scrollHeight - scrollEl.clientHeight),
      )
      : measuredPromptsRef.current;
    if (isVirtualized) {
      measuredPromptsRef.current = measured;
    }
    const next = activePromptForScroll(measured, scrollEl.scrollTop, anchors);
    if (next === null && isVirtualized) return;
    setActivePromptId((current) => current === next ? current : next);
  }, [scrollRef]);

  useEffect(() => {
    if (!enabled) {
      measuredPromptsRef.current = [];
      setMarkers((current) => current.length === 0 ? current : []);
      setActivePromptId(null);
      return;
    }
    updateMarkers();
  }, [bottomOffset, enabled, updateMarkers]);

  useEffect(() => {
    if (!enabled) return undefined;
    const scrollEl = scrollRef.current;
    if (!scrollEl) return undefined;

    let scrollFrame = 0;
    let resizeFrame = 0;
    const scheduleActivePrompt = () => {
      window.cancelAnimationFrame(scrollFrame);
      scrollFrame = window.requestAnimationFrame(updateActivePrompt);
    };
    const scheduleMeasurement = () => {
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => updateMarkersRef.current());
    };

    scrollEl.addEventListener("scroll", scheduleActivePrompt, { passive: true });
    window.addEventListener("resize", scheduleMeasurement);
    return () => {
      window.cancelAnimationFrame(scrollFrame);
      window.cancelAnimationFrame(resizeFrame);
      scrollEl.removeEventListener("scroll", scheduleActivePrompt);
      window.removeEventListener("resize", scheduleMeasurement);
    };
  }, [enabled, scrollRef, updateActivePrompt]);

  const railMounted = markers.length > 0;
  const hasWholeSessionSummaries = sessionPrompts.length > 0 || sessionTurns.length > 0;
  useEffect(() => {
    if (!enabled) return undefined;
    const scrollEl = scrollRef.current;
    if (!scrollEl || typeof ResizeObserver === "undefined") return undefined;
    let frame = 0;
    let timer = 0;
    let lastMeasuredAt = 0;
    const runMeasurement = () => {
      frame = 0;
      timer = 0;
      lastMeasuredAt = Date.now();
      updateMarkersRef.current();
    };
    const scheduleObservedMeasurement = () => {
      if (frame !== 0 || timer !== 0) return;
      const elapsed = Date.now() - lastMeasuredAt;
      if (elapsed >= RESIZE_MEASURE_THROTTLE_MS) {
        frame = window.requestAnimationFrame(runMeasurement);
        return;
      }
      timer = window.setTimeout(() => {
        timer = 0;
        frame = window.requestAnimationFrame(runMeasurement);
      }, RESIZE_MEASURE_THROTTLE_MS - elapsed);
    };
    const observer = new ResizeObserver(scheduleObservedMeasurement);
    observer.observe(scrollEl);
    // When whole-session prompt/turn summaries are already loaded, rail marker
    // positions depend only on rail height and summary count, not on content
    // height fluctuations from virtualized item measurement.
    if (!hasWholeSessionSummaries && scrollEl.firstElementChild) {
      observer.observe(scrollEl.firstElementChild);
    }
    if (railRef.current) observer.observe(railRef.current);
    return () => {
      if (frame !== 0) window.cancelAnimationFrame(frame);
      if (timer !== 0) window.clearTimeout(timer);
      observer.disconnect();
    };
  }, [enabled, hasWholeSessionSummaries, railMounted, scrollRef]);

  // Request the session prompt index immediately when older prompts exist
  // outside the loaded window so the whole-session rail populates right away.
  useEffect(() => {
    if (!enabled || !onNeedSessionTurns) return;
    if (sessionPrompts.length > 0 || sessionTurns.length > 0) return;
    if (effectiveTotalPrompts <= promptAnchors.length && !hasMoreBefore) return;
    onNeedSessionTurns();
  }, [
    effectiveTotalPrompts,
    enabled,
    hasMoreBefore,
    onNeedSessionTurns,
    promptAnchors.length,
    sessionPrompts.length,
    sessionTurns.length,
  ]);

  const waitingForInitialIndex =
    hasMoreBefore
    && loadingMorePrompts
    && sessionPrompts.length === 0
    && sessionTurns.length === 0;
  if (markers.length === 0 || waitingForInitialIndex) return null;

  const showLoadOlderFromIndex = hasMorePromptsBefore && Boolean(onLoadMorePrompts);
  const showLoadOlderTranscript =
    !showLoadOlderFromIndex
    && hasMoreBefore
    && sessionPrompts.length === 0
    && sessionTurns.length === 0
    && Boolean(onLoadOlder);
  const showLoadOlderButton = showLoadOlderFromIndex || showLoadOlderTranscript;
  const loadOlderButtonDisabled = showLoadOlderFromIndex ? loadingMorePrompts : loadingOlder;

  return (
    <div
      ref={railRef}
      aria-label={t("thread.promptNavigator.railAria")}
      className={cn(
        "thread-prompt-rail group pointer-events-auto absolute top-3 z-20 w-9 opacity-100",
        "transition-opacity duration-200",
        "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
      )}
      onPointerLeave={() => setFocusedMarkerIndex(null)}
      style={{ bottom: Math.max(80, bottomOffset) }}
    >
      {showLoadOlderButton ? (
        <button
          type="button"
          data-testid="prompt-rail-load-older"
          disabled={loadOlderButtonDisabled}
          aria-label={t("thread.promptNavigator.loadOlder", {
            defaultValue: "Load older prompts",
          })}
          onClick={() => {
            if (showLoadOlderFromIndex) {
              onLoadMorePrompts?.();
            } else {
              void onLoadOlder?.();
            }
          }}
          className={cn(
            "absolute left-0 top-0 z-30 flex h-5 w-7 items-center justify-center rounded-sm text-[11px] font-medium",
            "text-muted-foreground/80 hover:bg-accent/50 hover:text-foreground disabled:opacity-40",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/60",
          )}
        >
          …
        </button>
      ) : null}
      {markers.map((marker, index) => {
        const active = marker.ids.includes(activePromptId ?? "");
        const revealing = Boolean(
          pendingReveal?.status === "loading"
          && (
            (pendingReveal.promptId && marker.promptId === pendingReveal.promptId)
            || (pendingReveal.locator && marker.locator === pendingReveal.locator)
            || (pendingReveal.cursor && marker.cursor === pendingReveal.cursor)
          ),
        );
        const previewVisible = focusedMarkerIndex === index;
        const hoverDistance =
          focusedMarkerIndex === null ? null : Math.abs(index - focusedMarkerIndex);
        return (
          <Fragment key={marker.key}>
            <button
              type="button"
              aria-label={t("thread.promptNavigator.jumpTo", { label: marker.label })}
              aria-busy={revealing || undefined}
              onClick={() => {
                if (marker.renderedId) {
                  onJumpToPrompt(marker.renderedId);
                  return;
                }
                if (onRevealPrompt && (marker.locator || marker.promptId)) {
                  onRevealPrompt({
                    promptId: marker.promptId ?? "",
                    locator: marker.locator ?? null,
                    cursor: marker.cursor ?? null,
                  });
                  return;
                }
                if (marker.cursor && onJumpToTurn) {
                  onJumpToTurn(marker.cursor);
                  return;
                }
                if (marker.cursor && onRevealPrompt) {
                  onRevealPrompt({
                    promptId: marker.promptId ?? "",
                    locator: null,
                    cursor: marker.cursor,
                  });
                  return;
                }
                const fallbackRenderedId = marker.ids[marker.ids.length - 1];
                if (fallbackRenderedId) {
                  onJumpToPrompt(fallbackRenderedId);
                }
              }}
              onBlur={() => setFocusedMarkerIndex(null)}
              onFocus={() => setFocusedMarkerIndex(index)}
              onPointerEnter={() => setFocusedMarkerIndex(index)}
              onPointerLeave={() => setFocusedMarkerIndex(null)}
              className={cn(
                "absolute left-0 h-4 w-9 -translate-y-1/2 overflow-visible rounded-sm",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/60",
              )}
              style={{ top: `${marker.topPercent}%` }}
            >
              <span
                aria-hidden
                data-testid="prompt-rail-marker"
                data-revealing={revealing ? "true" : undefined}
                className={cn(
                  "absolute left-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full",
                  "transition-[width,background-color,opacity,height] duration-150",
                  revealing
                    ? "animate-pulse bg-primary"
                    : railMarkerTone(hoverDistance, active),
                )}
                style={{
                  height: markerHeight(hoverDistance),
                  width: markerWidth(hoverDistance),
                }}
              />
            </button>
            <div
              ref={makeInert}
              aria-hidden
              data-testid={previewVisible ? "prompt-rail-preview" : undefined}
              className={cn(
                "pointer-events-none absolute left-10 z-30 w-[34rem] max-w-[calc(100vw-4rem)] -translate-y-1/2 rounded-panel px-4 py-3 text-left",
                floatingSurfaceElevationClassName,
                "transition-[opacity,transform] duration-150",
                previewVisible
                  ? "translate-x-0 scale-100 opacity-100"
                  : "-translate-x-2 scale-[0.98] opacity-0",
              )}
              style={{ top: `${clamp(marker.topPercent, 14, 86)}%` }}
            >
              {previewVisible ? (
                <>
                  <div className="line-clamp-2 whitespace-pre-wrap break-words text-[15px] font-semibold leading-6">
                    {marker.preview}
                  </div>
                  {marker.answerPreview ? (
                    <div className="mt-1.5 max-h-[4.5rem] overflow-hidden break-words text-[14px] leading-6 text-muted-foreground dark:text-white/55">
                      <MarkdownText
                        className={cn(
                          "max-w-none text-[14px] leading-6 text-inherit",
                          "[--tw-prose-body:currentColor] [--tw-prose-headings:currentColor] [--tw-prose-bold:currentColor]",
                          "prose-headings:my-0 prose-h1:text-[14px] prose-h2:text-[14px] prose-h3:text-[14px] prose-h4:text-[14px]",
                          "prose-p:my-0 prose-ul:my-0 prose-ol:my-0 prose-li:my-0",
                        )}
                      >
                        {marker.answerPreview}
                      </MarkdownText>
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          </Fragment>
        );
      })}
    </div>
  );
});

function makeInert(node: HTMLDivElement | null): void {
  if (node) node.inert = true;
}

function measurePrompts(
  scrollEl: HTMLElement,
  anchors: PromptAnchor[],
  scrollRange: number,
): MeasuredPrompt[] {
  // One DOM scan for all anchors. Looking each id up separately makes this
  // O(prompts²) and runs on every measurement during streaming.
  const elements = new Map<string, HTMLElement>();
  for (const candidate of scrollEl.querySelectorAll<HTMLElement>("[data-user-prompt-id]")) {
    const id = candidate.dataset.userPromptId;
    if (id) elements.set(id, candidate);
  }
  if (elements.size === 0) return [];

  const scrollRect = scrollEl.getBoundingClientRect();
  const scrollTop = scrollEl.scrollTop;
  return anchors.flatMap((anchor) => {
    const target = elements.get(anchor.id);
    if (!target) return [];
    const targetRect = target.getBoundingClientRect();
    const top = scrollRect.top !== 0 || targetRect.top !== 0
      ? targetRect.top - scrollRect.top + scrollTop
      : target.offsetTop;
    const clamped = Math.max(0, Math.min(scrollRange, top - 16));
    return [{
      ...anchor,
      top: clamped,
      topPercent: clamp((clamped / scrollRange) * 100, 2, 98),
    }];
  });
}

function promptMarkersEqual(
  previous: PromptMarker[],
  next: PromptMarker[],
): boolean {
  if (previous === next) return true;
  if (previous.length !== next.length) return false;
  for (let index = 0; index < previous.length; index += 1) {
    const a = previous[index];
    const b = next[index];
    if (
      a.key !== b.key
      || a.topPercent !== b.topPercent
      || a.count !== b.count
      || a.label !== b.label
      || a.preview !== b.preview
      || a.answerPreview !== b.answerPreview
      || a.cursor !== b.cursor
      || a.promptId !== b.promptId
      || a.locator !== b.locator
      || a.renderedId !== b.renderedId
      || a.ids.length !== b.ids.length
      || a.ids.some((id, idIndex) => id !== b.ids[idIndex])
    ) {
      return false;
    }
  }
  return true;
}

function sessionPromptSummaryMarkers(
  sessionPrompts: SessionPromptSummary[],
  promptAnchors: PromptAnchor[],
): PromptMarker[] {
  const byPromptId = new Map<string, PromptAnchor>();
  const byOrdinal = new Map<number, PromptAnchor[]>();
  for (const anchor of promptAnchors) {
    if (anchor.promptId) byPromptId.set(anchor.promptId, anchor);
    byPromptId.set(anchor.id, anchor);
    if (anchor.ordinal !== undefined) {
      const list = byOrdinal.get(anchor.ordinal) ?? [];
      list.push(anchor);
      byOrdinal.set(anchor.ordinal, list);
    }
  }
  const countsByTurn = new Map<number, number>();
  for (const entry of sessionPrompts) {
    countsByTurn.set(entry.turn_ordinal, (countsByTurn.get(entry.turn_ordinal) ?? 0) + 1);
  }
  const matchedAnchorIds = new Set<string>();
  const markers: PromptMarker[] = sessionPrompts.map((entry) => {
    let rendered = byPromptId.get(entry.prompt_id);
    if (!rendered && (countsByTurn.get(entry.turn_ordinal) ?? 0) === 1) {
      const candidates = byOrdinal.get(entry.turn_ordinal);
      if (candidates && candidates.length === 1 && !candidates[0].promptId) {
        rendered = candidates[0];
      }
    }
    if (rendered) matchedAnchorIds.add(rendered.id);
    return {
      key: `prompt:${entry.prompt_id}`,
      answerPreview: rendered?.answerPreview ?? "",
      count: 1,
      ids: rendered ? [rendered.id] : [],
      renderedId: rendered?.id,
      promptId: entry.prompt_id,
      locator: entry.locator || undefined,
      label: rendered?.label ?? entry.label,
      preview: rendered?.preview ?? entry.label,
      topPercent: 0,
    };
  });
  for (const anchor of promptAnchors) {
    if (matchedAnchorIds.has(anchor.id)) continue;
    markers.push({
      key: `local:${anchor.promptId ?? anchor.id}`,
      answerPreview: anchor.answerPreview,
      count: 1,
      ids: [anchor.id],
      renderedId: anchor.id,
      promptId: anchor.promptId ?? anchor.id,
      label: anchor.label,
      preview: anchor.preview,
      topPercent: 0,
    });
  }
  return markers;
}

function bucketSummaryMarkers(
  markers: PromptMarker[],
  railHeight: number,
): PromptMarker[] {
  if (markers.length < DENSE_PROMPT_THRESHOLD) return markers;
  const bucketCount = railHeight > 0
    ? clamp(
      Math.floor(railHeight / DENSE_BUCKET_HEIGHT_PX),
      1,
      DENSE_BUCKET_MAX_COUNT,
    )
    : DENSE_BUCKET_FALLBACK_COUNT;
  const buckets = Array.from({ length: bucketCount }, () => [] as PromptMarker[]);
  for (let index = 0; index < markers.length; index += 1) {
    const bucketIndex = clamp(
      Math.floor((index / markers.length) * bucketCount),
      0,
      bucketCount - 1,
    );
    buckets[bucketIndex].push(markers[index]);
  }
  return buckets.flatMap((bucket, bucketIndex) => {
    if (bucket.length === 0) return [];
    const representative = bucket[bucket.length - 1];
    const allIds = bucket.flatMap((entry) => entry.ids);
    return [{
      key: `bucket:${bucketIndex}:${bucket[0].key}:${representative.key}`,
      count: bucket.length,
      ids: allIds,
      renderedId: representative.renderedId,
      promptId: representative.promptId,
      locator: representative.locator,
      cursor: representative.cursor,
      label: bucket.length === 1
        ? representative.label
        : groupedPromptLabel(bucket.length, representative.label),
      answerPreview: representative.answerPreview,
      preview: representative.preview,
      topPercent: 0,
    }];
  });
}

/**
 * Build one marker per turn of the whole session from the legacy turn index.
 */
function sessionPromptMarkers(
  sessionTurns: SessionTurnSummary[],
  renderedOrdinals: Map<number, PromptAnchor>,
): PromptMarker[] {
  return sessionTurns.map((turn) => {
    const rendered = renderedOrdinals.get(turn.ordinal);
    return {
      key: `turn:${turn.ordinal}:${turn.cursor}`,
      answerPreview: rendered?.answerPreview ?? "",
      count: 1,
      cursor: turn.cursor,
      ids: rendered ? [rendered.id] : [],
      renderedId: rendered?.id,
      label: rendered?.label ?? turn.label,
      preview: rendered?.preview ?? turn.label,
      topPercent: 0,
    };
  });
}


function groupPromptMarkers(
  measured: MeasuredPrompt[],
  railHeight: number,
): PromptMarker[] {
  if (measured.length === 0) return [];
  if (measured.length >= DENSE_PROMPT_THRESHOLD) {
    return bucketPromptMarkers(measured, railHeight);
  }

  const minGapPercent = railHeight > 0
    ? (MARKER_MIN_GAP_PX / railHeight) * 100
    : 2;
  const groups: PromptMarker[] = [];

  for (const prompt of measured) {
    const last = groups[groups.length - 1];
    if (last && prompt.topPercent - last.topPercent < minGapPercent) {
      last.count += 1;
      last.ids.push(prompt.id);
      last.renderedId = prompt.id;
      last.promptId = prompt.promptId ?? prompt.id;
      last.key = `local:${last.ids.join("|")}`;
      last.label = groupedPromptLabel(last.count, prompt.label);
      last.answerPreview = prompt.answerPreview;
      last.preview = prompt.preview;
      continue;
    }
    groups.push({
      key: `local:${prompt.id}`,
      answerPreview: prompt.answerPreview,
      count: 1,
      ids: [prompt.id],
      renderedId: prompt.id,
      promptId: prompt.promptId ?? prompt.id,
      label: prompt.label,
      preview: prompt.preview,
      topPercent: prompt.topPercent,
    });
  }

  return groups;
}

function bucketPromptMarkers(
  measured: MeasuredPrompt[],
  railHeight: number,
): PromptMarker[] {
  const bucketCount = railHeight > 0
    ? clamp(
      Math.floor(railHeight / DENSE_BUCKET_HEIGHT_PX),
      1,
      DENSE_BUCKET_MAX_COUNT,
    )
    : DENSE_BUCKET_FALLBACK_COUNT;
  const buckets = Array.from({ length: bucketCount }, () => [] as MeasuredPrompt[]);

  for (const prompt of measured) {
    const bucketIndex = clamp(
      Math.floor((prompt.topPercent / 100) * bucketCount),
      0,
      bucketCount - 1,
    );
    buckets[bucketIndex].push(prompt);
  }

  return buckets.flatMap((bucket, bucketIndex) => {
    if (bucket.length === 0) return [];
    const latest = bucket[bucket.length - 1];
    const topPercent =
      bucket.reduce((sum, prompt) => sum + prompt.topPercent, 0) / bucket.length;
    return [{
      key: `measured-bucket:${bucketIndex}:${bucket[0].id}:${latest.id}`,
      count: bucket.length,
      ids: bucket.map((prompt) => prompt.id),
      renderedId: latest.id,
      promptId: latest.promptId ?? latest.id,
      label: bucket.length === 1
        ? latest.label
        : groupedPromptLabel(bucket.length, latest.label),
      answerPreview: latest.answerPreview,
      preview: latest.preview,
      topPercent,
    }];
  });
}

function distributeMarkerPositions(
  markers: PromptMarker[],
  railHeight: number,
  topInsetPx = 0,
): PromptMarker[] {
  const height = railHeight > 0 ? railHeight : RAIL_FALLBACK_HEIGHT_PX;
  const usableHeight = Math.max(0, height - topInsetPx);
  if (markers.length <= 1) {
    return markers.map((marker) => ({
      ...marker,
      topPercent: ((topInsetPx + usableHeight / 2) / height) * 100,
    }));
  }

  const availableHeight = Math.max(0, usableHeight - MARKER_STACK_GAP_PX);
  const stepPx = Math.min(MARKER_STACK_GAP_PX, availableHeight / (markers.length - 1));
  const stackHeight = stepPx * (markers.length - 1);
  const firstMarkerPx = topInsetPx + (usableHeight - stackHeight) / 2;

  return markers.map((marker, index) => ({
    ...marker,
    topPercent: ((firstMarkerPx + stepPx * index) / height) * 100,
  }));
}

function localPromptAnchorMarkers(anchors: PromptAnchor[]): PromptMarker[] {
  return anchors.map((anchor) => ({
    key: `local:${anchor.promptId ?? anchor.id}`,
    answerPreview: anchor.answerPreview,
    count: 1,
    ids: [anchor.id],
    renderedId: anchor.id,
    promptId: anchor.promptId ?? anchor.id,
    label: anchor.label,
    preview: anchor.preview,
    topPercent: 0,
  }));
}

function activePromptForScroll(
  measured: MeasuredPrompt[],
  scrollTop: number,
  anchors?: PromptAnchor[],
): string | null {
  if (measured.length === 0) return null;
  const cursor = scrollTop + 96;
  if (anchors && measured[0].top > cursor) {
    const firstMountedIdx = anchors.findIndex((anchor) => anchor.id === measured[0].id);
    if (firstMountedIdx > 0) {
      return anchors[firstMountedIdx - 1].id;
    }
  }
  let lower = 0;
  let upper = measured.length - 1;
  let activeIndex = 0;
  while (lower <= upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (measured[middle].top <= cursor) {
      activeIndex = middle;
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  return measured[activeIndex].id;
}

function groupedPromptLabel(count: number, latestLabel: string): string {
  return `${count} prompts, latest: ${latestLabel}`;
}

function markerWidth(hoverDistance: number | null): number {
  if (hoverDistance === null) return MARKER_BASE_WIDTH_PX;
  return HOVER_MARKER_WIDTHS_PX[hoverDistance] ?? MARKER_BASE_WIDTH_PX;
}

function markerHeight(hoverDistance: number | null): number {
  return hoverDistance === 0 ? 3 : 2;
}

function railMarkerTone(hoverDistance: number | null, active: boolean): string {
  if (hoverDistance === 0) {
    return "bg-[#222222] opacity-100 dark:bg-white";
  }
  if (hoverDistance !== null && hoverDistance < HOVER_MARKER_WIDTHS_PX.length) {
    return "bg-[#d0d0d0] opacity-100 dark:bg-white/35";
  }
  if (active) {
    return "bg-[#6f6f6f] opacity-100 dark:bg-white/55";
  }
  return "bg-[#d8d8d8] opacity-100 dark:bg-white/25";
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
