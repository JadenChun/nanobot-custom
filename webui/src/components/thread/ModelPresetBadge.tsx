import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import { Check, ChevronLeft, ChevronRight, Search, Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  floatingItemClassName,
  floatingItemFocusClassName,
} from "@/components/ui/floating-surface";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useLogoFallback } from "@/hooks/useLogoFallback";
import { inferProviderFromModelName, providerBrand } from "@/lib/provider-brand";
import { cn } from "@/lib/utils";

const pickerWidthClassName = "w-[min(18rem,calc(100vw-2rem))]";
const QUICK_SWITCH_LIMIT = 3;
const LONG_PRESS_MS = 400;
const PRESS_SLOP_PX = 8;
const PILL_GAP_PX = 4;
const PILL_OFFSETS = [-2, -1, 0, 1, 2] as const;
const HANDOFF_THRESHOLD = 0.56;
const DOCK_MAX_SCALE = 1.08;
const DOCK_RADIUS = 1.5;
const SETTLE_MS = 200;

interface PresetGesture {
  active: boolean;
  baseIndex: number;
  latestY: number;
  pointerId: number;
  startY: number;
  step: number;
  target: HTMLElement;
  timer: ReturnType<typeof setTimeout> | null;
}

interface PresetMotion {
  index: number;
  remainder: number;
  settling: boolean;
}

function wrapIndex(index: number, length: number): number {
  return ((index % length) + length) % length;
}

function dockScale(distanceFromFocus: number): number {
  const distance = Math.abs(distanceFromFocus);
  if (distance >= DOCK_RADIUS) return 1;
  const influence = (1 + Math.cos(Math.PI * distance / DOCK_RADIUS)) / 2;
  return 1 + (DOCK_MAX_SCALE - 1) * influence;
}

function stepWithHysteresis(raw: number, current: number): number {
  let next = current;
  while (raw > next + HANDOFF_THRESHOLD) next += 1;
  while (raw < next - HANDOFF_THRESHOLD) next -= 1;
  return next;
}

function preventTouchScroll(event: TouchEvent) {
  if (event.cancelable) event.preventDefault();
}

function compactModelName(model?: string | null): string | null {
  const value = model?.trim();
  if (!value) return null;
  return value.split("/").at(-1) || value;
}

export interface ModelPresetOption {
  name: string;
  model?: string | null;
  provider?: string | null;
}

export interface ModelReasoningControl {
  value: string | null;
  defaultValue: string | null;
  options: string[];
  pending: boolean;
  failed: boolean;
  onChange: (effort: string | null) => void;
}

interface ModelPresetBadgeProps {
  label: string;
  modelDetail?: string | null;
  modelPreset?: string | null;
  modelPresets?: ModelPresetOption[];
  onPresetChange?: (name: string) => void;
  reasoning?: ModelReasoningControl;
  onManageModels?: () => void;
  onRequestComposerFocus?: () => void;
  provider?: string | null;
  providerLabel?: string | null;
  needsSetup?: boolean;
  attentionRequest?: number;
  fallbackModelName?: string | null;
  isHero: boolean;
  onClick?: () => void;
}

export function ModelPresetBadge({
  label,
  modelDetail,
  modelPreset,
  modelPresets = [],
  onPresetChange,
  reasoning,
  onRequestComposerFocus,
  provider,
  providerLabel,
  needsSetup = false,
  attentionRequest = 0,
  fallbackModelName,
  isHero,
  onClick,
}: ModelPresetBadgeProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [showAllModels, setShowAllModels] = useState(false);
  const [modelQuery, setModelQuery] = useState("");
  const [motion, setMotion] = useState<PresetMotion | null>(null);
  const [motionWidth, setMotionWidth] = useState<number | null>(null);
  const gestureRef = useRef<PresetGesture | null>(null);
  const suppressClickRef = useRef(false);
  const activeName = modelPreset?.trim() || "";
  const listedIndex = modelPresets.findIndex((preset) => preset.name === activeName);
  const activePreset: ModelPresetOption = {
    ...(listedIndex >= 0 ? modelPresets[listedIndex] : undefined),
    name: activeName,
    model: modelDetail ?? modelPresets[listedIndex]?.model,
    provider: provider || modelPresets[listedIndex]?.provider,
  };
  const fallbackPreset = fallbackModelName
    ? modelPresets.find((preset) => preset.model?.trim() === fallbackModelName.trim())
    : undefined;
  const fallbackDisplayLabel = fallbackPreset?.name
    || fallbackModelName?.trim().split(/[/:]/).pop()
    || null;
  const displayLabel = fallbackDisplayLabel || label;
  const displayModelDetail = fallbackPreset
    ? fallbackPreset.model
    : fallbackModelName
      ? null
      : modelDetail;
  const displayProvider = fallbackPreset?.provider
    || (fallbackModelName ? inferProviderFromModelName(fallbackModelName) : provider);
  const presets = !activeName
    ? modelPresets
    : listedIndex < 0
      ? [activePreset, ...modelPresets]
      : modelPresets.map((preset, index) => index === listedIndex ? activePreset : preset);
  const opensSetup = Boolean(onClick);
  const canSwitch = !opensSetup && Boolean(onPresetChange) && activeName !== "" && presets.length > 1;
  const currentIndex = Math.max(0, presets.findIndex((preset) => preset.name === activeName));
  const otherPresets = presets.filter((preset) => preset.name !== activeName);
  const hasMoreModels = otherPresets.length > QUICK_SWITCH_LIMIT;
  const quickSwitchPresets = hasMoreModels
    ? otherPresets.slice(0, QUICK_SWITCH_LIMIT)
    : otherPresets;
  const query = modelQuery.trim().toLowerCase();
  const filteredPresets = query
    ? presets.filter((preset) =>
        preset.name.toLowerCase().includes(query)
        || (preset.model ?? "").toLowerCase().includes(query))
    : presets;
  const activeProviderLabel = providerLabel
    || displayProvider
    || inferProviderFromModelName(displayModelDetail || displayLabel);
  // The composer label is the user-facing model name; the preset key is an
  // internal identifier, so the header never leads with it.
  const activeDisplayName = displayLabel;
  const activeDetail = compactModelName(displayModelDetail);
  const effortLabelFor = (effort: string) => t(`thread.composer.reasoningEfforts.${effort}`, {
    defaultValue: effortLabel(effort),
  });
  // A non-default effort is part of how the next turn will run, so the closed
  // pill names it next to the model.
  const activeEffortLabel = reasoning?.value ? effortLabelFor(reasoning.value) : null;
  const pillHeight = isHero ? 32 : 36;
  const pillStride = pillHeight + PILL_GAP_PX;
  const switchModelLabel = t("thread.composer.switchModel", {
    defaultValue: "Switch model for this chat",
  });

  const selectPreset = (name: string) => {
    if (reasoning?.pending) return;
    setOpen(false);
    if (name !== activeName) onPresetChange?.(name);
    requestAnimationFrame(() => onRequestComposerFocus?.());
  };

  const clearGesture = () => {
    const gesture = gestureRef.current;
    if (gesture?.timer) clearTimeout(gesture.timer);
    if (gesture?.active) gesture.target.removeEventListener("touchmove", preventTouchScroll);
    gestureRef.current = null;
  };

  const clearMotion = () => {
    setMotion(null);
    setMotionWidth(null);
  };

  useEffect(() => {
    if (!canSwitch) {
      clearGesture();
      clearMotion();
    }
    return clearGesture;
  }, [canSwitch]);

  useEffect(() => {
    if (!motion?.settling) return;
    const timer = setTimeout(clearMotion, SETTLE_MS + 80);
    return () => clearTimeout(timer);
  }, [motion?.settling]);

  const updateMotion = (gesture: PresetGesture, clientY: number) => {
    const raw = -(clientY - gesture.startY) / pillStride;
    gesture.step = stepWithHysteresis(raw, gesture.step);
    setMotion({
      index: gesture.baseIndex + gesture.step,
      remainder: raw - gesture.step,
      settling: false,
    });
  };

  const handlePointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    if (!canSwitch || gestureRef.current || motion) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    const gesture: PresetGesture = {
      active: false,
      baseIndex: currentIndex,
      latestY: event.clientY,
      pointerId: event.pointerId,
      startY: event.clientY,
      step: 0,
      target: event.currentTarget,
      timer: null,
    };
    gesture.timer = setTimeout(() => {
      if (gestureRef.current !== gesture) return;
      gesture.active = true;
      setMotionWidth(Math.round(gesture.target.getBoundingClientRect().width) || null);
      updateMotion(gesture, gesture.latestY);
      gesture.target.addEventListener("touchmove", preventTouchScroll, { passive: false });
      try {
        gesture.target.setPointerCapture(gesture.pointerId);
      } catch {
        // The pointer may already have ended.
      }
    }, LONG_PRESS_MS);
    gestureRef.current = gesture;
  };

  const handlePointerMove = (event: PointerEvent<HTMLButtonElement>) => {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    gesture.latestY = event.clientY;
    if (!gesture.active) {
      if (Math.abs(event.clientY - gesture.startY) > PRESS_SLOP_PX) clearGesture();
      return;
    }
    event.preventDefault();
    updateMotion(gesture, event.clientY);
  };

  const finishGesture = (event: PointerEvent<HTMLButtonElement>, commit: boolean) => {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    clearGesture();
    if (event.currentTarget.hasPointerCapture?.(gesture.pointerId)) {
      event.currentTarget.releasePointerCapture?.(gesture.pointerId);
    }
    if (!commit || !gesture.active) {
      clearMotion();
      return;
    }
    suppressClickRef.current = true;
    const selected = presets[wrapIndex(gesture.baseIndex + gesture.step, presets.length)];
    setMotion((current) => current && { ...current, remainder: 0, settling: true });
    if (selected && selected.name !== activeName) selectPreset(selected.name);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const targetByKey: Record<string, number> = {
      ArrowUp: currentIndex - 1,
      ArrowDown: currentIndex + 1,
      Home: 0,
      End: presets.length - 1,
    };
    const target = targetByKey[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = presets[wrapIndex(target, presets.length)];
    if (next?.name !== activeName) selectPreset(next.name);
  };

  const pill = (
    <PresetPill
      key={needsSetup ? attentionRequest : undefined}
      label={displayLabel}
      modelDetail={displayModelDetail}
      provider={displayProvider}
      providerLabel={fallbackModelName ? null : providerLabel}
      effortLabel={needsSetup || fallbackModelName ? null : activeEffortLabel}
      needsSetup={needsSetup}
      needsAttention={needsSetup && attentionRequest > 0}
      fallbackModelName={fallbackModelName}
      fallbackFromLabel={fallbackModelName ? label : null}
      isHero={isHero}
    />
  );

  if (!canSwitch && !reasoning) {
    const Container = opensSetup ? "button" : "span";
    return (
      <Container
        aria-label={fallbackModelName ? `${displayLabel} (fallback from ${label})` : label}
        type={opensSetup ? "button" : undefined}
        onClick={opensSetup ? onClick : undefined}
        className={cn(
          "thread-composer-model-badge group inline-flex w-fit min-w-0 max-w-[min(18rem,44vw)] appearance-none border-0 bg-transparent p-0 shadow-none",
          opensSetup && "cursor-pointer focus-visible:outline-none",
          isHero ? "h-8" : "h-9",
        )}
      >
        {pill}
      </Container>
    );
  }

  return (
    <Popover
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen);
        if (!nextOpen) requestAnimationFrame(() => onRequestComposerFocus?.());
      }}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          data-switching={motion ? "true" : undefined}
          aria-label={label}
          aria-expanded={open}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerLeave={(event) => {
            const gesture = gestureRef.current;
            if (gesture && gesture.pointerId === event.pointerId && !gesture.active) clearGesture();
          }}
          onPointerUp={(event) => finishGesture(event, true)}
          onPointerCancel={(event) => finishGesture(event, false)}
          onLostPointerCapture={(event) => finishGesture(event, false)}
          onContextMenu={(event) => {
            if (gestureRef.current?.active) event.preventDefault();
          }}
          onDragStart={(event) => event.preventDefault()}
          onKeyDown={canSwitch ? handleKeyDown : undefined}
          onClickCapture={(event) => {
            if (!suppressClickRef.current) return;
            suppressClickRef.current = false;
            event.preventDefault();
            event.stopPropagation();
          }}
          style={{
            touchAction: "manipulation",
            width: motionWidth ? `${motionWidth}px` : undefined,
          }}
          className={cn(
            "thread-composer-model-badge group relative inline-flex w-fit min-w-0 max-w-[min(18rem,44vw)] cursor-pointer appearance-none border-0 bg-transparent p-0 shadow-none focus-visible:outline-none",
            motion && "z-10 cursor-grabbing",
            !motion && "cursor-grab",
            isHero ? "h-8" : "h-9",
          )}
        >
          {motion ? (
            <>
              <span data-testid="composer-model-pill-layout" className="invisible inline-flex h-full shrink-0" aria-hidden>
                {pill}
              </span>
              <span
                data-testid="composer-model-pill-viewport"
                className={cn(
                  "composer-model-pill-viewport pointer-events-none absolute -left-2 right-0 overflow-hidden bg-transparent",
                  isHero ? "-bottom-2.5 -top-2.5" : "-bottom-3 -top-3",
                )}
                aria-hidden
              >
                <span
                  data-testid="composer-model-pill-track"
                  data-settling={motion.settling ? "true" : undefined}
                  className="composer-model-pill-track ml-auto flex w-[calc(100%-0.5rem)] flex-col items-end gap-1 will-change-transform"
                  onTransitionEnd={(event) => {
                    if (motion.settling && event.currentTarget === event.target) clearMotion();
                  }}
                  style={{
                    paddingTop: isHero ? "10px" : "12px",
                    transform: `translate3d(0, ${-pillStride * (2 + motion.remainder)}px, 0)`,
                  }}
                >
                  {PILL_OFFSETS.map((offset) => {
                    const preset = presets[wrapIndex(motion.index + offset, presets.length)];
                    return (
                      <PresetPill
                        key={motion.index + offset}
                        label={preset.name}
                        modelDetail={preset.model}
                        provider={preset.provider}
                        isHero={isHero}
                        offset={offset}
                        scale={motion.settling ? 1 : dockScale(offset - motion.remainder)}
                      />
                    );
                  })}
                </span>
              </span>
            </>
          ) : pill}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        side="top"
        sideOffset={10}
        role="dialog"
        aria-label={switchModelLabel}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          if (showAllModels) return;
          const content = event.currentTarget;
          if (!(content instanceof HTMLElement)) return;
          const selected = content.querySelector<HTMLElement>(
            '[role="option"][aria-selected="true"]',
          );
          selected?.focus();
        }}
        className={cn(
          pickerWidthClassName,
          "max-h-[var(--radix-popover-content-available-height)] origin-[var(--radix-popover-content-transform-origin)] overflow-y-auto p-1.5 scrollbar-thin scrollbar-track-transparent data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-bottom-1 data-[state=open]:slide-in-from-bottom-1 duration-200 ease-out motion-reduce:animate-none",
        )}
      >
        {showAllModels ? (
          <div data-testid="model-picker-all">
            <div className="flex items-center gap-1 pb-1">
              <button
                type="button"
                onClick={() => {
                  setShowAllModels(false);
                  setModelQuery("");
                }}
                aria-label={t("thread.composer.backToQuickModels", {
                  defaultValue: "Back to quick models",
                })}
                className={cn(
                  floatingItemClassName,
                  floatingItemFocusClassName,
                  "inline-flex min-h-8 items-center gap-1 rounded-lg px-2 text-xs font-medium text-muted-foreground hover:text-foreground",
                )}
              >
                <ChevronLeft className="size-3.5 shrink-0" strokeWidth={1.9} aria-hidden />
                <span>{t("thread.composer.back", { defaultValue: "Back" })}</span>
              </button>
            </div>
            <label className="flex items-center gap-2 rounded-lg bg-muted/45 px-2.5 py-1.5">
              <Search className="size-3.5 shrink-0 text-muted-foreground/70" strokeWidth={1.9} aria-hidden />
              <input
                autoFocus
                type="search"
                value={modelQuery}
                onChange={(event) => setModelQuery(event.target.value)}
                placeholder={t("thread.composer.searchModels", { defaultValue: "Search models" })}
                aria-label={t("thread.composer.searchModels", { defaultValue: "Search models" })}
                className="min-w-0 flex-1 bg-transparent text-[13px] text-foreground outline-none placeholder:text-muted-foreground/60"
              />
            </label>
            <div
              role="listbox"
              aria-label={switchModelLabel}
              className="max-h-[min(16rem,var(--radix-popover-content-available-height))] overflow-y-auto py-1 scrollbar-thin scrollbar-track-transparent"
            >
              {filteredPresets.map((preset) => (
                <PresetOption
                  key={preset.name}
                  preset={preset}
                  selected={preset.name === activeName}
                  onSelect={selectPreset}
                />
              ))}
            </div>
            {filteredPresets.length === 0 ? (
              <p className="px-2.5 py-2 text-xs text-muted-foreground">
                {t("thread.composer.noMatchingModels", { defaultValue: "No matching models" })}
              </p>
            ) : null}
          </div>
        ) : (
          <>
            <div
              data-testid="model-picker-active"
              className="flex min-w-0 items-center gap-2.5 rounded-lg bg-muted/55 px-2.5 py-2"
            >
              <PresetProviderIcon
                label={activeDisplayName}
                modelDetail={activeDetail}
                provider={displayProvider}
                isHero={false}
              />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="min-w-0 truncate text-[13px] font-semibold text-foreground">
                    {activeDisplayName}
                  </span>
                  <Check className="size-3.5 shrink-0 text-foreground/70" aria-hidden />
                </span>
                {activeProviderLabel ? (
                  <span className="min-w-0 truncate text-[11.5px] text-muted-foreground">
                    {activeProviderLabel}
                  </span>
                ) : activeDetail && activeDetail !== activeDisplayName ? (
                  <span className="min-w-0 truncate text-[11.5px] text-muted-foreground">
                    {activeDetail}
                  </span>
                ) : null}
              </span>
            </div>
            {reasoning ? (
              <ReasoningEffortPills
                key={activeName}
                control={reasoning}
                labelFor={effortLabelFor}
              />
            ) : null}
            <div
              role="listbox"
              aria-label={switchModelLabel}
              data-testid="model-picker-quick-switch"
              className="py-1"
            >
              {quickSwitchPresets.map((preset) => (
                <PresetOption
                  key={preset.name}
                  preset={preset}
                  selected={preset.name === activeName}
                  onSelect={selectPreset}
                />
              ))}
            </div>
            {hasMoreModels ? (
              <div className="border-t border-border/55 pt-1">
                <button
                  type="button"
                  data-testid="model-picker-more"
                  onClick={() => setShowAllModels(true)}
                  className={cn(
                    floatingItemClassName,
                    floatingItemFocusClassName,
                    "flex min-h-9 w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span className="min-w-0 flex-1 truncate text-left">
                    {t("thread.composer.moreModels", {
                      count: presets.length,
                      defaultValue: "More models ({{count}})",
                    })}
                  </span>
                  <ChevronRight className="size-4 shrink-0" strokeWidth={1.75} aria-hidden />
                </button>
              </div>
            ) : null}
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}

interface ReasoningEffortPillsProps {
  control: ModelReasoningControl;
  labelFor: (effort: string) => string;
}

/**
 * One-click segmented effort pills.
 *
 * Every available option stays visible, including middle values, so the user
 * never has to drag a slider to discover or select an intermediate effort. A
 * preset default is offered as a labelled reset option.
 */
function ReasoningEffortPills({ control, labelFor }: ReasoningEffortPillsProps) {
  const { t } = useTranslation();
  const sliderId = useId();
  const [preview, setPreview] = useState<number | null>(null);
  const pointer = useRef<number | null>(null);
  const efforts: (string | null)[] = [null, ...control.options];
  const committed = Math.max(0, efforts.indexOf(control.value));
  const selected = preview ?? committed;
  const last = efforts.length - 1;
  const selectedEffort = efforts[selected];
  const valueText = selectedEffort
    ? labelFor(selectedEffort)
    : control.defaultValue
      ? `${t("thread.composer.presetReasoningDefault", { defaultValue: "Preset default" })} (${labelFor(control.defaultValue)})`
      : t("thread.composer.presetReasoningDefault", { defaultValue: "Preset default" });

  const commit = (index: number) => {
    if (index !== committed && index >= 0 && index <= last) {
      control.onChange(efforts[index]);
    }
  };
  const endPreview = (event: PointerEvent<HTMLInputElement>, apply: boolean) => {
    if (pointer.current !== event.pointerId) return;
    pointer.current = null;
    setPreview(null);
    if (apply) commit(event.currentTarget.valueAsNumber);
  };

  return (
    <div
      data-testid="reasoning-effort-pills"
      role="group"
      aria-label={t("settings.models.reasoningEffort", { defaultValue: "Reasoning effort" })}
      aria-busy={control.pending}
      className="space-y-1 border-t border-border/55 px-1.5 py-2"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-2 px-1">
        <label htmlFor={sliderId} className="text-xs font-medium text-muted-foreground">
          {t("settings.models.reasoningEffort", { defaultValue: "Reasoning effort" })}
        </label>
        <span className="text-xs font-medium text-foreground">{valueText}</span>
      </div>
      <div className="px-1">
        <input
          id={sliderId}
          type="range"
          min={0}
          max={last}
          step={1}
          value={selected}
          aria-valuetext={valueText}
          onChange={(event) => {
            const index = event.target.valueAsNumber;
            if (pointer.current === null) commit(index);
            else setPreview(index);
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) {
              event.preventDefault();
              return;
            }
            pointer.current = event.pointerId;
            event.currentTarget.setPointerCapture?.(event.pointerId);
          }}
          onPointerUp={(event) => endPreview(event, true)}
          onPointerCancel={(event) => endPreview(event, false)}
          onLostPointerCapture={(event) => endPreview(event, false)}
          className="block h-6 w-full cursor-pointer touch-pan-y accent-foreground focus-visible:rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/50"
        />
      </div>
      <div className="flex items-center justify-between gap-0.5">
        {efforts.map((effort, index) => {
          const active = index === selected;
          return (
            <button
              key={effort ?? "default"}
              type="button"
              aria-pressed={index === committed}
              onClick={() => commit(index)}
              className={cn(
                "min-h-6 rounded px-1 text-[11px] transition-colors duration-150 motion-reduce:transition-none",
                "focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/50",
                active
                  ? "font-semibold text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {effort ? labelFor(effort) : t("settings.values.default", { defaultValue: "Default" })}
            </button>
          );
        })}
      </div>
      {control.failed ? (
        <p role="status" className="px-1 text-xs text-red-500 dark:text-red-400">
          {t("thread.composer.reasoningEffortFailed", { defaultValue: "Could not update reasoning effort. Try again." })}
        </p>
      ) : null}
    </div>
  );
}

function effortLabel(effort: string): string {
  return effort === "xhigh" ? "Extra high" : effort.charAt(0).toUpperCase() + effort.slice(1);
}

function PresetOption({
  preset,
  selected,
  onSelect,
}: {
  preset: ModelPresetOption;
  selected: boolean;
  onSelect: (name: string) => void;
}) {
  const detail = compactModelName(preset.model);
  return (
    <button
      type="button"
      role="option"
      aria-label={preset.name}
      aria-selected={selected}
      onClick={() => onSelect(preset.name)}
      className={cn(
        floatingItemClassName,
        floatingItemFocusClassName,
        "flex min-h-9 w-full cursor-pointer gap-2.5 px-2.5 py-1.5 text-left hover:bg-muted/55",
        selected && "bg-muted/55 text-foreground",
      )}
    >
      <PresetProviderIcon
        label={preset.name}
        modelDetail={detail}
        provider={preset.provider}
        isHero={false}
      />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-[13px] font-medium leading-snug text-foreground">
          {preset.name}
        </span>
        {detail && detail !== preset.name ? (
          <span className="truncate text-[11px] leading-tight text-muted-foreground/80">
            {detail}
          </span>
        ) : null}
      </span>
      {selected ? <Check className="h-4 w-4 shrink-0 text-foreground/80" aria-hidden /> : null}
    </button>
  );
}

function PresetPill({
  label,
  modelDetail,
  provider,
  providerLabel,
  effortLabel,
  needsSetup = false,
  needsAttention = false,
  fallbackModelName,
  fallbackFromLabel,
  isHero,
  offset,
  scale,
}: {
  label: string;
  modelDetail?: string | null;
  provider?: string | null;
  providerLabel?: string | null;
  effortLabel?: string | null;
  needsSetup?: boolean;
  needsAttention?: boolean;
  fallbackModelName?: string | null;
  fallbackFromLabel?: string | null;
  isHero: boolean;
  offset?: number;
  scale?: number;
}) {
  const labelRef = useRef<HTMLSpanElement | null>(null);
  const [labelOverflows, setLabelOverflows] = useState(false);
  const inferredProvider = needsSetup
    ? null
    : provider || inferProviderFromModelName(modelDetail || label);
  const title = [...new Set([label, modelDetail, providerLabel, effortLabel].filter(Boolean))]
    .join(" · ");
  const fallbackTitle = fallbackModelName
    ? `${fallbackFromLabel || label} · using ${fallbackModelName}`
    : title;

  useLayoutEffect(() => {
    const node = labelRef.current;
    if (!node) return;
    const update = () => setLabelOverflows(node.scrollWidth > node.clientWidth + 1);
    update();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(node);
    return () => observer?.disconnect();
  }, [label]);

  return (
    <span
      data-fallback={fallbackModelName ? "true" : undefined}
      data-needs-setup={needsSetup ? "true" : undefined}
      data-preset-offset={offset}
      title={fallbackTitle || undefined}
      className={cn(
        "composer-model-badge composer-model-pill inline-flex h-full max-w-full min-w-0 shrink-0 items-center rounded-full border border-border/55 bg-card font-medium text-foreground/70",
        "w-fit",
        "transition-[color,background-color,border-color,transform] duration-150 ease-out group-focus-visible:ring-2 group-focus-visible:ring-ring/45",
        isHero ? "gap-1.5 px-2.5 text-[12px]" : "gap-2 px-3 text-[12.5px]",
        needsSetup && "composer-model-pill-setup",
        needsAttention && "composer-model-pill-setup-attention",
        offset !== undefined && "composer-model-pill-dock",
      )}
      style={scale === undefined ? undefined : {
        height: `${isHero ? 32 : 36}px`,
        transform: `scale(${scale.toFixed(4)})`,
        zIndex: Math.round(scale * 100),
      }}
    >
      {!needsSetup ? (
        <PresetProviderIcon
          label={label}
          modelDetail={modelDetail}
          provider={inferredProvider}
          testId={`composer-model-logo${inferredProvider ? `-${inferredProvider}` : ""}`}
          isHero={isHero}
        />
      ) : null}
      <span
        ref={labelRef}
        className={cn(
          "thread-composer-model-label min-w-0 overflow-hidden whitespace-nowrap text-center",
          labelOverflows && "thread-composer-model-label-fade",
        )}
      >
        {needsSetup ? <SetupPromptLabel label={label} /> : label}
      </span>
      {effortLabel && !needsSetup ? (
        <span
          data-testid="composer-model-effort-tag"
          className={cn(
            "shrink-0 rounded-full bg-muted/70 px-1.5 text-[10.5px] font-semibold leading-4 text-muted-foreground",
            isHero && "px-1 text-[10px]",
          )}
        >
          {effortLabel}
        </span>
      ) : null}
    </span>
  );
}

function SetupPromptLabel({ label }: { label: string }) {
  const separator = label.indexOf(" ");
  if (separator < 0) {
    return <span className="text-foreground/80">{label}</span>;
  }

  return (
    <span data-testid="composer-model-setup-label">
      <span className="text-muted-foreground/90 transition-colors duration-150 group-hover:text-muted-foreground motion-reduce:transition-none">
        {label.slice(0, separator)}
      </span>
      {" "}
      <span className="text-foreground/80 transition-colors duration-150 group-hover:text-foreground/90 motion-reduce:transition-none">
        {label.slice(separator + 1)}
      </span>
    </span>
  );
}

function PresetProviderIcon({
  label,
  modelDetail,
  provider,
  testId,
  isHero,
}: {
  label: string;
  modelDetail?: string | null;
  provider?: string | null;
  testId?: string;
  isHero: boolean;
}) {
  const inferredProvider = provider || inferProviderFromModelName(modelDetail || label);
  const brand = providerBrand(inferredProvider);
  const { logoUrl, onLogoError, onLogoLoad } = useLogoFallback(brand?.logoUrls);
  return (
    <span
      data-testid={testId}
      className={cn(
        "composer-model-provider-icon grid shrink-0 place-items-center",
        isHero ? "h-4 w-4" : "h-[18px] w-[18px]",
      )}
      aria-hidden
    >
      {logoUrl ? (
        <img
          src={logoUrl}
          alt=""
          draggable={false}
          decoding="async"
          loading="lazy"
          className={cn("object-contain", isHero ? "h-3.5 w-3.5" : "h-[18px] w-[18px]")}
          onLoad={onLogoLoad}
          onError={onLogoError}
        />
      ) : brand ? (
        <span
          className={cn(
            "grid h-full w-full place-items-center rounded-full text-white",
            isHero ? "text-[7.5px]" : "text-[8px]",
          )}
          style={{ backgroundColor: brand.color }}
        >
          {brand.initials.slice(0, 2)}
        </span>
      ) : (
        <Sparkles className="h-3 w-3 text-muted-foreground/65" />
      )}
    </span>
  );
}
