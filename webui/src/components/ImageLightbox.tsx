import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { ChevronLeft, ChevronRight, Play, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { useMobileBack } from "@/hooks/useMobileBack";
import { useMobileMediaUrl } from "@/lib/mobile-media";
import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import { cn } from "@/lib/utils";
import type { UIImage } from "@/lib/types";

interface ImageLightboxProps {
  images: UIImage[];
  index: number | null;
  onIndexChange: (index: number) => void;
  onOpenChange: (open: boolean) => void;
}

/**
 * Modal image viewer. Uses the Radix Dialog primitives directly so we can
 * fill the viewport (the shared `DialogContent` wrapper caps at max-w-lg,
 * which is much too small for a photo preview).
 *
 * Implementation notes:
 * - `translate3d` + `will-change: transform` promote the image to a GPU
 *   compositing layer so open/swap stays at 60 FPS on long threads.
 * - Adjacent images are rendered in hidden `<img>` tags so the browser
 *   decodes them eagerly; pressing left/right feels instant.
 * - The thumbnail filmstrip keeps every image one click away and scrolls the
 *   active thumbnail into view without moving the document.
 * - Radix handles `Escape` + focus trapping; we only wire up ←/→ + Home/End.
 * - Touch uses horizontal swipes; the arrow buttons are desktop-only so they
 *   do not sit on top of the photo on phones (the filmstrip remains tappable).
 * - Respects `prefers-reduced-motion` by dropping the fade + zoom-in
 *   keyframes via `motion-reduce:*` variants.
 */
export function ImageLightbox({
  images,
  index,
  onIndexChange,
  onOpenChange,
}: ImageLightboxProps) {
  const { t } = useTranslation();
  const open = index !== null;
  const total = images.length;
  const current = index !== null ? images[index] : null;
  const resolvedCurrent = useMobileMediaUrl(current?.url);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const thumbnailRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const swipeRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    active: boolean;
  } | null>(null);
  const pendingIndexRef = useRef<number | null>(null);
  const slideTimerRef = useRef<number | null>(null);
  const [dragPx, setDragPx] = useState(0);
  const [slide, setSlide] = useState<-1 | 0 | 1>(0);
  const [snapBack, setSnapBack] = useState(false);

  const slideActionRef = useRef<(delta: 1 | -1) => void>(() => {});
  const go = useCallback((delta: number) => {
    slideActionRef.current(delta > 0 ? 1 : -1);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        go(1);
      } else if (e.key === "Home") {
        e.preventDefault();
        onIndexChange(0);
      } else if (e.key === "End") {
        e.preventDefault();
        onIndexChange(total - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, onIndexChange, open, total]);

  useEffect(() => {
    if (!open || index === null) return;
    const strip = stripRef.current;
    const thumbnail = thumbnailRefs.current[index];
    if (!strip || !thumbnail) return;
    const stripBounds = strip.getBoundingClientRect();
    const thumbnailBounds = thumbnail.getBoundingClientRect();
    const left = strip.scrollLeft
      + thumbnailBounds.left
      - stripBounds.left
      - (strip.clientWidth - thumbnailBounds.width) / 2;
    const reduceMotion = typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    strip.scrollTo({
      left: Math.max(0, left),
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }, [index, open]);

  // Neighbours we want the browser to decode eagerly.
  const preload = useMemo(() => {
    if (index === null || total <= 1) return [] as UIImage[];
    const prev = images[(index - 1 + total) % total];
    const next = images[(index + 1) % total];
    return [prev, next].filter((i) => i && i.url);
  }, [images, index, total]);

  const hasMany = total > 1;
  const touchFirst = isMobileRuntimeActive();
  const counter = hasMany && index !== null ? `${index + 1} / ${total}` : null;
  const slideImages = hasMany && index !== null
    ? [images[(index - 1 + total) % total], current, images[(index + 1) % total]]
    : current
      ? [current]
      : [];

  const prefersReducedMotion = () =>
    typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const clearSlideTimer = () => {
    if (slideTimerRef.current !== null) {
      window.clearTimeout(slideTimerRef.current);
      slideTimerRef.current = null;
    }
  };

  const completeSlide = () => {
    clearSlideTimer();
    const target = pendingIndexRef.current;
    pendingIndexRef.current = null;
    setSnapBack(true);
    setSlide(0);
    setDragPx(0);
    if (target !== null) onIndexChange(target);
  };

  const startSlide = (delta: 1 | -1) => {
    if (index === null || total <= 1 || slide !== 0) return;
    const target = (index + delta + total) % total;
    if (prefersReducedMotion()) {
      onIndexChange(target);
      return;
    }
    pendingIndexRef.current = target;
    setSlide(delta);
    clearSlideTimer();
    // `transitionend` is the normal commit path. The timer keeps the gallery
    // from sticking when a throttled/hidden tab never delivers the event.
    slideTimerRef.current = window.setTimeout(completeSlide, 400);
  };

  slideActionRef.current = startSlide;

  useEffect(() => {
    if (open) return;
    clearSlideTimer();
    pendingIndexRef.current = null;
    setSlide(0);
    setDragPx(0);
  }, [open]);

  useEffect(() => {
    if (!snapBack) return;
    const frame = window.requestAnimationFrame(() => setSnapBack(false));
    return () => window.cancelAnimationFrame(frame);
  }, [snapBack]);

  useMobileBack(() => {
    onOpenChange(false);
    return true;
  }, open);

  // Keep this below every hook: `index` is null while the dialog is closed,
  // and returning before a hook would change the hook order on open.
  if (!current || !resolvedCurrent) return null;

  const beginSwipe = (event: React.PointerEvent<HTMLDivElement>) => {
    if (total <= 1 || slide !== 0 || event.pointerType === "mouse") return;
    swipeRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      active: false,
    };
  };

  const moveSwipe = (event: React.PointerEvent<HTMLDivElement>) => {
    const swipe = swipeRef.current;
    if (!swipe || swipe.pointerId !== event.pointerId) return;
    const dx = event.clientX - swipe.startX;
    const dy = event.clientY - swipe.startY;
    if (!swipe.active) {
      // Let vertical gestures and taps win until the drag is clearly horizontal.
      if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 12) {
        swipeRef.current = null;
        setDragPx(0);
        return;
      }
      if (Math.abs(dx) < 10) return;
      swipe.active = true;
    }
    setDragPx(dx);
  };

  const finishSwipe = (event: React.PointerEvent<HTMLDivElement>) => {
    const swipe = swipeRef.current;
    if (!swipe || swipe.pointerId !== event.pointerId) return;
    swipeRef.current = null;
    const dx = event.clientX - swipe.startX;
    const dy = event.clientY - swipe.startY;
    const committed = swipe.active
      && Math.abs(dx) >= 56
      && Math.abs(dx) >= Math.abs(dy) * 1.2;
    if (committed) {
      startSlide(dx < 0 ? 1 : -1);
      return;
    }
    setDragPx(0);
  };

  const handleTrackTransitionEnd = (event: React.TransitionEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || slide === 0) return;
    completeSlide();
  };

  const trackStyle = total <= 1
    ? { transform: "translate3d(0,0,0)" }
    : {
        // The track is one viewport wide; slides overflow it as flex items, so
        // percentages below are relative to a single slide, not the 3-slide
        // content width.
        transform: slide === 1
          ? "translate3d(-200%,0,0)"
          : slide === -1
            ? "translate3d(0%,0,0)"
            : `translate3d(calc(-100% + ${dragPx}px),0,0)`,
        transition: slide !== 0
          ? "transform 260ms cubic-bezier(0.22, 1, 0.36, 1)"
          : snapBack || dragPx !== 0
            ? "none"
            : "transform 200ms ease-out",
        willChange: "transform",
      } as const;

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay
          className={cn(
            "fixed inset-0 z-50 bg-black/80 backdrop-blur-sm",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "motion-reduce:data-[state=open]:animate-none motion-reduce:data-[state=closed]:animate-none",
          )}
        />
        <DialogPrimitive.Content
          aria-label={current.name ?? t("lightbox.title")}
          className={cn(
            "fixed inset-0 z-50 flex flex-col",
            "focus:outline-none",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "motion-reduce:data-[state=open]:animate-none motion-reduce:data-[state=closed]:animate-none",
          )}
        >
          <DialogPrimitive.Title className="sr-only">
            {current.name ?? t("lightbox.title")}
          </DialogPrimitive.Title>

          <div className="relative flex min-h-0 flex-1 items-center justify-center px-4 pb-2 pt-16 sm:px-16">
            <div
              className="h-full w-full touch-pan-y overflow-hidden"
              onPointerDown={beginSwipe}
              onPointerMove={moveSwipe}
              onPointerUp={finishSwipe}
              onPointerCancel={finishSwipe}
            >
              <div
                className="flex h-full w-full"
                style={trackStyle}
                onTransitionEnd={handleTrackTransitionEnd}
              >
                {slideImages.map((image, slot) => (
                  <LightboxSlide key={`${image?.url ?? "empty"}-${slot}`} image={image} />
                ))}
              </div>
            </div>

            {hasMany && !touchFirst ? (
              <>
                <NavButton
                  side="left"
                  label={t("lightbox.prev")}
                  onClick={(e) => {
                    e.stopPropagation();
                    go(-1);
                  }}
                />
                <NavButton
                  side="right"
                  label={t("lightbox.next")}
                  onClick={(e) => {
                    e.stopPropagation();
                    go(1);
                  }}
                />
              </>
            ) : null}
          </div>

          {hasMany ? (
            <div
              className={cn(
                "relative z-10 shrink-0 border-t border-white/10 bg-black/70",
                "px-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] pt-2",
                "backdrop-blur-md",
              )}
            >
              <div className="mb-1.5 text-center text-[11px] font-medium tabular-nums text-white/70">
                {counter}
              </div>
              <div
                ref={stripRef}
                role="group"
                aria-label={t("lightbox.thumbnails")}
                className="mx-auto flex w-max max-w-[min(100%,48rem)] snap-x gap-2 overflow-x-auto pb-1 [scrollbar-width:thin]"
              >
                {images.map((image, i) => (
                  <button
                    key={`${image.url}-${i}`}
                    ref={(element) => {
                      thumbnailRefs.current[i] = element;
                    }}
                    type="button"
                    onClick={() => onIndexChange(i)}
                    aria-label={image.name
                      ? `${t("lightbox.goToImage", { index: i + 1 })}: ${image.name}`
                      : t("lightbox.goToImage", { index: i + 1 })}
                    aria-current={i === index ? "true" : undefined}
                    className={cn(
                      "relative h-14 w-14 shrink-0 snap-center overflow-hidden rounded-lg border",
                      "bg-black/30 transition-opacity motion-reduce:transition-none",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80",
                      i === index
                        ? "border-white/90 opacity-100 ring-1 ring-white/80"
                        : "border-white/15 opacity-60 hover:opacity-90",
                    )}
                  >
                    <LightboxThumbnail image={image} />
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <DialogPrimitive.Close
            aria-label={t("lightbox.close")}
            className={cn(
              "absolute right-4 top-4 grid h-9 w-9 place-items-center rounded-full",
              "bg-black/55 text-white/90 hover:bg-black/70 hover:text-white",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70",
              "transition-colors motion-reduce:transition-none",
            )}
          >
            <X className="h-4 w-4" aria-hidden />
          </DialogPrimitive.Close>

          {/* Invisible preload — browser decodes adjacent images so prev/next swap is instant. */}
          <div aria-hidden className="hidden">
            {preload.map((img, i) => (
              <PreloadImage key={`${img.url}-${i}`} image={img} />
            ))}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

interface NavButtonProps {
  side: "left" | "right";
  label: string;
  onClick: React.MouseEventHandler<HTMLButtonElement>;
}

function LightboxSlide({ image }: { image: UIImage | null }) {
  const src = useMobileMediaUrl(image?.url);
  if (!src) return <div className="h-full w-full shrink-0" aria-hidden />;
  if (image?.kind === "video") {
    return (
      <div className="flex h-full w-full shrink-0 items-center justify-center">
        <video
          src={src}
          controls
          playsInline
          preload="metadata"
          aria-label={image.name ?? ""}
          className="max-h-full max-w-full rounded-compact bg-black shadow-2xl"
        />
      </div>
    );
  }
  return (
    <div className="flex h-full w-full shrink-0 items-center justify-center">
      <img
        src={src}
        alt={image?.name ?? ""}
        decoding="async"
        draggable={false}
        className="max-h-full max-w-full select-none rounded-compact object-contain shadow-2xl"
      />
    </div>
  );
}

function PreloadImage({ image }: { image: UIImage }) {
  const src = useMobileMediaUrl(image.url);
  if (!src || image.kind === "video") return null;
  return <img src={src} alt="" />;
}

function LightboxThumbnail({ image }: { image: UIImage }) {
  const src = useMobileMediaUrl(image.url);
  if (!src) return null;
  if (image.kind === "video") {
    return (
      <span className="relative block h-full w-full">
        <video
          src={src}
          muted
          playsInline
          preload="metadata"
          className="h-full w-full object-cover"
        />
        <span className="absolute inset-0 grid place-items-center bg-black/20">
          <Play className="h-4 w-4 fill-white text-white" aria-hidden />
        </span>
      </span>
    );
  }
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      decoding="async"
      draggable={false}
      className="h-full w-full object-cover"
    />
  );
}

function NavButton({ side, label, onClick }: NavButtonProps) {
  const Icon = side === "left" ? ChevronLeft : ChevronRight;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className={cn(
        "absolute top-1/2 -translate-y-1/2 grid h-11 w-11 place-items-center rounded-full",
        "bg-black/55 text-white/90 hover:bg-black/70 hover:text-white",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70",
        "transition-colors motion-reduce:transition-none",
        side === "left" ? "left-4" : "right-4",
      )}
    >
      <Icon className="h-5 w-5" aria-hidden />
    </button>
  );
}
