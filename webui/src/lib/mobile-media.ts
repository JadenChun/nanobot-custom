import { useEffect, useState, type RefObject } from "react";

// Leave room in the Android broker (eight streams) for active playback.
const MOBILE_PREVIEW_LIMIT = 3;
const waitingPreviews = new Set<() => void>();
let activePreviews = 0;

export function queueMobilePreview(start: () => void) {
  const run = () => {
    activePreviews += 1;
    start();
  };
  if (activePreviews < MOBILE_PREVIEW_LIMIT) run();
  else waitingPreviews.add(run);

  return () => {
    if (waitingPreviews.delete(run)) return;
    activePreviews -= 1;
    const next = waitingPreviews.values().next().value;
    if (next) {
      waitingPreviews.delete(next);
      next();
    }
  };
}

export function useMobilePreviewVisibility(ref: RefObject<HTMLElement>, enabled: boolean) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const node = ref.current;
    if (!enabled || !node) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry?.isIntersecting) return;
      observer.disconnect();
      setVisible(true);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [enabled, ref]);
  return visible;
}

export type MediaSourceStatus = "ready" | "loading" | "error";

export interface MediaSource {
  src?: string;
  status: MediaSourceStatus;
}

/**
 * Resolve one signed media URL for the current surface.
 *
 * Both surfaces use the URL as-is: the browser fetches it directly, and the
 * Android WebView intercepts `/api/media/...` and streams the bytes through the
 * native transport (see `NativeMediaInterceptor`). No base64 bridge hop is
 * involved, so loading state is owned by the `<img>`/`<video>` element itself.
 */
export function useMediaSource(url: string | undefined): MediaSource {
  const [source, setSource] = useState<MediaSource>(() => (
    url ? { src: url, status: "ready" } : { status: "error" }
  ));
  useEffect(() => {
    setSource(url ? { src: url, status: "ready" } : { status: "error" });
  }, [url]);
  return source;
}

/** Backwards-compatible helper for callers that only need the resolved URL. */
export function useMobileMediaUrl(url: string | undefined): string | undefined {
  return useMediaSource(url).src;
}
