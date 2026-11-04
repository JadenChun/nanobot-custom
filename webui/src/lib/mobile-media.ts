import { useEffect, useState } from "react";

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
