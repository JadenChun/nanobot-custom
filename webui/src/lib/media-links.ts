import { isMobileRuntimeActive } from "@/lib/mobile-runtime";

/** Return the most useful shareable form of a signed media URL. */
export function mediaLink(url: string | undefined): string {
  if (!url) return "";
  if (isMobileRuntimeActive()) return url;
  try {
    return new URL(url, window.location.origin).href;
  } catch {
    return url;
  }
}
