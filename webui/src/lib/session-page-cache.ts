import type { UIMessage } from "@/lib/types";

/**
 * Renderer-side cache of a session's loaded history.
 *
 * Codex keeps the normalised turn store in the renderer (`entitiesByKey`) and
 * gates scroll-back with per-turn `hasLoadedOldest`, so returning to a session
 * renders from memory instead of re-fetching. This mirrors the part of that
 * model that applies to our flat message projection: the loaded window plus the
 * cursor needed to continue paging, keyed by session.
 *
 * It intentionally holds only the committed page, not the live turn. A streaming
 * turn is owned by the stream hook and merged on top; caching it here would let
 * a stale in-flight snapshot win after a reconnect. Entries are bounded and
 * evicted oldest-first, and a refresh clears the entry for that key.
 */
export interface CachedSessionPage {
  messages: UIMessage[];
  beforeCursor: string | null;
  hasMoreBefore: boolean;
  userMessageOffset: number;
  completedTurnIds: string[];
  forkBoundaryMessageCount: number | null;
  version: number;
}

const MAX_CACHED_SESSIONS = 8;

const pageCache = new Map<string, CachedSessionPage>();

export function readCachedSessionPage(key: string): CachedSessionPage | undefined {
  const cached = pageCache.get(key);
  if (cached) {
    // Refresh recency so revisiting recently used sessions survives eviction.
    pageCache.delete(key);
    pageCache.set(key, cached);
  }
  return cached;
}

export function writeCachedSessionPage(key: string, page: CachedSessionPage): void {
  pageCache.delete(key);
  pageCache.set(key, page);
  while (pageCache.size > MAX_CACHED_SESSIONS) {
    const oldest = pageCache.keys().next();
    if (oldest.done) break;
    pageCache.delete(oldest.value);
  }
}

export function clearCachedSessionPage(key: string): void {
  pageCache.delete(key);
}

/** Test seam: drop every cached page. */
export function resetSessionPageCache(): void {
  pageCache.clear();
}
