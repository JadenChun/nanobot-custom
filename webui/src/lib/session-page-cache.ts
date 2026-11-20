import type { NanobotClient } from "@/lib/nanobot-client";
import type { UIMessage } from "@/lib/types";

/**
 * Renderer-side cache of a session's loaded history.
 *
 * Codex keeps the normalised turn store in the renderer (`entitiesByKey`) and
 * gates scroll-back with per-turn `hasLoadedOldest`, so returning to a session
 * renders from memory instead of re-fetching. This mirrors the part of that
 * model that applies to our flat message projection: the loaded window plus the
 * cursor needed to continue paging, keyed by authenticated client and session.
 *
 * It intentionally holds only the committed page, not the live turn. A streaming
 * turn is owned by the stream hook and merged on top; caching it here would let
 * a stale in-flight snapshot win after a reconnect. Entries are bounded and
 * evicted oldest-first. A new authenticated client cannot reuse old entries.
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

let cacheByClient = new WeakMap<NanobotClient, Map<string, CachedSessionPage>>();

export function readCachedSessionPage(client: NanobotClient, key: string): CachedSessionPage | undefined {
  const pages = cacheByClient.get(client);
  const cached = pages?.get(key);
  if (pages && cached) {
    // Refresh recency so revisiting recently used sessions survives eviction.
    pages.delete(key);
    pages.set(key, cached);
  }
  return cached;
}

export function writeCachedSessionPage(client: NanobotClient, key: string, page: CachedSessionPage): void {
  let pages = cacheByClient.get(client);
  if (!pages) {
    pages = new Map();
    cacheByClient.set(client, pages);
  }
  pages.delete(key);
  pages.set(key, page);
  while (pages.size > MAX_CACHED_SESSIONS) {
    const oldest = pages.keys().next();
    if (oldest.done) break;
    pages.delete(oldest.value);
  }
}

export function clearCachedSessionPage(client: NanobotClient, key: string): void {
  cacheByClient.get(client)?.delete(key);
}

/** Test seam: drop every cached page. */
export function resetSessionPageCache(): void {
  cacheByClient = new WeakMap();
}
