/**
 * Window-independent context usage.
 *
 * The context meter follows the latest qualified completion observation of a
 * session, not the history window on screen. While an older page is viewed
 * (the tail is not in the window) the retained tail-derived value is kept; it
 * is never replaced by an older page's usage and never scanned for. Whenever
 * the tail is visible, the freshly computed value is authoritative, including
 * `null` after a compaction freshness invalidation.
 */
export interface RetainedContextUsage<T> {
  key: string | null;
  usage: T | null;
}

export function resolveRetainedContextUsage<T>(
  retained: RetainedContextUsage<T>,
  input: { key: string | null; viewingHistoricalWindow: boolean; computeFresh: () => T | null },
): { usage: T | null; retained: RetainedContextUsage<T> } {
  if (input.viewingHistoricalWindow) {
    return {
      usage: retained.key === input.key ? retained.usage : null,
      retained,
    };
  }
  const fresh = input.computeFresh();
  return { usage: fresh, retained: { key: input.key, usage: fresh } };
}
