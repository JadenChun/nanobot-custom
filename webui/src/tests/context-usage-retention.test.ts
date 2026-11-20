import { describe, expect, it } from "vitest";

import { resolveRetainedContextUsage, type RetainedContextUsage } from "@/lib/context-usage-retention";

type U = { contextTokens: number };

describe("window-independent context usage", () => {
  it("keeps tail usage while viewing an older window and never uses older-page usage", () => {
    let retained: RetainedContextUsage<U> = { key: null, usage: null };
    const tail = resolveRetainedContextUsage(retained, {
      key: "websocket:a", viewingHistoricalWindow: false, computeFresh: () => ({ contextTokens: 900 }),
    });
    retained = tail.retained;
    expect(tail.usage).toEqual({ contextTokens: 900 });
    let computed = 0;
    const older = resolveRetainedContextUsage(retained, {
      key: "websocket:a", viewingHistoricalWindow: true,
      computeFresh: () => { computed += 1; return { contextTokens: 10 }; },
    });
    expect(computed).toBe(0);
    expect(older.usage).toEqual({ contextTokens: 900 });
  });

  it("is unavailable for another session or before any tail observation", () => {
    const retained: RetainedContextUsage<U> = { key: "websocket:a", usage: { contextTokens: 5 } };
    expect(resolveRetainedContextUsage(retained, {
      key: "websocket:b", viewingHistoricalWindow: true, computeFresh: () => ({ contextTokens: 1 }),
    }).usage).toBeNull();
  });

  it("honors compaction invalidation and updates on the next qualified observation", () => {
    let retained: RetainedContextUsage<U> = { key: "websocket:a", usage: { contextTokens: 5 } };
    const invalid = resolveRetainedContextUsage(retained, {
      key: "websocket:a", viewingHistoricalWindow: false, computeFresh: () => null,
    });
    retained = invalid.retained;
    expect(invalid.usage).toBeNull();
    expect(resolveRetainedContextUsage(retained, {
      key: "websocket:a", viewingHistoricalWindow: true, computeFresh: () => ({ contextTokens: 1 }),
    }).usage).toBeNull();
    const next = resolveRetainedContextUsage(retained, {
      key: "websocket:a", viewingHistoricalWindow: false, computeFresh: () => ({ contextTokens: 7 }),
    });
    expect(next.usage).toEqual({ contextTokens: 7 });
  });
});
