import { describe, expect, it } from "vitest";

import { compactReasoningPreview } from "@/components/thread/activity/reasoning-preview";

describe("compactReasoningPreview", () => {
  it("normalizes links and emphasis for short reasoning", () => {
    expect(compactReasoningPreview("**Plan** [docs](https://example.com)")).toBe(
      "Plan docs",
    );
  });

  it("keeps only the recent tail and caps the rendered label", () => {
    const text = `START-OF-REASONING ${"old ".repeat(2_000)}**recent** [docs](https://example.com) ${"tail ".repeat(100)}`;
    const preview = compactReasoningPreview(text);

    expect(preview).not.toContain("START-OF-REASONING");
    expect(preview).not.toContain("**");
    expect(preview.length).toBeLessThanOrEqual(240);
    expect(preview.endsWith("...")).toBe(true);
  });
});
