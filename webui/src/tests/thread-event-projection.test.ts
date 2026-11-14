import { describe, expect, it } from "vitest";

import { stampLastAssistantCompletion } from "@/lib/thread-event-projection";
import type { UIMessage } from "@/lib/types";

const CREATED_AT = Date.UTC(2026, 6, 25, 12, 34, 0);

function message(
  role: UIMessage["role"],
  overrides: Partial<Omit<UIMessage, "role">> = {},
): UIMessage {
  return {
    id: role,
    role,
    content: role,
    createdAt: CREATED_AT,
    ...overrides,
  };
}

describe("stampLastAssistantCompletion", () => {
  it("leaves prior completion metadata intact when no message matches the turn", () => {
    const messages = [
      message("assistant", {
        id: "old-answer",
        content: "old answer",
        turnId: "turn-old",
        completedAt: 1,
      }),
    ];
    const original = messages[0];

    const stamped = stampLastAssistantCompletion(
      messages,
      { completedAt: 2 },
      "turn-empty",
    );

    expect(stamped).toBe(messages);
    expect(stamped).toEqual(messages);
    expect(stamped[0]).toBe(original);
    expect(stamped[0]?.completedAt).toBe(1);
  });

  it("stamps only the assistant message whose turn id matches", () => {
    const messages = [
      message("assistant", {
        id: "old-answer",
        content: "old answer",
        turnId: "turn-old",
        completedAt: 1,
      }),
      message("assistant", {
        id: "other-answer",
        content: "other answer",
        turnId: "turn-other",
      }),
    ];

    const stamped = stampLastAssistantCompletion(
      messages,
      { completedAt: 2 },
      "turn-old",
    );

    expect(stamped[0]?.completedAt).toBe(2);
    expect(stamped[1]?.completedAt).toBeUndefined();
  });

  it("keeps the previous behavior when no turn id is provided", () => {
    const messages = [
      message("assistant", {
        id: "first-answer",
        turnId: "turn-old",
        completedAt: 1,
      }),
      message("user", {
        id: "next-question",
        turnId: "turn-new",
      }),
      message("assistant", {
        id: "last-answer",
        turnId: "turn-new",
      }),
    ];

    const stamped = stampLastAssistantCompletion(messages, { completedAt: 2 });

    expect(stamped[0]?.completedAt).toBe(1);
    expect(stamped[2]?.completedAt).toBe(2);
  });

  // Requirements-only case C7 (use case U7): completion metadata stays inside
  // the originating turn. The current user boundary follows turn T1; T2 ends
  // with no assistant row; an event without a turn id must stamp nothing.
  // Source: a3-evidence/a3-reconciliation-2026-11-14/req-a34p34/CASES.md (C7).
  it("does not stamp an older assistant across the current user boundary when the event has no turn id", () => {
    const messages = [
      message("assistant", {
        id: "turn-one-answer",
        content: "answer one",
        turnId: "turn-1",
        completedAt: 1,
        latencyMs: 100,
      }),
      message("user", {
        id: "turn-two-question",
        content: "question two",
        turnId: "turn-2",
      }),
    ];
    const original = messages[0];

    const afterFirst = stampLastAssistantCompletion(messages, { completedAt: 2 });
    expect(afterFirst).toBe(messages);
    expect(afterFirst).toHaveLength(2);
    expect(afterFirst[0]).toBe(original);
    expect(afterFirst[0]?.completedAt).toBe(1);
    expect(afterFirst[0]?.latencyMs).toBe(100);
    expect(afterFirst.filter((item) => item.role === "assistant")).toHaveLength(1);

    const afterSecond = stampLastAssistantCompletion(afterFirst, {
      completedAt: 3,
      latencyMs: 999,
    });
    expect(afterSecond).toBe(afterFirst);
    expect(afterSecond[0]?.completedAt).toBe(1);
    expect(afterSecond[0]?.latencyMs).toBe(100);
  });
});
