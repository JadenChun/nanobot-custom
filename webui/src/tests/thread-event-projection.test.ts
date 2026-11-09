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
});
