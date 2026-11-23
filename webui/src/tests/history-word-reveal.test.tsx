import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ThreadMessages } from "@/components/thread/ThreadMessages";
import type { UIMessage } from "@/lib/types";

describe("history word reveal", () => {
  it("animates only the latest completed assistant body and stops when it changes", async () => {
    const messages: UIMessage[] = [
      { id: "user", role: "user", content: "Question", createdAt: 1 },
      {
        id: "older", role: "assistant", content: "Earlier answer", turnId: "older-turn", createdAt: 2,
      },
      {
        id: "latest", role: "assistant", content: "Latest answer", turnId: "latest-turn", createdAt: 3,
      },
    ];
    const { container, rerender } = render(
      <ThreadMessages messages={messages} revealLatestHistoryMessage />,
    );

    await waitFor(() => {
      expect(container.querySelectorAll(".markdown-content")).toHaveLength(2);
    });
    const bodies = container.querySelectorAll(".markdown-content");
    expect(bodies[0].querySelector("[data-sd-animate]")).not.toBeInTheDocument();
    expect(bodies[1].querySelectorAll("[data-sd-animate]").length).toBeGreaterThan(0);

    rerender(
      <ThreadMessages
        messages={messages.map((message) => message.id === "latest"
          ? { ...message, content: "Updated answer" }
          : message)}
        revealLatestHistoryMessage
      />,
    );

    await waitFor(() => {
      expect(container).toHaveTextContent("Updated answer");
      expect(container.querySelectorAll(".markdown-content [data-sd-animate]")).toHaveLength(0);
    });
  });
});
