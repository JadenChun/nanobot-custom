import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreadComposer } from "@/components/thread/ThreadComposer";

const queueKey = (chatId: string) => `nanobot.webui.composerQueuedGuidance.v1:${chatId}`;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("restored queued-row protection", () => {
  it("quarantines legacy and malformed provenance as read-only unknown rows", async () => {
    const chatId = "legacy-queue";
    localStorage.setItem(queueKey(chatId), JSON.stringify([
      { id: "legacy-row", text: "legacy submitted snapshot" },
      {
        id: "malformed-row",
        text: "unsupported submitted snapshot",
        submission: { version: 99, state: "ready" },
      },
    ]));
    const onSend = vi.fn();
    const { rerender } = render(
      <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
    );
    expect(await screen.findByText("legacy submitted snapshot")).toBeInTheDocument();
    expect(screen.getByText("unsupported submitted snapshot")).toBeInTheDocument();

    rerender(<ThreadComposer onSend={onSend} isStreaming={false} pendingQueueKey={chatId} />);

    const legacyRow = screen.getByText("legacy submitted snapshot")
      .closest("[data-queued-prompt-row='true']");
    const malformedRow = screen.getByText("unsupported submitted snapshot")
      .closest("[data-queued-prompt-row='true']");
    expect(legacyRow).toBeTruthy();
    expect(malformedRow).toBeTruthy();
    for (const row of [legacyRow!, malformedRow!]) {
      expect(within(row).getByRole("status")).toHaveTextContent("Awaiting confirmation");
      expect(within(row).getByRole("button", { name: "Send", exact: true })).toBeDisabled();
      expect(within(row).getByRole("button", { name: "Edit guidance" })).toBeDisabled();
      expect(within(row).getByRole("button", { name: "Delete guidance" })).toBeDisabled();
    }
    expect(onSend).not.toHaveBeenCalled();
  });

  it("does not dispatch when mandatory protection hits a storage quota error", async () => {
    const chatId = "quota-queue";
    const onSend = vi.fn();
    const { rerender } = render(
      <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
    );
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "must remain safe" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const savedBeforeAttempt = localStorage.getItem(queueKey(chatId));
    expect(savedBeforeAttempt).toContain('"state":"ready"');

    vi.spyOn(window.localStorage, "setItem").mockImplementationOnce(() => {
      throw new DOMException("quota exceeded", "QuotaExceededError");
    });
    rerender(<ThreadComposer onSend={onSend} pendingQueueKey={chatId} queueCompletionTurnId="completed" />);

    expect(await screen.findByText("must remain safe")).toBeInTheDocument();
    expect(screen.getByText("Queued messages could not be saved safely. Try again before sending."))
      .toBeInTheDocument();
    expect(onSend).not.toHaveBeenCalled();
    expect(localStorage.getItem(queueKey(chatId))).toBe(savedBeforeAttempt);
    expect(screen.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
  });

  it("does not dispatch when protection serialization fails", async () => {
    const chatId = "serialization-queue";
    const onSend = vi.fn();
    const { rerender } = render(
      <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
    );
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "must serialize first" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const savedBeforeAttempt = localStorage.getItem(queueKey(chatId));
    vi.spyOn(JSON, "stringify").mockImplementationOnce(() => {
      throw new TypeError("serialization failed");
    });

    rerender(<ThreadComposer onSend={onSend} pendingQueueKey={chatId} queueCompletionTurnId="completed" />);

    expect(await screen.findByText("must serialize first")).toBeInTheDocument();
    expect(screen.getByText("Queued messages could not be saved safely. Try again before sending."))
      .toBeInTheDocument();
    expect(onSend).not.toHaveBeenCalled();
    expect(localStorage.getItem(queueKey(chatId))).toBe(savedBeforeAttempt);
    expect(JSON.parse(savedBeforeAttempt!)[0].submission).toEqual({ version: 1, state: "ready" });
  });

  it.each(["missing", "truncated"] as const)(
    "blocks dispatch when the protected queue readback is %s",
    async (readback) => {
      const chatId = `readback-${readback}`;
      const onSend = vi.fn();
      const { rerender } = render(
        <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
      );
      const input = screen.getByLabelText("Message input");
      fireEvent.change(input, { target: { value: "readback must verify" } });
      fireEvent.keyDown(input, { key: "Enter" });

      const originalSetItem = window.localStorage.setItem.bind(window.localStorage);
      vi.spyOn(window.localStorage, "setItem").mockImplementationOnce((key, value) => {
        if (readback === "truncated") originalSetItem(key, value.slice(0, -1));
      });
      rerender(<ThreadComposer onSend={onSend} pendingQueueKey={chatId} queueCompletionTurnId="completed" />);

      expect(await screen.findByText("readback must verify")).toBeInTheDocument();
      expect(screen.getByText("Queued messages could not be saved safely. Try again before sending."))
        .toBeInTheDocument();
      expect(onSend).not.toHaveBeenCalled();
    },
  );

  it("keeps a full 20-row queue intact when protecting one row and preserves the separate draft", async () => {
    const chatId = "full-queue-protection";
    const rows = [
      { id: "ready-head", text: "ready head", submission: { version: 1, state: "ready" } },
      ...Array.from({ length: 19 }, (_, index) => ({
        id: `held-${index}`,
        text: `held snapshot ${index}`,
        submission: { version: 1, state: "unknown" },
      })),
    ];
    localStorage.setItem(queueKey(chatId), JSON.stringify(rows));
    const onSend = vi.fn();
    const { container } = render(
      <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
    );
    expect(await screen.findByText("ready head")).toBeInTheDocument();
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "rich later draft" } });
    const fileInput = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).toBeTruthy();
    fireEvent.change(fileInput!, {
      target: { files: [new File(["later attachment"], "later.txt", { type: "text/plain" })] },
    });
    await screen.findByText("later.txt");

    const head = screen.getByText("ready head").closest("[data-queued-prompt-row='true']");
    expect(head).toBeTruthy();
    fireEvent.click(within(head!).getByRole("button", { name: "Steer", exact: true }));

    const stored = JSON.parse(localStorage.getItem(queueKey(chatId))!);
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(stored).toHaveLength(20);
    expect(stored[0]).toMatchObject({
      id: "ready-head",
      submission: {
        version: 1,
        state: "submitted",
        binding: { queueKey: chatId, promptId: "ready-head", turnId: expect.any(String) },
      },
    });
    expect(stored.slice(1).every((row: { submission: { state: string } }) => (
      row.submission.state === "unknown"
    ))).toBe(true);
    expect(input).toHaveValue("rich later draft");
    expect(screen.getByText("later.txt")).toBeInTheDocument();
  });

  it("preserves the full accepted payload and its original binding in the queue snapshot", () => {
    const chatId = "normalized-held-binding";
    const longText = "x".repeat(4_500);
    const onSend = vi.fn();
    const { container } = render(
      <ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />,
    );
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: longText } });
    fireEvent.keyDown(input, { key: "Enter" });
    const queuedRow = container.querySelector("[data-queued-prompt-row='true']");
    expect(queuedRow).toBeTruthy();

    fireEvent.click(within(queuedRow!).getByRole("button", { name: "Steer", exact: true }));

    const saved = JSON.parse(localStorage.getItem(queueKey(chatId))!)[0];
    expect(onSend).toHaveBeenCalledWith(
      longText,
      undefined,
      expect.objectContaining({ queuedAttempt: expect.any(Object) }),
    );
    expect(saved.text).toBe(longText);
    expect(saved.submission).toMatchObject({
      version: 1,
      state: "submitted",
      binding: {
        queueKey: chatId,
        promptId: saved.id,
        turnId: expect.any(String),
      },
    });
  });

  it("protects a row only when every over-cap snapshot remains present", async () => {
    const chatId = "over-cap-queue";
    const rows = [
      { id: "ready-head", text: "ready head", submission: { version: 1, state: "ready" } },
      ...Array.from({ length: 20 }, (_, index) => ({
        id: `held-${index}`,
        text: `held snapshot ${index}`,
        submission: { version: 1, state: "unknown" },
      })),
    ];
    const encoded = JSON.stringify(rows);
    localStorage.setItem(queueKey(chatId), encoded);
    const onSend = vi.fn();
    render(<ThreadComposer onSend={onSend} isStreaming pendingQueueKey={chatId} />);
    expect(await screen.findByText("ready head")).toBeInTheDocument();

    const readyRow = screen.getByText("ready head").closest("[data-queued-prompt-row='true']");
    expect(readyRow).toBeTruthy();
    fireEvent.click(within(readyRow!).getByRole("button", { name: "Steer" }));

    expect(onSend).not.toHaveBeenCalled();
    expect(localStorage.getItem(queueKey(chatId))).toBe(encoded);
    const stored = JSON.parse(localStorage.getItem(queueKey(chatId))!);
    expect(stored).toHaveLength(21);
    expect(stored[0].submission).toEqual({ version: 1, state: "ready" });
    expect(stored.slice(1).every((row: { submission: { state: string } }) => (
      row.submission.state === "unknown"
    ))).toBe(true);
    expect(screen.getByText("Queued messages could not be saved safely. Try again before sending."))
      .toBeInTheDocument();
  });
});
