import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { useKohakuStream } from "@/hooks/useKohakuStream";
import { KohakuClient } from "@/lib/kohaku-client";
import { ClientProvider } from "@/providers/ClientProvider";

class SocketStub {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = SocketStub.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event?: { code?: number }) => void) | null = null;

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = SocketStub.CLOSED;
    this.onclose?.();
  }

  open() {
    this.readyState = SocketStub.OPEN;
    this.onopen?.();
  }

  receive(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function IntegrationComposer({ chatId }: { chatId: string }) {
  const stream = useKohakuStream(chatId);
  return (
    <ThreadComposer
      onSend={stream.send}
      onJoinQueuedAttempt={stream.joinQueuedAttempt}
      onConsumeQueuedAttempt={stream.consumeQueuedAttempt}
      isStreaming={stream.isStreaming}
      queueCompletionTurnId={stream.queueCompletionTurnId}
      automaticQueueSendBlocked={stream.hasUnknownMessageSend || !stream.connectionOpen}
      pendingQueueKey={chatId}
      placeholder="Type your message..."
    />
  );
}

function messageFrames(socket: SocketStub) {
  return socket.sent
    .map((frame) => JSON.parse(frame) as Record<string, unknown>)
    .filter((frame) => frame.type === "message");
}

describe("queued guidance owner admission over the actual client boundary", () => {
  let client: KohakuClient | null = null;

  afterEach(() => {
    act(() => client?.close());
    client = null;
  });

  it.each(["cancelled", "completed while offscreen"] as const)(
    "never drains on a turn %s",
    async (caseName) => {
      const chatId = `queue-not-natural-${caseName}`;
      const socket = new SocketStub();
      const current = client = new KohakuClient({
        url: "ws://test",
        reconnect: false,
        socketFactory: () => socket as unknown as WebSocket,
      });
      current.connect();
      socket.open();
      const view = (
        <ClientProvider client={current} token="test-token">
          <IntegrationComposer chatId={chatId} />
        </ClientProvider>
      );
      const mounted = render(view);
      act(() => socket.receive({
        event: "goal_status", chat_id: chatId, status: "running",
        started_at: 1_700_000_000, turn_id: "root-turn",
      }));
      const input = screen.getByLabelText("Message input");
      fireEvent.change(input, { target: { value: "leave this queued" } });
      fireEvent.keyDown(input, { key: "Enter" });

      if (caseName === "completed while offscreen") mounted.unmount();
      act(() => socket.receive({
        event: "turn_end", chat_id: chatId, turn_id: "root-turn",
        ...(caseName === "cancelled" ? { outcome: "cancelled" } : {}),
      }));
      if (caseName === "completed while offscreen") {
        render(view);
        // A delayed duplicate must not make the offscreen completion live.
        act(() => socket.receive({ event: "turn_end", chat_id: chatId, turn_id: "root-turn" }));
      }
      expect(await screen.findByText("leave this queued")).toBeInTheDocument();
      expect(messageFrames(socket)).toEqual([]);
    },
  );

  it.each(["reconnect", "missing turn identity"] as const)(
    "keeps the ready row when completion is not verified: %s",
    async (scenario) => {
      const chatId = `queue-proof-${scenario}`;
      const first = new SocketStub();
      const second = new SocketStub();
      const sockets = [first, second];
      const current = client = new KohakuClient({
        url: "ws://test",
        socketFactory: () => sockets.shift()! as unknown as WebSocket,
      });
      current.connect();
      first.open();
      render(<ClientProvider client={current} token="test-token">
        <IntegrationComposer chatId={chatId} />
      </ClientProvider>);

      const input = screen.getByLabelText("Message input");
      fireEvent.change(input, { target: { value: "start a run" } });
      fireEvent.keyDown(input, { key: "Enter" });
      const [root] = messageFrames(first);
      expect(root?.turn_id).toEqual(expect.any(String));
      act(() => first.receive({
        event: "goal_status", chat_id: chatId, status: "running",
        started_at: 1_700_000_000, turn_id: root.turn_id,
      }));
      fireEvent.change(input, { target: { value: "waiting follow-up" } });
      fireEvent.keyDown(input, { key: "Enter" });

      if (scenario === "reconnect") {
        await act(async () => {
          first.close();
          await current.retry();
          second.open();
          second.receive({ event: "attached", chat_id: chatId });
          second.receive({ event: "goal_status", chat_id: chatId, status: "idle", turn_id: root.turn_id });
        });
        expect(screen.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
        expect(messageFrames(second)).toEqual([]);
        act(() => second.receive({ event: "turn_end", chat_id: chatId, turn_id: root.turn_id }));
      } else {
        act(() => first.receive({ event: "turn_end", chat_id: chatId }));
      }

      expect(screen.getByText("waiting follow-up")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Send", exact: true })).toBeEnabled();
      expect(messageFrames(first)).toEqual([root]);
      expect(messageFrames(second)).toEqual([]);
    },
  );

  it("keeps the exact row through route acceptance and removes it only on its owner receipt", async () => {
    const chatId = "chat-queued-guidance-integration";
    const socket = new SocketStub();
    client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: () => socket as unknown as WebSocket,
    });
    client.connect();
    socket.open();

    render(
      <ClientProvider client={client} token="test-token">
        <IntegrationComposer chatId={chatId} />
      </ClientProvider>,
    );

    act(() => {
      socket.receive({
        event: "goal_status",
        chat_id: chatId,
        status: "running",
        started_at: 1_700_000_000,
        turn_id: "root-turn",
      });
    });

    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "queued follow-up" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.change(input, { target: { value: "independent draft" } });

    act(() => {
      socket.receive({
        event: "turn_end",
        chat_id: chatId,
        turn_id: "root-turn",
        outcome: "completed",
      });
    });

    let queuedAttempt: Record<string, unknown> | undefined;
    await waitFor(() => {
      queuedAttempt = messageFrames(socket).find((frame) => (
        frame.chat_id === chatId && frame.content === "queued follow-up"
      ));
      expect(queuedAttempt).toBeDefined();
    });
    const attemptId = queuedAttempt?.turn_id;
    expect(typeof attemptId).toBe("string");

    act(() => {
      socket.receive({
        event: "message_accepted",
        chat_id: chatId,
        turn_id: attemptId,
        active_turn_id: attemptId,
        starts_turn: true,
      });
    });
    expect(screen.getByText("queued follow-up")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Awaiting confirmation");
    expect(screen.getByLabelText("Message input")).toHaveValue("independent draft");

    act(() => {
      socket.receive({
        event: "message_admitted",
        chat_id: "another-chat",
        turn_id: attemptId,
      });
      socket.receive({
        event: "message_admitted",
        chat_id: chatId,
        turn_id: "unrelated-attempt",
        active_turn_id: attemptId,
      });
    });
    expect(screen.getByText("queued follow-up")).toBeInTheDocument();

    await act(async () => {
      socket.receive({
        event: "message_admitted",
        chat_id: chatId,
        turn_id: attemptId,
        active_turn_id: attemptId,
        starts_turn: true,
      });
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.queryByText("queued follow-up")).not.toBeInTheDocument();
    });
    expect(screen.getByLabelText("Message input")).toHaveValue("independent draft");
  });
});
