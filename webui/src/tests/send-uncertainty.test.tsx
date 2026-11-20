import { webcrypto } from "node:crypto";
import { act, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreadShell } from "@/components/thread/ThreadShell";
import { preloadMarkdownText } from "@/components/MarkdownText";
import { folderAccessThreadFixture } from "@/tests/fixtures/folder-access-request";
import { useNanobotStream } from "@/hooks/useNanobotStream";
import { NanobotClient } from "@/lib/nanobot-client";
import type { InboundEvent, UIMessage } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

class FakeSocket {
  static instances: FakeSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeSocket.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event?: { code?: number }) => void) | null = null;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(serialized: string): void {
    this.sent.push(serialized);
  }

  close(): void {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.();
  }

  fakeOpen(): void {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  fakeCloseWithCode(code: number): void {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.({ code });
  }

  fakeMessage(event: InboundEvent): void {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent);
  }
}

const clients: NanobotClient[] = [];

afterEach(() => {
  act(() => { for (const client of clients.splice(0)) client.close(); });
  vi.useRealTimers();
  FakeSocket.instances = [];
});

describe("send uncertainty integration", () => {
  it("projects an ACK-lost frame as unknown and reconnects without replaying it", () => {
    vi.useFakeTimers();
    FakeSocket.instances = [];
    const client = new NanobotClient({
      url: "ws://test",
      maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();

    function Wrapper({ children }: { children: ReactNode }) {
      return (
        <ClientProvider
          client={client}
          token="tok"
        >
          {children}
        </ClientProvider>
      );
    }

    const { result, unmount } = renderHook(
      () => useNanobotStream("chat-uncertain", []),
      { wrapper: Wrapper },
    );
    let submitted: ReturnType<typeof result.current.send> = null;
    act(() => {
      submitted = result.current.send("one attempt");
    });
    expect(submitted).not.toBeNull();
    expect(result.current.messages).toEqual([
      expect.objectContaining({
        content: "one attempt",
        deliveryStatus: "sending",
      }),
    ]);

    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));

    expect(result.current.messages).toEqual([
      expect.objectContaining({
        content: "one attempt",
        turnId: submitted!.turnId,
        deliveryStatus: "unknown",
        deliveryErrorKind: "connection_closed",
      }),
    ]);
    expect(client.hasUnsettledRun("chat-uncertain")).toBe(true);

    vi.advanceTimersByTime(500);
    FakeSocket.instances[1]!.fakeOpen();
    const messageFrames = FakeSocket.instances.flatMap((socket) => socket.sent)
      .map((serialized) => JSON.parse(serialized) as { type?: string })
      .filter((frame) => frame.type === "message");
    expect(messageFrames).toHaveLength(1);
    expect(client.hasUnsettledRun("chat-uncertain")).toBe(true);

    unmount();
    client.close();
  });
});


describe("admission refused for session recovery", () => {
  function renderChat(chatId: string) {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })));
    const client = new NanobotClient({ url: "ws://test", reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    const socket = FakeSocket.instances[0]!;
    socket.fakeOpen();
    render(<ClientProvider client={client} token="tok"><ThreadShell
      session={{ key: `websocket:${chatId}`, chatId, channel: "websocket", createdAt: null,
        updatedAt: null, preview: "" }} title="Preview chat" onToggleSidebar={() => {}}
    /></ClientProvider>);
    return { client, socket };
  }

  it.each([false, true])("shows the recovery notice and keeps the input (negotiated=%s)", async (negotiated) => {
    const chatId = "chat-needs-recovery";
    const { client, socket } = renderChat(chatId);
    if (negotiated) {
      vi.stubGlobal("crypto", webcrypto);
      act(() => socket.fakeMessage({
        event: "attached", chat_id: chatId, input_attempt_source: {
          version: 1,
          source: {
            store_identity: "a".repeat(64), session_key: `websocket:${chatId}`,
            incarnation: "chat-incarnation", generation: 0,
          },
          source_token: "b".repeat(32), principal: "c".repeat(64),
        },
      }));
    }
    const content = "Continue the preview.";
    fireEvent.change(screen.getByLabelText("Message input"), { target: { value: content } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    const sent = await waitFor(() => {
      const message = socket.sent.map((frame) => JSON.parse(frame)).find((frame) => frame.type === "message");
      expect(message?.turn_id).toEqual(expect.any(String));
      return message;
    });
    expect(!!sent.input_attempt).toBe(negotiated);
    act(() => socket.fakeMessage({ event: "error", chat_id: chatId, turn_id: sent.turn_id,
      detail: negotiated ? "input_attempt_unknown" : "not_admitted",
      reason: "session_recovery_required" }));

    expect(await screen.findByText("Chat needs recovery")).toBeVisible();
    expect(screen.getByText("Kohaku paused this chat after an interrupted turn. Start a new chat to continue.")).toBeVisible();
    expect(screen.getByText(content)).toBeVisible();
    expect(screen.queryByText("Sending")).not.toBeInTheDocument();
    expect(screen.queryByText("Working")).not.toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "Working" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Working" })).not.toBeInTheDocument();
    if (negotiated) {
      expect(client.hasUnknownMessageSend(chatId)).toBe(true);
      await act(async () => {
        socket.fakeCloseWithCode(1006);
        await client.retry();
        FakeSocket.instances.at(-1)!.fakeOpen();
      });
      const frames = FakeSocket.instances.flatMap((instance) => instance.sent)
        .map((frame) => JSON.parse(frame)).filter((frame) => frame.type === "message");
      expect(frames).toHaveLength(1);
      expect(client.hasUnknownMessageSend(chatId)).toBe(true);
      expect(screen.getByText("Chat needs recovery")).toBeVisible();
    }
  });

  it("keeps newer work active when an older turn receives a recovery rejection", async () => {
    const chatId = "chat-recovery-rejection-race";
    const { client, socket } = renderChat(chatId);
    act(() => client.sendMessage(chatId, "older attempt", undefined, { turnId: "turn-old" }));
    fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "newer attempt" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    const newer = socket.sent.map((frame) => JSON.parse(frame))
      .find((frame) => frame.type === "message" && frame.turn_id !== "turn-old");
    expect(newer).toBeDefined();
    act(() => {
      socket.fakeMessage({ event: "message_admitted", chat_id: chatId, turn_id: newer.turn_id,
        active_turn_id: newer.turn_id, starts_turn: true });
      socket.fakeMessage({ event: "goal_status", chat_id: chatId, turn_id: newer.turn_id,
        status: "running", started_at: Date.now() / 1000 });
    });
    expect(await screen.findByRole("status", { name: "Working" })).toBeVisible();
    act(() => socket.fakeMessage({ event: "error", chat_id: chatId, turn_id: "turn-old",
      detail: "not_admitted", reason: "session_recovery_required" }));
    expect(screen.getByRole("status", { name: "Working" })).toBeVisible();
  });
});

describe("canonical activity projection over the actual client and shell", () => {
  it.each([false, true])("repairs idle history without admitting/replaying an unknown attempt (omitted=%s)", async (omitted) => {
    await preloadMarkdownText();
    const chatId = "chat-title";
    const key = `websocket:${chatId}`;
    let historyReads = 0;
    let canonical: UIMessage[] = [];
    const scope = folderAccessThreadFixture.workspace_scope!;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes(`/api/sessions/${encodeURIComponent(key)}/webui-thread`)) {
        historyReads += 1;
        return { ok: true, status: 200, json: async () => ({
          ...folderAccessThreadFixture, sessionKey: key, messages: canonical,
          active_turn_id: null, has_pending_tool_calls: false, stop_target: null,
          workspace_permissions: { version: 1, runtime_id: "b".repeat(32), revision: 5,
            mode: "restricted", state: "active", can_enable_full: true,
            full_access_unavailable_reason: null, task_access: null, retirement: null },
        }) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
    const client = new NanobotClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();
    render(<ClientProvider client={client} token="tok"><ThreadShell
      session={{ key, chatId, channel: "websocket", createdAt: null, updatedAt: null, preview: "" }}
      title="Unknown projection" onToggleSidebar={() => {}} workspaceScope={scope}
      workspaceControls={{ can_change_project: true, can_use_full_access: true, permission_v2: true }}
      workspaces={{ schema_version: 2, default_access_mode: "restricted", default_scope: scope,
        managed_access: true, management: { grants_revision: 1, grants: [], repositories: [] },
        controls: { can_change_project: true, can_use_full_access: true, permission_v2: true, can_manage_folders: true,
          can_manage_catalog: true, can_discover_repositories: true } }}
    /></ClientProvider>);
    await waitFor(() => expect(historyReads).toBe(1));
    fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "one unknown attempt" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    const sent = FakeSocket.instances[0]!.sent.map((frame) => JSON.parse(frame))
      .find((frame) => frame.type === "message");
    expect(sent).toBeDefined();
    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));
    canonical = omitted
      ? [{ id: "other", role: "assistant", content: "latest reset reply", createdAt: 5 }]
      : [{ id: "user", role: "user", content: "one unknown attempt", turnId: sent.turn_id, createdAt: 1 },
         { id: "answer", role: "assistant", content: "one completed reply", turnId: sent.turn_id, createdAt: 2 }];
    await act(async () => { await client.retry(); FakeSocket.instances[1]!.fakeOpen();
      FakeSocket.instances[1]!.fakeMessage({ event: "attached", chat_id: chatId, stop_target: null }); });
    await waitFor(() => expect(historyReads).toBe(2));
    expect(await screen.findByText(omitted ? "latest reset reply" : "one completed reply")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("Working")).not.toBeInTheDocument());
    expect(screen.getAllByText("one unknown attempt")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Awaiting confirmation" })).toBeInTheDocument();
    expect(client.hasUnknownMessageSend(chatId)).toBe(true);
    expect(client.hasUnsettledRun(chatId)).toBe(false);
    expect(screen.getByRole("button", { name: "Choose folder" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Workspace access mode: Sandbox" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Stop response" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Retry/ })).not.toBeInTheDocument();
    const frames = FakeSocket.instances.flatMap((socket) => socket.sent).map((frame) => JSON.parse(frame));
    expect(frames.filter((frame) => frame.type === "message")).toHaveLength(1);
    act(() => FakeSocket.instances[1]!.fakeMessage({ event: "delta", chat_id: chatId,
      turn_id: sent.turn_id, text: "late stale output" }));
    expect(screen.queryByText("late stale output")).not.toBeInTheDocument();
    expect(client.hasUnknownMessageSend(chatId)).toBe(true);
    await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "message_admitted",
      chat_id: chatId, turn_id: sent.turn_id, active_turn_id: sent.turn_id, starts_turn: true,
      started_at: 123 }); });
    expect(screen.queryByRole("button", { name: "Awaiting confirmation" })).not.toBeInTheDocument();
    expect(client.hasUnknownMessageSend(chatId)).toBe(false);
    expect(screen.getByRole("button", { name: "Workspace access mode: Sandbox" })).toBeEnabled();
    expect(client.getRunStartedAt(chatId)).toBeNull();
    expect(screen.queryByText("Working")).not.toBeInTheDocument();
  });

  it("does not infer admission from attached or turn_end and accepts a later exact receipt", async () => {
    const client = new NanobotClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();
    const { result } = renderHook(() => useNanobotStream("held-admission"), {
      wrapper: ({ children }) => <ClientProvider client={client} token="tok">{children}</ClientProvider>,
    });
    let submitted!: NonNullable<ReturnType<typeof result.current.send>>;
    act(() => { submitted = result.current.send("held")!; });
    let admission: boolean | undefined;
    void submitted.admission!.then((accepted) => { admission = accepted; });
    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));
    await act(async () => { await client.retry(); FakeSocket.instances[1]!.fakeOpen();
      FakeSocket.instances[1]!.fakeMessage({ event: "attached", chat_id: "held-admission" }); });
    expect(admission).toBeUndefined();
    expect(result.current.hasUnknownMessageSend).toBe(true);
    await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "turn_end",
      chat_id: "held-admission", turn_id: submitted.turnId }); });
    expect(admission).toBeUndefined();
    expect(result.current.hasUnknownMessageSend).toBe(true);
    await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "message_admitted",
      chat_id: "held-admission", turn_id: submitted.turnId }); });
    expect(admission).toBe(true);
    expect(result.current.hasUnknownMessageSend).toBe(false);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.messages[0]).toMatchObject({ deliveryStatus: "accepted" });
  });
});

describe("live activity metadata conservatism", () => {
  it.each([
    ["non-null active owner", { active_turn_id: "active-attempt", has_pending_tool_calls: false }],
    ["incomplete persistence", { active_turn_id: null, has_pending_tool_calls: true }],
    ["legacy missing fields", {}],
  ])("keeps the cursor/Working for %s, then converges only on explicit idle", async (_label, liveFields) => {
    await preloadMarkdownText();
    const chatId = "live-conservative";
    let historyReads = 0;
    let messages: UIMessage[] = [];
    let fields: Record<string, unknown> = { active_turn_id: null, has_pending_tool_calls: false };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes(`websocket%3A${chatId}/webui-thread`)) {
        historyReads += 1;
        return { ok: true, status: 200, json: async () => ({ schemaVersion: 3,
          sessionKey: `websocket:${chatId}`, messages, stop_target: null, ...fields }) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
    const client = new NanobotClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();
    render(<ClientProvider client={client} token="tok"><ThreadShell
      session={{ key: `websocket:${chatId}`, chatId, channel: "websocket", createdAt: null,
        updatedAt: null, preview: "" }} title="Live activity" onToggleSidebar={() => {}}
    /></ClientProvider>);
    await waitFor(() => expect(historyReads).toBe(1));
    fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "active input" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    const sent = FakeSocket.instances[0]!.sent.map((frame) => JSON.parse(frame))
      .find((frame) => frame.type === "message");
    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));
    messages = [{ id: "u", role: "user", content: "active input", turnId: sent.turn_id, createdAt: 1 },
      { id: "a", role: "assistant", content: "immutable partial", turnId: sent.turn_id, createdAt: 2 }];
    fields = { ...liveFields, ...(liveFields.active_turn_id ? { active_turn_id: sent.turn_id } : {}) };
    await act(async () => { await client.retry(); FakeSocket.instances[1]!.fakeOpen();
      FakeSocket.instances[1]!.fakeMessage({ event: "attached", chat_id: chatId, stop_target: null }); });
    await waitFor(() => expect(historyReads).toBe(2));
    expect(screen.getByRole("button", { name: "Stop response" })).toBeDisabled();
    expect(screen.queryByText("immutable partial")).not.toBeInTheDocument();
    expect(client.hasUnsettledRun(chatId)).toBe(true);
    expect(client.hasUnknownMessageSend(chatId)).toBe(true);
    act(() => FakeSocket.instances[1]!.fakeMessage({ event: "delta", chat_id: chatId,
      turn_id: sent.turn_id, text: "resumed cursor" }));
    expect(await screen.findByText("resumed cursor")).toBeInTheDocument();
    messages = [messages[0]!, { id: "done", role: "assistant", content: "completed after attach",
      turnId: sent.turn_id, createdAt: 3 }];
    fields = { active_turn_id: null, has_pending_tool_calls: false };
    act(() => FakeSocket.instances[1]!.fakeMessage({ event: "session_updated", chat_id: chatId, scope: "thread" }));
    expect(await screen.findByText("completed after attach")).toBeInTheDocument();
    expect(screen.queryByText("resumed cursor")).not.toBeInTheDocument();
    expect(screen.getAllByText("active input")).toHaveLength(1);
    expect(screen.queryByText("Working")).not.toBeInTheDocument();
    const frames = FakeSocket.instances.flatMap((socket) => socket.sent).map((frame) => JSON.parse(frame));
    expect(frames.filter((frame) => frame.type === "message")).toHaveLength(1);
  });
});

describe("UNKNOWN queue and independent draft projection", () => {
  it.each(["late receipt", "independent send"])("holds the original queued attempt/settings across idle hydration and %s", async (action) => {
    await preloadMarkdownText();
    const chatId = "queue-idle-history";
    const storageKey = `nanobot.webui.composerQueuedGuidance.v1:${chatId}`;
    const ready = { version: 1, state: "ready" } as const;
    const original = { id: "original-queued", text: "original queued input", submission: ready,
      sessionMentions: [{ name: "context", session_key: "websocket:reference", title: "Reference" }] };
    localStorage.setItem(storageKey, JSON.stringify([original, {
      id: "next-queued", text: "later queued input", submission: ready,
    }]));
    let canonical: UIMessage[] = [];
    let historyReads = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes(`websocket%3A${chatId}/webui-thread`)) {
        historyReads += 1;
        return { ok: true, status: 200, json: async () => ({ schemaVersion: 3,
          sessionKey: `websocket:${chatId}`, messages: canonical, active_turn_id: null,
          has_pending_tool_calls: false, stop_target: null }) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
    const client = new NanobotClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();
    const { container } = render(<ClientProvider client={client} token="tok"><ThreadShell
      session={{ key: `websocket:${chatId}`, chatId, channel: "websocket", createdAt: null,
        updatedAt: null, preview: "" }} title="Queue projection" onToggleSidebar={() => {}}
    /></ClientProvider>);
    await screen.findByText("original queued input");
    await waitFor(() => expect(historyReads).toBe(1));
    act(() => FakeSocket.instances[0]!.fakeMessage({ event: "goal_status", chat_id: chatId,
      status: "running", started_at: 10, turn_id: "previous-root" }));
    act(() => FakeSocket.instances[0]!.fakeMessage({ event: "turn_end", chat_id: chatId,
      turn_id: "previous-root" }));
    const messageFrames = () => FakeSocket.instances.flatMap((socket) => socket.sent)
      .map((frame) => JSON.parse(frame) as Record<string, unknown>).filter((frame) => frame.type === "message");
    await waitFor(() => expect(messageFrames()).toHaveLength(1));
    const attempt = messageFrames()[0]!;
    expect(attempt.session_mentions).toEqual(original.sessionMentions);
    fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "rich independent draft" } });
    const fileInput = container.querySelector<HTMLInputElement>('input[type="file"]');
    fireEvent.change(fileInput!, { target: { files: [new File(["fixture"], "draft.txt", { type: "text/plain" })] } });
    await screen.findByText("draft.txt");
    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));
    canonical = [{ id: "queued-user", role: "user", content: "original queued input",
      turnId: String(attempt.turn_id), createdAt: 1 }, { id: "queued-answer", role: "assistant",
      content: "queued completed reply", turnId: String(attempt.turn_id), createdAt: 2 }];
    await act(async () => { await client.retry(); FakeSocket.instances[1]!.fakeOpen();
      FakeSocket.instances[1]!.fakeMessage({ event: "attached", chat_id: chatId, stop_target: null }); });
    await screen.findByText("queued completed reply");
    expect(messageFrames()).toHaveLength(1);
    const queue = screen.getByRole("group", { name: "Queued guidance" });
    expect(within(queue).getByText("original queued input")).toBeInTheDocument();
    expect(within(queue).getByText("later queued input")).toBeInTheDocument();
    expect(within(queue).getByRole("status")).toHaveTextContent("Awaiting confirmation");
    expect(screen.getByLabelText("Message input")).toHaveValue("rich independent draft");
    expect(screen.getByText("draft.txt")).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem(storageKey)!)[0]).toMatchObject({
      ...original,
      submission: {
        version: 1,
        state: "submitted",
        binding: { queueKey: chatId, promptId: original.id, turnId: attempt.turn_id },
      },
    });
    expect(screen.queryByText("Working")).not.toBeInTheDocument();
    if (action === "independent send") {
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      expect(within(queue).getByText("original queued input")).toBeInTheDocument();
      const upload = FakeSocket.instances[1]!.sent.map((frame) => JSON.parse(frame))
        .find((frame) => frame.type === "attachment");
      expect(upload).toBeDefined();
      await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "attachment_stored",
        request_id: upload.request_id, id: "fixture-attachment", name: "draft.txt", mime: "text/plain", bytes: 7 }); });
      await waitFor(() => expect(messageFrames()).toHaveLength(2));
      expect(messageFrames()[1]).toMatchObject({ request_intent: "new", content: "rich independent draft" });
    }
    const currentTurnId = client.getRunTurnId(chatId);
    const currentGeneration = client.getRunGeneration(chatId);
    await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "message_admitted",
      chat_id: chatId, turn_id: String(attempt.turn_id), active_turn_id: String(attempt.turn_id),
      starts_turn: true, started_at: 12 }); });
    expect(screen.queryAllByRole("status").filter((status) =>
      status.textContent?.includes("Awaiting confirmation"))).toHaveLength(0);
    expect(client.getRunTurnId(chatId)).toBe(currentTurnId);
    expect(client.getRunGeneration(chatId)).toBe(currentGeneration);
    expect(messageFrames()).toHaveLength(action === "independent send" ? 2 : 1);
    if (action === "late receipt") {
      expect(screen.queryByRole("button", { name: "Awaiting confirmation" })).not.toBeInTheDocument();
      expect(screen.getByRole("group", { name: "Queued guidance" })).toHaveTextContent("later queued input");
      expect(screen.getByLabelText("Message input")).toHaveValue("rich independent draft");
      expect(screen.getByText("draft.txt")).toBeInTheDocument();
      expect(screen.queryByText("Working")).not.toBeInTheDocument();
    }
  });
});

describe("chat isolation for mounted admission uncertainty", () => {
  it("resyncs the map query and does not leak a retired chat's late proof/output into a newer view", async () => {
    const client = new NanobotClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    clients.push(client);
    client.connect();
    FakeSocket.instances[0]!.fakeOpen();
    const { result, rerender } = renderHook(({ chatId }) => useNanobotStream(chatId), {
      initialProps: { chatId: "old-view" },
      wrapper: ({ children }) => <ClientProvider client={client} token="tok">{children}</ClientProvider>,
    });
    let old!: NonNullable<ReturnType<typeof result.current.send>>;
    act(() => { old = result.current.send("old input")!; });
    act(() => FakeSocket.instances[0]!.fakeCloseWithCode(1006));
    expect(result.current.hasUnknownMessageSend).toBe(true);
    await act(async () => { await client.retry(); FakeSocket.instances[1]!.fakeOpen(); });
    act(() => { client.reconcileCanonicalCompletion("old-view", client.getRunGeneration("old-view"),
      [old.turnId], { activitySnapshotAvailable: true, activeTurnId: null,
        hasPendingToolCalls: false, observedTurnIds: [old.turnId] }); });
    rerender({ chatId: "new-view" });
    expect(result.current.hasUnknownMessageSend).toBe(false);
    let newer!: NonNullable<ReturnType<typeof result.current.send>>;
    act(() => { newer = result.current.send("new view input")!; });
    act(() => FakeSocket.instances[1]!.fakeMessage({ event: "goal_status", chat_id: "new-view",
      status: "running", turn_id: newer.turnId, started_at: 200 }));
    await act(async () => { FakeSocket.instances[1]!.fakeMessage({ event: "message_admitted",
      chat_id: "old-view", turn_id: old.turnId, active_turn_id: old.turnId, starts_turn: true,
      started_at: 100 });
      FakeSocket.instances[1]!.fakeMessage({ event: "delta", chat_id: "old-view", turn_id: old.turnId,
        text: "old delayed output" }); });
    expect(result.current.hasUnknownMessageSend).toBe(false);
    expect(result.current.messages).toEqual([expect.objectContaining({ content: "new view input" })]);
    expect(result.current.isStreaming).toBe(true);
    expect(result.current.runStartedAt).toBe(200);
    expect(client.getRunTurnId("new-view")).toBe(newer.turnId);
    expect(client.hasUnknownMessageSend("old-view")).toBe(false);
  });
});
