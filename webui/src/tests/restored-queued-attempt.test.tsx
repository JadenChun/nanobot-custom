import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { useNanobotStream, type SendAttachment, type SendOptions, type SubmittedTurn } from "@/hooks/useNanobotStream";
import { NanobotClient, type QueuedAttemptBinding } from "@/lib/nanobot-client";
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

  disconnect(code = 1006) {
    this.readyState = SocketStub.CLOSED;
    this.onclose?.({ code });
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
  const stream = useNanobotStream(chatId);
  return (
    <ThreadComposer
      onSend={stream.send}
      onJoinQueuedAttempt={stream.joinQueuedAttempt}
      onConsumeQueuedAttempt={stream.consumeQueuedAttempt}
      isStreaming={stream.isStreaming}
      automaticQueueSendBlocked={stream.hasUnknownMessageSend || !stream.connectionOpen}
      pendingQueueKey={chatId}
      placeholder="Type your message..."
    />
  );
}

function OwnerBackedComposer({
  client,
  chatId,
  isStreaming,
}: {
  client: NanobotClient;
  chatId: string;
  isStreaming: boolean;
}) {
  const onSend = (
    content: string,
    _images?: SendAttachment[],
    options?: SendOptions,
  ): SubmittedTurn | null => {
    const binding = options?.queuedAttempt;
    if (!binding || !client.prepareQueuedAttempt(
      chatId,
      binding,
      !options?.continueActiveTurn,
    )) return null;
    client.sendMessage(chatId, content, undefined, {
      turnId: binding.turnId,
      startsNewRun: !options.continueActiveTurn,
      requestIntent: options.continueActiveTurn ? "steer" : "new",
      ...(options.continueActiveTurn && client.getRunTurnId(chatId)
        ? { targetTurnId: client.getRunTurnId(chatId)! }
        : {}),
    });
    return {
      turnId: binding.turnId,
      userMessageId: `queued-user-${binding.promptId}`,
      sideChannel: false,
    };
  };

  return (
    <ThreadComposer
      onSend={onSend}
      onJoinQueuedAttempt={(binding) => client.waitForQueuedAttempt(chatId, binding)}
      onConsumeQueuedAttempt={(binding) => client.consumeQueuedAttempt(chatId, binding)}
      isStreaming={isStreaming}
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

function queueKey(chatId: string) {
  return `nanobot.webui.composerQueuedGuidance.v1:${chatId}`;
}

describe('actual-client restored queue anti-replay',()=>{
 let clients: NanobotClient[]=[];
 afterEach(()=>{act(()=>clients.forEach(c=>c.close()));clients=[];localStorage.clear();});
 function connected(){const socket=new SocketStub();const client=new NanobotClient({url:'ws://test',reconnect:false,socketFactory:()=>socket as unknown as WebSocket});client.connect();socket.open();clients.push(client);return {client,socket};}
 it.each(['same client remount','recreated client'])('does not transmit the unresolved row again after %s',async mode=>{
  const chatId='restored-real-owner';let current=connected();const originalSocket=current.socket;
  const view=()=> <ClientProvider client={current.client} token="test-token"><IntegrationComposer chatId={chatId}/></ClientProvider>;
  const first=render(view());
  act(()=>current.socket.receive({event:'goal_status',chat_id:chatId,status:'running',started_at:1700000000,turn_id:'original-root'}));
  fireEvent.change(screen.getByLabelText('Message input'),{target:{value:'same queued snapshot'}});
  fireEvent.keyDown(screen.getByLabelText('Message input'),{key:'Enter'});
  fireEvent.click(screen.getByRole('button',{name:'Steer',exact:true}));
  expect(messageFrames(originalSocket)).toHaveLength(1);
  expect(screen.getByRole('button',{name:'Steer',exact:true})).toBeDisabled();
  first.unmount();
  if(mode==='recreated client'){act(()=>current.client.close());current=connected();}
  render(view());
  act(()=>current.socket.receive({event:'goal_status',chat_id:chatId,status:'running',started_at:1700000000,turn_id:'original-root'}));
  await screen.findByText('same queued snapshot');
  fireEvent.click(screen.getByRole('button',{name:'Steer',exact:true}));
  const all=mode==='same client remount'?messageFrames(originalSocket):[...messageFrames(originalSocket),...messageFrames(current.socket)];
  expect(all).toHaveLength(1);
 });

 it("keeps the original actual-client attempt held through an A-to-B-to-A chat switch", async ()=>{
  const chatA = "actual-client-a-to-b-to-a";
  const chatB = "actual-client-other-chat";
  const current = connected();
  const view = (chatId: string) => (
    <ClientProvider client={current.client} token="test-token">
      <IntegrationComposer chatId={chatId}/>
    </ClientProvider>
  );
  const mounted = render(view(chatA));
  act(() => current.socket.receive({
    event: "goal_status", chat_id: chatA, status: "running", started_at: 1_700_000_000,
    turn_id: "a-active-root",
  }));
  fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "A queued snapshot" } });
  fireEvent.keyDown(screen.getByLabelText("Message input"), { key: "Enter" });
  fireEvent.click(screen.getByRole("button", { name: "Steer", exact: true }));
  const original = messageFrames(current.socket)[0];
  expect(original).toBeDefined();
  expect(typeof original?.turn_id).toBe("string");

  mounted.rerender(view(chatB));
  mounted.rerender(view(chatA));
  expect(await screen.findByText("A queued snapshot")).toBeInTheDocument();
  const row = screen.getByText("A queued snapshot").closest("[data-queued-prompt-row='true']");
  expect(row).toBeTruthy();
  expect(within(row!).getByRole("button", { name: "Steer", exact: true })).toBeDisabled();
  expect(JSON.parse(localStorage.getItem(queueKey(chatA))!)[0].submission).toMatchObject({
    version: 1,
    state: "submitted",
    binding: {
      queueKey: chatA,
      promptId: expect.any(String),
      turnId: original?.turn_id,
    },
  });
  expect(messageFrames(current.socket)).toHaveLength(1);
 });
});

describe("actual-client queued outcome joins", () => {
  let clients: NanobotClient[] = [];
  afterEach(() => {
    act(() => clients.forEach((client) => client.close()));
    clients = [];
    localStorage.clear();
  });

  function connected() {
    const sockets: SocketStub[] = [];
    const client = new NanobotClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: () => {
        const socket = new SocketStub();
        sockets.push(socket);
        return socket as unknown as WebSocket;
      },
    });
    client.connect();
    sockets[0]!.open();
    clients.push(client);
    return { client, socket: sockets[0]!, sockets };
  }

  function startQueuedAttempt() {
    const chatId = "queued-outcome-owner";
    const current = connected();
    clients.push(current.client);
    const view = () => (
      <ClientProvider client={current.client} token="test-token">
        <IntegrationComposer chatId={chatId} />
      </ClientProvider>
    );
    const mounted = render(view());
    act(() => current.socket.receive({
      event: "goal_status",
      chat_id: chatId,
      status: "running",
      started_at: 1_700_000_000,
      turn_id: "owner-root-turn",
    }));
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "one queued owner attempt" },
    });
    fireEvent.keyDown(screen.getByLabelText("Message input"), { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "Steer", exact: true }));
    const frame = messageFrames(current.socket)[0];
    if (!frame || typeof frame.turn_id !== "string") {
      throw new Error("queued attempt did not produce a message frame");
    }
    return { chatId, current, frame, mounted, view };
  }

  async function startNewQueuedAttempt(chatId: string, promptText: string) {
    const current = connected();
    const view = (isStreaming: boolean) => (
      <ClientProvider client={current.client} token="test-token">
        <OwnerBackedComposer client={current.client} chatId={chatId} isStreaming={isStreaming} />
      </ClientProvider>
    );
    const mounted = render(view(true));
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: promptText } });
    fireEvent.keyDown(input, { key: "Enter" });
    const nextPromptText = "later queued item";
    fireEvent.change(input, { target: { value: nextPromptText } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.change(input, { target: { value: "rich later draft" } });
    const fileInput = mounted.container.querySelector<HTMLInputElement>('input[type="file"]');
    if (!fileInput) throw new Error("composer file input is missing");
    fireEvent.change(fileInput, {
      target: { files: [new File(["later attachment"], "later.txt", { type: "text/plain" })] },
    });
    await screen.findByText("later.txt");
    mounted.rerender(view(false));
    await waitFor(() => expect(messageFrames(current.socket)).toHaveLength(1));
    const frame = messageFrames(current.socket)[0];
    if (!frame || typeof frame.turn_id !== "string") {
      throw new Error("queued new turn did not produce a message frame");
    }
    return { chatId, current, frame, mounted, view };
  }

  function receiveAdmission(
    socket: SocketStub,
    chatId: string,
    turnId: string,
  ) {
    socket.receive({
      event: "message_admitted",
      chat_id: chatId,
      turn_id: turnId,
      active_turn_id: "owner-root-turn",
      starts_turn: false,
    });
  }

  function idleSnapshot(client: NanobotClient, chatId: string, turnId: string) {
    return client.reconcileCanonicalCompletion(
      chatId,
      client.getRunGeneration(chatId),
      [turnId],
      {
        activitySnapshotAvailable: true,
        activeTurnId: null,
        hasPendingToolCalls: false,
        observedTurnIds: [turnId],
      },
    );
  }

  it("keeps late running proof held after canonical idle, then consumes only the exact late receipt", async () => {
    const chatId = "late-running-after-idle";
    const { current, frame, mounted } = await startNewQueuedAttempt(chatId, "queued original");
    const turnId = String(frame.turn_id);
    const binding = JSON.parse(localStorage.getItem(queueKey(chatId))!)[0]
      .submission.binding as QueuedAttemptBinding;
    const observation = current.client.waitForQueuedAttempt(chatId, binding);
    expect(observation).not.toBeNull();
    expect(binding.turnId).toBe(turnId);
    expect(screen.getByRole("status")).toHaveTextContent("Awaiting confirmation");

    current.socket.disconnect();
    expect(current.client.hasUnknownMessageSend(chatId)).toBe(true);
    expect(idleSnapshot(current.client, chatId, turnId)).toBe(true);
    expect(current.client.hasUnknownMessageSend(chatId)).toBe(true);
    expect(current.client.getRunStartedAt(chatId)).toBeNull();
    current.client.connect();
    const resumedSocket = current.sockets[1];
    expect(resumedSocket).toBeDefined();
    resumedSocket!.open();

    await act(async () => {
      resumedSocket!.receive({
        event: "goal_status",
        chat_id: chatId,
        status: "running",
        started_at: 1_700_000_123,
        turn_id: turnId,
      });
      await Promise.resolve();
    });

    expect(screen.getByText("queued original")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Awaiting confirmation");
    expect(JSON.parse(localStorage.getItem(queueKey(chatId))!)[0].submission.state)
      .toBe("submitted");
    expect(current.client.hasUnknownMessageSend(chatId)).toBe(true);
    expect(current.client.getRunStartedAt(chatId)).toBeNull();

    await act(async () => {
      resumedSocket!.receive({
        event: "message_admitted",
        chat_id: chatId,
        turn_id: turnId,
        active_turn_id: turnId,
        starts_turn: true,
        started_at: 1_700_000_123,
      });
      expect(idleSnapshot(current.client, chatId, turnId)).toBe(true);
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.queryByText("queued original")).not.toBeInTheDocument();
    });
    expect(await observation).toEqual({ kind: "admitted" });
    expect(screen.getByText("later queued item")).toBeInTheDocument();
    expect(screen.getByLabelText("Message input")).toHaveValue("rich later draft");
    expect(screen.getByText("later.txt")).toBeInTheDocument();
    expect(current.client.hasUnsettledRun(chatId)).toBe(false);
    expect(current.client.getRunStartedAt(chatId)).toBeNull();
    expect(messageFrames(current.socket)).toHaveLength(1);
    expect(messageFrames(resumedSocket!)).toHaveLength(0);

    const replayed: unknown[] = [];
    const unsubscribe = current.client.onChat(chatId, (event) => replayed.push(event));
    expect(replayed).toEqual([]);
    unsubscribe();
    mounted.unmount();
  });

  it("requires a numeric started_at for running-proof admission", async () => {
    const chatId = "running-proof-timestamp";
    const { current, frame } = await startNewQueuedAttempt(chatId, "timestamped queue");
    const turnId = String(frame.turn_id);
    const binding = JSON.parse(localStorage.getItem(queueKey(chatId))!)[0]
      .submission.binding as QueuedAttemptBinding;
    const observation = current.client.waitForQueuedAttempt(chatId, binding);
    if (!observation) throw new Error("queued attempt observation is missing");
    let outcomeResolved = false;
    void observation.then(() => { outcomeResolved = true; });

    await act(async () => {
      current.socket.receive({
        event: "goal_status",
        chat_id: chatId,
        status: "running",
        turn_id: turnId,
      });
      await Promise.resolve();
    });
    expect(outcomeResolved).toBe(false);
    expect(screen.getByText("timestamped queue")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Awaiting confirmation");

    await act(async () => {
      current.socket.receive({
        event: "goal_status",
        chat_id: chatId,
        status: "running",
        started_at: 1_700_000_321,
        turn_id: turnId,
      });
      await observation;
    });
    expect(outcomeResolved).toBe(true);

    await waitFor(() => {
      expect(screen.queryByText("timestamped queue")).not.toBeInTheDocument();
    });
  });

  it.each(["proof after remount", "proof while offscreen"])(
    "removes only the persisted original when the exact owner receipt arrives %s",
    async (delivery) => {
      const { chatId, current, frame, mounted, view } = startQueuedAttempt();
      const turnId = String(frame.turn_id);
      const binding = JSON.parse(localStorage.getItem(queueKey(chatId))!)[0].submission.binding;
      mounted.unmount();

      if (delivery === "proof while offscreen") {
        const observation = current.client.waitForQueuedAttempt(chatId, binding);
        expect(observation).not.toBeNull();
        act(() => receiveAdmission(current.socket, chatId, turnId));
        await expect(observation).resolves.toEqual({ kind: "admitted" });
        expect(JSON.parse(localStorage.getItem(queueKey(chatId))!)[0].submission.state)
          .toBe("submitted");
        render(view());
        await waitFor(() => {
          expect(screen.queryByText("one queued owner attempt")).not.toBeInTheDocument();
        });
        expect(messageFrames(current.socket)).toHaveLength(1);
        expect(localStorage.getItem(queueKey(chatId))).toBeNull();
        return;
      }

      render(view());
      expect(await screen.findByText("one queued owner attempt")).toBeInTheDocument();
      if (delivery === "proof after remount") {
        const observation = current.client.waitForQueuedAttempt(chatId, binding);
        expect(observation).not.toBeNull();
        await act(async () => {
          receiveAdmission(current.socket, chatId, turnId);
          await observation;
        });
        await expect(observation).resolves.toEqual({ kind: "admitted" });
      }

      await waitFor(() => {
        expect(screen.queryByText("one queued owner attempt")).not.toBeInTheDocument();
      });
      expect(messageFrames(current.socket)).toHaveLength(1);
      expect(localStorage.getItem(queueKey(chatId))).toBeNull();
    },
  );

  it("retains an exact offscreen refusal as manual-only until the user retries", async () => {
    const { chatId, current, frame, mounted, view } = startQueuedAttempt();
    const originalTurnId = String(frame.turn_id);
    const binding = JSON.parse(localStorage.getItem(queueKey(chatId))!)[0].submission.binding;
    const observation = current.client.waitForQueuedAttempt(chatId, binding);
    expect(observation).not.toBeNull();
    mounted.unmount();

    act(() => current.socket.receive({
      event: "error",
      chat_id: chatId,
      turn_id: originalTurnId,
      detail: "not_admitted",
      reason: "busy target",
    }));
    await expect(observation).resolves.toEqual({ kind: "not_admitted", reason: "busy target" });

    render(view());
    const row = screen.getByText("one queued owner attempt")
      .closest("[data-queued-prompt-row='true']");
    expect(row).toBeTruthy();
    await waitFor(() => {
      expect(within(row!).queryByRole("status")).not.toBeInTheDocument();
    });
    expect(within(row!).getByRole("button", { name: "Steer", exact: true })).toBeEnabled();
    expect(messageFrames(current.socket)).toHaveLength(1);

    fireEvent.click(within(row!).getByRole("button", { name: "Steer", exact: true }));

    await waitFor(() => expect(messageFrames(current.socket)).toHaveLength(2));
    const retry = messageFrames(current.socket)[1];
    expect(retry).toBeDefined();
    expect(retry?.turn_id).not.toBe(originalTurnId);
    expect(JSON.parse(localStorage.getItem(queueKey(chatId))!)[0].submission).toMatchObject({
      version: 1,
      state: "submitted",
      binding: { queueKey: chatId, promptId: expect.any(String), turnId: retry?.turn_id },
    });
  });
});
