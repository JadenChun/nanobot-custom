import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import { KohakuClient } from "@/lib/kohaku-client";
import { ClientProvider } from "@/providers/ClientProvider";
import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { ThreadShell } from "@/components/thread/ThreadShell";
import { NextTurnSettingsTracker } from "@/lib/next-turn-settings";
import type {
  ChatSummary,
  FolderAccessRequest,
  UIMessage,
  WorkspacePermissionProjection,
  WorkspaceScopePayload,
  WorkspacesPayload,
} from "@/lib/types";

class FakeSocket {
  static instances: FakeSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  url: string;
  readyState = FakeSocket.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((ev?: { code?: number }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.();
  }

  fakeOpen() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  fakeMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  frames(): Array<Record<string, unknown>> {
    return this.sent.map((payload) => JSON.parse(payload) as Record<string, unknown>);
  }
}

function lastSocket(): FakeSocket {
  const socket = FakeSocket.instances.at(-1);
  if (!socket) throw new Error("no socket created yet");
  return socket;
}

const FOLDER_A: WorkspaceScopePayload = {
  project_path: "/projects/alpha",
  project_name: "alpha",
  access_mode: "restricted",
  restrict_to_workspace: true,
};
const FOLDER_B: WorkspaceScopePayload = {
  project_path: "/projects/beta",
  project_name: "beta",
  access_mode: "restricted",
  restrict_to_workspace: true,
};
const HOME: WorkspaceScopePayload = {
  project_path: "/agent/home",
  project_name: "Agent home",
  access_mode: "restricted",
  restrict_to_workspace: true,
};

function workspacesFixture(): WorkspacesPayload {
  const effectiveAccess = {
    policy: "managed" as const,
    source: "grant" as const,
    grant_root: "/projects",
    exists: true,
    read: true,
    write: true,
    commands: {
      requested: true,
      available: true,
      backend: "native" as const,
      reason: null,
    },
  };
  return {
    schema_version: 2,
    default_access_mode: "restricted",
    default_scope: HOME,
    managed_access: false,
    management: {
      grants: [],
      repositories: [
        {
          name: "alpha",
          path: FOLDER_A.project_path,
          purpose: "",
          kind: "project",
          knowledge_repos: [],
          auto_sync: false,
          sync_paths: [],
          exists: true,
          effective_access: effectiveAccess,
        },
        {
          name: "beta",
          path: FOLDER_B.project_path,
          purpose: "",
          kind: "project",
          knowledge_repos: [],
          auto_sync: false,
          sync_paths: [],
          exists: true,
          effective_access: effectiveAccess,
        },
      ],
      policy: {
        default_workspace: HOME.project_path,
        managed_access: false,
        can_manage_folders: true,
        can_manage_catalog: true,
        commands_enabled: true,
        commands_backend: "native",
      },
    },
    controls: {
      can_change_project: true,
      can_use_full_access: true,
      can_pick_folder: true,
    },
  };
}

function session(chatId: string): ChatSummary {
  return {
    key: `websocket:${chatId}`,
    channel: "websocket",
    chatId,
    createdAt: null,
    updatedAt: null,
    preview: "",
    workspaceScope: FOLDER_A,
  };
}

function transcript(): { schemaVersion: number; messages: UIMessage[] } {
  return {
    schemaVersion: 3,
    messages: [
      { id: "u1", role: "user", content: "start turn A", createdAt: 1, turnId: "turn-a" },
      { id: "a1", role: "assistant", content: "turn A reply", createdAt: 2, turnId: "turn-a" },
    ],
  };
}

function makeClient() {
  const chatHandlers = new Map<string, Set<(ev: import("@/lib/types").InboundEvent) => void>>();
  const sessionUpdateHandlers = new Set<(
    chatId: string,
    scope?: string,
    workspaceScope?: WorkspaceScopePayload,
    accessRequest?: FolderAccessRequest,
    permissions?: WorkspacePermissionProjection,
    nextTurn?: { workspaceScope?: WorkspaceScopePayload | null; revision?: number | null },
  ) => void>();
  const runStartedAtByChatId = new Map<string, number>();
  const runGenerationByChatId = new Map<string, number>();
  const latestRunTurnIdByChatId = new Map<string, string>();
  const statusHandlers = new Set<(status: import("@/lib/types").ConnectionStatus) => void>();
  const emitRunStatus = (chatId: string, startedAt: number | null) => {
    void chatId;
    void startedAt;
  };
  const client = {
    onChat: (chatId: string, handler: (ev: import("@/lib/types").InboundEvent) => void) => {
      let handlers = chatHandlers.get(chatId);
      if (!handlers) {
        handlers = new Set();
        chatHandlers.set(chatId, handlers);
      }
      handlers.add(handler);
      return () => {
        handlers?.delete(handler);
      };
    },
    onSessionUpdate: (handler: (
      chatId: string,
      scope?: string,
      workspaceScope?: WorkspaceScopePayload,
      accessRequest?: FolderAccessRequest,
      permissions?: WorkspacePermissionProjection,
      nextTurn?: { workspaceScope?: WorkspaceScopePayload | null; revision?: number | null },
    ) => void) => {
      sessionUpdateHandlers.add(handler);
      return () => {
        sessionUpdateHandlers.delete(handler);
      };
    },
    onError: () => () => {},
    onStatus: (handler: (status: import("@/lib/types").ConnectionStatus) => void) => {
      statusHandlers.add(handler);
      handler("open");
      return () => {
        statusHandlers.delete(handler);
      };
    },
    onRunStatus: () => () => {},
    onRuntimeModelUpdate: () => () => {},
    onTaskNotification: () => () => {},
    onSettingsUpdated: () => () => {},
    onSidebarStateUpdate: () => () => {},
    getRunStartedAt: (chatId: string) => runStartedAtByChatId.get(chatId) ?? null,
    getRunTurnId: (chatId: string) => latestRunTurnIdByChatId.get(chatId) ?? null,
    getRunGeneration: (chatId: string) => runGenerationByChatId.get(chatId) ?? 0,
    hasUnsettledRun: () => false,
    hasUnknownMessageSend: () => false,
    finishRunLocally: (chatId: string) => {
      runStartedAtByChatId.delete(chatId);
      latestRunTurnIdByChatId.delete(chatId);
    },
    getGoalState: () => undefined,
    canReconcileCanonicalCompletion: () => false,
    reconcileCanonicalCompletion: () => false,
    sendMessage: vi.fn(),
    sendSystemCommand: vi.fn().mockResolvedValue(undefined),
    sendApprovalDecision: vi.fn(() => "approval-1"),
    newChat: vi.fn(),
    forkChat: vi.fn(),
    attach: vi.fn(),
    connect: vi.fn(),
    close: vi.fn(),
    updateUrl: vi.fn(),
    requestMutation: vi.fn(async () => ({})),
    setNextTurnWorkspaceScope: vi.fn(),
    stop: vi.fn().mockResolvedValue(undefined),
    setWorkspaceScope: vi.fn(),
    controlDelegatedChild: vi.fn(),
    prepareQueuedAttempt: vi.fn(() => true),
    waitForQueuedAttempt: vi.fn(() => null),
    consumeQueuedAttempt: vi.fn(() => false),
    joinQueuedAttempt: vi.fn(() => null),
    transcribeAudio: vi.fn(),
    _emitChat(chatId: string, ev: import("@/lib/types").InboundEvent) {
      if (
        ev.event === "goal_status"
        && ev.status === "running"
        && typeof ev.started_at === "number"
      ) {
        runStartedAtByChatId.set(chatId, ev.started_at);
        if (ev.turn_id) latestRunTurnIdByChatId.set(chatId, ev.turn_id);
        runGenerationByChatId.set(chatId, (runGenerationByChatId.get(chatId) ?? 0) + 1);
        emitRunStatus(chatId, ev.started_at);
      } else if (
        (ev.event === "goal_status" && ev.status === "idle") || ev.event === "turn_end"
      ) {
        runStartedAtByChatId.delete(chatId);
        emitRunStatus(chatId, null);
      }
      for (const handler of chatHandlers.get(chatId) ?? []) handler(ev);
    },
    _emitSessionUpdate(
      chatId: string,
      scope?: string,
      workspaceScope?: WorkspaceScopePayload,
      accessRequest?: FolderAccessRequest,
      permissions?: WorkspacePermissionProjection,
      nextTurn?: { workspaceScope?: WorkspaceScopePayload | null; revision?: number | null },
    ) {
      for (const handler of sessionUpdateHandlers) {
        handler(chatId, scope, workspaceScope, accessRequest, permissions, nextTurn);
      }
    },
  };
  return client;
}

function wrap(client: ReturnType<typeof makeClient>, children: ReactNode) {
  return (
    <ClientProvider client={client as unknown as KohakuClient} token="tok" modelName={null}>
      {children}
    </ClientProvider>
  );
}

function stubThreadFetch(chatId = "next-turn-chat") {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "https://example.test");
    if (!url.pathname.endsWith("/webui-thread")) return new Response(null, { status: 404 });
    if (url.searchParams.get("view") === "controls") {
      return Response.json({
        view: "controls",
        sessionKey: `websocket:${chatId}`,
        schemaVersion: 3,
        stop_target: null,
        next_workspace_scope: null,
        next_turn_revision: 1,
      });
    }
    return Response.json({
      view: "history",
      sessionKey: `websocket:${chatId}`,
      ...transcript(),
      active_turn_id: null,
      has_pending_tool_calls: false,
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  FakeSocket.instances = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
  Reflect.deleteProperty(window, "nanobotHost");
});

describe("next-turn settings ordering tracker", () => {
  it("waits for the newest acknowledgement of each kind", async () => {
    const tracker = new NextTurnSettingsTracker();
    tracker.bindIncarnation("chat", "inc:1");
    let settle!: () => void;
    const workspace = new Promise<void>((resolve) => { settle = resolve; });
    tracker.begin("chat", "workspace", workspace);
    expect(tracker.hasPending("chat")).toBe(true);
    const pending = tracker.pending("chat");
    settle();
    await expect(pending).resolves.toBeUndefined();
    expect(tracker.hasPending("chat")).toBe(false);
  });

  it("rejects a refused choice instead of admitting under the old one", async () => {
    const tracker = new NextTurnSettingsTracker();
    tracker.bindIncarnation("chat", "inc:1");
    tracker.begin("chat", "workspace", Promise.reject(new Error("workspace_scope_rejected")));
    await expect(tracker.pending("chat")).rejects.toThrow("workspace_scope_rejected");
  });

  it("clears pending settings when the chat incarnation is replaced", () => {
    const tracker = new NextTurnSettingsTracker();
    tracker.bindIncarnation("chat", "inc:1");
    tracker.begin("chat", "workspace", new Promise<void>(() => {}));
    tracker.bindIncarnation("chat", "inc:2");
    expect(tracker.hasPending("chat")).toBe(false);
  });
});

describe("KohakuClient detached next-turn settings", () => {
  it("sends next_turn without mutating the active scope and resolves on its acknowledgement", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const acknowledged = client.setNextTurnWorkspaceScope("next-turn-chat", FOLDER_B);
    expect(socket.frames()).toContainEqual({
      type: "set_workspace_scope",
      chat_id: "next-turn-chat",
      next_turn: true,
      next_workspace_scope: FOLDER_B,
    });
    socket.fakeMessage({
      event: "attached",
      chat_id: "next-turn-chat",
      stop_target: null,
      next_turn_revision: 1,
    });
    socket.fakeMessage({
      event: "session_updated",
      chat_id: "next-turn-chat",
      scope: "metadata",
      workspace_scope: FOLDER_A,
      next_workspace_scope: FOLDER_B,
      next_turn_revision: 2,
    });

    await expect(acknowledged).resolves.toEqual({
      workspace_scope: FOLDER_A,
      next_workspace_scope: FOLDER_B,
      next_turn_revision: 2,
    });
    expect(client.getNextTurnSettings("next-turn-chat").workspaceScope).toEqual(FOLDER_B);
  });

  it("rejects the ordering promise when the gateway refuses the next folder", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const acknowledged = client.setNextTurnWorkspaceScope("next-turn-chat", FOLDER_B);
    socket.fakeMessage({
      event: "error",
      detail: "workspace_scope_rejected",
      reason: "workspace_controls_unavailable",
      chat_id: "next-turn-chat",
    });
    await expect(acknowledged).rejects.toThrow("workspace_controls_unavailable");
  });

  it("ignores a stale next-turn reply so it cannot overwrite a newer selection", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    socket.fakeMessage({
      event: "attached",
      chat_id: "next-turn-chat",
      stop_target: null,
      next_turn_revision: 7,
    });
    socket.fakeMessage({
      event: "session_updated",
      chat_id: "next-turn-chat",
      scope: "metadata",
      next_workspace_scope: FOLDER_B,
      next_turn_revision: 8,
    });
    expect(client.getNextTurnSettings("next-turn-chat").workspaceScope).toEqual(FOLDER_B);

    // An older acknowledgement arriving late must not replace folder B.
    socket.fakeMessage({
      event: "session_updated",
      chat_id: "next-turn-chat",
      scope: "metadata",
      next_workspace_scope: FOLDER_A,
      next_turn_revision: 5,
    });
    expect(client.getNextTurnSettings("next-turn-chat").workspaceScope).toEqual(FOLDER_B);
    expect(client.getNextTurnSettings("next-turn-chat").revision).toBe(8);
  });
});

describe("ThreadComposer next-turn picker", () => {
  it("keeps the picker enabled during a run and shows the next folder separately", async () => {
    const user = userEvent.setup();
    const onWorkspaceScopeChange = vi.fn();
    const workspaces = workspacesFixture();
    render(
      <ThreadComposer
        onSend={vi.fn()}
        placeholder="Ask anything..."
        pendingQueueKey="next-turn-chat"
        isStreaming
        workspaceScope={FOLDER_A}
        workspaceDefaultScope={HOME}
        workspaceControls={workspaces.controls}
        workspaces={workspaces}
        nextWorkspaceScope={FOLDER_B}
        onWorkspaceScopeChange={onWorkspaceScopeChange}
      />,
    );

    const trigger = screen.getByRole("button", { name: "Next turn: beta" });
    expect(trigger).toBeEnabled();
    await user.click(trigger);
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("button", { name: /beta/ })).toHaveAttribute("aria-current", "true");
    expect(within(dialog).getByRole("button", { name: "Clear project folder" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /alpha/ })).not.toHaveAttribute("aria-current");
  });

  it("does not request history to change the next folder", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const onWorkspaceScopeChange = vi.fn();
    const workspaces = workspacesFixture();
    render(
      <ThreadComposer
        onSend={vi.fn()}
        placeholder="Ask anything..."
        pendingQueueKey="next-turn-chat"
        isStreaming
        workspaceScope={FOLDER_A}
        workspaceDefaultScope={HOME}
        workspaceControls={workspaces.controls}
        workspaces={workspaces}
        onWorkspaceScopeChange={onWorkspaceScopeChange}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Working in: alpha" }));
    await user.click(await screen.findByRole("button", { name: /beta/i }));
    expect(onWorkspaceScopeChange).toHaveBeenCalledWith(
      expect.objectContaining({ project_path: FOLDER_B.project_path }),
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows a control-local error when the next folder was refused", () => {
    const workspaces = workspacesFixture();
    render(
      <ThreadComposer
        onSend={vi.fn()}
        placeholder="Ask anything..."
        pendingQueueKey="next-turn-chat"
        isStreaming
        workspaceScope={FOLDER_A}
        workspaceDefaultScope={HOME}
        workspaceControls={workspaces.controls}
        workspaces={workspaces}
        nextWorkspaceScope={FOLDER_B}
        nextTurnSettingsError="workspace_scope_rejected"
        onWorkspaceScopeChange={vi.fn()}
      />,
    );
    const banner = screen.getByTestId("composer-next-turn-error");
    expect(banner).toHaveTextContent(
      "The folder change was not saved.",
    );
    fireEvent.click(within(banner).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByTestId("composer-next-turn-error")).not.toBeInTheDocument();
  });
});

describe("ThreadShell next-turn container lock", () => {
  it("lifts the folder lock while a turn runs and leaves Stop/recovery untouched", async () => {
    const user = userEvent.setup();
    const client = makeClient();
    stubThreadFetch();
    const workspaces = workspacesFixture();
    const onWorkspaceScopeChange = vi.fn();
    let acknowledge!: (scope: WorkspaceScopePayload) => void;
    client.setNextTurnWorkspaceScope.mockImplementation(
      (_chatId: string, scope: WorkspaceScopePayload) => new Promise((resolve) => {
        acknowledge = (next: WorkspaceScopePayload) => resolve({
          workspace_scope: FOLDER_A,
          next_workspace_scope: next,
          next_turn_revision: 2,
        });
        void scope;
      }),
    );

    render(wrap(client, <ThreadShell
      session={session("next-turn-chat")}
      title="Next turn"
      onToggleSidebar={() => {}}
      workspaceScope={FOLDER_A}
      workspaceDefaultScope={HOME}
      workspaceControls={workspaces.controls}
      workspaces={workspaces}
      onWorkspaceScopeChange={onWorkspaceScopeChange}
    />));

    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message input" }))
      .toBeInTheDocument());
    act(() => client._emitChat("next-turn-chat", {
      event: "goal_status",
      chat_id: "next-turn-chat",
      status: "running",
      started_at: 1_700_000_000,
      turn_id: "turn-a",
    }));
    const stop = await screen.findByRole("button", { name: "Stop response" });

    const folderTrigger = screen.getByRole("button", { name: "Working in: alpha" });
    expect(folderTrigger).toBeEnabled();
    await user.click(folderTrigger);
    await user.click(await screen.findByRole("button", { name: /beta/i }));

    await waitFor(() => expect(client.setNextTurnWorkspaceScope)
      .toHaveBeenCalledWith("next-turn-chat", expect.objectContaining({
        project_path: FOLDER_B.project_path,
      })));
    // The active scope was not rewritten and Stop is still the active turn.
    expect(client.setWorkspaceScope).not.toHaveBeenCalled();
    expect(onWorkspaceScopeChange).not.toHaveBeenCalled();
    expect(stop).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next turn: beta" })).toBeEnabled();

    // The gateway acknowledgement keeps B displayed and turn A still owns Stop.
    acknowledge(FOLDER_B);
    await waitFor(() => expect(screen.getByRole("button", { name: "Next turn: beta" }))
      .toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Stop response" })).toBeInTheDocument();
  });

  it("waits for a held next-folder acknowledgement before admitting a new-turn Send", async () => {
    const user = userEvent.setup();
    const client = makeClient();
    stubThreadFetch();
    const workspaces = workspacesFixture();
    let acknowledge!: (scope: WorkspaceScopePayload) => void;
    client.setNextTurnWorkspaceScope.mockImplementation(
      () => new Promise((resolve) => {
        acknowledge = (next: WorkspaceScopePayload) => resolve({
          workspace_scope: FOLDER_A,
          next_workspace_scope: next,
          next_turn_revision: 2,
        });
      }),
    );

    render(wrap(client, <ThreadShell
      session={session("next-turn-chat")}
      title="Next turn"
      onToggleSidebar={() => {}}
      workspaceScope={FOLDER_A}
      workspaceDefaultScope={HOME}
      workspaceControls={workspaces.controls}
      workspaces={workspaces}
      onWorkspaceScopeChange={vi.fn()}
    />));

    const input = await screen.findByRole("textbox", { name: "Message input" });
    act(() => client._emitChat("next-turn-chat", {
      event: "goal_status",
      chat_id: "next-turn-chat",
      status: "running",
      started_at: 1_700_000_000,
      turn_id: "turn-a",
    }));

    await user.click(screen.getByRole("button", { name: "Working in: alpha" }));
    await user.click(await screen.findByRole("button", { name: /beta/i }));
    await waitFor(() => expect(client.setNextTurnWorkspaceScope).toHaveBeenCalled());

    // Turn A ends naturally while B's acknowledgement is still in flight.
    act(() => client._emitChat("next-turn-chat", {
      event: "turn_end",
      chat_id: "next-turn-chat",
      turn_id: "turn-a",
    }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop response" }))
      .toBeNull());

    fireEvent.change(input, { target: { value: "send under beta" } });
    fireEvent.keyDown(input, { key: "Enter" });

    // Admission is held behind the acknowledgement; nothing was sent yet.
    expect(client.sendMessage).not.toHaveBeenCalled();
    expect(input).toHaveValue("send under beta");

    acknowledge(FOLDER_B);
    await waitFor(() => expect(client.sendMessage).toHaveBeenCalledTimes(1));
    expect(client.sendMessage).toHaveBeenCalledWith(
      "next-turn-chat",
      "send under beta",
      undefined,
      expect.objectContaining({ turnId: expect.any(String) }),
    );
  });

  it("keeps the draft and refuses Send when the next folder is rejected", async () => {
    const user = userEvent.setup();
    const client = makeClient();
    stubThreadFetch();
    const workspaces = workspacesFixture();
    let reject!: (error: Error) => void;
    client.setNextTurnWorkspaceScope.mockImplementation(
      () => new Promise((_resolve, refuse) => {
        reject = refuse;
      }),
    );

    render(wrap(client, <ThreadShell
      session={session("next-turn-chat")}
      title="Next turn"
      onToggleSidebar={() => {}}
      workspaceScope={FOLDER_A}
      workspaceDefaultScope={HOME}
      workspaceControls={workspaces.controls}
      workspaces={workspaces}
      onWorkspaceScopeChange={vi.fn()}
    />));

    const input = await screen.findByRole("textbox", { name: "Message input" });
    act(() => client._emitChat("next-turn-chat", {
      event: "goal_status",
      chat_id: "next-turn-chat",
      status: "running",
      started_at: 1_700_000_000,
      turn_id: "turn-a",
    }));

    await user.click(screen.getByRole("button", { name: "Working in: alpha" }));
    await user.click(await screen.findByRole("button", { name: /beta/i }));
    await waitFor(() => expect(client.setNextTurnWorkspaceScope).toHaveBeenCalled());

    act(() => client._emitChat("next-turn-chat", {
      event: "turn_end",
      chat_id: "next-turn-chat",
      turn_id: "turn-a",
    }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop response" }))
      .toBeNull());

    fireEvent.change(input, { target: { value: "must not go under alpha" } });
    fireEvent.keyDown(input, { key: "Enter" });
    reject(new Error("workspace_scope_rejected"));

    await waitFor(() => expect(screen.getByTestId("composer-next-turn-error")).toBeInTheDocument());
    expect(client.sendMessage).not.toHaveBeenCalled();
    expect(input).toHaveValue("must not go under alpha");
    expect(screen.getByRole("button", { name: "Working in: alpha" })).toBeInTheDocument();
  });

  it("holds an automatic queue drain until the next-folder acknowledgement settles", async () => {
    const onSend = vi.fn();
    let acknowledge!: () => void;
    const onBeforeNewTurnSend = vi.fn(() => new Promise<void>((resolve) => {
      acknowledge = resolve;
    }));
    const view = (isStreaming: boolean, queueCompletionTurnId?: string) => (
      <ThreadComposer
        onSend={onSend}
        isStreaming={isStreaming}
        queueCompletionTurnId={queueCompletionTurnId}
        onBeforeNewTurnSend={onBeforeNewTurnSend}
      />
    );
    const { rerender } = render(view(true));
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "queued after A" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByText("queued after A")).toBeInTheDocument();

    rerender(view(false, "live-natural-turn"));
    await waitFor(() => expect(onBeforeNewTurnSend).toHaveBeenCalledTimes(1));
    // The row stays queued while the setting acknowledgement is in flight.
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByText("queued after A")).toBeInTheDocument();

    acknowledge();
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend).toHaveBeenCalledWith(
      "queued after A",
      undefined,
      expect.objectContaining({ queuedAttempt: expect.any(Object) }),
    );
  });

  it("keeps a queued row ready when the next-turn setting is refused", async () => {
    const onSend = vi.fn();
    const onBeforeNewTurnSend = vi.fn(() => Promise.reject(new Error("workspace_scope_rejected")));
    const view = (isStreaming: boolean, queueCompletionTurnId?: string) => (
      <ThreadComposer
        onSend={onSend}
        isStreaming={isStreaming}
        queueCompletionTurnId={queueCompletionTurnId}
        onBeforeNewTurnSend={onBeforeNewTurnSend}
      />
    );
    const { rerender } = render(view(true));
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "queued under beta" } });
    fireEvent.keyDown(input, { key: "Enter" });

    rerender(view(false, "live-natural-turn"));
    await waitFor(() => expect(onBeforeNewTurnSend).toHaveBeenCalledTimes(1));
    expect(onSend).not.toHaveBeenCalled();
    // The exact row is preserved and is still shown for a retry.
    expect(screen.getByText("queued under beta")).toBeInTheDocument();
  });

  it("keeps isCurrent accurate after settlement and clears tracker only on incarnation replacement", async () => {
    const tracker = new NextTurnSettingsTracker();
    let resolveFirst!: () => void;
    let resolveSecond!: () => void;
    const first = new Promise<void>((resolve) => { resolveFirst = resolve; });
    const second = new Promise<void>((resolve) => { resolveSecond = resolve; });

    const seq1 = tracker.begin("chat-1", "workspace", first);
    const seq2 = tracker.begin("chat-1", "workspace", second);
    expect(tracker.isCurrent("chat-1", "workspace", seq1)).toBe(false);
    expect(tracker.isCurrent("chat-1", "workspace", seq2)).toBe(true);

    // Initial incarnation hydration (null -> inc-1:0) must not drop in-flight waiters.
    tracker.bindIncarnation("chat-1", "inc-1:0");
    expect(tracker.hasPending("chat-1")).toBe(true);

    resolveFirst();
    await Promise.resolve();
    // Choice #2 is still in flight, so workspace remains pending.
    expect(tracker.pendingKinds("chat-1").has("workspace")).toBe(true);

    resolveSecond();
    await tracker.pending("chat-1");
    expect(tracker.hasPending("chat-1")).toBe(false);
    expect(tracker.isCurrent("chat-1", "workspace", seq2)).toBe(true);
    expect(tracker.isCurrent("chat-1", "workspace", seq1)).toBe(false);

    // Real incarnation replacement invalidates prior sequence tracking.
    tracker.bindIncarnation("chat-1", "inc-1:1");
    expect(tracker.isCurrent("chat-1", "workspace", seq2)).toBe(false);
  });

  it("fires Steer immediately without waiting on a held next-turn folder acknowledgement", async () => {
    const user = userEvent.setup();
    const client = makeClient();
    stubThreadFetch("steer-immediate");
    const workspaces = workspacesFixture();
    client.setNextTurnWorkspaceScope.mockReturnValue(new Promise(() => {}));

    render(wrap(client, <ThreadShell
      session={session("steer-immediate")}
      title="Steer immediate"
      onToggleSidebar={() => {}}
      workspaceScope={FOLDER_A}
      workspaceDefaultScope={HOME}
      workspaceControls={workspaces.controls}
      workspaces={workspaces}
      onWorkspaceScopeChange={vi.fn()}
    />));

    await screen.findByLabelText("Message input");
    act(() => {
      client._emitChat("steer-immediate", {
        event: "goal_status",
        chat_id: "steer-immediate",
        status: "running",
        started_at: 1_700_000_000,
        turn_id: "turn-a",
      });
    });
    await screen.findByRole("button", { name: "Stop response" });

    // Pick folder B while turn A is running; its ack stays held.
    await user.click(screen.getByRole("button", { name: "Working in: alpha" }));
    await user.click(await screen.findByRole("button", { name: /beta/i }));
    await waitFor(() => expect(client.setNextTurnWorkspaceScope).toHaveBeenCalledTimes(1));

    // Queue a prompt and click Steer: must send immediately without awaiting beta's ack.
    const input = screen.getByLabelText("Message input");
    fireEvent.change(input, { target: { value: "steer turn A now" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.click(await screen.findByRole("button", { name: "Steer" }));

    await waitFor(() => expect(client.sendMessage).toHaveBeenCalledWith(
      "steer-immediate",
      "steer turn A now",
      undefined,
      expect.objectContaining({
        requestIntent: "steer",
        startsNewRun: false,
        targetTurnId: "turn-a",
      }),
    ));
  });

});
