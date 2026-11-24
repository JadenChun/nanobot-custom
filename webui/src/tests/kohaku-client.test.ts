import { createElement, type ReactNode } from "react";
import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useTaskNotifications } from "@/hooks/useTaskNotifications";
import { KohakuClient } from "@/lib/kohaku-client";
import {
  DEFAULT_LOCAL_PREFS,
  LOCAL_PREFS_STORAGE_KEY,
  writeLocalPreferences,
} from "@/lib/local-preferences";
import type { SidebarStatePayload } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";
import { folderAccessSessionUpdatedFixture } from "./fixtures/folder-access-request";

/**
 * Minimal fake WebSocket implementing the subset KohakuClient touches.
 * Every instance is retrievable via ``FakeSocket.instances`` so tests can
 * drive open/close/message lifecycles deterministically.
 */
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

  /** Simulate a server-initiated drop with a specific wire-level close code
   * (e.g. ``1009`` for Message Too Big). */
  fakeCloseWithCode(code: number) {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.({ code });
  }

  fakeOpen() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  fakeMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function lastSocket(): FakeSocket {
  const s = FakeSocket.instances.at(-1);
  if (!s) throw new Error("no socket created yet");
  return s;
}

beforeEach(() => {
  FakeSocket.instances = [];
  vi.useFakeTimers();
  writeLocalPreferences({ ...DEFAULT_LOCAL_PREFS, browserNotifications: true });
  window.nanobotMobileRunKeepAlive = { start: vi.fn(), stop: vi.fn() };
});

afterEach(() => {
  Reflect.deleteProperty(window, "nanobotHost");
  delete window.nanobotMobileRunKeepAlive;
  window.localStorage.removeItem(LOCAL_PREFS_STORAGE_KEY);
  vi.useRealTimers();
});

describe("KohakuClient", () => {
  it("reconciles simultaneous client submissions to the gateway-owned turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    client.sendMessage("shared-chat", "from terminal B", undefined, {
      turnId: "turn-b",
    });
    expect(client.getRunTurnId("shared-chat")).toBe("turn-b");

    socket.fakeMessage({
      event: "message_accepted",
      chat_id: "shared-chat",
      turn_id: "turn-b",
      active_turn_id: "turn-a",
      starts_turn: false,
      started_at: 1_700_000_000,
    });

    expect(client.getRunTurnId("shared-chat")).toBe("turn-a");
    expect(client.getRunStartedAt("shared-chat")).toBe(1_700_000_000);
    expect(client.hasUnsettledRun("shared-chat")).toBe(true);
  });

  it("correlates successful WebUI mutation replies by request id", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const pending = client.requestMutation<{ saved: boolean }>(
      "settings.provider.update",
      { provider: "openrouter", apiKey: "secret" },
    );
    const frame = JSON.parse(socket.sent.at(-1) as string);
    expect(frame).toMatchObject({
      type: "webui_request",
      action: "settings.provider.update",
      payload: { provider: "openrouter", apiKey: "secret" },
    });
    expect(frame.request_id).toEqual(expect.any(String));

    socket.fakeMessage({
      event: "webui_response",
      request_id: frame.request_id,
      ok: true,
      result: { saved: true },
    });

    await expect(pending).resolves.toEqual({ saved: true });
  });

  it("sends an approval and save decision with only the approval identity", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const turnId = client.sendApprovalDecision("sandbox-chat", "sandbox-approval", "approve_and_save");
    const frame = JSON.parse(socket.sent.at(-1) as string);
    expect(frame.turn_id).toBe(turnId);
    expect(frame.approval_decision).toEqual({
      approval_id: "sandbox-approval",
      decision: "approve_and_save",
    });
  });

  it("sends a typed approval decision with the exact child source and run identity", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const firstTurnId = client.sendApprovalDecision("parent-chat", "approval-child-1", "approve", {
      sessionKey: "delegated:child-1",
      sessionIncarnation: "child-incarnation-1",
      runId: "child-run-1",
      generation: 7,
    });
    const secondTurnId = client.sendApprovalDecision("parent-chat", "approval-child-2", "deny", {
      sessionKey: "delegated:child-2",
      sessionIncarnation: "child-incarnation-2",
      runId: "child-run-2",
      generation: 3,
    });

    const firstFrame = JSON.parse(socket.sent.at(-2) as string);
    const secondFrame = JSON.parse(socket.sent.at(-1) as string);
    expect(firstFrame).toMatchObject({
      type: "message",
      chat_id: "parent-chat",
      content: "approval_decision",
      request_intent: "new",
      turn_id: firstTurnId,
      approval_decision: {
        approval_id: "approval-child-1",
        decision: "approve",
        session_key: "delegated:child-1",
        session_incarnation: "child-incarnation-1",
        run_id: "child-run-1",
        generation: 7,
      },
    });
    expect(secondFrame).toMatchObject({
      turn_id: secondTurnId,
      approval_decision: {
        approval_id: "approval-child-2",
        decision: "deny",
        session_key: "delegated:child-2",
        session_incarnation: "child-incarnation-2",
        run_id: "child-run-2",
        generation: 3,
      },
    });
    expect(firstTurnId).not.toBe(secondTurnId);
    expect(client.getRunTurnId("parent-chat")).toBeNull();
  });

  it("surfaces correlated WebUI mutation errors with status", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const pending = client.requestMutation("settings.channel.configure", {});
    const requestId = JSON.parse(socket.sent.at(-1) as string).request_id;
    socket.fakeMessage({
      event: "webui_response",
      request_id: requestId,
      ok: false,
      error: { status: 400, message: "missing channel name" },
    });

    await expect(pending).rejects.toMatchObject({
      status: 400,
      message: "missing channel name",
    });
  });

  it("times out WebUI mutations without replaying them", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const pending = expect(
      client.requestMutation("skill.install", { skill: "docs" }, 25),
    ).rejects.toMatchObject({
      status: 504,
      message: "WebUI request timed out after 25ms",
    });
    expect(socket.sent).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(25);

    await pending;
    expect(socket.sent).toHaveLength(1);
  });

  it("rejects in-flight WebUI mutations when the socket closes", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();

    const pending = client.requestMutation("session.delete", {
      key: "websocket:chat-1",
    });
    socket.fakeCloseWithCode(1006);

    await expect(pending).rejects.toMatchObject({
      status: 503,
      message: "Socket closed before WebUI response",
    });
  });

  it("replays the exact serialized mutation frame after reconnect", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 10,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const firstSocket = lastSocket();
    firstSocket.fakeOpen();

    const payload = { id: "daily-summary", options: { force: false } };
    const pending = client.requestMutation<{ ran: boolean }>("automation.run", payload);
    const frame = firstSocket.sent.at(-1) as string;
    const requestId = JSON.parse(frame).request_id;
    payload.id = "weekly-summary";
    payload.options.force = true;
    const settled = expect(pending).resolves.toEqual({ ran: true });
    firstSocket.fakeCloseWithCode(1006);

    await vi.advanceTimersByTimeAsync(20);
    const retrySocket = lastSocket();
    retrySocket.fakeOpen();
    expect(retrySocket.sent).toEqual([frame]);
    retrySocket.fakeMessage({
      event: "webui_response",
      request_id: requestId,
      ok: true,
      result: { ran: true },
    });

    await settled;
  });

  it("does not retry a WebUI mutation after its timeout expires", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 1_000,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const firstSocket = lastSocket();
    firstSocket.fakeOpen();

    const pending = expect(
      client.requestMutation("skill.install", { skill: "docs" }, 25),
    ).rejects.toMatchObject({ status: 504 });
    firstSocket.fakeCloseWithCode(1006);
    await vi.advanceTimersByTimeAsync(25);
    await pending;

    await vi.advanceTimersByTimeAsync(1_000);
    const retrySocket = lastSocket();
    retrySocket.fakeOpen();
    expect(retrySocket.sent).toEqual([]);
  });

  it("does not queue WebUI mutations before the authenticated socket opens", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();

    await expect(client.requestMutation("settings.agent.update", {})).rejects.toMatchObject({
      status: 503,
      message: "WebUI connection is not open",
    });
    expect(lastSocket().sent).toEqual([]);
  });

  it("keeps temporary chats out of attachment and reconnect state", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatId = "temp-server-id";
    client.connect();
    lastSocket().fakeOpen();
    const creation = client.newTemporaryChat();
    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toEqual({
      type: "new_temporary_chat",
    });
    lastSocket().fakeMessage({ event: "attached", chat_id: chatId, temporary: true });
    await expect(creation).resolves.toBe(chatId);
    lastSocket().sent = [];
    client.onChat(chatId, vi.fn());
    client.sendMessage(chatId, "hello", undefined, { turnId: "turn-1" });

    expect(lastSocket().sent.map((raw) => JSON.parse(raw))).toEqual([
      {
        type: "message",
        chat_id: chatId,
        content: "hello",
        turn_id: "turn-1",
        request_settings_version: 1,
      request_intent: "new",
      webui: true,
      },
    ]);

    client.discardTemporaryChat(chatId);
    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toEqual({
      type: "discard_temporary_chat",
      chat_id: chatId,
    });
  });

  it("waits for the temporary attachment when creating a temporary chat", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    const creation = client.newTemporaryChat();
    let resolved = false;
    void creation.then(() => { resolved = true; });
    lastSocket().fakeMessage({ event: "attached", chat_id: "ordinary-chat" });
    await Promise.resolve();
    expect(resolved).toBe(false);

    lastSocket().fakeMessage({
      event: "attached",
      chat_id: "server-temporary-chat",
      temporary: true,
    });
    await expect(creation).resolves.toBe("server-temporary-chat");
  });

  it("forgets every temporary chat when the socket drops", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 1,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const firstHandler = vi.fn();
    const secondHandler = vi.fn();
    client.connect();
    lastSocket().fakeOpen();
    const firstCreation = client.newTemporaryChat();
    lastSocket().fakeMessage({
      event: "attached",
      chat_id: "temp-drop-a",
      temporary: true,
    });
    await firstCreation;
    const secondCreation = client.newTemporaryChat();
    lastSocket().fakeMessage({
      event: "attached",
      chat_id: "temp-drop-b",
      temporary: true,
    });
    await secondCreation;
    lastSocket().sent = [];
    client.onChat("temp-drop-a", firstHandler);
    client.onChat("temp-drop-b", secondHandler);
    firstHandler.mockClear();
    secondHandler.mockClear();
    lastSocket().close();

    await vi.advanceTimersByTimeAsync(1);
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "message",
      chat_id: "temp-drop-a",
      text: "stale first chat",
    });
    lastSocket().fakeMessage({
      event: "message",
      chat_id: "temp-drop-b",
      text: "stale second chat",
    });

    expect(lastSocket().sent).toEqual([]);
    expect(firstHandler).not.toHaveBeenCalled();
    expect(secondHandler).not.toHaveBeenCalled();
  });

  it("routes events to the matching chat handler", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onChat("chat-a", handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({ event: "message", chat_id: "chat-a", text: "hi" });
    lastSocket().fakeMessage({ event: "message", chat_id: "chat-b", text: "no" });
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler.mock.calls[0][0]).toMatchObject({
      event: "message",
      chat_id: "chat-a",
      text: "hi",
    });
  });

  it("routes message acceptance acknowledgements to the matching chat handler", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onChat("chat-ack", handler);
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-ack", "hello", undefined, { turnId: "turn-ack" });

    lastSocket().fakeMessage({
      event: "message_accepted",
      chat_id: "chat-ack",
      turn_id: "turn-ack",
    });

    expect(handler).toHaveBeenCalledWith({
      event: "message_accepted",
      chat_id: "chat-ack",
      turn_id: "turn-ack",
    });
  });

  it("can swap the socket factory when the runtime URL changes", () => {
    const browserFactory = vi.fn(
      (url: string) => new FakeSocket(`browser:${url}`) as unknown as WebSocket,
    );
    const hostFactory = vi.fn(
      (url: string) => new FakeSocket(`host:${url}`) as unknown as WebSocket,
    );
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: browserFactory,
    });

    client.connect();
    expect(lastSocket().url).toBe("browser:ws://test");
    client.close();
    client.updateUrl("nanobot-host://engine/", hostFactory);
    client.connect();

    expect(hostFactory).toHaveBeenCalledWith("nanobot-host://engine/");
    expect(lastSocket().url).toBe("host:nanobot-host://engine/");
  });

  it("uses the host socket bridge for native host URLs", async () => {
    let socketEventHandler:
      | ((event: { id: string; type: "open" | "close" | "error"; message?: string }) => void)
      | null = null;
    const openSocket = vi.fn(async () => "host-socket-1");
    Object.defineProperty(window, "nanobotHost", {
      configurable: true,
      value: {
        openSocket,
        sendSocket: vi.fn(async () => undefined),
        closeSocket: vi.fn(async () => undefined),
        onSocketEvent: vi.fn((handler) => {
          socketEventHandler = handler;
          return vi.fn();
        }),
      },
    });
    const client = new KohakuClient({
      url: "nanobot-host://engine/",
      reconnect: false,
    });
    const status = vi.fn();
    client.onStatus(status);

    client.connect();
    await Promise.resolve();
    socketEventHandler?.({ id: "host-socket-1", type: "open" });

    expect(openSocket).toHaveBeenCalledWith("nanobot-host://engine/");
    expect(status).toHaveBeenLastCalledWith("open");
  });

  it("buffers chat events while no chat handler is registered and replays on subscribe", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    // Nobody listening yet — deltas must not be dropped (user switched away).
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-queue", text: "a" });
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-queue", text: "b" });
    const handler = vi.fn();
    client.onChat("chat-queue", handler);
    expect(handler).toHaveBeenCalledTimes(2);
    expect(handler.mock.calls[0][0]).toMatchObject({ event: "delta", text: "a" });
    expect(handler.mock.calls[1][0]).toMatchObject({ event: "delta", text: "b" });
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-queue", text: "c" });
    expect(handler).toHaveBeenCalledTimes(3);
  });

  it("shares one attach between listeners for the same chat", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const first = vi.fn();
    const second = vi.fn();
    client.onChat("shared-chat", first);
    client.onChat("shared-chat", second);
    expect(lastSocket().sent.map((frame) => JSON.parse(frame))).toEqual([
      { type: "attach", chat_id: "shared-chat" },
    ]);
    const attached = { event: "attached", chat_id: "shared-chat" };
    lastSocket().fakeMessage(attached);
    expect(first.mock.calls[0]?.[0]).toEqual(attached);
    expect(second.mock.calls[0]?.[0]).toEqual(attached);
  });

  it("reuses the live attachment when the same chat is revisited on one socket", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();
    const first = vi.fn();
    const unsubscribe = client.onChat("revisit-chat", first);
    socket.fakeMessage({
      event: "attached",
      chat_id: "revisit-chat",
      stop_target: null,
      recovery_state: { status: "failed", recovery_id: "r1" },
    });
    expect(socket.sent).toHaveLength(1);

    // A -> away -> A on the same socket: no duplicate attach or stale replay.
    unsubscribe();
    const warm = vi.fn();
    client.onChat("revisit-chat", warm);
    expect(socket.sent).toHaveLength(1);
    expect(warm).not.toHaveBeenCalled();

    socket.fakeMessage({ event: "delta", chat_id: "revisit-chat", text: "live" });
    expect(warm).toHaveBeenCalledTimes(1);
  });

  it("performs a fresh attach after an observed source replacement", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();
    const unsubscribe = client.onChat("replaced-chat", vi.fn());
    socket.fakeMessage({ event: "attached", chat_id: "replaced-chat", stop_target: null });
    expect(socket.sent).toHaveLength(1);

    client.invalidateAttachment("replaced-chat");
    unsubscribe();
    client.onChat("replaced-chat", vi.fn());
    expect(socket.sent.map((frame) => JSON.parse(frame))).toEqual([
      { type: "attach", chat_id: "replaced-chat" },
      { type: "attach", chat_id: "replaced-chat" },
    ]);
  });

  it("performs a fresh attach while an input attempt is still unproven", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();
    const unsubscribe = client.onChat("proof-chat", vi.fn());
    socket.fakeMessage({
      event: "attached",
      chat_id: "proof-chat",
      stop_target: null,
      input_attempt_source: {
        version: 1,
        source: {
          store_identity: "store",
          session_key: "websocket:proof-chat",
          incarnation: "inc",
          generation: 1,
        },
        source_token: "token",
        principal: "local",
      },
    });
    client.sendMessage("proof-chat", "uncertain input", undefined, { turnId: "turn-proof" });
    expect(client.hasUnsettledRun("proof-chat")).toBe(true);

    unsubscribe();
    client.onChat("proof-chat", vi.fn());
    expect(socket.sent.filter((frame) => (
      (JSON.parse(frame) as { type?: string }).type === "attach"
    ))).toHaveLength(2);
  });

  it("performs a fresh attach after a reconnect gap", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 10,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const unsubscribe = client.onChat("gap-chat", vi.fn());
    lastSocket().fakeMessage({ event: "attached", chat_id: "gap-chat", stop_target: null });
    expect(lastSocket().sent).toHaveLength(1);

    unsubscribe();
    lastSocket().close();
    await vi.advanceTimersByTimeAsync(20);
    const reconnected = lastSocket();
    reconnected.fakeOpen();
    client.onChat("gap-chat", vi.fn());
    expect(reconnected.sent.filter((frame) => (
      (JSON.parse(frame) as { type?: string }).type === "attach"
    ))).toHaveLength(2);
  });

  it("forwards owner admission receipts to the matching chat subscriber", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    const otherChatHandler = vi.fn();
    client.onChat("chat-admission", handler);
    client.onChat("chat-other", otherChatHandler);
    client.connect();
    lastSocket().fakeOpen();

    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-admission",
      turn_id: "attempt-steer",
      active_turn_id: "root-active",
      starts_turn: false,
    });
    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-admission",
      turn_id: "webui-system:internal",
    });

    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      event: "message_admitted",
      chat_id: "chat-admission",
      turn_id: "attempt-steer",
      active_turn_id: "root-active",
      starts_turn: false,
    }));
    expect(otherChatHandler).not.toHaveBeenCalled();
    expect(client.getRunTurnId("chat-admission")).toBe("root-active");
  });

  it("records goal_status run strip without an onChat subscriber", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-strip",
      status: "running",
      started_at: 12_345,
    });
    expect(client.getRunStartedAt("chat-strip")).toBe(12_345);
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-strip",
      status: "idle",
    });
    expect(client.getRunStartedAt("chat-strip")).toBeNull();
  });

  it("clears the local run strip immediately when a stop is requested", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onRunStatus(handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-stop",
      status: "running",
      started_at: 12_345,
      turn_id: "turn-stop",
    });

    client.finishRunLocally("chat-stop");

    expect(client.getRunStartedAt("chat-stop")).toBeNull();
    expect(client.hasUnsettledRun("chat-stop")).toBe(false);
    expect(handler).toHaveBeenLastCalledWith("chat-stop", null);
  });

  it("clears stale run strip when reconnecting after a dropped socket", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 10,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onRunStatus(handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-strip",
      status: "running",
      started_at: 12_345,
    });

    lastSocket().close();

    expect(client.getRunStartedAt("chat-strip")).toBeNull();
    expect(handler).toHaveBeenLastCalledWith("chat-strip", null);
    await vi.advanceTimersByTimeAsync(20);
    expect(FakeSocket.instances.length).toBeGreaterThan(1);
  });

  it("clears run strip when a turn_end arrives without idle", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onRunStatus(handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-strip",
      status: "running",
      started_at: 12_345,
    });
    lastSocket().fakeMessage({
      event: "turn_end",
      chat_id: "chat-strip",
    });
    expect(client.getRunStartedAt("chat-strip")).toBeNull();
    expect(handler).toHaveBeenLastCalledWith("chat-strip", null);
  });

  it("rejects a completed snapshot when a newer run is not represented", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const requestGeneration = client.getRunGeneration("chat-race");

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-race",
      status: "running",
      started_at: 12_345,
      turn_id: "turn-new",
    });

    expect(
      client.reconcileCanonicalCompletion("chat-race", requestGeneration, ["turn-old"]),
    ).toBe(false);
    expect(client.getRunStartedAt("chat-race")).toBe(12_345);
  });

  it("rejects a user-only snapshot for a submitted turn that has not completed", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-submitted", "question", undefined, { turnId: "turn-submitted" });
    const requestGeneration = client.getRunGeneration("chat-submitted");

    expect(
      client.reconcileCanonicalCompletion("chat-submitted", requestGeneration, []),
    ).toBe(false);
  });

  it("does not register injected guidance as an independently unsettled run", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-guidance",
      status: "running",
      started_at: 12_345,
      turn_id: "turn-active",
    });
    const requestGeneration = client.getRunGeneration("chat-guidance");

    client.sendMessage("chat-guidance", "focus on sources", undefined, {
      turnId: "turn-guidance",
      startsNewRun: false,
    });

    expect(client.getRunGeneration("chat-guidance")).toBe(requestGeneration);
    expect(
      client.reconcileCanonicalCompletion(
        "chat-guidance",
        requestGeneration,
        ["turn-active"],
      ),
    ).toBe(true);
  });

  it("accepts an explicitly completed turn with no assistant row", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-empty-answer", "question", undefined, {
      turnId: "turn-empty-answer",
    });
    const requestGeneration = client.getRunGeneration("chat-empty-answer");

    expect(
      client.reconcileCanonicalCompletion(
        "chat-empty-answer",
        requestGeneration,
        ["turn-empty-answer"],
      ),
    ).toBe(true);
  });

  it.each([
    "message_rejected",
    "attachment_rejected",
    "workspace_scope_rejected",
  ])("settles a specifically rejected outbound turn (%s)", (detail) => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-rejected", "question", undefined, {
      turnId: "turn-rejected",
    });
    const requestGeneration = client.getRunGeneration("chat-rejected");

    expect(
      client.reconcileCanonicalCompletion("chat-rejected", requestGeneration, []),
    ).toBe(false);

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-rejected",
      turn_id: "turn-rejected",
      detail,
      reason: "policy",
    });

    expect(
      client.reconcileCanonicalCompletion("chat-rejected", requestGeneration, []),
    ).toBe(true);
  });

  it("does not let an older rejection settle or stop a newer run", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-rejection-race", "first", undefined, {
      turnId: "turn-old",
    });
    client.sendMessage("chat-rejection-race", "second", undefined, {
      turnId: "turn-new",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-rejection-race",
      status: "running",
      started_at: 2_000,
      turn_id: "turn-new",
    });
    const requestGeneration = client.getRunGeneration("chat-rejection-race");

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-rejection-race",
      turn_id: "turn-old",
      detail: "message_rejected",
      reason: "text_too_large",
    });

    expect(client.getRunStartedAt("chat-rejection-race")).toBe(2_000);
    expect(
      client.reconcileCanonicalCompletion("chat-rejection-race", requestGeneration, []),
    ).toBe(false);
  });

  it("restores the previous turn clock when the newer running turn is rejected", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-reject-newer-clock", "first", undefined, {
      turnId: "turn-clock-first",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-reject-newer-clock",
      status: "running",
      started_at: 1_000,
      turn_id: "turn-clock-first",
    });
    client.sendMessage("chat-reject-newer-clock", "second", undefined, {
      turnId: "turn-clock-second",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-reject-newer-clock",
      status: "running",
      started_at: 2_000,
      turn_id: "turn-clock-second",
    });

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-reject-newer-clock",
      turn_id: "turn-clock-second",
      detail: "message_rejected",
    });

    expect(client.getRunStartedAt("chat-reject-newer-clock")).toBe(1_000);
    expect(client.hasUnsettledRun("chat-reject-newer-clock")).toBe(true);
  });

  it("rolls back lifecycle sends that close 1009 before server acceptance", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-too-big", "oversized", undefined, {
      turnId: "turn-too-big",
    });
    const requestGeneration = client.getRunGeneration("chat-too-big");

    lastSocket().fakeCloseWithCode(1009);

    expect(errors).toEqual([{
      kind: "message_too_big",
      chatId: "chat-too-big",
      turnId: "turn-too-big",
    }]);
    expect(
      client.reconcileCanonicalCompletion("chat-too-big", requestGeneration, []),
    ).toBe(true);
  });

  it("preserves an accepted older run when a newer send closes 1009", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-too-big-race", "first", undefined, {
      turnId: "turn-accepted",
    });
    lastSocket().fakeMessage({
      event: "message_accepted",
      chat_id: "chat-too-big-race",
      turn_id: "turn-accepted",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-too-big-race",
      status: "running",
      started_at: 1_000,
      turn_id: "turn-accepted",
    });
    client.sendMessage("chat-too-big-race", "oversized", undefined, {
      turnId: "turn-rejected",
    });
    const requestGeneration = client.getRunGeneration("chat-too-big-race");

    lastSocket().fakeCloseWithCode(1009);

    expect(errors).toEqual([{
      kind: "message_too_big",
      chatId: "chat-too-big-race",
      turnId: "turn-rejected",
    }]);
    expect(client.getRunStartedAt("chat-too-big-race")).toBe(1_000);
    expect(
      client.reconcileCanonicalCompletion(
        "chat-too-big-race",
        requestGeneration,
        [],
      ),
    ).toBe(false);
  });

  it("does not roll back a lifecycle send after its acceptance ACK", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-accepted", "question", undefined, {
      turnId: "turn-accepted",
    });
    const requestGeneration = client.getRunGeneration("chat-accepted");
    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-accepted",
      turn_id: "turn-accepted",
    });

    lastSocket().fakeCloseWithCode(1009);

    expect(
      client.reconcileCanonicalCompletion("chat-accepted", requestGeneration, []),
    ).toBe(false);
  });

  it("preflights exact websocket frame bytes and rejects only the oversized turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      maxFrameBytes: 180,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    const sentBefore = lastSocket().sent.length;

    client.sendMessage("chat-preflight-size", "x".repeat(500), undefined, {
      turnId: "turn-preflight-size",
    });

    expect(lastSocket().sent).toHaveLength(sentBefore);
    expect(client.hasUnsettledRun("chat-preflight-size")).toBe(false);
    expect(errors).toEqual([{
      kind: "message_too_big",
      chatId: "chat-preflight-size",
      turnId: "turn-preflight-size",
    }]);
  });

  it("does not attribute a fallback 1009 close across multiple unacknowledged chats", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-size-a", "first", undefined, { turnId: "turn-size-a" });
    client.sendMessage("chat-size-b", "second", undefined, { turnId: "turn-size-b" });

    lastSocket().fakeCloseWithCode(1009);

    expect(errors).toEqual([{ kind: "message_too_big" }]);
    expect(client.hasUnsettledRun("chat-size-a")).toBe(true);
    expect(client.hasUnsettledRun("chat-size-b")).toBe(true);
  });

  it("does not attribute 1009 to an unacknowledged message when another frame followed it", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-before-audio", "question", undefined, {
      turnId: "turn-before-audio",
    });
    const transcription = client.transcribeAudio("data:audio/webm;base64,AAAA");

    lastSocket().fakeCloseWithCode(1009);

    await expect(transcription).rejects.toThrow("socket closed");
    expect(errors).toEqual([{ kind: "message_too_big" }]);
    expect(client.hasUnsettledRun("chat-before-audio")).toBe(true);
  });

  it("holds an unknown send through idle and history evidence until owner admission", () => {
    const client = new KohakuClient({
      url: "ws://test",
      maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-never-arrived", "question", undefined, {
      turnId: "turn-never-arrived",
    });
    const requestGeneration = client.getRunGeneration("chat-never-arrived");

    lastSocket().close();

    const snapshot = {
      observedTurnIds: ["turn-never-arrived"],
      activitySnapshotAvailable: true, hasPendingToolCalls: false,
      activeTurnId: null,
    };
    lastSocket().fakeMessage({
      event: "message_accepted",
      chat_id: "chat-never-arrived",
      turn_id: "turn-never-arrived",
    });
    lastSocket().fakeMessage({
      event: "user_message",
      chat_id: "chat-never-arrived",
      turn_id: "turn-never-arrived",
      text: "question",
      starts_turn: true,
    });

    expect(
      client.canReconcileCanonicalCompletion(
        "chat-never-arrived",
        requestGeneration,
        ["turn-never-arrived"],
        snapshot,
      ),
    ).toBe(true);
    expect(
      client.reconcileCanonicalCompletion(
        "chat-never-arrived",
        requestGeneration,
        ["turn-never-arrived"],
        snapshot,
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-never-arrived")).toBe(false);
    expect(client.hasUnknownMessageSend("chat-never-arrived")).toBe(true);

    vi.advanceTimersByTime(500);
    lastSocket().fakeOpen();
    const messageFrames = FakeSocket.instances.flatMap((socket) => socket.sent)
      .map((serialized) => JSON.parse(serialized) as { type?: string })
      .filter((frame) => frame.type === "message");
    expect(messageFrames).toHaveLength(1);

    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-never-arrived",
      turn_id: "turn-never-arrived",
    });
    expect(
      client.reconcileCanonicalCompletion(
        "chat-never-arrived",
        requestGeneration,
        [],
        { observedTurnIds: [], activitySnapshotAvailable: true, hasPendingToolCalls: false, activeTurnId: null },
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-never-arrived")).toBe(false);
    expect(client.hasUnknownMessageSend("chat-never-arrived")).toBe(false);
  });

  it.each(["message_accepted", "user_message"] as const)(
    "keeps an unknown send when a replacement socket reports %s for another owner's run",
    (event) => {
      const client = new KohakuClient({
        url: "ws://test",
        maxBackoffMs: 500,
        socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
      });
      client.connect();
      const originalSocket = lastSocket();
      originalSocket.fakeOpen();
      client.sendMessage("chat-owner-remap", "question", undefined, {
        turnId: "turn-uncertain-owner-remap",
      });
      const generation = client.getRunGeneration("chat-owner-remap");

      originalSocket.close();
      vi.advanceTimersByTime(500);
      const replacementSocket = lastSocket();
      expect(replacementSocket).not.toBe(originalSocket);
      replacementSocket.fakeOpen();
      replacementSocket.fakeMessage({
        event,
        chat_id: "chat-owner-remap",
        turn_id: "turn-uncertain-owner-remap",
        active_turn_id: "turn-owned-by-another-client",
        starts_turn: false,
        ...(event === "user_message" ? { text: "question" } : {}),
      });

      expect(client.getRunTurnId("chat-owner-remap")).toBe("turn-uncertain-owner-remap");
      expect(client.hasUnsettledRun("chat-owner-remap")).toBe(true);
      expect(
        client.canReconcileCanonicalCompletion(
          "chat-owner-remap",
          generation,
          ["turn-owned-by-another-client"],
          { observedTurnIds: [], activitySnapshotAvailable: true, hasPendingToolCalls: false, activeTurnId: null },
        ),
      ).toBe(true);
    },
  );

  it.each([
    ["goal_status", { status: "idle" }],
    ["turn_end", {}],
  ] as const)(
    "does not infer an unknown attempt from unkeyed %s traffic",
    (event, payload) => {
      const client = new KohakuClient({
        url: "ws://test",
        maxBackoffMs: 500,
        socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
      });
      client.connect();
      const originalSocket = lastSocket();
      originalSocket.fakeOpen();
      client.sendMessage("chat-unkeyed-unknown", "question", undefined, {
        turnId: "turn-unkeyed-unknown",
      });
      const generation = client.getRunGeneration("chat-unkeyed-unknown");

      originalSocket.close();
      vi.advanceTimersByTime(500);
      const replacementSocket = lastSocket();
      expect(replacementSocket).not.toBe(originalSocket);
      replacementSocket.fakeOpen();
      replacementSocket.fakeMessage({
        event,
        chat_id: "chat-unkeyed-unknown",
        ...payload,
      });

      expect(client.getRunTurnId("chat-unkeyed-unknown")).toBe("turn-unkeyed-unknown");
      expect(client.hasUnsettledRun("chat-unkeyed-unknown")).toBe(true);
      expect(
        client.canReconcileCanonicalCompletion(
          "chat-unkeyed-unknown",
          generation,
          ["turn-from-other-client"],
          { observedTurnIds: [], activitySnapshotAvailable: true, hasPendingToolCalls: false, activeTurnId: null },
        ),
      ).toBe(true);
    },
  );

  it("keeps an ACK-lost observed turn unresolved when only idle history is available", () => {
    const client = new KohakuClient({
      url: "ws://test",
      maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-ack-lost", "question", undefined, {
      turnId: "turn-ack-lost",
    });
    const requestGeneration = client.getRunGeneration("chat-ack-lost");
    lastSocket().close();

    expect(
      client.canReconcileCanonicalCompletion(
        "chat-ack-lost",
        requestGeneration,
        [],
        {
          observedTurnIds: ["turn-ack-lost"],
          activitySnapshotAvailable: true,
          hasPendingToolCalls: true,
          activeTurnId: "turn-ack-lost",
        },
      ),
    ).toBe(false);
    expect(
      client.reconcileCanonicalCompletion(
        "chat-ack-lost",
        requestGeneration,
        ["turn-ack-lost"],
        {
          observedTurnIds: ["turn-ack-lost"],
          activitySnapshotAvailable: true, hasPendingToolCalls: false,
          activeTurnId: null,
        },
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-ack-lost")).toBe(false);
    expect(client.hasUnknownMessageSend("chat-ack-lost")).toBe(true);

    vi.advanceTimersByTime(500);
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "delta",
      chat_id: "chat-ack-lost",
      turn_id: "turn-ack-lost",
      text: "reply",
    });
    expect(client.hasUnknownMessageSend("chat-ack-lost")).toBe(true);
    expect(
      client.reconcileCanonicalCompletion(
        "chat-ack-lost",
        requestGeneration,
        ["turn-ack-lost"],
        { observedTurnIds: [], activitySnapshotAvailable: true, hasPendingToolCalls: false, activeTurnId: null },
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-ack-lost")).toBe(false);
    expect(client.hasUnknownMessageSend("chat-ack-lost")).toBe(true);
  });

  it("settles an accepted turn that never reached running from canonical idle", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-accepted-idle", "question", undefined, {
      turnId: "turn-accepted-idle",
    });
    const requestGeneration = client.getRunGeneration("chat-accepted-idle");
    lastSocket().fakeMessage({
      event: "message_accepted",
      chat_id: "chat-accepted-idle",
      turn_id: "turn-accepted-idle",
    });

    expect(
      client.reconcileCanonicalCompletion(
        "chat-accepted-idle",
        requestGeneration,
        [],
        {
          observedTurnIds: ["turn-accepted-idle"],
          activitySnapshotAvailable: true,
          hasPendingToolCalls: false,
          activeTurnId: null,
        },
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-accepted-idle")).toBe(false);
  });

  it("keeps known accepted settlement available beside an unresolved side-channel send", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-known-and-unknown", "accepted", undefined, {
      turnId: "turn-known",
    });
    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-known-and-unknown",
      turn_id: "turn-known",
    });
    client.sendMessage("chat-known-and-unknown", "side-channel", undefined, {
      turnId: "turn-side-channel",
      startsNewRun: false,
    });
    const requestGeneration = client.getRunGeneration("chat-known-and-unknown");
    lastSocket().close();

    expect(
      client.reconcileCanonicalCompletion(
        "chat-known-and-unknown",
        requestGeneration,
        ["turn-known"],
        {
          observedTurnIds: ["turn-known"],
          activitySnapshotAvailable: true,
          hasPendingToolCalls: false,
          activeTurnId: null,
        },
      ),
    ).toBe(true);
    expect(client.hasUnsettledRun("chat-known-and-unknown")).toBe(false);
  });

  it("does not let a pre-send idle response erase a newly accepted turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const requestGeneration = client.getRunGeneration("chat-stale-idle");
    client.sendMessage("chat-stale-idle", "question", undefined, {
      turnId: "turn-after-request",
    });
    lastSocket().fakeMessage({
      event: "message_accepted",
      chat_id: "chat-stale-idle",
      turn_id: "turn-after-request",
    });

    expect(
      client.reconcileCanonicalCompletion(
        "chat-stale-idle",
        requestGeneration,
        [],
        {
          observedTurnIds: [],
          activitySnapshotAvailable: true,
          hasPendingToolCalls: false,
          activeTurnId: null,
        },
      ),
    ).toBe(false);
    expect(client.hasUnsettledRun("chat-stale-idle")).toBe(true);
  });

  it("retires omitted unknown activity without admitting it and fences late proof from a newer run", () => {
    const client = new KohakuClient({
      url: "ws://test",
      maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-retired", "first", undefined, { turnId: "old-attempt" });
    const generation = client.getRunGeneration("chat-retired");
    lastSocket().close();
    vi.advanceTimersByTime(500);
    lastSocket().fakeOpen();
    const idle = { observedTurnIds: [], activitySnapshotAvailable: true,
      hasPendingToolCalls: false, activeTurnId: null };
    expect(client.reconcileCanonicalCompletion("chat-retired", generation, [], idle)).toBe(true);
    expect(client.hasUnknownMessageSend("chat-retired")).toBe(true);
    expect(client.hasUnsettledRun("chat-retired")).toBe(false);
    const received = vi.fn();
    client.onChat("chat-retired", received);
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-retired",
      turn_id: "old-attempt", text: "delayed" });
    expect(client.hasUnknownMessageSend("chat-retired")).toBe(true);
    expect(received).not.toHaveBeenCalled();
    client.sendMessage("chat-retired", "second", undefined, { turnId: "new-attempt" });
    const newerGeneration = client.getRunGeneration("chat-retired");
    lastSocket().fakeMessage({ event: "message_admitted", chat_id: "chat-retired",
      turn_id: "old-attempt", active_turn_id: "old-attempt", starts_turn: true,
      started_at: 100 });
    expect(client.hasUnknownMessageSend("chat-retired")).toBe(false);
    expect(client.getRunTurnId("chat-retired")).toBe("new-attempt");
    expect(client.getRunGeneration("chat-retired")).toBe(newerGeneration);
    expect(client.getRunStartedAt("chat-retired")).toBeNull();
    expect(received).toHaveBeenCalledWith(expect.objectContaining({ event: "message_admitted" }));
  });

  it("repairs a superseded receipt without rewriting the newer owner's clock", () => {
    const client = new KohakuClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-old-receipt", "first", undefined, { turnId: "old" });
    lastSocket().close();
    vi.advanceTimersByTime(500);
    lastSocket().fakeOpen();
    client.sendMessage("chat-old-receipt", "new", undefined, { turnId: "new" });
    lastSocket().fakeMessage({ event: "goal_status", chat_id: "chat-old-receipt",
      status: "running", started_at: 200, turn_id: "new" });
    const generation = client.getRunGeneration("chat-old-receipt");
    lastSocket().fakeMessage({ event: "message_admitted", chat_id: "chat-old-receipt",
      turn_id: "old", active_turn_id: "new", starts_turn: false, started_at: 100 });
    expect(client.hasUnknownMessageSend("chat-old-receipt")).toBe(false);
    expect(client.getRunTurnId("chat-old-receipt")).toBe("new");
    expect(client.getRunGeneration("chat-old-receipt")).toBe(generation);
    expect(client.getRunStartedAt("chat-old-receipt")).toBe(200);
  });

  it("rejects a newer unrepresented unknown run and snapshots without explicit idle authority", () => {
    const client = new KohakuClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    client.connect();
    lastSocket().fakeOpen();
    const generation = client.getRunGeneration("chat-newer-unknown");
    client.sendMessage("chat-newer-unknown", "newer", undefined, { turnId: "newer" });
    lastSocket().close();
    const idle = { observedTurnIds: [], activitySnapshotAvailable: true,
      hasPendingToolCalls: false, activeTurnId: null };
    expect(client.canReconcileCanonicalCompletion("chat-newer-unknown", generation, [], idle)).toBe(false);
    for (const snapshot of [
      { ...idle, activitySnapshotAvailable: false },
      { ...idle, activeTurnId: "newer" },
      { ...idle, hasPendingToolCalls: true },
    ]) {
      expect(client.canReconcileCanonicalCompletion("chat-newer-unknown",
        client.getRunGeneration("chat-newer-unknown"), ["newer"], snapshot)).toBe(false);
    }
    expect(client.hasUnknownMessageSend("chat-newer-unknown")).toBe(true);
  });

  it("correlates a legacy rejection only to one currently sent turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-legacy-reject", "question", undefined, {
      turnId: "turn-legacy-reject",
    });

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-legacy-reject",
      detail: "message_rejected",
      reason: "text_too_large",
    });

    expect(client.hasUnsettledRun("chat-legacy-reject")).toBe(false);
    expect(errors).toEqual([expect.objectContaining({
      kind: "turn_rejected",
      chatId: "chat-legacy-reject",
      turnId: "turn-legacy-reject",
    })]);
  });

  it("correlates legacy lifecycle completion when exactly one turn is unsettled", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onChat("chat-legacy-idle", handler);
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-legacy-idle", "question", undefined, {
      turnId: "turn-legacy-idle",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-legacy-idle",
      status: "running",
      started_at: 4321,
    });

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-legacy-idle",
      status: "idle",
    });

    expect(client.hasUnsettledRun("chat-legacy-idle")).toBe(false);
    expect(client.getRunStartedAt("chat-legacy-idle")).toBeNull();
    expect(handler).toHaveBeenLastCalledWith(expect.objectContaining({
      event: "goal_status",
      status: "idle",
      turn_id: "turn-legacy-idle",
    }));
  });

  it("does not apply an uncorrelated legacy idle to multiple unsettled turns", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onChat("chat-legacy-ambiguous", handler);
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-legacy-ambiguous", "first", undefined, {
      turnId: "turn-legacy-first",
    });
    client.sendMessage("chat-legacy-ambiguous", "second", undefined, {
      turnId: "turn-legacy-second",
    });
    handler.mockClear();

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-legacy-ambiguous",
      status: "idle",
    });

    expect(client.hasUnsettledRun("chat-legacy-ambiguous")).toBe(true);
    expect(handler).not.toHaveBeenCalled();
  });

  it("does not correlate a legacy scope error to an already accepted turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; chatId?: string; turnId?: string }> = [];
    client.onError((error) => errors.push(error));
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-legacy-scope", "question", undefined, {
      turnId: "turn-already-accepted",
    });
    lastSocket().fakeMessage({
      event: "message_admitted",
      chat_id: "chat-legacy-scope",
      turn_id: "turn-already-accepted",
    });

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-legacy-scope",
      detail: "workspace_scope_rejected",
      reason: "chat_running",
    });

    expect(client.hasUnsettledRun("chat-legacy-scope")).toBe(true);
    expect(errors).toEqual([{
      kind: "workspace_scope_rejected",
      reason: "chat_running",
      chatId: "chat-legacy-scope",
      turnId: undefined,
    }]);
  });

  it("does not correlate a scope-control rejection to a preceding unacknowledged message", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-scope-control", "question", undefined, {
      turnId: "turn-before-scope-control",
    });
    client.setWorkspaceScope("chat-scope-control", {
      project_path: "/tmp/project",
      project_name: "project",
      access_mode: "restricted",
    });

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-scope-control",
      detail: "workspace_scope_rejected",
      reason: "chat_running",
    });

    expect(client.hasUnsettledRun("chat-scope-control")).toBe(true);
  });

  it("sends large sidebar ordering state as a correlated WebUI request", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const sessionOrder = Array.from(
      { length: 160 },
      (_, index) => `websocket:${index.toString().padStart(4, "0")}-${"x".repeat(48)}`,
    );
    const state: SidebarStatePayload = {
      schema_version: 1,
      pinned_keys: [],
      archived_keys: [],
      session_order: sessionOrder,
      title_overrides: {},
      project_name_overrides: {},
      tags_by_key: {},
      collapsed_groups: {},
      workbench: { version: 1, tabs: {} },
      view: {
        density: "comfortable",
        show_previews: false,
        show_timestamps: false,
        show_archived: false,
        sort: "manual",
      },
      updated_at: null,
    };

    client.connect();
    lastSocket().fakeOpen();
    const pending = client.setSidebarState(state);

    const [serialized] = lastSocket().sent;
    expect(new TextEncoder().encode(serialized).byteLength).toBeGreaterThan(8_192);
    const request = JSON.parse(serialized) as {
      type: string;
      request_id: string;
      action: string;
      payload: { state: SidebarStatePayload };
    };
    expect(request).toEqual({
      type: "webui_request",
      request_id: expect.any(String),
      action: "sidebar.update",
      payload: { state },
    });
    lastSocket().fakeMessage({
      event: "webui_response",
      request_id: request.request_id,
      ok: true,
      result: state,
    });
    await expect(pending).resolves.toEqual(state);
  });

  it("delivers backend sidebar state updates to every subscriber", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    const state: SidebarStatePayload = {
      schema_version: 1,
      pinned_keys: [],
      archived_keys: [],
      session_order: [],
      title_overrides: {},
      project_name_overrides: {},
      tags_by_key: {},
      collapsed_groups: {},
      workbench: {
        version: 1,
        tabs: {
          "tab:websocket:a": {
            explicit: true,
            title: "Research",
            paneKeys: ["websocket:a", "websocket:b"],
            layout: "columns",
          },
        },
      },
      view: {
        density: "comfortable",
        show_previews: false,
        show_timestamps: false,
        show_archived: false,
        sort: "updated_desc",
      },
      updated_at: "2026-08-11T08:00:00Z",
    };

    client.onSidebarStateUpdate(handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({ event: "sidebar_state_updated", state });

    expect(handler).toHaveBeenCalledWith(state);
  });

  it("does not correlate a new-chat scope rejection to an unrelated sent turn", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-unrelated-scope", "question", undefined, {
      turnId: "turn-unrelated-scope",
    });
    const pendingChat = client.newChat(5_000, {
      project_path: "/missing",
      project_name: "missing",
      access_mode: "restricted",
    });

    lastSocket().fakeMessage({
      event: "error",
      detail: "workspace_scope_rejected",
      reason: "project_path must be an existing directory",
    });

    await expect(pendingChat).rejects.toThrow("workspace_scope_rejected");
    expect(client.hasUnsettledRun("chat-unrelated-scope")).toBe(true);
  });

  it("rejects a correlated system command instead of leaving it pending", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const pending = client.sendSystemCommand("chat-system-reject", "/model invalid");
    const sent = JSON.parse(lastSocket().sent.at(-1) ?? "{}") as { turn_id?: string };
    expect(sent.turn_id).toMatch(/^webui-system:/);

    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-system-reject",
      turn_id: sent.turn_id,
      detail: "message_rejected",
      reason: "invalid_command",
    });

    await expect(pending).rejects.toThrow("message_rejected:invalid_command");
  });

  it("ignores a delayed idle event from an older turn after a new run starts", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatHandler = vi.fn();
    const runHandler = vi.fn();
    client.onChat("chat-delayed-idle", chatHandler);
    client.onRunStatus(runHandler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-delayed-idle",
      status: "running",
      started_at: 1_000,
      turn_id: "turn-old",
    });
    client.sendMessage("chat-delayed-idle", "next question", undefined, {
      turnId: "turn-new",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-delayed-idle",
      status: "running",
      started_at: 2_000,
      turn_id: "turn-new",
    });
    chatHandler.mockClear();
    runHandler.mockClear();

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-delayed-idle",
      status: "idle",
      turn_id: "turn-old",
    });

    expect(client.getRunStartedAt("chat-delayed-idle")).toBe(2_000);
    expect(runHandler).not.toHaveBeenCalled();
    expect(chatHandler).not.toHaveBeenCalled();
  });

  it("accepts a completed snapshot that represents a delayed running frame", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const requestGeneration = client.getRunGeneration("chat-delayed-run");

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-delayed-run",
      status: "running",
      started_at: 12_345,
      turn_id: "turn-complete",
    });

    expect(
      client.reconcileCanonicalCompletion(
        "chat-delayed-run",
        requestGeneration,
        ["turn-complete"],
      ),
    ).toBe(true);
    expect(client.getRunStartedAt("chat-delayed-run")).toBeNull();
  });

  it("preflights canonical completion without fencing or settling the turn", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatHandler = vi.fn();
    client.onChat("chat-preflight", chatHandler);
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-preflight", "question", undefined, {
      turnId: "turn-preflight",
    });
    const requestGeneration = client.getRunGeneration("chat-preflight");

    expect(
      client.canReconcileCanonicalCompletion(
        "chat-preflight",
        requestGeneration,
        ["turn-preflight"],
      ),
    ).toBe(true);
    expect(
      client.canReconcileCanonicalCompletion("chat-preflight", requestGeneration, []),
    ).toBe(false);

    lastSocket().fakeMessage({
      event: "delta",
      chat_id: "chat-preflight",
      turn_id: "turn-preflight",
      text: "still live",
    });
    expect(chatHandler).toHaveBeenCalledWith(
      expect.objectContaining({ event: "delta", text: "still live" }),
    );
  });

  it("notifies accepted canonical idle without a local clock, but not preflight or stale rejection", () => {
    const client = new KohakuClient({ url: "ws://test", maxBackoffMs: 500,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-http-running", "input", undefined, { turnId: "unknown-input" });
    const requestGeneration = client.getRunGeneration("chat-http-running");
    lastSocket().close();
    vi.advanceTimersByTime(500);
    lastSocket().fakeOpen();
    // App can have learned running from HTTP without observing a WS clock.
    let sidebarRunning = true;
    const runStatus = vi.fn((_chatId: string, startedAt: number | null) => {
      sidebarRunning = startedAt !== null;
    });
    client.onRunStatus(runStatus);
    expect(client.getRunStartedAt("chat-http-running")).toBeNull();
    const idle = { activitySnapshotAvailable: true, observedTurnIds: ["unknown-input"],
      hasPendingToolCalls: false, activeTurnId: null };
    expect(client.canReconcileCanonicalCompletion("chat-http-running", requestGeneration,
      ["unknown-input"], idle)).toBe(true);
    expect(runStatus).not.toHaveBeenCalled();
    expect(client.reconcileCanonicalCompletion("chat-http-running", requestGeneration,
      ["unknown-input"], idle)).toBe(true);
    expect(runStatus).toHaveBeenCalledTimes(1);
    expect(runStatus).toHaveBeenLastCalledWith("chat-http-running", null);
    expect(sidebarRunning).toBe(false);
    expect(client.hasUnknownMessageSend("chat-http-running")).toBe(true);
    client.sendMessage("chat-http-running", "new input", undefined, { turnId: "new-input" });
    sidebarRunning = true;
    expect(client.reconcileCanonicalCompletion("chat-http-running", requestGeneration,
      ["unknown-input"], idle)).toBe(false);
    expect(runStatus).toHaveBeenCalledTimes(1);
    expect(sidebarRunning).toBe(true);
    expect(client.hasUnknownMessageSend("chat-http-running")).toBe(true);
    expect(client.hasUnsettledRun("chat-http-running")).toBe(true);
  });

  it("clears the run cache and fences delayed frames after canonical completion", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatHandler = vi.fn();
    const runHandler = vi.fn();
    client.onChat("chat-canonical", chatHandler);
    client.onRunStatus(runHandler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-canonical",
      status: "running",
      started_at: 12_345,
      turn_id: "turn-canonical",
    });
    const requestGeneration = client.getRunGeneration("chat-canonical");

    expect(
      client.reconcileCanonicalCompletion(
        "chat-canonical",
        requestGeneration,
        ["turn-canonical"],
      ),
    ).toBe(true);
    expect(client.getRunStartedAt("chat-canonical")).toBeNull();
    expect(runHandler).toHaveBeenLastCalledWith("chat-canonical", null);
    const deliveredBeforeLateFrames = chatHandler.mock.calls.length;

    lastSocket().fakeMessage({
      event: "delta",
      chat_id: "chat-canonical",
      text: " delayed",
      turn_id: "turn-canonical",
    });
    lastSocket().fakeMessage({
      event: "turn_end",
      chat_id: "chat-canonical",
      turn_id: "turn-canonical",
    });
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-canonical",
      status: "idle",
      turn_id: "turn-canonical",
    });

    expect(chatHandler).toHaveBeenCalledTimes(deliveredBeforeLateFrames);
    expect(client.getRunStartedAt("chat-canonical")).toBeNull();
  });

  it("notifies run status subscribers and replays running chats", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onRunStatus(handler);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-status",
      status: "running",
      started_at: 12_345,
    });
    expect(handler).toHaveBeenCalledWith("chat-status", 12_345);

    const lateHandler = vi.fn();
    client.onRunStatus(lateHandler);
    expect(lateHandler).toHaveBeenCalledWith("chat-status", 12_345);

    lastSocket().fakeMessage({
      event: "goal_status",
      chat_id: "chat-status",
      status: "idle",
    });
    expect(handler).toHaveBeenCalledWith("chat-status", null);
    expect(lateHandler).toHaveBeenCalledWith("chat-status", null);
  });

  it("records goal_state per chat_id without an onChat subscriber", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "goal_state",
      chat_id: "chat-goal-a",
      goal_state: { active: true, ui_summary: "Docs" },
    });
    lastSocket().fakeMessage({
      event: "goal_state",
      chat_id: "chat-goal-b",
      goal_state: { active: true, objective: "Ship API" },
    });
    expect(client.getGoalState("chat-goal-a")).toEqual({ active: true, ui_summary: "Docs" });
    expect(client.getGoalState("chat-goal-b")).toEqual({
      active: true,
      objective: "Ship API",
    });
    lastSocket().fakeMessage({
      event: "goal_state",
      chat_id: "chat-goal-a",
      goal_state: { active: false },
    });
    expect(client.getGoalState("chat-goal-a")).toEqual({ active: false });
  });

  it("records goal_state from turn_end payload when present", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "turn_end",
      chat_id: "chat-te",
      goal_state: { active: true, objective: "Long task" },
    });
    expect(client.getGoalState("chat-te")).toEqual({ active: true, objective: "Long task" });
  });

  it("buffers after unsubscribe until the chat is subscribed again", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const h1 = vi.fn();
    const unsub = client.onChat("chat-rejoin", h1);
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-rejoin", text: "live" });
    expect(h1).toHaveBeenCalledTimes(1);
    unsub();
    lastSocket().fakeMessage({ event: "delta", chat_id: "chat-rejoin", text: "queued" });
    expect(h1).toHaveBeenCalledTimes(1);
    const h2 = vi.fn();
    client.onChat("chat-rejoin", h2);
    expect(h2).toHaveBeenCalledTimes(1);
    expect(h2.mock.calls[0][0]).toMatchObject({ event: "delta", text: "queued" });
  });

  it("dispatches runtime model updates globally", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onRuntimeModelUpdate(handler);
    client.connect();
    lastSocket().fakeOpen();

    lastSocket().fakeMessage({
      event: "runtime_model_updated",
      model_name: "openai/gpt-4.1",
      model_preset: "fast",
    });

    expect(handler).toHaveBeenCalledWith("openai/gpt-4.1", "fast");
  });

  it("dispatches turn model updates to the active chat", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatHandler = vi.fn();
    client.onChat("chat-a", chatHandler);
    client.connect();
    lastSocket().fakeOpen();

    lastSocket().fakeMessage({
      event: "turn_model_updated",
      chat_id: "chat-a",
      model_name: "deepseek/deepseek-chat",
      model_preset: "Deep Research",
      fallback: true,
    });

    expect(chatHandler).toHaveBeenCalledWith({
      event: "turn_model_updated",
      chat_id: "chat-a",
      model_name: "deepseek/deepseek-chat",
      model_preset: "Deep Research",
      fallback: true,
    });
  });

  it("dispatches session updates globally", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const globalHandler = vi.fn();
    const chatHandler = vi.fn();
    client.onSessionUpdate(globalHandler);
    client.onChat("chat-title", chatHandler);
    client.connect();
    lastSocket().fakeOpen();

    lastSocket().fakeMessage(folderAccessSessionUpdatedFixture);

    expect(globalHandler).toHaveBeenCalledWith(
      "chat-title",
      "metadata",
      expect.objectContaining({ project_path: "/agent/home" }),
      expect.objectContaining({
        access_request_id: "0123456789abcdef0123456789abcdef",
        status: "pending",
      }),
    );
    expect(chatHandler).not.toHaveBeenCalled();
  });

  it("resolves newChat() via the server-assigned chat_id", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const promise = client.newChat(1_000);
    expect(lastSocket().sent).toContain(JSON.stringify({ type: "new_chat" }));
    lastSocket().fakeMessage({ event: "attached", chat_id: "fresh-id" });
    await expect(promise).resolves.toBe("fresh-id");
  });

  it("serializes workspace scope only for new chats", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const workspaceScope = {
      project_path: "/tmp/project",
      project_name: "project",
      access_mode: "full" as const,
      restrict_to_workspace: false,
    };
    client.connect();
    lastSocket().fakeOpen();

    const promise = client.newChat(1_000, workspaceScope);
    expect(lastSocket().sent).toContain(
      JSON.stringify({ type: "new_chat", workspace_scope: workspaceScope }),
    );
    lastSocket().fakeMessage({ event: "attached", chat_id: "fresh-id" });
    await expect(promise).resolves.toBe("fresh-id");

    client.sendMessage("fresh-id", "hello");
    expect(lastSocket().sent).toContain(
      JSON.stringify({
        type: "message",
        chat_id: "fresh-id",
        content: "hello",
        request_settings_version: 1,
        request_intent: "new",
        webui: true,
      }),
    );
  });

  it("sends transcription requests and resolves transcription results outside chat dispatch", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const handler = vi.fn();
    client.onChat("chat-a", handler);
    client.connect();
    lastSocket().fakeOpen();

    const promise = client.transcribeAudio("data:audio/webm;base64,AAAA", {
      durationMs: 1234,
      timeoutMs: 1_000,
    });
    const frame = JSON.parse(lastSocket().sent.at(-1) as string);
    expect(frame).toMatchObject({
      type: "transcribe_audio",
      data_url: "data:audio/webm;base64,AAAA",
      duration_ms: 1234,
    });
    expect(typeof frame.request_id).toBe("string");

    lastSocket().fakeMessage({
      event: "transcription_result",
      request_id: frame.request_id,
      text: "hello from voice",
    });
    await expect(promise).resolves.toBe("hello from voice");
    expect(handler).not.toHaveBeenCalled();
  });

  it("rejects pending transcription requests on server errors and socket close", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    const errored = client.transcribeAudio("data:audio/webm;base64,AAAA", { timeoutMs: 1_000 });
    const errorFrame = JSON.parse(lastSocket().sent.at(-1) as string);
    lastSocket().fakeMessage({
      event: "transcription_error",
      request_id: errorFrame.request_id,
      detail: "not_configured",
    });
    await expect(errored).rejects.toThrow("not_configured");

    const dropped = client.transcribeAudio("data:audio/webm;base64,BBBB", { timeoutMs: 1_000 });
    lastSocket().close();
    await expect(dropped).rejects.toThrow("socket closed");
  });

  it("queues sends while connecting and flushes on open", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    client.sendMessage("chat-x", "hello");
    expect(lastSocket().sent).toEqual([]);
    lastSocket().fakeOpen();
    // Attach is sent first because sendMessage adds to knownChats, which
    // handleOpen re-attaches; then the queued message follows.
    expect(lastSocket().sent).toContain(
      JSON.stringify({ type: "message", chat_id: "chat-x", content: "hello", request_settings_version: 1, request_intent: "new", webui: true }),
    );
  });

  it("includes an explicit turn id on outbound WebUI messages", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-x", "hello", undefined, { turnId: "turn-1" });
    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toEqual({
      type: "message",
      chat_id: "chat-x",
      content: "hello",
      turn_id: "turn-1",
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    });
  });

  it("handles the silent system-command lifecycle without hiding concurrent events", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const chatHandler = vi.fn();
    client.onChat("chat-x", chatHandler);
    client.connect();
    lastSocket().fakeOpen();

    const pending = client.sendSystemCommand("chat-x", "  /model fast  ", 1_000);
    const frame = JSON.parse(lastSocket().sent.at(-1) as string);
    expect(frame).toMatchObject({
      type: "message",
      chat_id: "chat-x",
      content: "/model fast",
      webui: true,
    });
    expect(frame).not.toHaveProperty("request_settings_version");
    expect(frame).not.toHaveProperty("request_intent");
    expect(frame.turn_id).toMatch(/^webui-system:/);

    lastSocket().fakeMessage({
      event: "message",
      chat_id: "chat-x",
      text: "normal reply",
      turn_id: "normal-turn",
    });
    lastSocket().fakeMessage({
      event: "message",
      chat_id: "chat-x",
      text: "Switched model preset to fast.",
      turn_id: frame.turn_id,
    });

    await expect(pending).resolves.toBeUndefined();
    expect(chatHandler).toHaveBeenCalledTimes(1);
    expect(chatHandler).toHaveBeenCalledWith(expect.objectContaining({
      text: "normal reply",
      turn_id: "normal-turn",
    }));
    const interrupted = client.sendSystemCommand("chat-x", "/model fast", 1_000);
    lastSocket().close();
    await expect(interrupted).rejects.toThrow("socket closed");
  });

  it("sends selected assistant text as separate quoted context", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    client.sendMessage("chat-x", "What does this mean?", undefined, {
      quotedContext: "  selected answer excerpt  ",
    });

    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toEqual({
      type: "message",
      chat_id: "chat-x",
      content: "What does this mean?",
      quoted_context: "selected answer excerpt",
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    });
  });

  it("includes CLI app attachments in outbound messages", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    client.sendMessage(
      "chat-cli",
      "@drawio please make this diagram",
      undefined,
      {
        cliApps: [{
          name: "drawio",
          display_name: "Draw.io",
          category: "diagrams",
          entry_point: "cli-anything-drawio",
          logo_url: null,
          brand_color: "#F08705",
        }],
      },
    );

    expect(lastSocket().sent).toContain(
      JSON.stringify({
        type: "message",
        chat_id: "chat-cli",
        content: "@drawio please make this diagram",
        cli_apps: [{
          name: "drawio",
          display_name: "Draw.io",
          category: "diagrams",
          entry_point: "cli-anything-drawio",
          logo_url: null,
          brand_color: "#F08705",
        }],
        request_settings_version: 1,
      request_intent: "new",
      webui: true,
      }),
    );
  });

  it("includes MCP preset attachments in outbound messages", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    client.sendMessage(
      "chat-mcp",
      "@browserbase check this page",
      undefined,
      {
        mcpPresets: [{
          name: "browserbase",
          display_name: "Browserbase",
          category: "browser",
          transport: "streamableHttp",
          status: "configured",
          configured: true,
          logo_url: "https://example.invalid/browserbase.svg",
          brand_color: "#111827",
        }],
      },
    );

    expect(lastSocket().sent).toContain(
      JSON.stringify({
        type: "message",
        chat_id: "chat-mcp",
        content: "@browserbase check this page",
        mcp_presets: [{
          name: "browserbase",
          display_name: "Browserbase",
          category: "browser",
          transport: "streamableHttp",
          status: "configured",
          configured: true,
          logo_url: "https://example.invalid/browserbase.svg",
          brand_color: "#111827",
        }],
        request_settings_version: 1,
      request_intent: "new",
      webui: true,
      }),
    );
  });

  it("includes session mentions in outbound messages", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    client.sendMessage("chat-current", "Use @pricing", undefined, {
      sessionMentions: [{
        name: "pricing",
        session_key: "websocket:pricing",
        title: "Pricing",
      }],
    });

    expect(lastSocket().sent).toContain(JSON.stringify({
      type: "message",
      chat_id: "chat-current",
      content: "Use @pricing",
      session_mentions: [{
        name: "pricing",
        session_key: "websocket:pricing",
        title: "Pricing",
      }],
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    }));
  });

  it("re-attaches known chats after a reconnect", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 10,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.onChat("chat-z", () => {});
    client.connect();
    lastSocket().fakeOpen();
    expect(lastSocket().sent).toContain(
      JSON.stringify({ type: "attach", chat_id: "chat-z" }),
    );
    // Drop the socket.
    lastSocket().close();
    // Advance the backoff timer.
    await vi.advanceTimersByTimeAsync(20);
    const reconnected = lastSocket();
    expect(reconnected).not.toBe(FakeSocket.instances[0]);
    reconnected.fakeOpen();
    expect(reconnected.sent).toContain(
      JSON.stringify({ type: "attach", chat_id: "chat-z" }),
    );
  });

  it("reports status transitions through onStatus", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const seen: string[] = [];
    client.onStatus((s) => seen.push(s));
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().close();
    expect(seen).toEqual(["idle", "connecting", "open", "closed"]);
  });

  it("does not schedule a reconnect when close() is called explicitly", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 10,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const seen: string[] = [];
    client.onStatus((s) => seen.push(s));
    client.connect();
    lastSocket().fakeOpen();
    client.close();
    // Advance past any possible backoff window to prove no reconnect was scheduled.
    await vi.advanceTimersByTimeAsync(200);
    expect(FakeSocket.instances).toHaveLength(1);
    // "reconnecting" must never appear after an intentional close.
    expect(seen).not.toContain("reconnecting");
    expect(seen.at(-1)).toBe("closed");
  });

  it("passes media through into the message envelope", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-x", "look", [
      { data_url: "data:image/png;base64,AAAA", name: "shot.png" },
    ]);
    const lastFrame = JSON.parse(lastSocket().sent.at(-1) as string);
    expect(lastFrame).toEqual({
      type: "message",
      chat_id: "chat-x",
      content: "look",
      media: [{ data_url: "data:image/png;base64,AAAA", name: "shot.png" }],
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    });
  });

  it("omits media from the envelope when no images are attached", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-x", "hello");
    const lastFrame = JSON.parse(lastSocket().sent.at(-1) as string);
    expect(lastFrame).not.toHaveProperty("media");
    expect(lastFrame).toEqual({
      type: "message",
      chat_id: "chat-x",
      content: "hello",
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    });
  });

  it("emits a message_too_big error when the socket closes with code 1009", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string }> = [];
    client.onError((e) => errors.push(e));
    client.connect();
    lastSocket().fakeOpen();
    // Server rejected an outbound frame as too large.
    lastSocket().fakeCloseWithCode(1009);
    expect(errors).toEqual([{ kind: "message_too_big" }]);
  });

  it("emits workspace scope rejection errors from server frames", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string; reason?: string; chatId?: string }> = [];
    client.onError((e) => errors.push(e));
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeMessage({
      event: "error",
      chat_id: "chat-a",
      detail: "workspace_scope_rejected",
      reason: "chat_running",
    });
    expect(errors).toEqual([
      {
        kind: "workspace_scope_rejected",
        reason: "chat_running",
        chatId: "chat-a",
      },
    ]);
  });

  it("rejects pending new chats when workspace scope is rejected", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    const pending = client.newChat(5_000, {
      project_path: "/missing",
      project_name: "missing",
      access_mode: "restricted",
    });
    lastSocket().fakeMessage({
      event: "error",
      detail: "workspace_scope_rejected",
      reason: "project_path must be an existing directory",
    });
    await expect(pending).rejects.toThrow("workspace_scope_rejected");
  });

  it("isolates throwing error handlers so reconnect bookkeeping still runs", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 5,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    // First handler explodes; subsequent reconnect state must be untouched.
    client.onError(() => {
      throw new Error("subscriber blew up");
    });
    const seenStatuses: string[] = [];
    client.onStatus((s) => seenStatuses.push(s));
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().fakeCloseWithCode(1009);
    // Despite the throwing handler, the client must still schedule a reconnect.
    expect(seenStatuses).toContain("reconnecting");
    await vi.advanceTimersByTimeAsync(20);
    expect(FakeSocket.instances.length).toBeGreaterThan(1);
  });

  it("does not emit a stream error on a vanilla socket close", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const errors: Array<{ kind: string }> = [];
    client.onError((e) => errors.push(e));
    client.connect();
    lastSocket().fakeOpen();
    lastSocket().close();
    expect(errors).toEqual([]);
  });

  it("surfaces 'reconnecting' only on an unexpected drop", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 5,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    const seen: string[] = [];
    client.onStatus((s) => seen.push(s));
    client.connect();
    lastSocket().fakeOpen();
    // Simulate the remote side hanging up (no client.close() call).
    lastSocket().close();
    await vi.advanceTimersByTimeAsync(50);
    expect(seen).toContain("reconnecting");
    expect(FakeSocket.instances.length).toBeGreaterThan(1);
  });
});

describe("versioned request settings", () => {
  it("sends new intent and no steer target for a normal message", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-x", "hello", undefined, {
      turnId: "turn-new",
    });
    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toEqual({
      type: "message",
      chat_id: "chat-x",
      content: "hello",
      turn_id: "turn-new",
      request_settings_version: 1,
      request_intent: "new",
      webui: true,
    });
  });

  it("sends exact steer target identity for guidance", () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();
    client.sendMessage("chat-x", "guide", undefined, {
      turnId: "turn-steer",
      requestIntent: "steer",
      targetTurnId: "turn-active",
    });
    expect(JSON.parse(lastSocket().sent.at(-1) as string)).toMatchObject({
      request_settings_version: 1,
      request_intent: "steer",
      target_turn_id: "turn-active",
    });
  });
});

describe("mobile run keepalive and centralized task notifications", () => {
  function createClient() {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();
    return { client, socket };
  }

  describe("desktop task completion alerts", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
    });

    function openBackgroundDesktop() {
      const { client, socket } = createClient();
      vi.stubGlobal("__TAURI_INTERNALS__", {});
      vi.spyOn(document, "hasFocus").mockReturnValue(false);
      const notify = vi.spyOn(client, "requestMutation").mockResolvedValue({ ok: true });
      renderHook(useTaskNotifications, {
        wrapper: ({ children }: { children: ReactNode }) => createElement(ClientProvider, {
          client,
          token: "token",
          children,
        }),
      });
      return { client, socket, notify };
    }

    it("alerts once for the owned completion after progress, retries and another turn", () => {
      const { client, socket, notify } = openBackgroundDesktop();
      const chat_id = "chat-1";
      client.sendMessage(chat_id, "task", undefined, { turnId: "owned-1" });

      socket.fakeMessage({
        event: "message", kind: "progress", text: "Working", chat_id, turn_id: "owned-1",
      });
      expect(notify).not.toHaveBeenCalled();
      socket.fakeMessage({
        event: "retry_status", state: "waiting", attempt: 1, error_kind: "connection",
        retry_after_s: 2, chat_id, turn_id: "owned-1",
      });
      expect(notify).not.toHaveBeenCalled();
      socket.fakeMessage({ event: "turn_end", chat_id, turn_id: "other-1", outcome: "completed" });
      expect(notify).not.toHaveBeenCalled();

      socket.fakeMessage({ event: "turn_end", chat_id, turn_id: "owned-1", outcome: "completed" });
      expect(notify).toHaveBeenCalledTimes(1);
      expect(notify.mock.calls[0]?.[0]).toBe("desktop.notify");
    });

    it("stays quiet while recovery waits and alerts for a later legacy completion", () => {
      const { client, socket, notify } = openBackgroundDesktop();
      const chat_id = "chat-1";
      client.sendMessage(chat_id, "task", undefined, { turnId: "waiting-1" });
      socket.fakeMessage({ event: "recovery_state", chat_id, status: "awaiting_user", recovery_id: "waiting" });
      expect(notify).not.toHaveBeenCalled();

      client.sendMessage(chat_id, "new task", undefined, { turnId: "legacy-1" });
      socket.fakeMessage({ event: "turn_end", chat_id, turn_id: "legacy-1" });
      expect(notify).toHaveBeenCalledTimes(1);
      expect(notify.mock.calls[0]?.[0]).toBe("desktop.notify");
    });

    it("alerts on failures and stays quiet when a desktop task stops without a result", () => {
      const { client, socket, notify } = openBackgroundDesktop();
      const chat_id = "chat-1";
      client.sendMessage(chat_id, "task", undefined, { turnId: "recovery-failed-1" });
      socket.fakeMessage({ event: "recovery_state", chat_id, status: "failed", recovery_id: "failed" });
      expect(notify).toHaveBeenCalledTimes(1);

      for (const outcome of ["failed", "cancelled", "interrupted"] as const) {
        const turn_id = `${outcome}-1`;
        client.sendMessage(chat_id, "task", undefined, { turnId: turn_id });
        socket.fakeMessage({ event: "turn_end", chat_id, turn_id, outcome });
        expect(notify).toHaveBeenCalledTimes(2);
      }
      expect(notify.mock.calls.map(([action]) => action)).toEqual(["desktop.notify", "desktop.notify"]);
    });
  });

  it("starts for a phone-started run and notifies centrally on turn_end", () => {
    const start = vi.fn();
    const stop = vi.fn();
    window.nanobotMobileRunKeepAlive = { start, stop };
    const { client, socket } = createClient();
    const notifications: unknown[] = [];
    const chatEvents: string[] = [];
    client.onTaskNotification((event) => notifications.push(event));
    client.onChat("chat-1", () => chatEvents.push("chat"));

    client.sendMessage("chat-1", "hello", undefined, { turnId: "turn-1" });
    expect(start).toHaveBeenCalledWith("chat-1");

    socket.fakeMessage({
      event: "turn_end",
      chat_id: "chat-1",
      turn_id: "turn-1",
      outcome: "completed",
    });

    expect(notifications).toEqual([{
      kind: "turn_end",
      chatId: "chat-1",
      turnId: "turn-1",
      outcome: "completed",
    }]);
    expect(chatEvents).toEqual(["chat"]);
    expect(stop).toHaveBeenCalledWith("chat-1");
  });

  it("still notifies when no chat view is mounted", () => {
    const { client, socket } = createClient();
    const notification = vi.fn();
    client.onTaskNotification(notification);
    client.sendMessage("chat-2", "hello", undefined, { turnId: "turn-2" });

    socket.fakeMessage({
      event: "turn_end",
      chat_id: "chat-2",
      turn_id: "turn-2",
      outcome: "completed",
    });

    expect(notification).toHaveBeenCalledWith(expect.objectContaining({
      kind: "turn_end",
      chatId: "chat-2",
      turnId: "turn-2",
    }));
  });

  it("does not start for side-channel or steer messages", () => {
    const start = vi.fn();
    window.nanobotMobileRunKeepAlive = { start, stop: vi.fn() };
    const { client, socket } = createClient();

    client.sendMessage("chat-3", "side", undefined, {
      turnId: "side-1",
      sideChannel: true,
      startsNewRun: false,
    });
    client.sendMessage("chat-3", "steer", undefined, {
      turnId: "steer-1",
      startsNewRun: false,
    });

    expect(start).not.toHaveBeenCalled();
    socket.fakeMessage({
      event: "turn_end",
      chat_id: "chat-3",
      turn_id: "side-1",
      outcome: "completed",
    });
    expect(start).not.toHaveBeenCalled();
  });

  it("emits one owned completion across reconciliation and a terminal replay", () => {
    const { client, socket } = createClient();
    const notify = vi.fn();
    client.onTaskNotification(notify);
    client.sendMessage("chat-17", "work", undefined, { turnId: "turn-42" });
    socket.fakeCloseWithCode(1006);
    client.connect();
    lastSocket().fakeOpen();
    const snapshot = { activitySnapshotAvailable: true, observedTurnIds: ["turn-42"], activeTurnId: null, hasPendingToolCalls: false };
    for (const id of ["chat-17", "chat-17", "chat-18"]) {
      client.reconcileCanonicalCompletion(id, client.getRunGeneration(id), ["turn-42"], snapshot);
    }
    lastSocket().fakeMessage({ event: "turn_end", chat_id: "chat-17", turn_id: "turn-42", outcome: "completed" });
    expect(notify.mock.calls).toEqual([[{ kind: "turn_end", chatId: "chat-17", turnId: "turn-42", outcome: "completed" }]]);
  });

  it("does not stop keepalive on a socket drop", () => {
    const stop = vi.fn();
    window.nanobotMobileRunKeepAlive = { start: vi.fn(), stop };
    const { client, socket } = createClient();
    client.sendMessage("chat-4", "hello", undefined, { turnId: "turn-4" });

    socket.fakeCloseWithCode(1006);

    expect(stop).not.toHaveBeenCalled();
  });

  it("stops keepalive when a new run is rejected", () => {
    const stop = vi.fn();
    window.nanobotMobileRunKeepAlive = { start: vi.fn(), stop };
    const { client, socket } = createClient();
    const notification = vi.fn();
    client.onTaskNotification(notification);
    client.sendMessage("chat-5", "hello", undefined, { turnId: "turn-5" });

    socket.fakeMessage({
      event: "error",
      chat_id: "chat-5",
      turn_id: "turn-5",
      detail: "message_rejected",
    });

    expect(stop).toHaveBeenCalledWith("chat-5");
    expect(notification).not.toHaveBeenCalled();
  });

  it("stops keepalive when the user stops the run locally", () => {
    const stop = vi.fn();
    window.nanobotMobileRunKeepAlive = { start: vi.fn(), stop };
    const { client } = createClient();
    client.sendMessage("chat-6", "hello", undefined, { turnId: "turn-6" });

    client.finishRunLocally("chat-6");

    expect(stop).toHaveBeenCalledWith("chat-6");
  });

  it("starts keepalive for phone-started runs even when task notifications are disabled", () => {
    const start = vi.fn();
    window.nanobotMobileRunKeepAlive = { start, stop: vi.fn() };
    writeLocalPreferences({ ...DEFAULT_LOCAL_PREFS, browserNotifications: false });
    const { client } = createClient();

    client.sendMessage("chat-7", "hello", undefined, { turnId: "turn-7" });

    expect(start).toHaveBeenCalledWith("chat-7");
  });

  it("notifies and stops keepalive when recovery fails", () => {
    const stop = vi.fn();
    window.nanobotMobileRunKeepAlive = { start: vi.fn(), stop };
    const { client, socket } = createClient();
    const notification = vi.fn();
    client.onTaskNotification(notification);
    client.sendMessage("chat-8", "hello", undefined, { turnId: "turn-8" });

    socket.fakeMessage({
      event: "recovery_state",
      chat_id: "chat-8",
      status: "failed",
      recovery_id: "recovery-1",
    });

    expect(notification).toHaveBeenCalledWith({
      kind: "recovery",
      chatId: "chat-8",
      status: "failed",
    });
    expect(stop).toHaveBeenCalledWith("chat-8");
  });

  it("does not notify for a recovered recovery state", () => {
    const { client, socket } = createClient();
    const notification = vi.fn();
    client.onTaskNotification(notification);
    client.sendMessage("chat-9", "hello", undefined, { turnId: "turn-9" });

    socket.fakeMessage({
      event: "recovery_state",
      chat_id: "chat-9",
      status: "recovered",
      recovery_id: "recovery-2",
    });

    expect(notification).not.toHaveBeenCalled();
  });
});

describe("bounded mobile recovery", () => {
  it("stops silent retrying and exposes an actionable retry after repeated reauth failure", async () => {
    const reauth = vi.fn().mockResolvedValue(null);
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 2_000,
      onReauth: reauth,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    for (let attempt = 0; attempt < 3; attempt += 1) {
      lastSocket().fakeCloseWithCode(1008);
      await vi.advanceTimersByTimeAsync(2_000);
    }

    expect(client.status).toBe("error");
    const socketsBeforeRetry = FakeSocket.instances.length;

    reauth.mockResolvedValue("ws://refreshed");
    await client.retry();
    expect(FakeSocket.instances.length).toBe(socketsBeforeRetry + 1);
    expect(lastSocket().url).toBe("ws://refreshed");
    expect(client.status).toBe("connecting");
    lastSocket().fakeOpen();
    expect(client.status).toBe("open");
  });

  it("keeps retrying across transient reauth network errors beyond 3 attempts", async () => {
    const reauth = vi.fn().mockRejectedValue(new Error("network offline"));
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      maxBackoffMs: 1_000,
      onReauth: reauth,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    lastSocket().fakeOpen();

    lastSocket().fakeCloseWithCode(1006);
    for (let attempt = 0; attempt < 4; attempt += 1) {
      await vi.advanceTimersByTimeAsync(1_000);
    }

    expect(reauth).toHaveBeenCalledTimes(4);
    expect(client.status).toBe("reconnecting");

    reauth.mockResolvedValue("ws://recovered");
    await vi.advanceTimersByTimeAsync(1_000);
    expect(lastSocket().url).toBe("ws://recovered");
    lastSocket().fakeOpen();
    expect(client.status).toBe("open");
  });

  it("force-closes a half-open socket when retry() is invoked", async () => {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: true,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const first = lastSocket();
    first.fakeOpen();
    const closeSpy = vi.spyOn(first, "close");

    await client.retry();

    expect(closeSpy).toHaveBeenCalledOnce();
    expect(FakeSocket.instances.length).toBeGreaterThan(1);
    expect(lastSocket()).not.toBe(first);
    expect(client.status).toBe("connecting");
  });
});

describe("WebUI mutation replay safety", () => {
  type MutationRejection = Error & {
    status?: number;
    outcome?: string;
    requestId?: string;
  };

  function connectedClient() {
    const client = new KohakuClient({
      url: "ws://test",
      reconnect: false,
      socketFactory: (url) => new FakeSocket(url) as unknown as WebSocket,
    });
    client.connect();
    const socket = lastSocket();
    socket.fakeOpen();
    return { client, socket };
  }

  async function rejectionOf(failure: Promise<unknown>): Promise<MutationRejection> {
    const rejection = await failure.then(
      () => null,
      (error: unknown) => error,
    );
    expect(rejection).toBeInstanceOf(Error);
    expect((rejection as Error).name).toBe("WebUIMutationError");
    return rejection as MutationRejection;
  }

  it("sends an explicit request id on the mutation frame", async () => {
    const { client, socket } = connectedClient();

    const pending = client.requestMutation<{ saved: boolean }>(
      "sidebar.update",
      {},
      1_000,
      "request-fixed",
    );
    const frame = JSON.parse(socket.sent.at(-1) as string) as {
      type: string;
      action: string;
      request_id: string;
    };
    expect(frame).toEqual({
      type: "webui_request",
      action: "sidebar.update",
      request_id: "request-fixed",
      payload: {},
    });

    socket.fakeMessage({
      event: "webui_response",
      request_id: "request-fixed",
      ok: true,
      result: { saved: true },
    });
    await expect(pending).resolves.toEqual({ saved: true });
  });

  it("rejects with the server outcome and the originating request id", async () => {
    const { client, socket } = connectedClient();

    const pending = client.requestMutation(
      "sidebar.update",
      {},
      1_000,
      "request-fixed",
    );
    const rejection = rejectionOf(pending);
    socket.fakeMessage({
      event: "webui_response",
      request_id: "request-fixed",
      ok: false,
      error: { status: 409, message: "replay_uncertain", outcome: "uncertain" },
    });

    const error = await rejection;
    expect(error.status).toBe(409);
    expect(error.message).toBe("replay_uncertain");
    expect(error.outcome).toBe("uncertain");
    expect(error.requestId).toBe("request-fixed");
  });

  it("rejects a timed-out mutation as uncertain with status 504 and its request id", async () => {
    const { client } = connectedClient();

    const rejection = rejectionOf(
      client.requestMutation(
        "sidebar.update",
        {},
        10,
        "request-timeout",
      ),
    );
    await vi.advanceTimersByTimeAsync(10);

    const error = await rejection;
    expect(error.status).toBe(504);
    expect(error.outcome).toBe("uncertain");
    expect(error.requestId).toBe("request-timeout");
  });

  it("forwards an explicit request id through setSidebarState", async () => {
    const { client, socket } = connectedClient();
    const state: SidebarStatePayload = {
      schema_version: 1,
      pinned_keys: [],
      archived_keys: [],
      session_order: [],
      title_overrides: {},
      project_name_overrides: {},
      tags_by_key: {},
      collapsed_groups: {},
      workbench: { version: 1, tabs: {} },
      view: {
        density: "comfortable",
        show_previews: false,
        show_timestamps: false,
        show_archived: false,
        sort: "manual",
      },
      updated_at: null,
    };

    const pending = client.setSidebarState(state, "sidebar-fixed");
    const frame = JSON.parse(socket.sent.at(-1) as string);
    expect(frame).toEqual({
      type: "webui_request",
      request_id: "sidebar-fixed",
      action: "sidebar.update",
      payload: { state },
    });

    socket.fakeMessage({
      event: "webui_response",
      request_id: "sidebar-fixed",
      ok: true,
      result: state,
    });
    await expect(pending).resolves.toEqual(state);
  });
});
