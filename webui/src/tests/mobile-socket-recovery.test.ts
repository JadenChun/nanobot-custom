import { afterEach, describe, expect, it, vi } from "vitest";

import { createMobileRuntime, type MobileEvent, type MobileTransport } from "@/lib/mobile-runtime";
import { KohakuClient } from "@/lib/kohaku-client";

async function flushSocketWork(): Promise<void> {
  for (let index = 0; index < 40; index += 1) await Promise.resolve();
}

afterEach(() => vi.useRealTimers());

describe("KohakuClient with the actual mobile socket adapter", () => {
  it.each(["recover", "renew"] as const)("owns reconnect after %s and never replays an unacknowledged message", async (operation) => {
    vi.useFakeTimers();
    let listener: ((event: MobileEvent) => void) | null = null;
    let nextId = 0;
    let activeId = 0;
    let finishClose!: () => void;
    const nativeClose = new Promise<void>((resolve) => { finishClose = resolve; });
    const transport: MobileTransport = {
      start: vi.fn(async () => undefined),
      status: vi.fn(async () => ({})),
      http: vi.fn(async () => 1),
      cancel: vi.fn(async () => undefined),
      onEvent: vi.fn(async (callback) => {
        listener = callback;
        return () => { listener = null; };
      }),
      openLogin: vi.fn(async () => undefined),
      refresh: vi.fn(async () => true),
      wsOpen: vi.fn(async () => {
        if (activeId) throw new Error("single native socket still occupied");
        activeId = ++nextId;
        // The real native event pump may deliver ready before wsOpen returns.
        listener?.({ type: "ws_frame", ws_id: activeId,
          data_base64: btoa(JSON.stringify({ event: "ready", chat_id: "mobile-chat" })) });
        return activeId;
      }),
      wsSend: vi.fn(async () => undefined),
      wsClose: vi.fn(async (id) => {
        if (id === 1) await nativeClose;
        if (activeId === id) activeId = 0;
      }),
      wsPing: vi.fn(async () => false),
      stop: vi.fn(async () => undefined),
    };
    const runtime = createMobileRuntime(transport);
    const adapters: WebSocket[] = [];
    const client = new KohakuClient({
      url: "wss://host/mobile/v1/ws",
      maxBackoffMs: 50,
      socketFactory: (url) => {
        const socket = runtime.createWebSocket(url);
        adapters.push(socket);
        return socket;
      },
    });
    const statuses: string[] = [];
    client.onStatus((status) => statuses.push(status));
    try {
      client.connect();
      await flushSocketWork();
      expect(client.status).toBe("open");
      expect(client.defaultChatId).toBe("mobile-chat");
      client.onChat("mobile-chat", () => undefined);
      client.sendMessage("mobile-chat", "do not duplicate", undefined, { turnId: "mobile-attempt" });
      const generation = client.getRunGeneration("mobile-chat");
      const idleSnapshot = { observedTurnIds: [], hasPendingToolCalls: false, activeTurnId: null };
      const canSettleFromIdle = () => client.canReconcileCanonicalCompletion(
        "mobile-chat", generation, [], idleSnapshot,
      );
      // Read-only fixture inspection, not a new transport admission authority.
      const pendingAttempt = () => [...client["pendingMessageSends"].values()]
        .find(({ chatId, turnId }) => chatId === "mobile-chat" && turnId === "mobile-attempt");
      // No owner ACK was emitted: this is still sent, not accepted or unknown.
      expect(pendingAttempt()).toMatchObject({ state: "sent" });
      expect(canSettleFromIdle()).toBe(false);

      const maintenance = runtime[operation]();
      await flushSocketWork();
      expect(adapters[0].readyState).toBe(3);
      expect(client.status).toBe("reconnecting");
      expect(pendingAttempt()).toMatchObject({ state: "unknown" });
      // Idle/absent history alone cannot resolve the attempt's admission.
      expect(canSettleFromIdle()).toBe(false);
      expect(client.hasUnsettledRun("mobile-chat")).toBe(true);
      expect(adapters).toHaveLength(1);
      expect(transport.wsOpen).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(50);
      expect(adapters).toHaveLength(2);
      expect(client.status).toBe("connecting");
      expect(transport.wsOpen).toHaveBeenCalledTimes(1);
      finishClose();
      await maintenance;
      await flushSocketWork();
      expect(client.status).toBe("open");
      expect(statuses).toEqual(["idle", "connecting", "open", "reconnecting", "connecting", "open"]);
      expect(transport.wsClose).toHaveBeenCalledTimes(1);
      expect(transport.wsClose).toHaveBeenCalledWith(1);
      expect(transport.wsOpen).toHaveBeenCalledTimes(2);
      expect(pendingAttempt()).toMatchObject({ state: "unknown" });
      expect(canSettleFromIdle()).toBe(false);
      expect(client.hasUnsettledRun("mobile-chat")).toBe(true);

      const sends = vi.mocked(transport.wsSend).mock.calls.map(([id, bytes]) => ({
        id, frame: JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>,
      }));
      expect(sends.filter(({ id }) => id === 2)).toEqual([
        { id: 2, frame: { type: "attach", chat_id: "mobile-chat" } },
      ]);
      expect(sends.filter(({ frame }) => frame.type === "message")).toHaveLength(1);
      expect(sends.find(({ frame }) => frame.type === "message")).toMatchObject({
        id: 1, frame: { turn_id: "mobile-attempt", content: "do not duplicate" },
      });
    } finally {
      finishClose();
      client.close();
      runtime.dispose();
      await flushSocketWork();
    }
  });
});
