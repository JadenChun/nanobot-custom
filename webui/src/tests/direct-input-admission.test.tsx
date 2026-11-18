/** Real hook/client admission handoff; no listener, source mutation or model. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useNanobotStream } from "@/hooks/useNanobotStream";
import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { NanobotClient } from "@/lib/nanobot-client";
import { ClientProvider } from "@/providers/ClientProvider";
import type { UIMessage } from "@/lib/types";

const capability = {
  version: 1, source: { store_identity: "a".repeat(64), session_key: "websocket:attempt-chat",
    incarnation: "original-incarnation", generation: 0 },
  source_token: "b".repeat(32), principal: "c".repeat(64),
};
const proof = { ...capability, turn_id: "attempt-1",
  fingerprint: createHash("sha256").update("nanobot.input-attempt.v1\0new\0exact submitted text").digest("hex"),
  event_id: "original-event", run_id: "original-run" };
const attached = { event: "attached", chat_id: "attempt-chat", input_attempt_source: capability,
  input_attempt_result: { version: 1, turn_id: "attempt-1", disposition: "admitted", proof } };
// Qualification supplies the unmodified real backend bridge. The fallback keeps
// the repository suite independently runnable without a machine-local QA path.
const bridgePath = process.env.NANOBOT_INPUT_ATTEMPT_BRIDGE;
const bridge = bridgePath ? JSON.parse(readFileSync(bridgePath, "utf8")) : {
  capability: { event: "attached", chat_id: "attempt-chat", input_attempt_source: capability },
  query: { type: "attach", chat_id: "attempt-chat", input_attempt_query: {
    version: 1, turn_id: proof.turn_id, source: proof.source,
    source_token: proof.source_token, fingerprint: proof.fingerprint } },
  attached, attached_running: attached,
};
const CHAT = "attempt-chat";
const EMPTY: UIMessage[] = [];
const clients: NanobotClient[] = [];
class Socket {
  static all: Socket[] = [];
  readyState = 0;
  sent: Record<string, unknown>[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  send(value: string) { this.sent.push(JSON.parse(value)); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(value: unknown) { this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent); }
  close() { this.readyState = 3; this.onclose?.(); }
  constructor() { Socket.all.push(this); }
}
async function flush() { for (let i = 0; i < 8; i += 1) await Promise.resolve(); }
function make(negotiated = true) {
  const client = new NanobotClient({ url: "ws://qa-no-network", maxBackoffMs: 1,
    socketFactory: () => new Socket() as unknown as WebSocket });
  clients.push(client);
  client.connect();
  const socket = Socket.all.at(-1)!;
  socket.open();
  function Wrapper({ children }: { children: ReactNode }) {
    return <ClientProvider client={client} token="qa-fixture">{children}</ClientProvider>;
  }
  const hook = renderHook(({ chatId, messages }) => useNanobotStream(chatId, messages), {
    wrapper: Wrapper, initialProps: { chatId: CHAT, messages: EMPTY },
  });
  if (negotiated) act(() => socket.message(bridge.capability));
  return { client, socket, hook };
}
async function sendDirect(env: ReturnType<typeof make>) {
  let submitted: ReturnType<typeof env.hook.result.current.send> = null;
  let admitted: boolean | null = null;
  await act(async () => {
    submitted = env.hook.result.current.send("exact submitted text");
    if (submitted?.admission) void submitted.admission.then((value) => { admitted = value; });
    await flush();
  });
  expect(submitted, "invalid harness: direct submission absent").not.toBeNull();
  expect(submitted!.turnId, "invalid harness: backend bridge identity differs").toBe("attempt-1");
  expect(submitted!.admission, "invalid harness: direct admission waiter absent").toBeDefined();
  const frames = env.socket.sent.filter((f) => f.type === "message");
  expect(frames, "invalid harness: send did not reach real client transport").toHaveLength(1);
  return { submitted: submitted!, admitted: () => admitted, frame: frames[0] };
}
async function reconnect(env: ReturnType<typeof make>) {
  act(() => env.socket.close());
  expect(env.client.hasUnknownMessageSend(CHAT)).toBe(true);
  expect(env.hook.result.current.messages.find((m) => m.turnId === "attempt-1")?.deliveryStatus).toBe("unknown");
  await act(async () => { await env.client.retry(); Socket.all.at(-1)!.open(); await flush(); });
  return Socket.all.at(-1)!;
}
beforeEach(() => {
  vi.useFakeTimers(); Socket.all = [];
  let uuid = 0;
  vi.stubGlobal("crypto", {
    randomUUID: () => ++uuid === 1 ? "attempt-1" : `qa-row-${uuid}`,
    subtle: { digest: (_algorithm: unknown, bytes: Uint8Array) => {
      const hash = createHash("sha256").update(bytes).digest();
      return Promise.resolve(hash.buffer.slice(hash.byteOffset, hash.byteOffset + hash.byteLength));
    } },
  });
});
afterEach(() => {
  cleanup();
  act(() => { for (const client of clients.splice(0)) client.close(); });
  vi.unstubAllGlobals(); vi.useRealTimers();
});

describe("real direct SubmittedTurn admission consumer", () => {
  it("negotiated unqualified running keeps direct admission and row unresolved, while preserving activity", async () => {
    const env = make(); const direct = await sendDirect(env);
    expect(direct.frame.input_attempt, "invalid harness: no negotiated wire binding").toBeDefined();
    const next = await reconnect(env);
    await act(async () => { next.message({ event: "goal_status", chat_id: CHAT,
      turn_id: "attempt-1", status: "running", started_at: 123 }); await flush(); });
    expect(env.client.hasUnknownMessageSend(CHAT), "invalid harness: private client proof gate was bypassed").toBe(true);
    expect(env.hook.result.current.isStreaming, "activity hydration must not be suppressed").toBe(true);
    expect(env.hook.result.current.runStartedAt).toBe(123);
    expect(env.hook.result.current.streamError).toMatchObject({ kind: "connection_closed", turnId: "attempt-1" });
    expect(direct.admitted(), "unqualified activity must not prove negotiated direct admission").toBeNull();
    expect(env.hook.result.current.messages[0].deliveryStatus).toBe("unknown");
  });
  it("unqualified delta may project output but cannot repair the negotiated direct user row", async () => {
    const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
    await act(async () => { next.message({ event: "delta", chat_id: CHAT,
      turn_id: "attempt-1", text: "legitimate projected activity" }); await flush(); });
    expect(env.client.hasUnknownMessageSend(CHAT)).toBe(true);
    expect(env.hook.result.current.streamError).toMatchObject({ kind: "connection_closed", turnId: "attempt-1" });
    await act(async () => { await vi.advanceTimersByTimeAsync(34); });
    expect(env.hook.result.current.messages.some((row) => row.content === "legitimate projected activity")).toBe(true);
    expect(direct.admitted()).toBeNull();
    expect(env.hook.result.current.messages.find((m) => m.role === "user")?.deliveryStatus,
      "unqualified projected output must not repair canonical admission").toBe("unknown");
  });
  it.each([false, true])("actual positive attach proof reaches direct waiter/visible disposition without historical run revival (complete=%s)", async (complete) => {
    const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
    expect(next.sent.find((f) => f.type === "attach"), "invalid harness: real handleOpen query differs from backend").toEqual(bridge.query);
    if (complete) await act(async () => {
      next.message({ event: "goal_status", chat_id: CHAT, turn_id: "attempt-1", status: "idle" });
      expect(env.client.reconcileCanonicalCompletion(CHAT, env.client.getRunGeneration(CHAT), ["attempt-1"], {
        activitySnapshotAvailable: true, activeTurnId: null, hasPendingToolCalls: false, observedTurnIds: [],
      })).toBe(true);
      await flush();
    });
    const generation = env.client.getRunGeneration(CHAT);
    await act(async () => { next.message(complete ? bridge.attached : bridge.attached_running); await flush(); });
    expect(env.client.hasUnknownMessageSend(CHAT), "invalid harness: private proof acceptance failed").toBe(false);
    expect(env.client.getRunGeneration(CHAT)).toBe(generation);
    expect(Socket.all.flatMap((s) => s.sent).filter((f) => f.type === "message")).toHaveLength(1);
    if (complete) { expect(env.hook.result.current.isStreaming).toBe(false); expect(env.client.hasUnsettledRun(CHAT)).toBe(false); }
    expect(direct.admitted(), "validated attach proof must reach the real direct admission waiter").toBe(true);
    expect(env.hook.result.current.streamError).toBeNull();
    expect(env.hook.result.current.messages.find((m) => m.role === "user")?.deliveryStatus).toBe("accepted");
  });
  it("actual qualified live receipt resolves the real direct waiter", async () => {
    const env = make(); const direct = await sendDirect(env);
    await act(async () => { env.socket.message({ event: "message_admitted", chat_id: CHAT,
      turn_id: "attempt-1", starts_turn: true, active_turn_id: "attempt-1",
      input_attempt_proof: bridge.attached.input_attempt_result.proof }); await flush(); });
    expect(direct.admitted()).toBe(true);
    expect(env.hook.result.current.messages[0].deliveryStatus).toBe("accepted");
  });
  it("legacy unbound running retains its current direct admission behavior", async () => {
    const env = make(false); const direct = await sendDirect(env);
    expect(direct.frame.input_attempt).toBeUndefined();
    const next = await reconnect(env);
    await act(async () => { next.message({ event: "goal_status", chat_id: CHAT,
      turn_id: "attempt-1", status: "running", started_at: 123 }); await flush(); });
    expect(direct.admitted()).toBe(true);
    expect(env.hook.result.current.messages[0].deliveryStatus).toBe("accepted");
    expect(env.hook.result.current.isStreaming).toBe(true);
  });
  it("a principal-mismatched attach proof cannot settle the real direct waiter", async () => {
    const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
    const wrong = JSON.parse(JSON.stringify(bridge.attached));
    wrong.input_attempt_result.proof.principal = "f".repeat(64);
    await act(async () => { next.message(wrong); await flush(); });
    expect(direct.admitted()).toBeNull();
    expect(env.client.hasUnknownMessageSend(CHAT)).toBe(true);
    expect(env.hook.result.current.messages[0].deliveryStatus).toBe("unknown");
  });
});

describe.each(["attach", "live"])("exact local admission fact from %s", (delivery) => {
  it.each(["principal", "fingerprint", "origin", "turn", "token", "version", "event", "run", "extra", "missing", "unknown"])(
    "does not admit a %s proof/result", async (change) => {
      const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
      const frame = structuredClone(bridge.attached);
      const claim = frame.input_attempt_result.proof;
      if (change === "principal") claim.principal = "f".repeat(64);
      if (change === "fingerprint") claim.fingerprint = "f".repeat(64);
      if (change === "origin") claim.source.generation += 1;
      if (change === "turn") claim.turn_id = "different-attempt";
      if (change === "token") claim.source_token = "f".repeat(32);
      if (change === "version") claim.version = 2;
      if (change === "event") claim.event_id = "";
      if (change === "run") claim.run_id = "";
      if (change === "extra") claim.untrusted = true;
      if (change === "missing") delete frame.input_attempt_result;
      if (change === "unknown") frame.input_attempt_result = {
        version: 1, turn_id: "attempt-1", disposition: "unknown",
      };
      const wire = delivery === "attach" ? frame : {
        event: "message_admitted", chat_id: CHAT, turn_id: "attempt-1",
        input_attempt_proof: frame.input_attempt_result?.proof,
      };
      await act(async () => { next.message(wire); await flush(); });
      expect(direct.admitted()).toBeNull();
      expect(env.client.hasUnknownMessageSend(CHAT)).toBe(true);
      expect(env.hook.result.current.messages[0].deliveryStatus).toBe("unknown");
      expect(env.hook.result.current.streamError).toMatchObject({ kind: "connection_closed", turnId: "attempt-1" });
    },
  );
});

it("ignores old-socket positive proof and caller-supplied local facts", async () => {
  const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
  const callback = vi.fn();
  const unsubscribe = env.client.onChat(CHAT, callback);
  await act(async () => {
    env.socket.message(bridge.attached);
    env.socket.message({ event: "message_admitted", chat_id: CHAT, turn_id: "attempt-1",
      input_attempt_proof: bridge.attached.input_attempt_result.proof });
    next.message({ event: "goal_status", chat_id: CHAT, turn_id: "attempt-1", status: "running", started_at: 42,
      admission: { turnId: "attempt-1", disposition: "admitted" } });
    await flush();
  });
  expect(direct.admitted()).toBeNull();
  expect(callback).toHaveBeenCalledTimes(1);
  expect(callback.mock.calls[0][1]).toEqual({ turnId: "attempt-1", disposition: "unresolved" });
  expect(env.hook.result.current.messages[0].deliveryStatus).toBe("unknown");
  unsubscribe();
});

it("qualifies the exact preparing owner without waiting for asynchronous fingerprint completion", async () => {
  vi.stubGlobal("crypto", { ...globalThis.crypto,
    subtle: { digest: () => new Promise<ArrayBuffer>(() => {}) } });
  const env = make();
  const callback = vi.fn();
  const unsubscribe = env.client.onChat(CHAT, callback);
  act(() => {
    env.hook.result.current.send("exact submitted text");
    env.socket.message({ event: "goal_status", chat_id: CHAT, turn_id: "attempt-1", status: "running", started_at: 42 });
  });
  expect(callback).toHaveBeenCalledWith(expect.objectContaining({ event: "goal_status" }), {
    turnId: "attempt-1", disposition: "unresolved",
  });
  expect(env.hook.result.current.messages[0].deliveryStatus).toBe("sending");
  expect(env.hook.result.current.isStreaming).toBe(true);
  expect(env.socket.sent.filter((frame) => frame.type === "message")).toHaveLength(0);
  unsubscribe();
});

it("cannot create admitted context for an attach query whose exact pending owner was replaced", async () => {
  const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
  const owners = Reflect.get(env.client, "pendingMessageSends") as Map<string, Record<string, unknown>>;
  const key = `${CHAT}\0attempt-1`;
  const callback = vi.fn(); const unsubscribe = env.client.onChat(CHAT, callback);
  // onChat reattaches with the original owner before an exact-key replacement.
  owners.set(key, { ...owners.get(key)! });
  await act(async () => { next.message(bridge.attached); await flush(); });
  expect(callback).toHaveBeenCalledWith(bridge.attached);
  expect(callback.mock.calls.every(([, admission]) => admission?.disposition !== "admitted")).toBe(true);
  expect(direct.admitted()).toBeNull();
  expect(env.hook.result.current.messages[0].deliveryStatus).toBe("unknown");
  unsubscribe();
});

it("keeps a buffered negotiated user echo unresolved when the old row is not mounted", async () => {
  const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
  env.hook.rerender({ chatId: "other-chat", messages: EMPTY });
  next.message({ event: "user_message", chat_id: CHAT, turn_id: "attempt-1", text: "exact submitted text" });
  env.hook.rerender({ chatId: CHAT, messages: EMPTY });
  expect(env.hook.result.current.messages).toEqual([expect.objectContaining({
    role: "user", content: "exact submitted text", turnId: "attempt-1", deliveryStatus: "unknown",
  })]);
  expect(env.client.hasUnknownMessageSend(CHAT)).toBe(true);
  expect(direct.admitted()).toBeNull();
});

it("preserves proven-unsent preparation failure for the actual direct waiter and row", async () => {
  vi.stubGlobal("crypto", { ...globalThis.crypto,
    subtle: { digest: () => Promise.reject(new Error("local crypto unavailable")) } });
  const env = make();
  let submitted: ReturnType<typeof env.hook.result.current.send> = null;
  await act(async () => { submitted = env.hook.result.current.send("exact submitted text"); await flush(); });
  await expect(submitted!.admission).resolves.toBe(false);
  expect(env.socket.sent.filter((frame) => frame.type === "message")).toHaveLength(0);
  expect(env.hook.result.current.messages[0]).toMatchObject({ content: "exact submitted text", deliveryStatus: "failed" });
  expect(env.hook.result.current.streamError).toMatchObject({
    kind: "turn_rejected", reason: "input_attempt_preparation_failed",
  });
});

it("pairs buffered unresolved traffic with its original fact even after the owner is proven", async () => {
  const env = make(); await sendDirect(env); const next = await reconnect(env);
  env.hook.unmount();
  next.message({ event: "delta", chat_id: CHAT, turn_id: "attempt-1", text: "buffered output" });
  next.message(bridge.attached);
  expect(env.client.hasUnknownMessageSend(CHAT)).toBe(false);
  const callback = vi.fn();
  const unsubscribe = env.client.onChat(CHAT, callback);
  expect(callback.mock.calls.map(([event, admission]) => [event.event, admission])).toEqual([
    ["delta", { turnId: "attempt-1", disposition: "unresolved" }],
    ["attached", { turnId: "attempt-1", disposition: "admitted" }],
  ]);
  expect(callback.mock.calls.every(([event]) => !Object.hasOwn(event, "admission"))).toBe(true);
  unsubscribe();
});

it("retains admitted attach context after canonical pruning and private pending cleanup", async () => {
  const env = make(); await sendDirect(env); const next = await reconnect(env);
  env.hook.unmount();
  next.message({ event: "delta", chat_id: CHAT, turn_id: "attempt-1", text: "pruned output" });
  expect(env.client.reconcileCanonicalCompletion(CHAT, env.client.getRunGeneration(CHAT), ["attempt-1"], {
    activitySnapshotAvailable: true, activeTurnId: null, hasPendingToolCalls: false, observedTurnIds: [],
  })).toBe(true);
  next.message(bridge.attached);
  const pending = Reflect.get(env.client, "pendingMessageSends") as Map<string, unknown>;
  expect(pending.has(`${CHAT}\0attempt-1`)).toBe(false);
  const generation = env.client.getRunGeneration(CHAT);
  const callback = vi.fn(); const other = vi.fn();
  const unsubscribeOther = env.client.onChat("other-chat", other);
  const unsubscribe = env.client.onChat(CHAT, callback);
  expect(callback).toHaveBeenCalledTimes(1);
  expect(callback).toHaveBeenCalledWith(bridge.attached, { turnId: "attempt-1", disposition: "admitted" });
  expect(other).not.toHaveBeenCalled();
  expect(env.client.hasUnsettledRun(CHAT)).toBe(false);
  expect(env.client.getRunGeneration(CHAT)).toBe(generation);
  unsubscribe(); unsubscribeOther();
});

it("preserves bounded buffer eviction and per-turn pruning with paired local context", async () => {
  const env = make(); await sendDirect(env); const next = await reconnect(env);
  env.hook.unmount();
  const log = vi.spyOn(console, "log").mockImplementation(() => {});
  for (let index = 0; index < 2001; index += 1) next.message({
    event: "delta", chat_id: CHAT, turn_id: "attempt-1", text: String(index),
  });
  const callback = vi.fn();
  const unsubscribe = env.client.onChat(CHAT, callback);
  expect(callback).toHaveBeenCalledTimes(2000);
  expect(callback.mock.calls[0]).toEqual([expect.objectContaining({ text: "1" }), {
    turnId: "attempt-1", disposition: "unresolved",
  }]);
  expect(callback.mock.calls.at(-1)).toEqual([expect.objectContaining({ text: "2000" }), {
    turnId: "attempt-1", disposition: "unresolved",
  }]);
  unsubscribe();
  next.message({ event: "delta", chat_id: CHAT, turn_id: "attempt-1", text: "prune this" });
  next.message({ event: "delta", chat_id: CHAT, turn_id: "independent-turn", text: "keep this" });
  env.client.reconcileCanonicalCompletion(CHAT, env.client.getRunGeneration(CHAT), ["attempt-1"], {
    activitySnapshotAvailable: true, activeTurnId: null, hasPendingToolCalls: false, observedTurnIds: [],
  });
  const later = vi.fn(); const unsubscribeLater = env.client.onChat(CHAT, later);
  expect(later).toHaveBeenCalledTimes(1);
  expect(later).toHaveBeenCalledWith(expect.objectContaining({ text: "keep this" }));
  unsubscribeLater(); log.mockRestore();
});

it("does not transfer buffered admission across a mounted chat switch", async () => {
  const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
  const originalRows = env.hook.result.current.messages;
  const otherRows: UIMessage[] = [{ id: "other", role: "user", content: "other draft", turnId: "other-turn", createdAt: 1 }];
  env.hook.rerender({ chatId: "other-chat", messages: otherRows });
  await act(async () => { next.message(bridge.attached); await flush(); });
  expect(env.hook.result.current.messages).toEqual(otherRows);
  expect(env.hook.result.current.isStreaming).toBe(false);
  expect(direct.admitted()).toBeNull(); // The old hook waiter was explicitly retired on chat switch.
  env.hook.rerender({ chatId: CHAT, messages: originalRows });
  expect(env.hook.result.current.messages[0].deliveryStatus).toBe("accepted");
  expect(env.hook.result.current.isStreaming).toBe(false);
});

it("repairs only the old direct row while a newer direct owner, error and real composer draft survive", async () => {
  const env = make(); const direct = await sendDirect(env); const next = await reconnect(env);
  let laterAdmitted: boolean | null = null;
  let laterTurn = "";
  await act(async () => {
    const submitted = env.hook.result.current.send("independent later input")!;
    laterTurn = submitted.turnId;
    void submitted.admission!.then((value) => { laterAdmitted = value; });
    await flush();
  });
  // Mount the unchanged actual composer with this direct hook's callback.
  render(<ThreadComposer onSend={env.hook.result.current.send} />);
  fireEvent.change(screen.getByLabelText("Message input"), { target: { value: "independent draft" } });
  const rowBefore = env.hook.result.current.messages.find((row) => row.turnId === laterTurn);
  // The existing error callback is the same correlated transport-close seam;
  // leave the live query/socket intact so the old proof can legitimately return.
  act(() => Reflect.get(env.client, "emitError").call(env.client, {
    kind: "connection_closed", chatId: CHAT, turnId: laterTurn,
  }));
  const laterRowBeforeProof = env.hook.result.current.messages.find((row) => row.turnId === laterTurn);
  expect(rowBefore?.deliveryStatus).toBe("sending");
  expect(laterRowBeforeProof?.deliveryStatus).toBe("unknown");
  expect(env.hook.result.current.streamError).toMatchObject({ kind: "connection_closed", turnId: laterTurn });
  const generation = env.client.getRunGeneration(CHAT);
  expect(env.client.getRunTurnId(CHAT)).toBe(laterTurn);
  await act(async () => { next.message(bridge.attached); await flush(); });
  expect(direct.admitted()).toBe(true);
  expect(laterAdmitted).toBeNull();
  expect(env.hook.result.current.messages.find((row) => row.turnId === "attempt-1")?.deliveryStatus).toBe("accepted");
  expect(env.hook.result.current.messages.find((row) => row.turnId === laterTurn)).toEqual(laterRowBeforeProof);
  expect(env.hook.result.current.streamError).toMatchObject({ kind: "connection_closed", turnId: laterTurn });
  expect(screen.getByLabelText("Message input")).toHaveValue("independent draft");
  expect(env.client.getRunGeneration(CHAT)).toBe(generation);
  expect(env.client.getRunTurnId(CHAT)).toBe(laterTurn);
  expect(Socket.all.flatMap((socket) => socket.sent).filter((frame) => frame.type === "message")).toHaveLength(2);
});

it("preserves legacy route acknowledgement and live owner receipt semantics", async () => {
  const env = make(false); const direct = await sendDirect(env);
  await act(async () => {
    env.socket.message({ event: "message_accepted", chat_id: CHAT, turn_id: "attempt-1" });
    await flush();
  });
  expect(direct.admitted()).toBeNull();
  expect(env.hook.result.current.messages[0].deliveryStatus).toBe("accepted");
  await act(async () => {
    env.socket.message({ event: "message_admitted", chat_id: CHAT, turn_id: "attempt-1" });
    await flush();
  });
  expect(direct.admitted()).toBe(true);
});
