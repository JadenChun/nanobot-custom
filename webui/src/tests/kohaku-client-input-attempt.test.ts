import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { KohakuClient } from "@/lib/kohaku-client";
import {
  inputAttemptFingerprint,
  matchingAttemptProof,
  parseAttemptCapability,
  validAttemptTurnId,
} from "@/lib/input-attempt";
import type { InputAttemptCapability, InputAttemptProof } from "@/lib/types";
import vectors from "./fixtures/input-attempt-vectors.json";

class Socket {
  static all: Socket[] = [];
  readyState = 0;
  sent: Record<string, unknown>[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor() { Socket.all.push(this); }
  send(text: string) { this.sent.push(JSON.parse(text)); }
  open() { this.readyState = 1; this.onopen?.(); }
  message(value: unknown) { this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent); }
  close() { this.readyState = 3; this.onclose?.(); }
}

const capability: InputAttemptCapability = {
  version: 1, source: { store_identity: "a".repeat(64), session_key: "websocket:attempt-chat",
    incarnation: "original-incarnation", generation: 0 },
  source_token: "b".repeat(32), principal: "c".repeat(64),
};
const binding = { queueKey: "attempt-chat", promptId: "row-one", turnId: "attempt-1" };
let clients: KohakuClient[];

function makeClient() {
  const client = new KohakuClient({ url: "ws://offline-test", maxBackoffMs: 1,
    socketFactory: () => new Socket() as unknown as WebSocket });
  clients.push(client);
  client.connect();
  const socket = Socket.all.at(-1)!;
  socket.open();
  client.attach("attempt-chat");
  return { client, socket };
}
function advertise(socket: Socket, cap = capability) {
  socket.message({ event: "attached", chat_id: "attempt-chat", input_attempt_source: cap });
}
async function flush() { for (let i = 0; i < 6; i += 1) await Promise.resolve(); }
function proofFor(socket: Socket, cap = capability): InputAttemptProof {
  const message = socket.sent.find((frame) => frame.type === "message")!;
  const attempt = message.input_attempt as { fingerprint: string };
  return { ...cap, turn_id: "attempt-1", fingerprint: attempt.fingerprint,
    event_id: "original-event", run_id: "original-run" };
}
function queuedSend(client: KohakuClient, text = "exact submitted text") {
  expect(client.prepareQueuedAttempt("attempt-chat", binding, true)).toBe(true);
  const outcome = client.waitForQueuedAttempt("attempt-chat", binding)!;
  client.sendMessage("attempt-chat", text, undefined, { turnId: binding.turnId });
  return outcome;
}
function digest(_algorithm: unknown, data: Uint8Array) {
  const bytes = createHash("sha256").update(data).digest();
  return Promise.resolve(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

beforeEach(() => {
  clients = [];
  Socket.all = [];
  vi.useFakeTimers();
  vi.stubGlobal("crypto", { subtle: { digest: vi.fn(digest) } });
});
afterEach(() => {
  clients.forEach((client) => client.close());
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("negotiated canonical input attempts", () => {
  it.each(vectors)("shares exact Python/WebCrypto fingerprint: $utf8_hex", async (vector) => {
    vi.stubGlobal("crypto", webcrypto);
    expect(await inputAttemptFingerprint(vector.text)).toBe(vector.fingerprint);
  });
  it.each(["\ud800", "\udc00", "x\ud800y", "x\udc00y"])("rejects an unpaired surrogate", async (text) => {
    await expect(inputAttemptFingerprint(text)).rejects.toThrow("Invalid input attempt text");
  });

  it.each(["\n", "\r", "\u2028", "\u2029"])("rejects final line terminators at the strict wire edge", (suffix) => {
    expect(validAttemptTurnId(`attempt-1${suffix}`)).toBe(false);
    expect(parseAttemptCapability({ ...capability, principal: capability.principal + suffix }, "attempt-chat")).toBeNull();
    expect(parseAttemptCapability({ ...capability, source_token: capability.source_token + suffix }, "attempt-chat")).toBeNull();
    expect(parseAttemptCapability({ ...capability,
      source: { ...capability.source, store_identity: capability.source.store_identity + suffix } }, "attempt-chat")).toBeNull();
    const frozen = { ...capability, turn_id: "attempt-1", fingerprint: "d".repeat(64) };
    const proof = { ...frozen, event_id: "event-1", run_id: "run-1" };
    expect(matchingAttemptProof({ ...proof, event_id: proof.event_id + suffix }, frozen, "attempt-chat")).toBe(false);
    expect(matchingAttemptProof({ ...proof, run_id: proof.run_id + suffix }, frozen, "attempt-chat")).toBe(false);
  });

  it.each([false, true])("reads actual frozen admission before/after completion (%s), without resend or run revival", async (complete) => {
    const { client, socket } = makeClient();
    advertise(socket);
    const outcome = queuedSend(client);
    await flush();
    const proof = proofFor(socket);
    socket.close();
    expect(client.hasUnknownMessageSend("attempt-chat")).toBe(true);
    if (complete) {
      expect(client.reconcileCanonicalCompletion("attempt-chat", client.getRunGeneration("attempt-chat"), ["attempt-1"], {
        activitySnapshotAvailable: true, activeTurnId: null, hasPendingToolCalls: false, observedTurnIds: [],
      })).toBe(true);
    }
    const runGeneration = client.getRunGeneration("attempt-chat");
    await vi.advanceTimersByTimeAsync(1);
    const reconnected = Socket.all.at(-1)!;
    reconnected.open();
    expect(reconnected.sent.filter((frame) => frame.type === "message")).toEqual([]);
    expect(reconnected.sent.filter((frame) => frame.type === "attach")).toEqual([{
      type: "attach", chat_id: "attempt-chat", input_attempt_query: {
        version: 1, turn_id: proof.turn_id, source: proof.source,
        source_token: proof.source_token, fingerprint: proof.fingerprint,
      },
    }]);
    reconnected.message({ event: "attached", chat_id: "attempt-chat", input_attempt_source: capability,
      input_attempt_result: { version: 1, turn_id: "attempt-1", disposition: "admitted", proof } });
    await expect(outcome).resolves.toEqual({ kind: "admitted" });
    expect(client.hasUnknownMessageSend("attempt-chat")).toBe(false);
    expect(client.getRunGeneration("attempt-chat")).toBe(runGeneration);
    expect(client.consumeQueuedAttempt("attempt-chat", { ...binding, promptId: "later-row" })).toBe(false);
    expect(client.consumeQueuedAttempt("attempt-chat", binding)).toBe(true);
  });

  it.each([false, true])("connects real authenticated backend attach wire to the actual pending consumer (%s)", async (complete) => {
    const path = process.env.KOHAKU_INPUT_ATTEMPT_BRIDGE;
    const wire = path ? JSON.parse(readFileSync(path, "utf8")) : null;
    const { client, socket } = makeClient();
    const cap: InputAttemptCapability = wire?.capability.input_attempt_source ?? capability;
    advertise(socket, cap);
    const outcome = queuedSend(client);
    await flush();
    const serverAttached = wire ? (complete ? wire.attached : wire.attached_running) : null;
    const proof = serverAttached?.input_attempt_result.proof ?? proofFor(socket, cap);
    expect(proof.fingerprint).toBe(proofFor(socket, cap).fingerprint);
    socket.close();
    await vi.advanceTimersByTimeAsync(1);
    const next = Socket.all.at(-1)!;
    next.open();
    if (complete) client.reconcileCanonicalCompletion("attempt-chat", client.getRunGeneration("attempt-chat"), ["attempt-1"], {
      activitySnapshotAvailable: true, activeTurnId: null, hasPendingToolCalls: false, observedTurnIds: [],
    });
    if (wire) expect(next.sent[0]).toEqual(wire.query);
    next.message(serverAttached ?? { event: "attached", chat_id: "attempt-chat",
      input_attempt_result: { version: 1, turn_id: "attempt-1", disposition: "admitted", proof } });
    await expect(outcome).resolves.toEqual({ kind: "admitted" });
    expect(client.consumeQueuedAttempt("attempt-chat", binding)).toBe(true);
  });

  it.each(["principal", "fingerprint", "source_token", "turn_id", "event_id", "run_id", "version", "origin", "extra"])("holds mismatched %s proof, live traffic and route refusal", async (field) => {
    const { client, socket } = makeClient();
    advertise(socket);
    const outcome = queuedSend(client);
    const settled = vi.fn();
    void outcome.then(settled);
    await flush();
    const proof: Record<string, unknown> = { ...proofFor(socket) };
    if (field === "origin") proof.source = { ...capability.source, generation: 1 };
    else if (field === "extra") proof.extra = true;
    else proof[field] = field === "version" ? 2 : ["event_id", "run_id"].includes(field) ? "" : "wrong";
    for (const frame of [
      { event: "message_accepted", active_turn_id: "attempt-1", starts_turn: true },
      { event: "user_message", content: "echo", active_turn_id: "attempt-1" },
      { event: "goal_status", status: "running", started_at: 123 },
      { event: "message_admitted", input_attempt_proof: proof },
      { event: "error", detail: "not_admitted", reason: "busy_new_request" },
    ]) socket.message({ chat_id: "attempt-chat", turn_id: "attempt-1", ...frame });
    await flush();
    expect(settled).not.toHaveBeenCalled();
    socket.close();
    expect(client.hasUnknownMessageSend("attempt-chat")).toBe(true);
  });

  it("rejects UNKNOWN and stale sockets/results without upgrading a legacy UNKNOWN", async () => {
    const { client, socket } = makeClient();
    const outcome = queuedSend(client); // No capability at invocation.
    const settled = vi.fn();
    void outcome.then(settled);
    socket.close();
    await vi.advanceTimersByTimeAsync(1);
    const next = Socket.all.at(-1)!;
    next.open();
    expect(next.sent[0]).toEqual({ type: "attach", chat_id: "attempt-chat" });
    advertise(next);
    next.message({ event: "attached", chat_id: "attempt-chat",
      input_attempt_result: { version: 1, turn_id: "attempt-1", disposition: "unknown" } });
    await flush();
    expect(settled).not.toHaveBeenCalled();
    expect(client.hasUnknownMessageSend("attempt-chat")).toBe(true);
  });

  it("never acquires a later capability for a previously queued unsent frame", async () => {
    const { client, socket } = makeClient();
    socket.close();
    const outcome = queuedSend(client);
    void outcome;
    await vi.advanceTimersByTimeAsync(1);
    const next = Socket.all.at(-1)!;
    next.open();
    advertise(next);
    const message = next.sent.find((frame) => frame.type === "message")!;
    expect(message.input_attempt).toBeUndefined();
  });

  it.each([false, true])("fences delayed preparation after disposal/replacement (%s)", async (replace) => {
    let release!: (value: ArrayBuffer) => void;
    vi.stubGlobal("crypto", { subtle: { digest: () => new Promise<ArrayBuffer>((resolve) => { release = resolve; }) } });
    const { client, socket } = makeClient();
    advertise(socket);
    queuedSend(client);
    if (replace) {
      // A later same-chat owner is distinct and must retain its own frozen state.
      client.sendMessage("attempt-chat", "later text", undefined, { turnId: "later-attempt" });
    } else client.close();
    const oldRelease = release;
    if (replace) {
      // Both owners are still valid, but close explicitly disposes all preparation.
      client.close();
    }
    oldRelease(new ArrayBuffer(32));
    await flush();
    expect(socket.sent.filter((frame) => frame.type === "message")).toEqual([]);
  });

  it("preserves the queued row on crypto failure, without an unstamped send", async () => {
    vi.stubGlobal("crypto", { subtle: { digest: () => Promise.reject(new Error("denied")) } });
    const { client, socket } = makeClient();
    advertise(socket);
    const outcome = queuedSend(client);
    await flush();
    await expect(outcome).resolves.toEqual({ kind: "not_admitted", reason: "input_attempt_preparation_failed" });
    expect(socket.sent.filter((frame) => frame.type === "message")).toEqual([]);
    expect(client.waitForQueuedAttempt("attempt-chat", binding)).not.toBeNull();
    expect(client.consumeQueuedAttempt("attempt-chat", binding)).toBe(true);
  });

  it("keeps the original capability while crypto crosses reconnect", async () => {
    let release!: (value: ArrayBuffer) => void;
    vi.stubGlobal("crypto", { subtle: { digest: () => new Promise<ArrayBuffer>((resolve) => { release = resolve; }) } });
    const { client, socket } = makeClient();
    advertise(socket);
    queuedSend(client);
    socket.close();
    await vi.advanceTimersByTimeAsync(1);
    const next = Socket.all.at(-1)!;
    next.open();
    advertise(next, { ...capability, source_token: "d".repeat(32) });
    release(new ArrayBuffer(32));
    await flush();
    const messages = next.sent.filter((frame) => frame.type === "message");
    expect(messages).toHaveLength(1);
    expect((messages[0].input_attempt as Record<string, unknown>).source_token).toBe(capability.source_token);
  });
});

it("accepts only a qualified live receipt and preserves a later queued owner", async () => {
  const { client, socket } = makeClient();
  advertise(socket);
  const first = queuedSend(client);
  await flush();
  const proof = proofFor(socket);
  const later = { ...binding, promptId: "later-row", turnId: "later-attempt" };
  expect(client.prepareQueuedAttempt("attempt-chat", later, true)).toBe(true);
  const laterSettled = vi.fn();
  void client.waitForQueuedAttempt("attempt-chat", later)!.then(laterSettled);
  socket.message({ event: "message_admitted", chat_id: "attempt-chat", turn_id: "attempt-1",
    input_attempt_proof: proof, starts_turn: true, active_turn_id: "attempt-1" });
  await expect(first).resolves.toEqual({ kind: "admitted" });
  await flush();
  expect(laterSettled).not.toHaveBeenCalled();
  expect(client.consumeQueuedAttempt("attempt-chat", later)).toBe(false);
  expect(client.consumeQueuedAttempt("attempt-chat", binding)).toBe(true);
});

it("fences the original pending object during asynchronous crypto preparation", async () => {
  let release!: (value: ArrayBuffer) => void;
  vi.stubGlobal("crypto", { subtle: { digest: () => new Promise<ArrayBuffer>((resolve) => { release = resolve; }) } });
  const { client, socket } = makeClient();
  advertise(socket);
  queuedSend(client);
  // Deliberately replace this private owner seam to test the identity fence,
  // even when the replacement reuses every visible key and row field.
  const owners = Reflect.get(client, "pendingMessageSends") as Map<string, Record<string, unknown>>;
  const key = "attempt-chat\0attempt-1";
  const replacement = { ...owners.get(key)!, recovery: undefined };
  owners.set(key, replacement);
  release(new ArrayBuffer(32));
  await flush();
  expect(socket.sent.filter((frame) => frame.type === "message")).toEqual([]);
  expect(owners.get(key)).toBe(replacement);
  expect(replacement.recovery).toBeUndefined();
});

it("rejects old-socket proof and an attach result for a replaced exact-key owner", async () => {
  const { client, socket } = makeClient();
  advertise(socket);
  const outcome = queuedSend(client);
  const settled = vi.fn();
  void outcome.then(settled);
  await flush();
  const proof = proofFor(socket);
  socket.close();
  await vi.advanceTimersByTimeAsync(1);
  const next = Socket.all.at(-1)!;
  next.open();
  socket.message({ event: "message_admitted", chat_id: "attempt-chat", turn_id: "attempt-1", input_attempt_proof: proof });
  await flush();
  expect(settled).not.toHaveBeenCalled();
  const owners = Reflect.get(client, "pendingMessageSends") as Map<string, Record<string, unknown>>;
  const key = "attempt-chat\0attempt-1";
  const replacement = { ...owners.get(key)! };
  owners.set(key, replacement);
  next.message({ event: "attached", chat_id: "attempt-chat",
    input_attempt_result: { version: 1, turn_id: "attempt-1", disposition: "admitted", proof } });
  await flush();
  expect(settled).not.toHaveBeenCalled();
  expect(client.hasUnknownMessageSend("attempt-chat")).toBe(true);
  expect(owners.get(key)).toBe(replacement);
});

it.each(["unknown", "unsupported", "extra"])("keeps %s attach readback non-authoritative", async (kind) => {
  const { client, socket } = makeClient();
  advertise(socket);
  const outcome = queuedSend(client);
  const settled = vi.fn();
  void outcome.then(settled);
  await flush();
  const proof = proofFor(socket);
  socket.close();
  await vi.advanceTimersByTimeAsync(1);
  const next = Socket.all.at(-1)!;
  next.open();
  const result = kind === "unknown"
    ? { version: 1, turn_id: "attempt-1", disposition: "unknown" }
    : { version: kind === "unsupported" ? 2 : 1, turn_id: "attempt-1", disposition: "admitted", proof,
      ...(kind === "extra" ? { extra: true } : {}) };
  next.message({ event: "attached", chat_id: "attempt-chat", input_attempt_result: result });
  await flush();
  expect(settled).not.toHaveBeenCalled();
  expect(client.hasUnknownMessageSend("attempt-chat")).toBe(true);
  expect(next.sent.filter((frame) => frame.type === "message")).toEqual([]);
});
