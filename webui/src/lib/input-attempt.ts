/** Positive-only v1 evidence for the surviving KohakuClient pending owner. */
import type {
  InputAttemptBinding,
  InputAttemptCapability,
  InputAttemptProof,
  InputAttemptResult,
  SourceOrigin,
} from "./types";

const TURN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const HEX = /^[0-9a-f]{64}$/;
const TOKEN = /^[0-9a-f]{32}$/;

function object(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).length === keys.length
    && keys.every((key) => Object.hasOwn(value, key));
}

function source(value: unknown, chatId: string): value is SourceOrigin {
  if (!object(value, ["store_identity", "session_key", "incarnation", "generation"])) return false;
  return typeof value.store_identity === "string" && HEX.test(value.store_identity)
    && value.session_key === `websocket:${chatId}`
    && typeof value.incarnation === "string" && value.incarnation.length >= 1
    && value.incarnation.length <= 128
    && Array.from(value.incarnation).every((char) => char.charCodeAt(0) >= 32 && char.charCodeAt(0) !== 127)
    && typeof value.generation === "number" && Number.isSafeInteger(value.generation)
    && value.generation >= 0;
}

export function validAttemptTurnId(value: string): boolean {
  return TURN.test(value);
}

export function parseAttemptCapability(value: unknown, chatId: string): InputAttemptCapability | null {
  if (!object(value, ["version", "source", "source_token", "principal"]) || value.version !== 1
    || !source(value.source, chatId) || typeof value.source_token !== "string"
    || !TOKEN.test(value.source_token) || typeof value.principal !== "string"
    || !HEX.test(value.principal)) return null;
  return Object.freeze({ version: 1, source: Object.freeze({ ...value.source }),
    source_token: value.source_token, principal: value.principal });
}

export async function inputAttemptFingerprint(text: string): Promise<string> {
  // TextEncoder replaces lone surrogates; replacement is NOT our wire contract.
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(i + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error("Invalid input attempt text");
      i += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new Error("Invalid input attempt text");
    }
  }
  const bytes = new TextEncoder().encode(`nanobot.input-attempt.v1\0new\0${text}`);
  const hash = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function sameSource(left: SourceOrigin, right: SourceOrigin): boolean {
  return left.store_identity === right.store_identity && left.session_key === right.session_key
    && left.incarnation === right.incarnation && left.generation === right.generation;
}

export function matchingAttemptProof(
  value: unknown, binding: InputAttemptBinding, chatId: string,
): value is InputAttemptProof {
  return object(value, ["version", "turn_id", "source", "source_token", "fingerprint", "principal", "event_id", "run_id"])
    && value.version === 1 && value.turn_id === binding.turn_id
    && source(value.source, chatId) && sameSource(value.source, binding.source)
    && value.source_token === binding.source_token && value.fingerprint === binding.fingerprint
    && value.principal === binding.principal
    && typeof value.event_id === "string" && TURN.test(value.event_id)
    && typeof value.run_id === "string" && TURN.test(value.run_id);
}

export function matchingAttemptResult(
  value: unknown, binding: InputAttemptBinding, chatId: string,
): value is Extract<InputAttemptResult, { disposition: "admitted" }> {
  return object(value, ["version", "turn_id", "disposition", "proof"])
    && value.version === 1 && value.turn_id === binding.turn_id && value.disposition === "admitted"
    && matchingAttemptProof(value.proof, binding, chatId);
}
