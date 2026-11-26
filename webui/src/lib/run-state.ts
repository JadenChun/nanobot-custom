import type { RunLifecycleState, RunState } from "@/lib/types";

const RUN_STATES: ReadonlySet<RunLifecycleState> = new Set([
  "running",
  "waiting_approval",
  "stopping",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

/** Validate a live or hydrated run state at the WebUI boundary. */
export function parseRunState(value: unknown): RunState | null | undefined {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;
  const state = value.state;
  const startedAt = value.started_at;
  if (
    typeof state !== "string"
    || !RUN_STATES.has(state as RunLifecycleState)
    || !nullableString(value.turn_id)
    || !nullableString(value.run_id)
    || !nullableString(value.source_incarnation)
    || !(startedAt === null || (typeof startedAt === "number" && Number.isFinite(startedAt)))
    || typeof value.can_stop !== "boolean"
    || typeof value.can_steer !== "boolean"
  ) return undefined;
  return {
    state: state as RunLifecycleState,
    turn_id: value.turn_id,
    run_id: value.run_id,
    source_incarnation: value.source_incarnation,
    started_at: startedAt,
    can_stop: value.can_stop,
    can_steer: value.can_steer,
  };
}
