import { describe, expect, it } from "vitest";

import { hasPendingAgentActivity, normalizeActivityTimeline } from "@/lib/activity-timeline";
import {
  delegatedChildRows,
  mergeDelegatedChildRows,
  parseDelegatedChildPayload,
  parseDelegatedChildSummaryPayload,
  type DelegatedChildProjectionMode,
} from "@/lib/delegated-child-projection";
import type { DelegatedChildPayload, SourceOrigin, UIMessage } from "@/lib/types";

const root: SourceOrigin = {
  store_identity: "root-store",
  session_key: "websocket:parent",
  incarnation: "root-incarnation",
  generation: 4,
};

function child(overrides: Partial<DelegatedChildPayload> = {}): DelegatedChildPayload {
  return {
    parent_session_key: root.session_key,
    child_session_key: "websocket:child-a",
    child_id: "child-a",
    profile: "worker",
    task_label: "Inspect the API contract",
    task_label_truncated: false,
    state: "active",
    outcome: null,
    runtime_generation: 1,
    turn_id: null,
    parent_run_id: "root-run-1",
    child_run_id: "child-run-1",
    parent_store_identity: root.store_identity,
    parent_incarnation: root.incarnation,
    parent_generation: root.generation,
    root_key: root.session_key,
    root_store_identity: root.store_identity,
    root_incarnation: root.incarnation,
    root_generation: root.generation,
    child_store_identity: "child-store-a",
    child_incarnation: "child-incarnation-a",
    child_generation: 2,
    run_sequence: 1,
    state_sequence: 1,
    control_epoch: 0,
    holds_capacity: true,
    can_stop: true,
    owned_work: true,
    pending_approval_id: null,
    blocked_run_id: null,
    result_disposition: null,
    result_summary: null,
    result_summary_truncated: false,
    ...overrides,
  };
}

function row(payload: DelegatedChildPayload): UIMessage {
  return {
    id: `delegated-child-${payload.child_id}`,
    role: "assistant",
    kind: "delegated_child",
    content: "",
    createdAt: 10,
    delegatedChild: payload,
  };
}

function merge(
  current: UIMessage[],
  incoming: unknown[],
  mode: DelegatedChildProjectionMode = "live",
) {
  return mergeDelegatedChildRows(current, incoming, {
    expectedRoot: root,
    expectedRootKey: root.session_key,
    mode,
  });
}

describe("delegated child projection ordering", () => {
  it("retains an explicit source-less summary beside a canonical sibling without control authority", () => {
    const summary = {
      history_kind: "summary",
      parent_session_key: root.session_key,
      child_session_key: "delegated:legacy-child",
      child_id: "legacy-child",
      profile: "worker",
      state: "completed",
      task_label: null,
      task_label_truncated: false,
      result_summary: "A retained source-less result.",
      result_summary_truncated: false,
      outcome: "completed",
      runtime_generation: null,
      parent_run_id: null,
      child_run_id: null,
      parent_store_identity: null,
      parent_incarnation: null,
      parent_generation: null,
      root_key: root.session_key,
      root_store_identity: null,
      root_incarnation: null,
      root_generation: null,
      child_store_identity: null,
      child_incarnation: null,
      child_generation: null,
      run_sequence: null,
      state_sequence: null,
      control_epoch: null,
      holds_capacity: null,
      can_stop: false,
      owned_work: null,
      pending_approval_id: null,
      blocked_run_id: null,
      result_disposition: null,
    };
    const canonical = child();
    const result = merge([], [canonical, summary], "hydration");

    expect(result.refreshRequired).toBe(false);
    expect(result.messages).toHaveLength(2);
    const summaryRow = result.messages.find((message) => (
      (message.delegatedChild as unknown as Record<string, unknown> | undefined)?.history_kind
        === "summary"
    ));
    expect(summaryRow?.delegatedChild).toMatchObject({
      history_kind: "summary",
      task_label: null,
      child_store_identity: null,
      run_sequence: null,
      can_stop: false,
    });
    expect(summaryRow?.delegatedChild).not.toHaveProperty("turn_id");
    expect(parseDelegatedChildSummaryPayload(summary)).toMatchObject({ history_kind: "summary" });
    expect(parseDelegatedChildSummaryPayload(summary)).not.toHaveProperty("turn_id");
    expect(parseDelegatedChildSummaryPayload({ ...summary, turn_id: null }))
      .toHaveProperty("turn_id", null);
    expect(parseDelegatedChildPayload(summary)).toBeNull();
    expect(delegatedChildRows(result.messages).map((item) => item.child_id)).toEqual(["child-a"]);
  });

  it("updates an existing stable row for newer state and treats exact duplicates as no-ops", () => {
    const first = child();
    const newer = child({
      state: "waiting_approval",
      state_sequence: 2,
      pending_approval_id: "approval-1",
    });
    const initial = row(first);

    const duplicate = merge([initial], [first]);
    const original = [initial];
    expect(merge(original, [first]).messages).toBe(original);
    expect(duplicate.refreshRequired).toBe(false);

    const updated = merge([initial], [newer]);
    expect(updated.messages).toHaveLength(1);
    expect(updated.messages[0]?.id).toBe(initial.id);
    expect(updated.messages[0]?.delegatedChild).toEqual(newer);
    expect(updated.refreshRequired).toBe(false);
  });

  it("accepts hydration without the parent delivery turn id", () => {
    const payload = child();
    const hydratedPayload = { ...payload };
    delete hydratedPayload.turn_id;
    const result = merge([], [hydratedPayload], "hydration");

    expect(result.refreshRequired).toBe(false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]?.delegatedChild?.turn_id).toBeNull();
  });

  it("does not treat parent delivery turn metadata as an equal-version fact conflict", () => {
    const current = child({ turn_id: "live-delivery-turn" });
    const hydrated = child({ turn_id: null });
    const result = merge([row(current)], [hydrated], "hydration");

    expect(result.refreshRequired).toBe(false);
    expect(result.messages[0]?.delegatedChild).toEqual(current);
  });

  it("applies task and result limits by Unicode code point like the backend", () => {
    const valid = parseDelegatedChildPayload(child({
      task_label: "🚀".repeat(160),
      result_summary: "🧪".repeat(1000),
    }));
    expect(valid?.task_label).toBe("🚀".repeat(160));
    expect(valid?.result_summary).toBe("🧪".repeat(1000));

    expect(parseDelegatedChildPayload(child({ task_label: "🚀".repeat(161) }))).toBeNull();
    expect(parseDelegatedChildPayload(child({ result_summary: "🧪".repeat(1001) }))).toBeNull();
  });

  it("ignores lower run or state versions and requests refresh for equal-version conflicts", () => {
    const current = child({ state_sequence: 4 });
    const lowerState = child({ state: "active", state_sequence: 3 });
    const lowerRun = child({
      run_sequence: 0,
      state_sequence: 99,
      child_run_id: "old-run",
    });
    expect(merge([row(current)], [lowerState]).messages[0]?.delegatedChild).toEqual(current);
    expect(merge([row(current)], [lowerRun]).messages[0]?.delegatedChild).toEqual(current);

    const conflict = merge([row(current)], [child({
      outcome: "conflicting equal version",
      state_sequence: 4,
    })]);
    expect(conflict.messages[0]?.delegatedChild).toEqual(current);
    expect(conflict.refreshRequired).toBe(true);
  });

  it("rejects malformed or wrong-root dynamic events before they enter the timeline", () => {
    const malformed = child({ state_sequence: -1 });
    const wrongRoot = child({ root_generation: root.generation + 1 });
    const result = merge([], [malformed, wrongRoot]);

    expect(result.messages).toEqual([]);
    expect(result.refreshRequired).toBe(true);
  });

  it("refuses a changed child lifetime from a live event and accepts it from fresh hydration", () => {
    const first = child();
    const replacement = child({
      child_store_identity: "replacement-store",
      child_incarnation: "replacement-incarnation",
      child_generation: 0,
      run_sequence: 0,
      state_sequence: 0,
      child_run_id: null,
      state: "reserved",
    });
    const live = merge([row(first)], [replacement]);
    expect(live.messages[0]?.delegatedChild).toEqual(first);
    expect(live.refreshRequired).toBe(true);

    const hydrated = merge([row(first)], [replacement], "hydration");
    expect(hydrated.messages).toHaveLength(1);
    expect(hydrated.messages[0]?.id).toBe(row(first).id);
    expect(hydrated.messages[0]?.delegatedChild).toEqual(replacement);
    expect(hydrated.refreshRequired).toBe(false);
  });

  it("merges every sibling while preserving a live update that outran hydration", () => {
    const siblingA = child({ state_sequence: 5 });
    const siblingB = child({
      child_id: "child-b",
      child_session_key: "websocket:child-b",
      child_store_identity: "child-store-b",
      child_incarnation: "child-incarnation-b",
      task_label: "Check the docs",
      child_run_id: "child-run-b",
    });
    const stalePage = child({ state: "active", state_sequence: 4 });
    const result = merge([row(siblingA)], [stalePage, siblingB], "hydration");

    expect(result.messages.map((item) => item.delegatedChild?.child_id)).toEqual([
      "child-a",
      "child-b",
    ]);
    expect(result.messages[0]?.delegatedChild).toEqual(siblingA);
    expect(result.refreshRequired).toBe(false);
  });
});

describe("delegated child anchoring", () => {
  const user = (turnId: string): UIMessage => ({
    id: `user-${turnId}`, role: "user", content: "go", createdAt: 1, turnId,
  });
  const trace = (id: string, turnId: string, callIds: string[], turnSeq?: number): UIMessage => ({
    id, role: "tool", kind: "trace", content: `${id}()`, traces: [`${id}()`], createdAt: 2, turnId,
    ...(turnSeq === undefined ? {} : { turnSeq }),
    toolEvents: callIds.map((call_id) => ({ phase: "start", call_id, name: "explore" })),
  });
  const answer = (turnId: string, id = `answer-${turnId}`): UIMessage => ({
    id, role: "assistant", content: "done", createdAt: 3, turnId, turnPhase: "answer",
  });
  const transcript = [
    user("t1"),
    trace("spawn", "t1", ["call-spawn"]),
    trace("later", "t1", ["call-later"]),
    answer("t1"),
    user("t2"),
    trace("next", "t2", ["call-next"]),
    answer("t2"),
  ];
  const ids = (messages: UIMessage[]) => messages.map((message) => message.id);
  const childId = "delegated-child-child-a";

  it("inserts a child right after the row that holds its spawning call", () => {
    const result = merge(transcript, [child({ turn_id: "t1", spawn_call_id: "call-spawn" })]);

    expect(ids(result.messages)).toEqual([
      "user-t1", "spawn", childId, "later", "answer-t1", "user-t2", "next", "answer-t2",
    ]);
    expect(result.messages[2]?.turnId).toBe("t1");
  });

  it("anchors by spawning call without any turn id, as API hydration delivers it", () => {
    const result = merge(transcript, [child({ spawn_call_id: "call-spawn" })], "hydration");

    expect(ids(result.messages).indexOf(childId)).toBe(2);
    expect(result.messages[2]?.turnId).toBe("t1");
  });

  it("orders siblings from one tool batch by their calls, not by arrival", () => {
    const batch = [user("t1"), trace("batch", "t1", ["call-1", "call-2"]), answer("t1")];
    const second = child({
      child_id: "child-b",
      child_session_key: "websocket:child-b",
      child_store_identity: "child-store-b",
      spawn_call_id: "call-2",
    });
    const result = merge(batch, [second, child({ spawn_call_id: "call-1" })], "hydration");

    expect(ids(result.messages)).toEqual([
      "user-t1", "batch", childId, "delegated-child-child-b", "answer-t1",
    ]);
  });

  it.each([
    ["turn_id", { turn_id: "t1" }],
    ["parent_run_id", { turn_id: null, parent_run_id: "t1" }],
    ["an unmatched spawn call", { turn_id: "t1", spawn_call_id: "call-paged-out" }],
  ])("falls back to the end of the turn's activity using %s", (_name, facts) => {
    const result = merge(transcript, [child(facts)], "hydration");

    expect(ids(result.messages)).toEqual([
      "user-t1", "spawn", "later", childId, "answer-t1", "user-t2", "next", "answer-t2",
    ]);
  });

  it("keeps a fallback child ahead of an answer-only turn's answer", () => {
    const result = merge([user("t1"), answer("t1"), user("t2")], [child({ turn_id: "t1" })]);

    expect(ids(result.messages)).toEqual(["user-t1", childId, "answer-t1", "user-t2"]);
  });

  it("appends a legacy child only when nothing in the transcript matches", () => {
    const result = merge(transcript, [child()], "hydration");

    expect(ids(result.messages)).toEqual([...ids(transcript), childId]);
  });

  it("keeps the spawn anchor when a follow-up run reports from a later turn", () => {
    const spawned = merge(transcript, [child({ turn_id: "t1" })]).messages;
    const followedUp = merge(spawned, [child({
      turn_id: "t2", run_sequence: 2, child_run_id: "child-run-2",
    })]).messages;

    expect(ids(followedUp).indexOf(childId)).toBe(3);
    expect(followedUp[3]?.delegatedChild).toMatchObject({ turn_id: "t1", run_sequence: 2 });
  });

  it("keeps known anchor facts across partial payloads without requesting a refresh", () => {
    const live = merge(transcript, [child({ turn_id: "t1", spawn_call_id: "call-spawn" })]).messages;
    const newer = merge(live, [child({ state_sequence: 2 })], "hydration");

    expect(newer.messages[2]?.delegatedChild).toMatchObject({
      turn_id: "t1", spawn_call_id: "call-spawn", state_sequence: 2,
    });
    const same = merge(newer.messages, [child({ state_sequence: 2 })], "hydration");
    expect(same.refreshRequired).toBe(false);
    expect(same.messages).toBe(newer.messages);
  });

  it("learns a missing anchor from an equal-version payload", () => {
    const appended = merge(transcript, [child()], "hydration").messages;
    const result = merge(appended, [child({ spawn_call_id: "call-spawn" })], "hydration");

    expect(result.refreshRequired).toBe(false);
    expect(ids(result.messages).indexOf(childId)).toBe(2);
  });

  it("requests a refresh when two payloads name different spawning calls", () => {
    const live = merge(transcript, [child({ spawn_call_id: "call-spawn" })]).messages;
    const result = merge(live, [child({ spawn_call_id: "call-later" })]);

    expect(result.refreshRequired).toBe(true);
    expect(result.messages).toBe(live);
  });

  it("gives live updates and reload hydration the same order", () => {
    const payload = child({ turn_id: "t1", spawn_call_id: "call-spawn" });
    // Live: the child event follows the spawn row; later rows arrive afterwards.
    const early = merge(transcript.slice(0, 2), [payload]).messages;
    const live = merge(
      [...early, ...transcript.slice(2)],
      [child({ state: "completed", state_sequence: 2, spawn_call_id: "call-spawn" })],
    ).messages;
    // Reload: history arrives whole and the API payload has no turn id.
    const reloaded = merge(
      transcript,
      [child({ state: "completed", state_sequence: 2, spawn_call_id: "call-spawn" })],
      "hydration",
    ).messages;

    expect(ids(live)).toEqual(ids(reloaded));
  });

  it("parses the spawning call id and rejects a malformed one", () => {
    expect(parseDelegatedChildPayload(child({ spawn_call_id: "call-spawn" })))
      .toMatchObject({ spawn_call_id: "call-spawn" });
    expect(parseDelegatedChildPayload(child({ spawn_call_id: null })))
      .not.toHaveProperty("spawn_call_id");
    expect(parseDelegatedChildPayload({ ...child(), spawn_call_id: 7 })).toBeNull();
  });

  it("projects one activity group with the child as a step after its call", () => {
    const rows = [
      user("t1"),
      trace("spawn", "t1", ["call-spawn"], 2),
      trace("later", "t1", ["call-later"], 4),
      // Arrived before its spawn row was known, so state order is stale.
      row(child({ turn_id: "t1", spawn_call_id: "call-spawn" })),
      answer("t1"),
    ];
    rows[4] = { ...rows[4]!, turnSeq: 6 };
    const units = normalizeActivityTimeline(rows);

    expect(units.map((unit) => unit.type)).toEqual(["message", "activity", "message"]);
    const activity = units[1]!;
    expect(activity.type === "activity" && ids(activity.messages))
      .toEqual(["spawn", childId, "later"]);
    expect(units.reduce((total, unit) => total + unit.sourceMessageCount, 0)).toBe(rows.length);
  });

  it("ignores child rows when timing a turn and detecting unfinished work", () => {
    const childRow = { ...row(child({ turn_id: "t1" })), createdAt: 9_999_999 };
    const units = normalizeActivityTimeline([...transcript.slice(0, 4), childRow]);
    const activity = units.find((unit) => unit.type === "activity");

    expect(activity?.type === "activity" && activity.turnLatencyMs).toBe(2);
    expect(hasPendingAgentActivity([...transcript.slice(0, 4), childRow])).toBe(false);
    expect(hasPendingAgentActivity([...transcript.slice(0, 3), childRow])).toBe(true);
    expect(hasPendingAgentActivity([row(child())])).toBe(false);
  });

  it("keeps completion facts that turn_end stamped on a trailing child row", () => {
    const childRow = { ...row(child({ turn_id: "t1" })), latencyMs: 4_200 };
    const units = normalizeActivityTimeline([...transcript.slice(0, 3), childRow]);

    expect(units.at(-1)).toMatchObject({ type: "activity", turnLatencyMs: 4_200 });
  });

  it("does not move the turn duration onto an unanchored child appended after history", () => {
    const units = normalizeActivityTimeline([...transcript, row(child())]);
    const timed = units.filter((unit) => unit.type === "activity" && unit.showTurnDuration);

    expect(timed).toHaveLength(2);
    expect(units.at(-1)).toMatchObject({ type: "activity" });
    expect(units.at(-1)).not.toHaveProperty("showTurnDuration");
  });
});
