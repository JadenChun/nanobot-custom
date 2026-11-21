import { describe, expect, it } from "vitest";

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
