import { describe, expect, test } from "bun:test"

import {
  DelegatedChildProjection,
  parseDelegatedChildActivity,
  parseDelegatedChildSummary,
  parseStopExpectation,
  type DelegatedChildActivity,
} from "./delegated-children"

function child(overrides: Partial<DelegatedChildActivity> = {}): DelegatedChildActivity {
  return {
    parent_session_key: "websocket:root",
    child_session_key: "delegated:child-a",
    child_id: "child-a",
    profile: "delegated",
    state: "active",
    task_label: "Inspect the new route",
    task_label_truncated: false,
    result_summary: null,
    result_summary_truncated: false,
    outcome: null,
    runtime_generation: 0,
    turn_id: null,
    parent_run_id: "root-run",
    child_run_id: "child-run-1",
    parent_store_identity: "store",
    parent_incarnation: "parent-inc-1",
    parent_generation: 1,
    root_key: "websocket:root",
    root_store_identity: "store",
    root_incarnation: "root-inc-1",
    root_generation: 1,
    child_store_identity: "store",
    child_incarnation: "child-inc-1",
    child_generation: 1,
    run_sequence: 1,
    state_sequence: 1,
    control_epoch: 0,
    holds_capacity: true,
    can_stop: true,
    owned_work: true,
    pending_approval_id: null,
    blocked_run_id: null,
    result_disposition: null,
    ...overrides,
  }
}

function stopTarget(): unknown {
  return {
    version: 1,
    source: {
      store_identity: "store",
      session_key: "websocket:root",
      incarnation: "root-inc-1",
      generation: 1,
    },
    control_epoch: 0,
    active_root_present: false,
    active_turn_id: null,
    active_root_run_ids: [],
    children: [],
    pending_work_digest: "a".repeat(64),
  }
}

describe("delegated child wire projection", () => {
  test("keeps an explicit source-less summary beside canonical rows without control authority", () => {
    const summary = {
      history_kind: "summary",
      parent_session_key: "websocket:root",
      child_session_key: "delegated:legacy-child",
      child_id: "legacy-child",
      profile: "delegated",
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
      root_key: "websocket:root",
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
    }
    const projection = new DelegatedChildProjection()
    const startedAt = projection.revisionSnapshot()
    expect(projection.applyHydration([child(), summary], "websocket:root", null, startedAt)).toBe(true)
    expect(projection.rows.map((row) => row.child_id)).toEqual(["child-a"])
    expect(projection.summaryRows).toEqual([expect.objectContaining({
      history_kind: "summary",
      child_id: "legacy-child",
      task_label: null,
      child_store_identity: null,
      run_sequence: null,
      can_stop: false,
    })])
    expect(parseDelegatedChildActivity(summary)).toBeNull()
    const parsedSummary = parseDelegatedChildSummary(summary)
    expect(parsedSummary).toMatchObject({ history_kind: "summary" })
    expect(parsedSummary).not.toHaveProperty("turn_id")
    expect(parseDelegatedChildSummary({ ...summary, turn_id: null }))
      .toHaveProperty("turn_id", null)
  })

  test("validates bounded display fields by Unicode code points and accepts missing delivery turn", () => {
    const row = child({ task_label: "😀".repeat(160) })
    const wire = { ...row } as Record<string, unknown>
    delete wire.turn_id

    expect(parseDelegatedChildActivity(wire)).toEqual(row)
    expect(parseDelegatedChildActivity({ ...wire, task_label: "😀".repeat(161) })).toBeNull()
    expect(parseDelegatedChildActivity({ ...wire, result_summary: "x".repeat(1001) })).toBeNull()
    expect(parseDelegatedChildActivity({ ...wire, run_sequence: true })).toBeNull()
  })

  test("orders rows by run and state sequence and retains a conflicting equal version", () => {
    const projection = new DelegatedChildProjection()
    expect(projection.applyLive(child(), "websocket:root")).toBe("inserted")
    expect(projection.applyLive(child({ state_sequence: 0, state: "reserved" }), "websocket:root"))
      .toBe("stale")
    expect(projection.applyLive(child(), "websocket:root")).toBe("duplicate")
    expect(projection.applyLive(child({ state: "failed" }), "websocket:root")).toBe("conflict")
    expect(projection.get("child-a")?.state).toBe("active")
    expect(projection.refreshRequired).toBe(true)
    expect(projection.applyLive(child({ state_sequence: 2, state: "waiting_approval" }), "websocket:root"))
      .toBe("updated")
    expect(projection.get("child-a")?.state).toBe("waiting_approval")
  })

  test("requires authoritative hydration before replacing a child lifetime", () => {
    const projection = new DelegatedChildProjection()
    projection.applyLive(child(), "websocket:root")
    const revisionAtStart = projection.revisionSnapshot()
    const recreated = child({
      child_incarnation: "child-inc-2",
      child_generation: 2,
      child_run_id: "child-run-1",
      run_sequence: 0,
      state_sequence: 0,
    })
    expect(projection.applyLive(recreated, "websocket:root")).toBe("lifetime_mismatch")
    expect(projection.get("child-a")?.child_incarnation).toBe("child-inc-1")
    expect(projection.applyHydration([recreated], "websocket:root", null, revisionAtStart)).toBe(true)
    expect(projection.get("child-a")?.child_incarnation).toBe("child-inc-2")
  })

  test("keeps live rows that change during all-page hydration and accepts exact stop snapshots", () => {
    const projection = new DelegatedChildProjection()
    projection.applyLive(child(), "websocket:root")
    const revisionAtStart = projection.revisionSnapshot()
    projection.applyLive(child({ state_sequence: 2, state: "waiting_approval" }), "websocket:root")
    const recreated = child({
      child_incarnation: "child-inc-2",
      child_generation: 2,
      state_sequence: 0,
      run_sequence: 0,
    })
    expect(projection.applyHydration([recreated], "websocket:root", null, revisionAtStart)).toBe(true)
    expect(projection.get("child-a")?.state).toBe("waiting_approval")
    expect(projection.refreshRequired).toBe(true)
    expect(parseStopExpectation(stopTarget())).not.toBeNull()
    expect(parseStopExpectation({ ...stopTarget() as object, active_root_present: true })).toBeNull()
  })
})
