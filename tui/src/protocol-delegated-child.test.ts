import { afterEach, describe, expect, test } from "bun:test"

import {
  NanobotClient,
  fetchDelegatedChildHistory,
  fetchDelegatedChildren,
  fetchHistory,
  type InboundEvent,
  type MessageOptions,
} from "./protocol"
import type {
  DelegatedChildActivity,
  DelegatedChildSummary,
  StopExpectation,
} from "./delegated-children"

class FakeSocket {
  static instances: FakeSocket[] = []
  static readonly OPEN = 1
  readyState = 0
  sent: string[] = []
  private readonly listeners = new Map<string, Array<(event: { data?: string }) => void>>()

  constructor() { FakeSocket.instances.push(this) }

  addEventListener(name: string, listener: (event: { data?: string }) => void): void {
    const listeners = this.listeners.get(name) || []
    listeners.push(listener)
    this.listeners.set(name, listeners)
  }

  send(value: string): void { this.sent.push(value) }
  close(): void { this.readyState = 3 }
  emit(name: string, event: { data?: string } = {}): void {
    for (const listener of this.listeners.get(name) || []) listener(event)
  }
  open(): void { this.readyState = FakeSocket.OPEN; this.emit("open") }
  deliver(value: unknown): void { this.emit("message", { data: JSON.stringify(value) }) }
}

const originalSocket = globalThis.WebSocket
const originalFetch = globalThis.fetch

afterEach(() => {
  Object.defineProperty(globalThis, "WebSocket", { configurable: true, value: originalSocket })
  globalThis.fetch = originalFetch
  FakeSocket.instances = []
})

function child(overrides: Partial<DelegatedChildActivity> = {}): DelegatedChildActivity {
  return {
    parent_session_key: "websocket:chat",
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
    child_run_id: "child-run",
    parent_store_identity: "store",
    parent_incarnation: "parent-inc",
    parent_generation: 1,
    root_key: "websocket:chat",
    root_store_identity: "store",
    root_incarnation: "root-inc",
    root_generation: 1,
    child_store_identity: "store",
    child_incarnation: "child-inc",
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

function summary(overrides: Partial<DelegatedChildSummary> = {}): DelegatedChildSummary {
  return {
    history_kind: "summary",
    parent_session_key: "websocket:chat",
    child_session_key: "delegated:legacy-child",
    child_id: "legacy-child",
    profile: "delegated",
    state: "completed",
    task_label: null,
    task_label_truncated: false,
    result_summary: "An old source-less result.",
    result_summary_truncated: false,
    outcome: "completed",
    runtime_generation: null,
    parent_run_id: null,
    child_run_id: null,
    parent_store_identity: null,
    parent_incarnation: null,
    parent_generation: null,
    root_key: "websocket:chat",
    root_store_identity: null,
    root_incarnation: null,
    root_generation: null,
    child_store_identity: null,
    child_incarnation: null,
    child_generation: null,
    run_sequence: null,
    state_sequence: null,
    control_epoch: null,
    can_stop: false,
    ...overrides,
  }
}

function stopTarget(): StopExpectation {
  return {
    version: 1,
    source: {
      store_identity: "store",
      session_key: "websocket:chat",
      incarnation: "root-inc",
      generation: 1,
    },
    control_epoch: 0,
    active_root_present: true,
    active_turn_id: "root-turn",
    active_root_run_ids: ["root-run"],
    children: [],
    pending_work_digest: "a".repeat(64),
  }
}

async function connectClient(): Promise<{
  client: NanobotClient
  socket: FakeSocket
  events: InboundEvent[]
  statuses: string[]
}> {
  const events: InboundEvent[] = []
  const statuses: string[] = []
  Object.defineProperty(globalThis, "WebSocket", {
    configurable: true,
    value: class extends FakeSocket {},
  })
  const client = new NanobotClient({
    url: "ws://tui.test",
    chatId: "chat",
    reconnect: false,
    onEvent: (event) => events.push(event),
    onStatus: (_status, detail) => { if (detail) statuses.push(detail) },
  })
  client.connect()
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  socket.deliver({ event: "ready", chat_id: "", client_id: "tui" })
  socket.deliver({ event: "attached", chat_id: "chat", stop_target: null })
  return { client, socket, events, statuses }
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status })
}

describe("delegated child protocol", () => {
  test("routes validated child and Stop events while preserving unknown additive events", async () => {
    const { client, socket, events, statuses } = await connectClient()
    const row = child({ task_label: "😀".repeat(160) })
    socket.deliver({ event: "delegated_child", chat_id: "chat", ...row })
    socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: stopTarget() })
    socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: { version: 2 } })
    socket.deliver({ event: "delegated_child", chat_id: "chat", ...row, state_sequence: true })
    socket.deliver({ event: "future_activity", chat_id: "chat", payload: true })

    expect(events.map((event) => event.event)).toEqual([
      "ready", "attached", "delegated_child", "stop_target",
    ])
    expect(events[1]).toHaveProperty("stop_target", null)
    expect(events[2]).toMatchObject({ event: "delegated_child", child_id: "child-a" })
    expect(events[3]).toMatchObject({ event: "stop_target", stop_target: stopTarget() })
    expect(statuses).toContain("gateway sent an invalid event")
    client.close()
  })

  test("echoes a modern Stop snapshot and a typed exact approval in existing messages", async () => {
    const { client, socket } = await connectClient()
    const target = stopTarget()
    const approval = {
      approval_id: "approval-child-a",
      decision: "approve" as const,
      session_key: "delegated:child-a",
      session_incarnation: "child-inc",
      run_id: "child-run",
      generation: 1,
    }
    client.send("/stop", { stopExpectation: target })
    client.send("approval_decision", { approvalDecision: approval } satisfies MessageOptions)

    const stopFrame = JSON.parse(socket.sent.at(-2)!) as Record<string, unknown>
    const approvalFrame = JSON.parse(socket.sent.at(-1)!) as Record<string, unknown>
    expect(stopFrame.stop_expectation).toEqual(target)
    expect(stopFrame).not.toHaveProperty("target_turn_id")
    expect(approvalFrame.turn_id).toEqual(expect.any(String))
    expect(approvalFrame.turn_id).not.toBe(stopFrame.turn_id)
    expect(approvalFrame.approval_decision).toEqual(approval)
    expect(approvalFrame.request_intent).toBe("new")
    client.close()
  })

  test("hydrates every delegated-child page and retains explicit modern null", async () => {
    const requested: string[] = []
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = String(input)
      requested.push(url)
      if (url.includes("before=cursor-1")) {
        return jsonResponse({
          schemaVersion: 3,
          parentSessionKey: "websocket:chat",
          children: [summary()],
          page: { before_cursor: null, has_more_before: false, loaded_message_count: 1 },
          stop_target: null,
        })
      }
      return jsonResponse({
        schemaVersion: 3,
        parentSessionKey: "websocket:chat",
        children: [child()],
        page: { before_cursor: "cursor-1", has_more_before: true, loaded_message_count: 1 },
        stop_target: null,
      })
    }) as unknown as typeof fetch

    const result = await fetchDelegatedChildren("http://gateway", "token", "websocket:chat")
    expect(result?.children.map((item) => item.child_id)).toEqual(["child-a", "legacy-child"])
    expect(result?.children[1]).toMatchObject({ history_kind: "summary", can_stop: false })
    expect(result?.stopTargetSnapshotAvailable).toBe(true)
    expect(result?.stopTarget).toBeNull()
    expect(requested).toHaveLength(2)
    expect(requested[1]).toContain("before=cursor-1")
  })

  test("loads full read-only child tool history and authorized source-less summary detail", async () => {
    const requested: string[] = []
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = String(input)
      requested.push(url)
      const older = url.includes("before=history-cursor")
      return jsonResponse({
        schemaVersion: 3,
        read_only: true,
        history_kind: "canonical",
        parent_session_key: "websocket:chat",
        root_key: "websocket:chat",
        source: {
          store_identity: "store",
          session_key: "delegated:child-a",
          incarnation: "child-inc",
          generation: 1,
        },
        page: {
          before_cursor: older ? null : "history-cursor",
          has_more_before: !older,
          loaded_message_count: 2,
        },
        messages: older
          ? [{ id: "tool-1", role: "tool", kind: "trace", content: "", toolEvents: [
              { call_id: "call-1", phase: "end", name: "read_file", result: "full tool result" },
            ] }]
          : [{ id: "assistant-1", role: "assistant", content: "latest answer" }],
      })
    }) as unknown as typeof fetch
    const history = await fetchDelegatedChildHistory(
      "http://gateway", "token", "chat", child(),
    )
    expect(history.messages.map((message) => message.content)).toEqual(["", "latest answer"])
    expect(history.messages[0]?.toolEvents?.[0]?.result).toBe("full tool result")
    expect(requested).toHaveLength(2)

    globalThis.fetch = (async () => jsonResponse({
      schemaVersion: 3,
      read_only: true,
      history_kind: "summary",
      parent_session_key: "websocket:chat",
      root_key: "websocket:chat",
      source: null,
      page: { before_cursor: null, has_more_before: false, loaded_message_count: 1 },
      messages: [{ id: "summary-1", role: "assistant", content: "read-only summary" }],
    })) as unknown as typeof fetch
    const summaryHistory = await fetchDelegatedChildHistory(
      "http://gateway", "token", "chat", summary(),
    )
    expect(summaryHistory.historyKind).toBe("summary")
    expect(summaryHistory.readOnly).toBe(true)
    expect(summaryHistory.messages[0]?.content).toBe("read-only summary")
  })

  test("does not accept source-less summary detail for a canonical child reference", async () => {
    globalThis.fetch = (async () => jsonResponse({
      schemaVersion: 3,
      read_only: true,
      history_kind: "summary",
      parent_session_key: "websocket:chat",
      root_key: "websocket:chat",
      source: null,
      page: { before_cursor: null, has_more_before: false, loaded_message_count: 0 },
      messages: [],
    })) as unknown as typeof fetch
    await expect(fetchDelegatedChildHistory("http://gateway", "token", "chat", child()))
      .rejects.toThrow("source-less")
  })

  test("refuses a changed canonical source during child detail pagination", async () => {
    let page = 0
    globalThis.fetch = (async () => {
      page += 1
      return jsonResponse({
        schemaVersion: 3,
        read_only: true,
        history_kind: "canonical",
        parent_session_key: "websocket:chat",
        root_key: "websocket:chat",
        source: {
          store_identity: "store",
          session_key: "delegated:child-a",
          incarnation: page === 1 ? "child-inc" : "recreated-child",
          generation: 1,
        },
        page: {
          before_cursor: page === 1 ? "history-cursor" : null,
          has_more_before: page === 1,
          loaded_message_count: 1,
        },
        messages: [],
      })
    }) as unknown as typeof fetch
    await expect(fetchDelegatedChildHistory("http://gateway", "token", "chat", child()))
      .rejects.toThrow("source changed")
  })

  test("parent history retains compact child rows and current Stop field presence", async () => {
    globalThis.fetch = (async () => jsonResponse({
      schemaVersion: 3,
      stop_target: null,
      page: { before_cursor: null, has_more_before: false, loaded_message_count: 1 },
      messages: [{
        id: "child-a",
        role: "assistant",
        kind: "delegated_child",
        content: "",
        delegatedChild: { ...child(), turn_id: undefined },
      }],
    })) as unknown as typeof fetch
    const history = await fetchHistory("http://gateway", "token", "chat")
    expect(history.stopTargetSnapshotAvailable).toBe(true)
    expect(history.stopTarget).toBeNull()
    expect(history.messages[0]?.delegatedChild?.child_id).toBe("child-a")
    expect(history.messages[0]?.delegatedChild?.turn_id).toBeNull()
  })
})
