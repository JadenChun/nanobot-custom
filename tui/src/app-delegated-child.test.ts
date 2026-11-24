import { afterEach, expect, test } from "bun:test"
import { BoxRenderable, TextareaRenderable, TextRenderable } from "@opentui/core"
import {
  MockTreeSitterClient,
  createTestRenderer,
  type TestRendererSetup,
} from "@opentui/core/testing"

import { KohakuTui, type AppOptions } from "./app"
import { ComposerDraft } from "./composer-draft"
import type {
  DelegatedChildActivity,
  DelegatedChildReference,
  DelegatedChildSummary,
  StopExpectation,
} from "./delegated-children"
import {
  KohakuClient,
  fetchDelegatedChildren,
  fetchHistory,
  fetchSessionList,
  type ConnectionStatus,
  type ConnectionStatusInfo,
  type InboundEvent,
} from "./protocol"
import type { Transcript } from "./transcript"

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

  open(): void {
    this.readyState = FakeSocket.OPEN
    this.emit("open")
  }

  deliver(value: unknown): void {
    this.emit("message", { data: JSON.stringify(value) })
  }
}

interface AppState {
  connected: boolean
  ready: boolean
  activeTurn: boolean
  activeTurnId: string | null
  stopTarget: StopExpectation | null
  stopInFlight: boolean
  childRefreshPending: boolean
  delegatedChildren: {
    get(childId: string): DelegatedChildActivity | undefined
    getReference(childId: string): DelegatedChildReference | undefined
    rows: DelegatedChildActivity[]
    summaryRows: DelegatedChildSummary[]
    references: DelegatedChildReference[]
  }
  transcript: Transcript
  childDetailTranscript: Transcript | null
  composer: TextareaRenderable
  composerFrame: BoxRenderable
  draft: ComposerDraft
  presentedApprovals: Map<string, unknown>
  pendingApprovalTurns: Map<string, string>
  pendingApprovalIds: Set<string>
  refusedApprovalIds: Set<string>
  handleStatus(status: ConnectionStatus, detail?: string, info?: ConnectionStatusInfo): void
}

interface GatewayApp {
  app: KohakuTui
  state: AppState
  socket: FakeSocket
  setup: TestRendererSetup
}

const originalWebSocket = globalThis.WebSocket
const originalFetch = globalThis.fetch
let setup: TestRendererSetup | undefined

const appOptions: AppOptions = {
  wsUrl: "ws://nanobot.test/ws",
  apiUrl: "",
  apiToken: "",
  model: "test/model",
  modelPreset: "default",
  workspace: "/tmp/nanobot-workspace",
  version: "test",
  access: "workspace access",
  theme: "dark",
}

afterEach(() => {
  if (setup && !setup.renderer.isDestroyed) setup.renderer.destroy()
  setup = undefined
  globalThis.fetch = originalFetch
  Object.defineProperty(globalThis, "WebSocket", {
    configurable: true,
    value: originalWebSocket,
  })
  FakeSocket.instances = []
})

async function waitUntil(predicate: () => boolean, timeout = 1_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate() && Date.now() < deadline) await Bun.sleep(5)
  if (!predicate()) throw new Error(`condition was not met within ${timeout}ms`)
}

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value))
}

function state(app: KohakuTui): AppState {
  return app as unknown as AppState
}

function child(
  childId: string,
  overrides: Partial<DelegatedChildActivity> = {},
): DelegatedChildActivity {
  return {
    parent_session_key: "websocket:chat",
    child_session_key: `delegated:${childId}`,
    child_id: childId,
    profile: "delegated",
    state: "active",
    task_label: `Inspect ${childId}`,
    task_label_truncated: false,
    result_summary: null,
    result_summary_truncated: false,
    outcome: null,
    runtime_generation: 0,
    turn_id: null,
    parent_run_id: "root-run",
    child_run_id: `run-${childId}`,
    parent_store_identity: "store",
    parent_incarnation: "parent-inc",
    parent_generation: 1,
    root_key: "websocket:chat",
    root_store_identity: "store",
    root_incarnation: "root-inc",
    root_generation: 1,
    child_store_identity: "store",
    child_incarnation: `inc-${childId}`,
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

function summary(
  childId: string,
  overrides: Partial<DelegatedChildSummary> = {},
): DelegatedChildSummary {
  return {
    history_kind: "summary",
    parent_session_key: "websocket:chat",
    child_session_key: `delegated:${childId}`,
    child_id: childId,
    profile: "delegated",
    state: "completed",
    task_label: `Retained ${childId}`,
    task_label_truncated: false,
    result_summary: `Retained result for ${childId}.`,
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

function stopTarget(
  children: readonly DelegatedChildActivity[],
  activeRootPresent = false,
  controlEpoch = 1,
): StopExpectation {
  return {
    version: 1,
    source: {
      store_identity: "store",
      session_key: "websocket:chat",
      incarnation: "root-inc",
      generation: 1,
    },
    control_epoch: controlEpoch,
    active_root_present: activeRootPresent,
    active_turn_id: activeRootPresent ? "root-turn" : null,
    active_root_run_ids: activeRootPresent ? ["root-run"] : [],
    children: [...children]
      .sort((left, right) => left.child_id.localeCompare(right.child_id))
      .map((item) => ({
        child_id: item.child_id,
        source: {
          store_identity: item.child_store_identity,
          session_key: item.child_session_key,
          incarnation: item.child_incarnation,
          generation: item.child_generation,
        },
        child_run_id: item.child_run_id,
        run_sequence: item.run_sequence,
        control_epoch: item.control_epoch,
      })),
    pending_work_digest: String(controlEpoch).padStart(64, "0"),
  }
}

function delegatedChildrenSnapshot(
  children: DelegatedChildReference[],
  target: StopExpectation | null = null,
  page: { before_cursor: string | null; has_more_before: boolean } = {
    before_cursor: null,
    has_more_before: false,
  },
) {
  return {
    schemaVersion: 3,
    parentSessionKey: "websocket:chat",
    children,
    page: { ...page, loaded_message_count: children.length },
    stop_target: target,
  }
}

function childHistoryPage(
  item: DelegatedChildActivity,
  messages: unknown[],
  page: { before_cursor: string | null; has_more_before: boolean },
) {
  return {
    schemaVersion: 3,
    read_only: true,
    history_kind: "canonical",
    parent_session_key: item.parent_session_key,
    root_key: item.root_key,
    source: {
      store_identity: item.child_store_identity,
      session_key: item.child_session_key,
      incarnation: item.child_incarnation,
      generation: item.child_generation,
    },
    page: { ...page, loaded_message_count: messages.length },
    messages,
  }
}

function summaryHistoryPage(
  item: DelegatedChildSummary,
  messages: unknown[],
  page: { before_cursor: string | null; has_more_before: boolean },
) {
  return {
    schemaVersion: 3,
    read_only: true,
    history_kind: "summary",
    parent_session_key: item.parent_session_key,
    root_key: item.root_key,
    source: null,
    page: { ...page, loaded_message_count: messages.length },
    messages,
  }
}

async function mountGateway(
  target: StopExpectation | null | undefined,
  options: Partial<AppOptions> = {},
  fetcher: (url: string) => Response | Promise<Response> = () => jsonResponse(
    delegatedChildrenSnapshot([]),
  ),
): Promise<GatewayApp> {
  setup = await createTestRenderer({
    width: 100,
    height: 24,
    screenMode: "alternate-screen",
    consoleMode: "disabled",
    exitOnCtrlC: false,
  })
  globalThis.fetch = (async (input: RequestInfo | URL) => fetcher(String(input))) as unknown as typeof fetch
  Object.defineProperty(globalThis, "WebSocket", {
    configurable: true,
    value: class extends FakeSocket {},
  })

  let app: KohakuTui
  const client = new KohakuClient({
    url: appOptions.wsUrl,
    chatId: "chat",
    reconnect: false,
    onEvent: (event) => app.accept(event),
    onStatus: (status, detail, info) => state(app).handleStatus(status, detail, info),
  })
  app = KohakuTui.mount(
    setup.renderer,
    { ...appOptions, ...options },
    client,
    new MockTreeSitterClient({ autoResolveTimeout: 0 }),
  )
  client.connect()
  await waitUntil(() => FakeSocket.instances.length > 0)
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  socket.deliver({ event: "ready", chat_id: "", client_id: "tui-test" })
  socket.deliver({
    event: "attached",
    chat_id: "chat",
    ...(target !== undefined ? { stop_target: target } : {}),
  })
  await waitUntil(() => state(app).ready)
  await setup.renderOnce()
  return { app, state: state(app), socket, setup }
}

function messageFrames(socket: FakeSocket): Array<Record<string, unknown>> {
  return socket.sent
    .map((frame) => JSON.parse(frame) as Record<string, unknown>)
    .filter((frame) => frame.type === "message")
}

test("FakeSocket child events hydrate all pages without replacing newer live rows", async () => {
  let hydrationCalls = 0
  let releaseFirst: ((response: Response) => void) | undefined
  const firstPage = new Promise<Response>((resolve) => { releaseFirst = resolve })
  const root = child("child-a", { task_label: "Read the route", state_sequence: 1 })
  const sibling = child("child-b", { task_label: "Check the tests", state_sequence: 1 })
  const updatedSibling = child("child-b", {
    task_label: "Check the tests",
    state: "completed",
    state_sequence: 2,
    holds_capacity: false,
    can_stop: false,
    owned_work: false,
    result_summary: "Four focused tests passed",
    outcome: "completed",
  })
  const nested = child("child-c", {
    parent_session_key: "delegated:child-a",
    parent_incarnation: "inc-child-a",
    task_label: "Inspect the nested provider",
    state: "reserved",
    state_sequence: 1,
  })
  const gateway = await mountGateway(null, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    if (hydrationCalls === 1) return firstPage
    if (hydrationCalls === 2) {
      return jsonResponse(delegatedChildrenSnapshot(
        [updatedSibling, nested],
        null,
      ))
    }
    return jsonResponse(delegatedChildrenSnapshot([root, updatedSibling, nested], null))
  })
  await waitUntil(() => hydrationCalls === 1)

  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...root })
  gateway.socket.deliver({
    event: "delegated_child",
    chat_id: "chat",
    ...child("child-a", {
      task_label: "Read the route",
      state: "active",
      state_sequence: 2,
    }),
  })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...sibling })

  releaseFirst!(jsonResponse(delegatedChildrenSnapshot(
    [root],
    null,
    { before_cursor: "older-children", has_more_before: true },
  )))
  await waitUntil(() => (
    gateway.state.delegatedChildren.get("child-c") !== undefined
    && gateway.state.delegatedChildren.get("child-b")?.state === "completed"
  ))

  gateway.socket.deliver({
    event: "delegated_child",
    chat_id: "chat",
    ...child("child-a", {
      task_label: "Read the route",
      state: "reserved",
      state_sequence: 1,
    }),
  })
  gateway.socket.deliver({
    event: "delegated_child",
    chat_id: "chat",
    ...child("child-a", {
      task_label: "Read the route",
      state: "failed",
      state_sequence: 2,
    }),
  })
  await waitUntil(() => hydrationCalls >= 3)
  await waitUntil(() => gateway.state.delegatedChildren.get("child-a")?.state === "active")

  expect(gateway.state.delegatedChildren.rows.map((row) => row.child_id).sort())
    .toEqual(["child-a", "child-b", "child-c"])
  expect(gateway.state.delegatedChildren.get("child-a")?.state).toBe("active")
  expect(gateway.state.delegatedChildren.get("child-b")?.result_summary)
    .toBe("Four focused tests passed")
  expect(gateway.state.delegatedChildren.get("child-c")?.parent_session_key)
    .toBe("delegated:child-a")
  await gateway.setup.renderOnce()
  const frame = gateway.setup.captureCharFrame()
  expect(frame).toContain("Read the route")
  expect(frame).toContain("Check the tests")
  expect(frame).toContain("Four focused tests passed")
  expect(frame).toContain("Inspect the nested provider")
})

test("source-less summaries hydrate separately and open only as read-only summary history", async () => {
  const canonical = child("current-child")
  const retained = summary("legacy-child")
  const requested: string[] = []
  const gateway = await mountGateway(null, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    requested.push(url)
    if (url.includes("/delegated-children?")) {
      return jsonResponse(delegatedChildrenSnapshot([canonical, retained], null))
    }
    return jsonResponse(summaryHistoryPage(retained, [
      { id: "legacy-summary", role: "assistant", content: "A retained source-less result." },
    ], { before_cursor: null, has_more_before: false }))
  })

  await waitUntil(() => gateway.state.delegatedChildren.getReference("legacy-child") !== undefined)
  expect(gateway.state.delegatedChildren.rows.map((row) => row.child_id)).toEqual(["current-child"])
  expect(gateway.state.delegatedChildren.summaryRows).toEqual([expect.objectContaining({
    history_kind: "summary",
    child_id: "legacy-child",
    child_store_identity: null,
    can_stop: false,
  })])
  expect(gateway.state.delegatedChildren.get("legacy-child")).toBeUndefined()
  await gateway.setup.renderOnce()
  expect(gateway.setup.captureCharFrame()).toContain("Retained result for legacy-child.")

  gateway.state.composer.setText("/child open legacy-child")
  gateway.state.composer.submit()
  await waitUntil(() => gateway.state.childDetailTranscript !== null)
  await gateway.setup.renderOnce()

  expect(requested.filter((url) => url.includes("/webui-thread?"))).toHaveLength(1)
  expect(gateway.setup.captureCharFrame()).toContain("Read-only summary · inspect-only history")
  const detailMarkdown = (gateway.state.childDetailTranscript as unknown as {
    markdown: Set<{ content: string }>
  }).markdown
  expect([...detailMarkdown].map((renderable) => renderable.content))
    .toContain("A retained source-less result.")
  await waitUntil(() => gateway.setup.captureCharFrame().includes("A retained source-less result."))
  expect(messageFrames(gateway.socket)).toHaveLength(0)

  gateway.setup.mockInput.pressEscape()
  await gateway.setup.flush()
  await waitUntil(() => gateway.state.childDetailTranscript === null)
  await gateway.setup.renderOnce()
  expect(gateway.setup.captureCharFrame()).toContain("Retained result for legacy-child.")
  expect(messageFrames(gateway.socket)).toHaveLength(0)
})

test("an empty old hydration cannot replace a newer live root target", async () => {
  const oldTarget = stopTarget([], true, 1)
  const currentChild = child("child-b", {
    parent_incarnation: "root-inc-b",
    parent_generation: 2,
    root_incarnation: "root-inc-b",
    root_generation: 2,
  })
  const currentTarget: StopExpectation = {
    ...stopTarget([currentChild], false, 2),
    source: {
      ...oldTarget.source,
      incarnation: "root-inc-b",
      generation: 2,
    },
  }
  let hydrationCalls = 0
  let releaseOldHydration: ((response: Response) => void) | undefined
  const oldHydration = new Promise<Response>((resolve) => { releaseOldHydration = resolve })
  const freshHydration = new Promise<Response>(() => {})
  const gateway = await mountGateway(oldTarget, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    if (hydrationCalls === 1) return oldHydration
    return freshHydration
  })
  await waitUntil(() => hydrationCalls === 1)

  gateway.socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: currentTarget })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...currentChild })
  expect(gateway.state.stopTarget).toEqual(currentTarget)
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(currentChild)

  releaseOldHydration!(jsonResponse(delegatedChildrenSnapshot([], oldTarget)))
  // A queued replacement may begin after the old request finishes. Keep it
  // deferred so it cannot hide an intermediate stale target application.
  await waitUntil(() => hydrationCalls > 1 || !gateway.state.childRefreshPending)

  expect(gateway.state.stopTarget).toEqual(currentTarget)
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(currentChild)
})

test("a source-less summary from an old hydration cannot reappear beside a newer root", async () => {
  const oldTarget = stopTarget([], true, 1)
  const oldSummary = summary("retired-child")
  const currentChild = child("child-b", {
    parent_incarnation: "root-inc-b",
    parent_generation: 2,
    root_incarnation: "root-inc-b",
    root_generation: 2,
  })
  const currentTarget: StopExpectation = {
    ...stopTarget([currentChild], false, 2),
    source: {
      ...oldTarget.source,
      incarnation: "root-inc-b",
      generation: 2,
    },
  }
  let hydrationCalls = 0
  let releaseOldHydration: ((response: Response) => void) | undefined
  const oldHydration = new Promise<Response>((resolve) => { releaseOldHydration = resolve })
  const gateway = await mountGateway(oldTarget, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    if (hydrationCalls === 1) return oldHydration
    return new Promise<Response>(() => {})
  })
  await waitUntil(() => hydrationCalls === 1)

  gateway.socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: currentTarget })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...currentChild })
  releaseOldHydration!(jsonResponse(delegatedChildrenSnapshot([oldSummary], oldTarget)))
  await waitUntil(() => !gateway.state.childRefreshPending)

  expect(gateway.state.stopTarget).toEqual(currentTarget)
  expect(gateway.state.delegatedChildren.summaryRows).toEqual([])
  expect(gateway.state.delegatedChildren.getReference(oldSummary.child_id)).toBeUndefined()
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(currentChild)
})

test("an old hydration cannot revive a target after a newer explicit null", async () => {
  const oldTarget = stopTarget([], true, 1)
  const currentChild = child("child-b", {
    parent_incarnation: "root-inc-b",
    parent_generation: 2,
    root_incarnation: "root-inc-b",
    root_generation: 2,
  })
  let hydrationCalls = 0
  let releaseOldHydration: ((response: Response) => void) | undefined
  const oldHydration = new Promise<Response>((resolve) => { releaseOldHydration = resolve })
  const gateway = await mountGateway(oldTarget, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    if (hydrationCalls === 1) return oldHydration
    return new Promise<Response>(() => {})
  })
  await waitUntil(() => hydrationCalls === 1)

  gateway.socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: null })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...currentChild })
  releaseOldHydration!(jsonResponse(delegatedChildrenSnapshot([], oldTarget)))
  await waitUntil(() => hydrationCalls > 1 || !gateway.state.childRefreshPending)

  expect(gateway.state.stopTarget).toBeNull()
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(currentChild)
})

test("a delayed final canonical page cannot replace a newer live root", async () => {
  const oldTarget = stopTarget([], true, 1)
  const oldChildPage = child("child-a")
  const oldChildTail = child("child-old-tail")
  const currentChild = child("child-b", {
    parent_incarnation: "root-inc-b",
    parent_generation: 2,
    root_incarnation: "root-inc-b",
    root_generation: 2,
  })
  const currentTarget: StopExpectation = {
    ...stopTarget([currentChild], false, 2),
    source: {
      ...oldTarget.source,
      incarnation: "root-inc-b",
      generation: 2,
    },
  }
  let hydrationCalls = 0
  let releaseFirstPage: ((response: Response) => void) | undefined
  let releaseFinalPage: ((response: Response) => void) | undefined
  const firstPage = new Promise<Response>((resolve) => { releaseFirstPage = resolve })
  const finalPage = new Promise<Response>((resolve) => { releaseFinalPage = resolve })
  const gateway = await mountGateway(oldTarget, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    if (hydrationCalls === 1) return firstPage
    if (hydrationCalls === 2) return finalPage
    return jsonResponse(delegatedChildrenSnapshot([currentChild], currentTarget))
  })
  await waitUntil(() => hydrationCalls === 1)

  gateway.socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: currentTarget })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...currentChild })
  releaseFirstPage!(jsonResponse(delegatedChildrenSnapshot(
    [oldChildPage],
    oldTarget,
    { before_cursor: "older-old-root", has_more_before: true },
  )))
  await waitUntil(() => hydrationCalls === 2)
  releaseFinalPage!(jsonResponse(delegatedChildrenSnapshot([oldChildTail], oldTarget)))
  await waitUntil(() => hydrationCalls === 3 || !gateway.state.childRefreshPending)

  expect(gateway.state.stopTarget).toEqual(currentTarget)
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(currentChild)
  expect(gateway.state.delegatedChildren.get("child-a")).toBeUndefined()
  expect(gateway.state.delegatedChildren.get("child-old-tail")).toBeUndefined()
})

test("background Stop echoes the exact current snapshot and a stale refusal never retries", async () => {
  const initialChild = child("child-a")
  const initialTarget = stopTarget([initialChild], true)
  const currentTarget = stopTarget([initialChild], false, 2)
  const refreshedTarget = stopTarget([], true, 3)
  const refreshPaths: string[] = []
  let hydrationCalls = 0
  const gateway = await mountGateway(initialTarget, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    refreshPaths.push(url)
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    return jsonResponse(delegatedChildrenSnapshot(
      hydrationCalls === 1 ? [initialChild] : [],
      hydrationCalls === 1 ? initialTarget : refreshedTarget,
    ))
  })
  await waitUntil(() => hydrationCalls === 1)
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...initialChild })
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Start the parent task",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.socket.deliver({ event: "turn_end", chat_id: "chat", turn_id: "root-turn" })
  gateway.socket.deliver({ event: "stop_target", chat_id: "chat", stop_target: currentTarget })
  expect(gateway.state.activeTurn).toBe(false)
  expect(gateway.state.stopTarget).toEqual(currentTarget)

  gateway.state.composer.setText("keep this parent draft")
  gateway.setup.mockInput.pressCtrlC()
  await gateway.setup.flush()
  await waitUntil(() => messageFrames(gateway.socket).length === 1)
  expect(gateway.state.composer.plainText).toBe("keep this parent draft")
  const firstStop = messageFrames(gateway.socket)[0]!
  expect(firstStop.content).toBe("/stop")
  expect(firstStop.stop_expectation).toEqual(currentTarget)
  expect(firstStop).not.toHaveProperty("target_turn_id")
  const refusedTurn = firstStop.turn_id
  gateway.socket.deliver({
    event: "error",
    chat_id: "chat",
    turn_id: refusedTurn,
    detail: "not_admitted",
    reason: "stale_stop_target",
  })
  await waitUntil(() => hydrationCalls === 2)
  await waitUntil(() => !gateway.state.stopInFlight)
  await Bun.sleep(10)
  expect(refreshPaths).toHaveLength(2)
  expect(messageFrames(gateway.socket)).toHaveLength(1)
  expect(gateway.state.delegatedChildren.get("child-a")).toBeUndefined()
  expect(gateway.state.stopTarget).toEqual(refreshedTarget)

  gateway.state.composer.setText("/stop")
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 2)
  const secondStop = messageFrames(gateway.socket)[1]!
  expect(secondStop.stop_expectation).toEqual(refreshedTarget)
  expect(secondStop).not.toHaveProperty("target_turn_id")
})

test("modern explicit null never downgrades to root-only Stop while streaming", async () => {
  const gateway = await mountGateway(null)
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Root work",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.setup.mockInput.pressCtrlC()
  await gateway.setup.flush()
  await Bun.sleep(10)
  expect(messageFrames(gateway.socket)).toHaveLength(0)
  expect(gateway.state.activeTurn).toBe(true)
  expect(gateway.state.activeTurnId).toBe("root-turn")
})

test("a malformed modern HTTP target cannot fall back to root-only Stop", async () => {
  let hydrationCalls = 0
  const gateway = await mountGateway(undefined, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (!url.includes("/delegated-children?")) return jsonResponse({})
    hydrationCalls += 1
    return jsonResponse({
      schemaVersion: 3,
      parentSessionKey: "websocket:chat",
      children: [],
      page: { before_cursor: null, has_more_before: false, loaded_message_count: 0 },
      stop_target: { version: 1, source: "malformed" },
    })
  })
  await waitUntil(() => hydrationCalls === 1)
  await waitUntil(() => !gateway.state.childRefreshPending)
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Root work",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.setup.mockInput.pressCtrlC()
  await gateway.setup.flush()
  await Bun.sleep(10)

  expect(messageFrames(gateway.socket)).toHaveLength(0)
  expect(gateway.state.activeTurn).toBe(true)
  expect(gateway.state.activeTurnId).toBe("root-turn")
})

test("an attached Stop target for another chat cannot authorize a send", async () => {
  const target = stopTarget([], true)
  const gateway = await mountGateway({
    ...target,
    source: { ...target.source, session_key: "websocket:other" },
  })
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Root work in this chat",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.state.composer.setText("/stop")
  gateway.state.composer.submit()
  await Bun.sleep(20)

  expect(gateway.state.stopTarget).toBeNull()
  expect(gateway.state.activeTurn).toBe(true)
  expect(messageFrames(gateway.socket)).toHaveLength(0)
})

test("legacy Stop uses only a real live root turn ID and refuses child-owned fallback", async () => {
  const gateway = await mountGateway(undefined)
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Root work",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.setup.mockInput.pressCtrlC()
  await gateway.setup.flush()
  await waitUntil(() => messageFrames(gateway.socket).length === 1)
  expect(messageFrames(gateway.socket)[0]).toMatchObject({
    content: "/stop",
    target_turn_id: "root-turn",
  })
  expect(messageFrames(gateway.socket)[0]).not.toHaveProperty("stop_expectation")
})

test("legacy Stop refuses missing root identity and visible child-owned work", async () => {
  const gateway = await mountGateway(undefined)
  const row = child("child-a")
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...row })
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Root work without an ID",
    starts_turn: true,
  })
  gateway.setup.mockInput.pressCtrlC()
  await gateway.setup.flush()
  await Bun.sleep(10)
  expect(messageFrames(gateway.socket)).toHaveLength(0)
  expect(gateway.state.activeTurn).toBe(true)
})

test("sibling approvals stay individually typed and ordinary text carries no consent", async () => {
  const approvalA = "a".repeat(32)
  const approvalB = "b".repeat(32)
  const childA = child("child-a", {
    state: "waiting_approval",
    pending_approval_id: approvalA,
    blocked_run_id: "run-child-a",
  })
  const childB = child("child-b", {
    state: "waiting_approval",
    pending_approval_id: approvalB,
    blocked_run_id: "run-child-b",
  })
  const prompts = new Map([
    [childA.child_session_key, `I'm about to take a risky action and want your approval first. Planned action: Replace config.yml. Approval ID: ${approvalA}. Reply with the approval control to continue or deny.`],
    [childB.child_session_key, `I'm about to take a risky action and want your approval first. Planned action: Run the migration script. Approval ID: ${approvalB}. Reply with the approval control to continue or deny.`],
  ])
  const gateway = await mountGateway(null, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (url.includes("/delegated-children?")) {
      return jsonResponse(delegatedChildrenSnapshot([]))
    }
    const encodedKey = new URL(url).pathname.split("/")[3] || ""
    const key = decodeURIComponent(encodedKey)
    const item = key === childA.child_session_key ? childA : childB
    const prompt = prompts.get(item.child_session_key) || ""
    return jsonResponse(childHistoryPage(item, [
      { id: `approval-${item.child_id}`, role: "assistant", content: prompt },
    ], { before_cursor: null, has_more_before: false }))
  })
  gateway.socket.deliver({
    event: "user_message",
    chat_id: "chat",
    text: "Keep working on the parent task",
    turn_id: "root-turn",
    starts_turn: true,
  })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...childA })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...childB })
  await waitUntil(() => gateway.state.presentedApprovals.size === 2)

  gateway.state.composer.setText("/child approvals")
  gateway.state.composer.submit()
  await waitUntil(() => gateway.state.composer.plainText === "")
  await Bun.sleep(0)
  await gateway.setup.flush()
  await gateway.setup.renderOnce()
  const frame = gateway.setup.captureCharFrame()
  expect(frame).toContain("Replace config.yml")
  expect(frame).toContain("Run the migration script")
  expect(frame).toContain("delegated:child-a")
  expect(frame).toContain(approvalA)
  expect(frame).toContain(approvalB)

  gateway.state.composer.setText(`/child approve ${approvalA}`)
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 1)
  let sent = messageFrames(gateway.socket)
  expect(sent).toHaveLength(1)
  expect(sent[0]?.approval_decision).toEqual({
    approval_id: approvalA,
    decision: "approve",
    session_key: childA.child_session_key,
    session_incarnation: childA.child_incarnation,
    run_id: childA.blocked_run_id,
    generation: childA.child_generation,
  })
  expect(gateway.state.activeTurn).toBe(true)

  gateway.state.composer.setText(`/child approve ${approvalA}`)
  gateway.state.composer.submit()
  await Bun.sleep(10)
  expect(messageFrames(gateway.socket)).toHaveLength(1)

  gateway.state.composer.setText(`/child deny ${approvalB}`)
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 2)
  sent = messageFrames(gateway.socket)
  expect(sent).toHaveLength(2)
  expect(sent[1]?.approval_decision).toEqual({
    approval_id: approvalB,
    decision: "deny",
    session_key: childB.child_session_key,
    session_incarnation: childB.child_incarnation,
    run_id: childB.blocked_run_id,
    generation: childB.child_generation,
  })

  gateway.state.composer.setText("yes")
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 3)
  sent = messageFrames(gateway.socket)
  expect(sent).toHaveLength(3)
  expect(sent[2]?.content).toBe("yes")
  expect(sent[2]).not.toHaveProperty("approval_decision")
  expect(sent[2]).toMatchObject({ request_intent: "steer", target_turn_id: "root-turn" })
})

test("approval refusal, admission, disconnect, and refresh recover only the matching card", async () => {
  const approvalA = "c".repeat(32)
  const approvalB = "d".repeat(32)
  const childA = child("child-a", {
    state: "waiting_approval",
    pending_approval_id: approvalA,
    blocked_run_id: "run-child-a",
  })
  const childB = child("child-b", {
    state: "waiting_approval",
    pending_approval_id: approvalB,
    blocked_run_id: "run-child-b",
  })
  const prompts = new Map([
    [childA.child_session_key, `I'm about to take a risky action and want your approval first. Planned action: Replace the config. Approval ID: ${approvalA}. Reply with the approval control to continue or deny.`],
    [childB.child_session_key, `I'm about to take a risky action and want your approval first. Planned action: Run the migration. Approval ID: ${approvalB}. Reply with the approval control to continue or deny.`],
  ])
  let hydrationCalls = 0
  let releaseRefusalRefresh: ((response: Response) => void) | undefined
  const refusalRefresh = new Promise<Response>((resolve) => { releaseRefusalRefresh = resolve })
  const gateway = await mountGateway(null, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    if (url.includes("/delegated-children?")) {
      hydrationCalls += 1
      if (hydrationCalls === 1) {
        return jsonResponse(delegatedChildrenSnapshot([childA, childB], null))
      }
      if (hydrationCalls === 2) return refusalRefresh
      return jsonResponse(delegatedChildrenSnapshot([childA, childB], null))
    }
    if (!url.includes("/webui-thread?")) return jsonResponse({})
    const key = decodeURIComponent(new URL(url).pathname.split("/")[3] || "")
    const item = key === childA.child_session_key ? childA : childB
    return jsonResponse(childHistoryPage(item, [
      { id: `approval-${item.child_id}`, role: "assistant", content: prompts.get(key) || "" },
    ], { before_cursor: null, has_more_before: false }))
  })
  await waitUntil(() => hydrationCalls === 1)
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...childA })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...childB })
  await waitUntil(() => gateway.state.presentedApprovals.size === 2)

  gateway.state.composer.setText(`/child approve ${approvalA}`)
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 1)
  gateway.state.composer.setText(`/child approve ${approvalB}`)
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 2)
  const [firstDecision, siblingDecision] = messageFrames(gateway.socket)
  const refusedTurn = String(firstDecision?.turn_id)
  const siblingTurn = String(siblingDecision?.turn_id)
  expect(firstDecision?.approval_decision).toMatchObject({ approval_id: approvalA })
  expect(siblingDecision?.approval_decision).toMatchObject({ approval_id: approvalB })
  expect(refusedTurn).not.toBe(siblingTurn)

  gateway.socket.deliver({
    event: "error",
    chat_id: "chat",
    turn_id: refusedTurn,
    detail: "not_admitted",
    reason: "approval_refused",
  })
  await waitUntil(() => hydrationCalls === 2)
  expect(gateway.state.pendingApprovalTurns.has(approvalA)).toBe(false)
  expect(gateway.state.pendingApprovalIds.has(approvalA)).toBe(false)
  expect(gateway.state.refusedApprovalIds.has(approvalA)).toBe(true)
  expect(gateway.state.pendingApprovalTurns.get(approvalB)).toBe(siblingTurn)
  expect(gateway.state.pendingApprovalIds.has(approvalB)).toBe(true)

  gateway.socket.deliver({
    event: "message_accepted",
    chat_id: "chat",
    turn_id: refusedTurn,
    starts_turn: false,
  })
  expect(gateway.state.pendingApprovalTurns.get(approvalB)).toBe(siblingTurn)
  gateway.socket.deliver({
    event: "message_accepted",
    chat_id: "chat",
    turn_id: siblingTurn,
    starts_turn: false,
  })
  expect(gateway.state.pendingApprovalTurns.get(approvalB)).toBe(siblingTurn)
  expect(gateway.state.pendingApprovalIds.has(approvalB)).toBe(true)

  gateway.state.handleStatus("reconnecting")
  expect(gateway.state.pendingApprovalTurns.size).toBe(0)
  expect(gateway.state.pendingApprovalIds.size).toBe(0)
  expect(gateway.state.refusedApprovalIds).toEqual(new Set([approvalA, approvalB]))

  gateway.state.handleStatus("connected")
  gateway.socket.deliver({ event: "attached", chat_id: "chat", stop_target: null })
  await waitUntil(() => gateway.state.ready)
  releaseRefusalRefresh!(jsonResponse(delegatedChildrenSnapshot([], null)))
  await waitUntil(() => hydrationCalls === 3)
  await waitUntil(() => !gateway.state.childRefreshPending)
  expect(gateway.state.delegatedChildren.get("child-a")).toEqual(childA)
  expect(gateway.state.delegatedChildren.get("child-b")).toEqual(childB)
  expect(gateway.state.refusedApprovalIds.size).toBe(0)

  gateway.state.composer.setText(`/child approve ${approvalA}`)
  gateway.state.composer.submit()
  await waitUntil(() => messageFrames(gateway.socket).length === 3)
  const retry = messageFrames(gateway.socket)[2]!
  const retryTurn = String(retry.turn_id)
  expect(retry.approval_decision).toMatchObject({ approval_id: approvalA, decision: "approve" })
  expect(retryTurn).not.toBe(refusedTurn)
  expect(gateway.state.pendingApprovalTurns.get(approvalA)).toBe(retryTurn)
})

test("child detail is read-only and Escape restores parent draft, image, scroll, and focus", async () => {
  const item = child("child-a")
  const requested: string[] = []
  const gateway = await mountGateway(null, {
    apiUrl: "http://gateway.test",
    apiToken: "test-token",
  }, (url) => {
    requested.push(url)
    if (url.includes("/delegated-children?")) {
      return jsonResponse(delegatedChildrenSnapshot([]))
    }
    const older = url.includes("before=older-history")
    return jsonResponse(childHistoryPage(
      item,
      older
        ? [{
            id: "child-tool",
            role: "tool",
            kind: "trace",
            content: "",
            traces: ["Read the child workspace"],
            toolEvents: [{
              call_id: "read-child-file",
              phase: "end",
              name: "read_file",
              result: "full child tool result",
            }],
          }]
        : [{ id: "child-answer", role: "assistant", content: "Child latest answer" }],
      older
        ? { before_cursor: null, has_more_before: false }
        : { before_cursor: "older-history", has_more_before: true },
    ))
  })
  gateway.socket.deliver({ event: "delegated_child", chat_id: "chat", ...item })

  for (let index = 0; index < 24; index += 1) {
    gateway.state.transcript.assistant(`Parent transcript row ${index}`)
  }
  await gateway.setup.renderOnce()
  gateway.state.transcript.root.scrollTop = 5
  await gateway.setup.renderOnce()
  const parentScroll = gateway.state.transcript.root.scrollTop
  const image = gateway.state.draft.image({
    data_url: "data:image/png;base64,AA==",
    name: "parent.png",
  })!
  const parentDraft = `Keep this draft ${image.text.trim()}`
  gateway.state.composer.setText(parentDraft)
  gateway.state.composer.focus()

  const row = gateway.state.transcript.root.getChildren()
    .find((entry) => entry.id.includes("delegated-child-row"))
  expect(row).toBeDefined()
  await gateway.setup.mockMouse.click(row!.x + 1, row!.y)
  await waitUntil(() => gateway.state.childDetailTranscript !== null)
  await gateway.setup.renderOnce()

  expect(gateway.state.transcript.root.visible).toBe(false)
  expect(gateway.state.composerFrame.visible).toBe(false)
  expect(gateway.state.composer.plainText).toBe(parentDraft)
  expect(gateway.state.draft.imageCount).toBe(1)
  expect(gateway.state.childDetailTranscript?.root.getChildren().length).toBeGreaterThan(2)
  expect(requested.filter((url) => url.includes("/webui-thread?"))).toHaveLength(2)
  expect(gateway.setup.captureCharFrame()).toContain("Read-only canonical child history")
  expect(gateway.setup.captureCharFrame()).toContain("Child latest answer")
  expect(gateway.setup.captureCharFrame()).not.toContain("Parent transcript row 0")
  expect(messageFrames(gateway.socket)).toHaveLength(0)

  gateway.setup.mockInput.pressEscape()
  await gateway.setup.flush()
  await waitUntil(() => gateway.state.childDetailTranscript === null)
  await gateway.setup.renderOnce()
  expect(gateway.state.transcript.root.visible).toBe(true)
  expect(gateway.state.composerFrame.visible).toBe(true)
  expect(gateway.state.composer.plainText).toBe(parentDraft)
  expect(gateway.state.draft.imageCount).toBe(1)
  expect(gateway.state.transcript.root.scrollTop).toBe(parentScroll)
  expect(gateway.state.composer.focused).toBe(true)
  expect(gateway.setup.captureCharFrame()).toContain("Parent transcript row 0")
  expect(gateway.setup.captureCharFrame()).not.toContain("Child latest answer")
  expect(messageFrames(gateway.socket)).toHaveLength(0)
})

const claudeAvailable = { claude_sdk: { available: true } }
const claudeUnavailable = { claude_sdk: { available: false, reason: "sdk_missing" } }
const claudeBinding = {
  backend: "external",
  harness: "claude_sdk",
  external_session_id: null,
  locked_workspace: null,
  runtime_generation: 0,
} as const
const nativeBinding = { ...claudeBinding, backend: "native", harness: "native" } as const

function sentFrames(socket: FakeSocket) {
  return socket.sent.map((frame) => JSON.parse(frame) as Record<string, unknown>)
}

function submitText(gateway: GatewayApp, text: string): void {
  gateway.state.composer.setText(text)
  gateway.state.composer.submit()
}

async function rendered(gateway: GatewayApp): Promise<string> {
  await gateway.setup.renderOnce()
  return gateway.setup.captureCharFrame()
}

async function waitForText(gateway: GatewayApp, text: string): Promise<string> {
  const deadline = Date.now() + 1_000
  let frame = await rendered(gateway)
  while (!frame.includes(text) && Date.now() < deadline) {
    await Bun.sleep(5)
    frame = await rendered(gateway)
  }
  expect(frame).toContain(text)
  return frame
}

test("parses harness facts across socket, session list, and thread boundaries", async () => {
  const events: InboundEvent[] = []
  const diagnostics: string[] = []
  Object.defineProperty(globalThis, "WebSocket", { configurable: true, value: FakeSocket })
  const client = new KohakuClient({
    url: appOptions.wsUrl,
    reconnect: false,
    onEvent: (event) => events.push(event),
    onStatus: (_, detail) => { if (detail) diagnostics.push(detail) },
  })
  client.connect()
  await waitUntil(() => FakeSocket.instances.length === 1)
  const socket = FakeSocket.instances[0]!
  socket.open()
  const saved = {
    ...claudeBinding, external_session_id: "sdk-session", locked_workspace: "/repo", runtime_generation: 2,
  }
  const liveChild = child("wire", { harness: "native" })
  const ready = { event: "ready", chat_id: "", client_id: "test", harness_capabilities: claudeUnavailable } as const
  const attached = { event: "attached", chat_id: "chat", harness_binding: claudeBinding, harness_capabilities: claudeAvailable } as const
  const updated = { event: "session_updated", chat_id: "chat", harness_binding: saved } as const
  const legacy = { event: "attached", chat_id: "chat", model_preset: "old-preset" } as const
  try {
    for (const event of [ready, attached, updated, legacy]) socket.deliver(event)
    socket.deliver({ ...liveChild, event: "delegated_child", chat_id: "chat" })
    socket.deliver({ ...attached, harness_binding: { ...saved, runtime_generation: "2" } })

    expect(events.slice(0, 4)).toEqual([ready, attached, updated, legacy])
    expect(events[4]).toMatchObject({ event: "delegated_child", harness: "native" })
    expect(events).toHaveLength(5)
    expect(diagnostics).toContain("gateway sent an invalid event")
  } finally {
    client.close()
  }

  const retained = summary("retained", { harness: "claude_sdk" })
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.includes("/delegated-children")) return jsonResponse(delegatedChildrenSnapshot([retained]))
    if (url.includes("webui-thread")) return jsonResponse({
      ...(url.includes("legacy") ? {} : {
        harness: "claude_sdk", harness_binding: saved, harness_capabilities: claudeAvailable,
      }),
      messages: [{ role: "user", content: "Hello" }],
    })
    return jsonResponse({
      harness_capabilities: claudeAvailable,
      sessions: [
        { key: "websocket:claude", harness_binding: saved },
        { key: "websocket:native", harness_binding: nativeBinding },
        { key: "websocket:legacy", title: "Legacy chat" },
      ],
    })
  }) as typeof fetch

  const api = "http://nanobot.test"
  const list = await fetchSessionList(api, "test-token")
  expect(list.harnessCapabilities).toEqual(claudeAvailable)
  expect(list.sessions.map((row) => row.harnessBinding)).toEqual([saved, nativeBinding, undefined])
  expect(list.sessions[2]?.title).toBe("Legacy chat")
  expect(await fetchHistory(api, "test-token", "child")).toMatchObject({
    harness: "claude_sdk", harnessCapabilities: claudeAvailable, harnessBinding: saved,
  })
  const history = await fetchHistory(api, "test-token", "legacy")
  expect(history.harness).toBeUndefined()
  expect(history.harnessBinding).toBeUndefined()
  expect(history.messages[0]?.content).toBe("Hello")
  expect((await fetchDelegatedChildren(api, "test-token", "websocket:chat"))?.children[0]?.harness).toBe("claude_sdk")
})

test("offers a default harness and excludes Claude from unavailable and temporary chats", async () => {
  const gateway = await mountGateway(undefined)
  const { socket, setup } = gateway
  socket.deliver({ event: "attached", chat_id: "chat", harness_capabilities: claudeAvailable })
  await waitUntil(() => gateway.state.ready)

  submitText(gateway, "/new-chat")
  const chooser = await waitForText(gateway, "Harness · Kohaku")
  expect(chooser).toContain("› Harness · Kohaku")
  expect(chooser).toContain("Claude")
  setup.mockInput.pressEnter()
  await waitUntil(() => sentFrames(socket).some((frame) => frame.type === "new_chat"))
  expect(sentFrames(socket).at(-1)).not.toHaveProperty("harness")

  socket.deliver({ event: "attached", chat_id: "native" })
  await waitUntil(() => gateway.state.ready)
  submitText(gateway, "/new-chat")
  await waitForText(gateway, "Harness · Kohaku")
  setup.mockInput.pressArrow("down")
  setup.mockInput.pressEnter()
  await waitUntil(() => sentFrames(socket).some((frame) => frame.harness === "claude_sdk"))
  expect(sentFrames(socket).at(-1)).toMatchObject({ type: "new_chat", harness: "claude_sdk" })

  socket.deliver({ event: "error", detail: "harness_unavailable" })
  await waitForText(gateway, "Claude Code is unavailable on this gateway")
  socket.deliver({ event: "attached", chat_id: "native", harness_capabilities: claudeUnavailable })
  await waitUntil(() => gateway.state.ready)
  submitText(gateway, "/new-chat")
  await waitForText(gateway, "Claude Agent SDK is missing")
  const count = socket.sent.length
  setup.mockInput.pressArrow("down")
  setup.mockInput.pressEnter()
  await waitForText(gateway, "Unavailable · Claude Agent SDK is missing.")
  expect(socket.sent).toHaveLength(count)

  setup.mockInput.pressEscape()
  await waitForText(gateway, "Ask Kohaku anything")
  socket.deliver({ event: "attached", chat_id: "native", harness_capabilities: claudeAvailable })
  await waitUntil(() => gateway.state.ready)
  submitText(gateway, "/new-temporary-chat")
  const temporary = await waitForText(gateway, "Claude Code requires a saved chat")
  expect(temporary).toContain("› Harness · Kohaku")
  setup.mockInput.pressArrow("down")
  setup.mockInput.pressEnter()
  await waitUntil(() => sentFrames(socket).some((frame) => frame.type === "new_temporary_chat"))
  expect(sentFrames(socket).at(-1)).toEqual({ type: "new_temporary_chat" })
})

test("projects the Claude badge from saved chat and child harnesses", async () => {
  const sdkChild = child("sdk", { harness: "claude_sdk", task_label: "SDK work" })
  const nativeChild = child("native", { harness: "native", task_label: "Native work" })
  const page = { before_cursor: null, has_more_before: false }
  let retainedChildren: DelegatedChildActivity[] = []
  const gateway = await mountGateway(undefined, {
    apiUrl: "http://nanobot.test", apiToken: "test-token",
  }, (url) => {
    if (url.endsWith("/api/sessions")) return jsonResponse({
      harness_capabilities: claudeAvailable,
      sessions: [
        { key: "websocket:chat", title: "SDK session", model_preset: "Other model", harness_binding: claudeBinding },
        { key: "websocket:native", title: "Native session", model_preset: "Claude preset", harness_binding: nativeBinding },
      ],
    })
    if (url.includes("/delegated-children")) return jsonResponse(delegatedChildrenSnapshot(retainedChildren))
    if (url.includes("delegated%3Asdk")) return jsonResponse({
      ...childHistoryPage(sdkChild, [], page), harness: "claude_sdk",
    })
    if (url.includes("delegated%3Anative")) return jsonResponse({
      ...childHistoryPage(nativeChild, [], page), harness: "native",
    })
    if (url.includes("webui-thread")) {
      if (url.includes("websocket%3Anative")) {
        return jsonResponse({ messages: [], harness_binding: nativeBinding })
      }
      return jsonResponse({
        messages: [{ kind: "delegated_child", delegatedChild: { ...sdkChild, harness: undefined } }],
        harness_binding: claudeBinding,
      })
    }
    return jsonResponse({})
  })
  const { socket, setup } = gateway
  await waitUntil(() => !gateway.state.childRefreshPending)
  socket.deliver({ event: "delegated_child", chat_id: "chat", ...sdkChild })
  socket.deliver({ event: "delegated_child", chat_id: "chat", ...nativeChild })
  const live = await waitForText(gateway, "[Claude] SDK work")
  expect(live).toContain("Native work")
  expect(live).not.toContain("[Claude] Native work")

  retainedChildren = [sdkChild, nativeChild]
  socket.deliver({ event: "attached", chat_id: "chat", model_preset: "Other model" })
  const history = await waitForText(gateway, "[Claude] SDK work")
  expect(history).toContain("[Claude]  Other model")
  submitText(gateway, "/sessions")
  const list = await waitForText(gateway, "[Claude] SDK session")
  expect(list).toContain("Native session")
  expect(list).not.toContain("[Claude] Native session")
  setup.mockInput.pressEscape()
  await waitForText(gateway, "Ask Kohaku anything")

  submitText(gateway, "/child open sdk")
  expect(await waitForText(gateway, "Read-only canonical child history")).toContain("vtest  [Claude]")
  setup.mockInput.pressEscape()
  await waitForText(gateway, "Ask Kohaku anything")
  socket.deliver({ event: "turn_model_updated", chat_id: "chat", model_name: "Claude model", model_preset: "Claude preset" })
  submitText(gateway, "/child open native")
  expect(await waitForText(gateway, "Read-only canonical child history")).not.toContain("vtest  [Claude]")
  setup.mockInput.pressEscape()
  await waitForText(gateway, "Ask Kohaku anything")
  socket.deliver({ event: "attached", chat_id: "native", model_preset: "Claude preset", harness_binding: nativeBinding })
  await waitUntil(() => gateway.state.ready)
  expect(await waitForText(gateway, "Claude preset ▾")).not.toContain("[Claude]  Claude preset")
  socket.deliver({ event: "attached", chat_id: "chat", model_preset: "Other model" })
  expect(await waitForText(gateway, "[Claude] SDK work")).toContain("[Claude]  Other model")
})

test("shows the locked workspace and explains lock and steering refusals", async () => {
  const gateway = await mountGateway(undefined, {
    apiUrl: "http://nanobot.test", apiToken: "test-token",
  }, (url) => url.includes("available-models")
    ? jsonResponse({ models: [], canUseFullAccess: true })
    : jsonResponse({ sessions: [], messages: [] }))
  const { socket, setup } = gateway
  socket.deliver({
    event: "attached", chat_id: "chat", harness_binding: claudeBinding,
  })
  await waitUntil(() => gateway.state.ready)
  const scope = { project_path: "/chosen/workspace", access_mode: "restricted" }
  socket.deliver({ event: "session_updated", chat_id: "chat", workspace_scope: scope })
  expect(await rendered(gateway)).not.toContain("Workspace locked")

  socket.deliver({
    event: "session_updated", chat_id: "chat",
    harness_binding: { ...claudeBinding, locked_workspace: scope.project_path, runtime_generation: 1 },
  })
  await waitForText(gateway, "Workspace locked · /chosen/workspace")
  const lock = setup.renderer.root.findDescendantById("kohaku-tui-workspace-lock-text")
  if (!(lock instanceof TextRenderable)) throw new Error("workspace lock control is missing")
  await setup.mockMouse.click(lock.x + 5, lock.y)
  await waitForText(gateway, "Start a new chat to change it")

  socket.deliver({ event: "error", chat_id: "chat", detail: "harness_workspace_locked" })
  await waitForText(gateway, "Claude Code locks the workspace")
  socket.deliver({ event: "error", chat_id: "chat", detail: "harness_steering_unavailable" })
  await waitForText(gateway, "Claude Code cannot change an active turn")
})

test("asks for a fresh message for each external recovery reason", async () => {
  const gateway = await mountGateway(undefined)
  for (const [reason, title] of [
    ["external_turn_interrupted", "Claude Code turn interrupted"],
    ["external_tool_state_unknown", "Claude Code tool status is unknown"],
  ]) {
    gateway.socket.deliver({
      event: "recovery_state", chat_id: "chat", status: "awaiting_user", recovery_id: reason,
      reason, can_continue: false,
    })
    const frame = await waitForText(gateway, title!)
    expect(frame).toContain("Send a fresh message")
    expect(frame).not.toContain("Continue")
    submitText(gateway, "/continue")
    await waitForText(gateway, "Send a fresh message to continue.")
    expect(sentFrames(gateway.socket).some((request) => request.type === "webui_request")).toBe(false)
  }
})

function externalApproval(id: string, item?: DelegatedChildActivity, canSave = false) {
  return {
    event: "external_approval",
    chat_id: "chat",
    approval_id: id,
    ...(item ? { child_id: item.child_id } : {}),
    approval: {
      session_key: item?.child_session_key ?? "websocket:chat",
      session_incarnation: item?.child_incarnation ?? "root-inc",
      run_id: item?.child_run_id ?? "root-run",
      generation: 2,
      effect_id: `effect-${id}`,
      summary: `Review request ${id}`,
      command: `echo ${id}`,
      save_exclusion_command: canSave ? "echo" : null,
    },
  }
}

test("sends each external approval decision with only its live identity", async () => {
  const item = child("worker", { harness: "claude_sdk" })
  const gateway = await mountGateway(undefined)
  const { socket } = gateway
  for (const [id, action, decision, owner, canSave] of [
    ["a", "allow", "approve", undefined, false],
    ["b", "save", "approve_and_save", item, true],
    ["c", "deny", "deny", item, false],
  ] as const) {
    const event = externalApproval(id, owner, canSave)
    socket.deliver(event)
    const frame = await waitForText(gateway, event.approval.summary)
    expect(frame).toContain("Allow once")
    expect(frame).toContain("Deny")
    expect(frame.includes("Allow & save")).toBe(canSave)
    submitText(gateway, `/approval ${action} ${id}`)
    await waitForText(gateway, "Waiting for the gateway to resolve the approval")
    const { session_key, session_incarnation, run_id, generation, effect_id } = event.approval
    expect(messageFrames(socket).at(-1)).toMatchObject({
      chat_id: "chat", content: "approval_decision", webui: true,
      approval_decision: { approval_id: id, decision, session_key, session_incarnation, run_id, generation, effect_id },
    })
    expect(Object.keys(messageFrames(socket).at(-1)!.approval_decision as object).sort()).toEqual(
      ["approval_id", "decision", "effect_id", "generation", "run_id", "session_incarnation", "session_key"],
    )
    socket.deliver({ ...event, approval: null })
  }
  expect(new Set(messageFrames(socket).map((frame) => frame.turn_id)).size).toBe(3)
})

test("keeps child approvals after parent settlement and uses the normal Stop target", async () => {
  const item = child("worker", { harness: "claude_sdk" })
  const sibling = child("sibling", { harness: "claude_sdk" })
  const target = stopTarget([item, sibling], true)
  const gateway = await mountGateway(target)
  const { socket, setup } = gateway
  const direct = externalApproval("direct")
  const first = externalApproval("first", item)
  const second = externalApproval("second", sibling)
  socket.deliver({ event: "goal_status", chat_id: "chat", turn_id: "root-turn", status: "running" })
  for (const event of [direct, first, second]) socket.deliver(event)
  expect(await waitForText(gateway, direct.approval.summary)).toContain("(+2 more)")
  setup.mockInput.pressCtrlC()
  await waitUntil(() => messageFrames(socket).some((frame) => frame.content === "/stop"))
  expect(messageFrames(socket).at(-1)).toMatchObject({
    chat_id: "chat", content: "/stop", stop_expectation: target,
  })

  socket.deliver({ event: "turn_end", chat_id: "chat", turn_id: "root-turn" })
  const afterRoot = await waitForText(gateway, first.approval.summary)
  expect(afterRoot).not.toContain(direct.approval.summary)
  expect(afterRoot).toContain("(+1 more)")
  socket.deliver({ ...first, approval: null })
  expect(await waitForText(gateway, second.approval.summary)).not.toContain(first.approval.summary)
  socket.deliver({ event: "attached", chat_id: "chat" })
  expect(await rendered(gateway)).not.toContain(second.approval.summary)
  socket.deliver(second)
  expect(await waitForText(gateway, second.approval.summary)).not.toContain("(+1 more)")
})
