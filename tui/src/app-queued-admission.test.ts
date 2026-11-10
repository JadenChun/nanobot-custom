import { afterEach, describe, expect, test } from "bun:test"
import { TextareaRenderable } from "@opentui/core"
import {
  MockTreeSitterClient,
  createTestRenderer,
  type TestRendererSetup,
} from "@opentui/core/testing"

import { NanobotTui, type AppOptions } from "./app"
import type { MessageOptions, WorkspaceScopePayload } from "./protocol"

const options: AppOptions = {
  wsUrl: "ws://localhost.invalid/ws",
  apiUrl: "",
  apiToken: "",
  model: "test/model",
  modelPreset: "default",
  workspace: "/tmp/nanobot-workspace",
  version: "test",
  access: "workspace access",
  theme: "auto",
}

async function waitUntil(predicate: () => boolean, timeout = 1_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate() && Date.now() < deadline) await Bun.sleep(5)
  if (!predicate()) throw new Error(`condition was not met within ${timeout}ms`)
}

function makeClient(
  sent: string[],
  sentOptions: MessageOptions[] = [],
  failContent?: string,
  turnIds: string[] = [],
) {
  let index = 0
  return {
    activeChatId: "chat",
    connect() {},
    close() {},
    send(content: string, options: MessageOptions = {}) {
      if (failContent && content === failContent) throw new Error("transport down")
      sent.push(content)
      sentOptions.push(options)
      const turnId = turnIds[index] ?? `turn-${index + 1}`
      index += 1
      return turnId
    },
    attach(_chatId: string) {},
    newChat(_scope?: WorkspaceScopePayload) {},
    forkChat(_source: string, _before: number, _title?: string) {},
    setWorkspaceScope(_scope: WorkspaceScopePayload) {},
    updateRecovery() {
      return Promise.resolve({ status: "recovered" as const, recovery_id: "r" })
    },
  }
}

type InspectableApp = {
  ready: boolean
  composer: TextareaRenderable
  promptQueue: { length: number }
  queuePreview: { root: { visible: boolean } }
  activeTurnId: string | null
}

let setup: TestRendererSetup | null = null

afterEach(() => {
  setup?.renderer.destroy()
  setup = null
})

async function mount(
  client: ReturnType<typeof makeClient>,
): Promise<{ app: NanobotTui; ui: InspectableApp }> {
  setup = await createTestRenderer({ width: 88, height: 24, screenMode: "alternate-screen" })
  const app = NanobotTui.mount(
    setup.renderer,
    options,
    client,
    new MockTreeSitterClient({ autoResolveTimeout: 0 }),
  )
  app.accept({ event: "attached", chat_id: "chat" })
  await waitUntil(() => (app as unknown as InspectableApp).ready)
  return { app, ui: app as unknown as InspectableApp }
}

async function queueFollowUp(app: NanobotTui, ui: InspectableApp, content: string): Promise<void> {
  app.accept({ event: "goal_status", chat_id: "chat", status: "running", turn_id: "turn-1" })
  ui.composer.setText(content)
  setup!.mockInput.pressTab()
  await waitUntil(() => ui.composer.plainText === "")
}

describe("queued prompt admission", () => {
  test("failed send retains the queued item", async () => {
    const sent: string[] = []
    const sentOptions: MessageOptions[] = []
    const { app, ui } = await mount(makeClient(sent, sentOptions, "follow up"))

    await queueFollowUp(app, ui, "follow up")
    expect(ui.promptQueue.length).toBe(1)

    app.accept({ event: "turn_end", chat_id: "chat", turn_id: "turn-1" })
    await Bun.sleep(50)

    expect(ui.promptQueue.length).toBe(1)
  })

  test("route-level acceptance does not commit; owner admission commits", async () => {
    const sent: string[] = []
    const sentOptions: MessageOptions[] = []
    const { app, ui } = await mount(makeClient(sent, sentOptions, undefined, ["turn-follow"]))

    await queueFollowUp(app, ui, "follow up")
    app.accept({ event: "turn_end", chat_id: "chat", turn_id: "turn-1" })
    await waitUntil(() => sent.includes("follow up"))

    expect(ui.promptQueue.length).toBe(1)

    app.accept({
      event: "message_accepted",
      chat_id: "chat",
      turn_id: "turn-follow",
      active_turn_id: "turn-follow",
      starts_turn: true,
    })
    expect(ui.promptQueue.length).toBe(1)

    app.accept({
      event: "message_admitted",
      chat_id: "chat",
      turn_id: "turn-follow",
      attempt_id: "turn-follow",
      session_key: "websocket:chat",
      session_incarnation: "inc-1",
      active_turn_id: "turn-follow",
      starts_turn: true,
    })
    expect(ui.promptQueue.length).toBe(0)

    app.accept({
      event: "message_admitted",
      chat_id: "chat",
      turn_id: "turn-follow",
      attempt_id: "turn-follow",
      session_key: "websocket:chat",
      session_incarnation: "inc-1",
      active_turn_id: "turn-follow",
      starts_turn: true,
    })
    expect(ui.promptQueue.length).toBe(0)
  })
})
