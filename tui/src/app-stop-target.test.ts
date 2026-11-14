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
      // /stop is a control frame; only prompt sends consume a turn id.
      if (content === "/stop") return "stop"
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

function stopOptions(sent: string[], sentOptions: MessageOptions[]): MessageOptions[] {
  const result: MessageOptions[] = []
  for (let index = 0; index < sent.length; index += 1) {
    if (sent[index] === "/stop") result.push(sentOptions[index]!)
  }
  return result
}

describe("targeted stop", () => {
  test("a stop targets the current active turn and not a stale one", async () => {
    const sent: string[] = []
    const sentOptions: MessageOptions[] = []
    // Keep Ctrl+C for the app; the renderer must not exit on it.
    setup = await createTestRenderer({
      width: 88,
      height: 24,
      screenMode: "alternate-screen",
      exitOnCtrlC: false,
    })
    const app = NanobotTui.mount(
      setup.renderer,
      options,
      makeClient(sent, sentOptions, undefined, ["turn-1", "turn-2"]),
      new MockTreeSitterClient({ autoResolveTimeout: 0 }),
    )
    const connectionState = app as unknown as {
      handleStatus(status: "connected"): void
    }
    connectionState.handleStatus("connected")
    app.accept({ event: "attached", chat_id: "chat" })
    await waitUntil(() => (app as unknown as InspectableApp).ready)
    const ui = app as unknown as InspectableApp

    ui.composer.setText("long task")
    ui.composer.submit()
    await waitUntil(() => ui.activeTurnId === "turn-1")

    setup.mockInput.pressCtrlC()
    await waitUntil(() => sent.includes("/stop"))

    let stops = stopOptions(sent, sentOptions)
    expect(stops.at(-1)?.targetTurnId).toBe("turn-1")

    app.accept({ event: "turn_end", chat_id: "chat", turn_id: "turn-1" })
    await waitUntil(() => ui.activeTurnId === null)

    ui.composer.setText("later task")
    ui.composer.submit()
    await waitUntil(() => ui.activeTurnId === "turn-2")

    const stopCount = stops.length
    setup.mockInput.pressCtrlC()
    await waitUntil(() => stopOptions(sent, sentOptions).length > stopCount)

    stops = stopOptions(sent, sentOptions)
    expect(stops.at(-1)?.targetTurnId).toBe("turn-2")
  })
})
