import { afterEach, describe, expect, test } from "bun:test"
import { TextareaRenderable } from "@opentui/core"
import {
  MockTreeSitterClient,
  createTestRenderer,
  type TestRendererSetup,
} from "@opentui/core/testing"

import { NanobotTui, type AppOptions } from "./app"
import type { MessageOptions, WorkspaceScopePayload } from "./protocol"
import type { ClipboardImageReader } from "./clipboard-image"

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
  status: { plainText: string }
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
  appOptions: AppOptions = options,
  clipboard?: ClipboardImageReader,
): Promise<{ app: NanobotTui; ui: InspectableApp }> {
  setup = await createTestRenderer({ width: 88, height: 24, screenMode: "alternate-screen" })
  const app = NanobotTui.mount(
    setup.renderer,
    appOptions,
    client,
    new MockTreeSitterClient({ autoResolveTimeout: 0 }),
    undefined,
    clipboard,
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

function admitOwner(app: NanobotTui, turnId: string): void {
  app.accept({
    event: "message_admitted",
    chat_id: "chat",
    turn_id: turnId,
    attempt_id: turnId,
    session_key: "websocket:chat",
    session_incarnation: "inc-1",
    active_turn_id: turnId,
    starts_turn: true,
  })
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

  test("a follow-up with an image is queued and sent once", async () => {
    const sent: string[] = []
    const sentOptions: MessageOptions[] = []
    const clipboard: ClipboardImageReader = {
      read: async () => ({
        mimeType: "image/png",
        dataUrl: "data:image/png;base64,AAEC/w==",
      }),
      dispose: async () => undefined,
    }
    const { app, ui } = await mount(
      makeClient(sent, sentOptions, undefined, ["turn-media"]),
      options,
      clipboard,
    )

    app.accept({ event: "goal_status", chat_id: "chat", status: "running", turn_id: "turn-1" })
    await setup!.mockInput.pressKey("v", { ctrl: true })
    await waitUntil(() => ui.composer.plainText === "[Image #1] ")

    await setup!.mockInput.typeText("describe this")
    await setup!.mockInput.pressTab()
    await waitUntil(() => ui.composer.plainText === "")

    expect(ui.status.plainText).toContain("1 queued")

    app.accept({ event: "turn_end", chat_id: "chat", turn_id: "turn-1" })
    await waitUntil(() => sent.length === 1)

    expect(sent[0]).toContain("describe this")
    expect(sentOptions[0]?.media).toHaveLength(1)
    expect(sentOptions[0]?.media?.[0]?.data_url).toBe("data:image/png;base64,AAEC/w==")

    admitOwner(app, "turn-media")
    expect(ui.status.plainText).not.toContain("queued")

    admitOwner(app, "turn-media")
    expect(sent).toHaveLength(1)
    expect(ui.status.plainText).not.toContain("queued")
  })

  test("reconnect reconciles a lost admission once and keeps order", async () => {
    const sent: string[] = []
    const sentOptions: MessageOptions[] = []
    const clipboard: ClipboardImageReader = {
      read: async () => ({
        mimeType: "image/png",
        dataUrl: "data:image/png;base64,AAEC/w==",
      }),
      dispose: async () => undefined,
    }
    const reconnectOptions: AppOptions = {
      ...options,
      apiUrl: "http://nanobot.test",
      apiToken: "token",
    }

    const originalFetch = globalThis.fetch
    const reconcileBody = {
      messages: [],
      active_turn_id: "turn-a",
      completed_turn_ids: [],
      page: { has_more_before: false, before_cursor: null, user_message_offset: 0 },
    }
    globalThis.fetch = (async (input: string | URL | Request, _init?: RequestInit) => {
      const raw =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url
      if (raw.includes("/webui-thread")) {
        return new Response(JSON.stringify(reconcileBody), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      }
      return new Response("{}", { status: 404 })
    }) as unknown as typeof fetch

    try {
      const { app, ui } = await mount(
        makeClient(sent, sentOptions, undefined, ["turn-a", "turn-b"]),
        reconnectOptions,
        clipboard,
      )

      app.accept({ event: "goal_status", chat_id: "chat", status: "running", turn_id: "turn-1" })

      await setup!.mockInput.typeText("first")
      await setup!.mockInput.pressTab()
      await waitUntil(() => ui.composer.plainText === "")

      await setup!.mockInput.typeText("second")
      await setup!.mockInput.pressTab()
      await waitUntil(() => ui.composer.plainText === "")

      expect(ui.status.plainText).toContain("2 queued")

      app.accept({ event: "turn_end", chat_id: "chat", turn_id: "turn-1" })
      await waitUntil(() => sent.length === 1)
      expect(sent[0]).toBe("first")

      // The owner admission receipts for both messages are lost.
      app.accept({ event: "attached", chat_id: "chat" })
      await waitUntil(() => sent.length === 2)

      expect(sent).toEqual(["first", "second"])
      expect(ui.status.plainText).toContain("1 queued")

      admitOwner(app, "turn-a")
      admitOwner(app, "turn-b")

      expect(sent).toEqual(["first", "second"])
      expect(ui.status.plainText).not.toContain("queued")
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})
