import { afterEach, describe, expect, test } from "bun:test"
import { NanobotClient, type InboundEvent } from "./protocol"

class FakeSocket {
  static instances: FakeSocket[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  readyState = 0
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readonly listeners = new Map<string, Array<(event: unknown) => void>>()

  constructor(readonly url: string) {
    FakeSocket.instances.push(this)
  }

  addEventListener(type: string, handler: (event: unknown) => void): void {
    const list = this.listeners.get(type) ?? []
    list.push(handler)
    this.listeners.set(type, list)
  }

  send(value: string): void {
    this.sent.push(value)
  }

  close(): void {
    this.readyState = 3
  }

  emit(type: string, event: unknown): void {
    for (const handler of this.listeners.get(type) ?? []) handler(event)
  }

  open(): void {
    this.readyState = 1
    this.emit("open", {})
  }

  deliver(payload: unknown): void {
    this.emit("message", { data: JSON.stringify(payload) })
  }
}

const originalWebSocket = globalThis.WebSocket

afterEach(() => {
  globalThis.WebSocket = originalWebSocket
  FakeSocket.instances = []
})

async function waitUntil(predicate: () => boolean, timeout = 1_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate() && Date.now() < deadline) await Bun.sleep(5)
  if (!predicate()) throw new Error(`condition was not met within ${timeout}ms`)
}

async function connectClient(): Promise<{
  client: NanobotClient
  socket: FakeSocket
  events: InboundEvent[]
}> {
  const events: InboundEvent[] = []
  // The client builds its first socket from the global constructor.
  globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket
  const client = new NanobotClient({
    url: "ws://test",
    reconnect: false,
    onEvent: (event) => events.push(event),
    onStatus: () => undefined,
  })
  client.connect()
  await waitUntil(() => FakeSocket.instances.length > 0)
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  socket.deliver({ event: "ready", chat_id: "", client_id: "client" })
  socket.deliver({ event: "attached", chat_id: "chat" })
  return { client, socket, events }
}

function lastFrame(socket: FakeSocket): Record<string, unknown> {
  return JSON.parse(socket.sent.at(-1)!) as Record<string, unknown>
}

describe("control envelope", () => {
  test("a steer request carries the settings version and target turn", async () => {
    const { client, socket } = await connectClient()

    client.send("guide", { requestIntent: "steer", targetTurnId: "turn-active" })

    const frame = lastFrame(socket)
    expect(frame.request_settings_version).toBe(1)
    expect(frame.request_intent).toBe("steer")
    expect(frame.target_turn_id).toBe("turn-active")
  })

  test("a default request is a new request with the settings version", async () => {
    const { client, socket } = await connectClient()

    client.send("/stop", { targetTurnId: "turn-active" })

    const frame = lastFrame(socket)
    expect(frame.request_settings_version).toBe(1)
    expect(frame.request_intent).toBe("new")
    expect(frame.target_turn_id).toBe("turn-active")
    expect(frame.content).toBe("/stop")
  })

  test("message_admitted reaches the event callback", async () => {
    const { socket, events } = await connectClient()

    socket.deliver({
      event: "message_admitted",
      chat_id: "chat",
      turn_id: "turn-1",
      attempt_id: "turn-1",
      session_key: "websocket:chat",
      session_incarnation: "inc-1",
      active_turn_id: "turn-1",
      starts_turn: true,
    })

    expect(events.some((event) => event.event === "message_admitted")).toBe(true)
  })
})
