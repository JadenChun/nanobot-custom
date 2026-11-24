export type ChatHarness = "native" | "claude_sdk"

export interface HarnessCapabilities {
  claude_sdk?: { available: boolean; reason?: string }
}

export interface HarnessBinding {
  backend: "native" | "external"
  harness: ChatHarness | null
  external_session_id: string | null
  locked_workspace: string | null
  runtime_generation: number
}

export function isChatHarness(value: unknown): value is ChatHarness {
  return value === "native" || value === "claude_sdk"
}

export function harnessBadge(harness: ChatHarness | null | undefined): string {
  return harness === "claude_sdk" ? "[Claude]" : ""
}

export const WORKSPACE_LOCKED = "Claude Code locks the workspace after the first turn starts. Start a new chat to change it."
export const STEERING_UNAVAILABLE = "Claude Code cannot change an active turn. Queue a follow-up or press Ctrl+C to stop."
export const TEMPORARY_HARNESS_HINT = "Claude Code requires a saved chat."

export function harnessErrorMessage(code: string | undefined): string | undefined {
  switch (code) {
    case "harness_workspace_locked": return WORKSPACE_LOCKED
    case "harness_steering_unavailable": return STEERING_UNAVAILABLE
    case "harness_requires_persistent_session": return TEMPORARY_HARNESS_HINT
    case "harness_unavailable": return "Claude Code is unavailable on this gateway."
    case "harness_session_create_failed": return "The gateway could not create the Claude Code chat."
    case "invalid_harness": return "The gateway rejected the harness selection."
    default: return undefined
  }
}
