import { describe, expect, it, vi } from "vitest";

import {
  prepareWorkspaceFullAccess,
  setWorkspaceAccessMode,
} from "@/lib/api";

function capturingTransport() {
  const calls: Array<{ action: string; payload: Record<string, unknown> }> = [];
  return {
    calls,
    transport: {
      requestMutation: vi.fn(async (action: string, payload: Record<string, unknown>) => {
        calls.push({ action, payload });
        throw new Error("captured");
      }),
    },
  };
}

describe("workspace v1 control payloads", () => {
  it("sends version, source token and confirmation on mode set", async () => {
    const { calls, transport } = capturingTransport();
    await expect(setWorkspaceAccessMode(transport, {
      chat_id: "chat-a",
      mode: "full",
      expected_runtime_id: "b".repeat(32),
      expected_permission_revision: 3,
      source_token: "tok-live",
      full_confirmation_id: "confirm-1",
    })).rejects.toThrow("captured");
    expect(calls[0]).toEqual({
      action: "workspace.access.mode.set",
      payload: {
        chat_id: "chat-a",
        mode: "full",
        expected_runtime_id: "b".repeat(32),
        expected_permission_revision: 3,
        request_settings_version: 1,
        source_token: "tok-live",
        full_confirmation_id: "confirm-1",
      },
    });
  });

  it("requests a bound confirmation with the current source", async () => {
    const { calls, transport } = capturingTransport();
    await expect(prepareWorkspaceFullAccess(transport, "chat-a", "tok-live"))
      .rejects.toThrow("captured");
    expect(calls[0]).toEqual({
      action: "workspace.access.prepare_full",
      payload: {
        chat_id: "chat-a",
        request_settings_version: 1,
        source_token: "tok-live",
      },
    });
  });
});


describe("workspace permission projection parsing", () => {
  it("preserves the server source token and top-level prepare token", async () => {
    const transport = {
      requestMutation: vi.fn().mockResolvedValue({
        workspace_scope: {
          project_path: "/tmp/project",
          project_name: "project",
          access_mode: "restricted",
          restrict_to_workspace: true,
        },
        workspace_permissions: {
          version: 1,
          runtime_id: "b".repeat(32),
          revision: 2,
          mode: "restricted",
          state: "active",
          can_enable_full: true,
          full_access_unavailable_reason: null,
          task_access: null,
          retirement: null,
          source_token: "a".repeat(32),
        },
        source_token: "a".repeat(32),
        full_confirmation_id: "c".repeat(32),
      }),
    };
    const result = await prepareWorkspaceFullAccess(transport, "chat-a", "a".repeat(32));
    expect(result.source_token).toBe("a".repeat(32));
    expect(result.workspace_permissions.source_token).toBe("a".repeat(32));
    expect(result.full_confirmation_id).toBe("c".repeat(32));
  });
});
