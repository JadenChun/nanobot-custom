import type {
  FolderAccessActionResult,
  FolderAccessRequest,
  InboundEvent,
  WebuiThreadPersistedPayload,
} from "@/lib/types";

export const folderAccessRequestFixture: FolderAccessRequest = {
  version: 1,
  access_request_id: "0123456789abcdef0123456789abcdef",
  revision: 1,
  chat_id: "chat-title",
  task: {
    turn_id: "turn-original",
    excerpt: "Check the website and inventory projects in Freelance.",
  },
  blocked_turn_id: "turn-blocked",
  tool_call_id: "tool-1",
  created_at_ms: 1_798_228_800_000,
  status: "pending",
  reason: "folder_not_granted",
  operation: "read",
  target: { kind: "unknown", path: null },
  purpose: "Inspect the project files and inventory.",
  selected_path: null,
  selected_access: null,
  can_continue: false,
  unavailable_reason: null,
  continuation_turn_id: null,
};

const workspaceScope = {
  project_path: "/agent/home",
  project_name: "Agent home",
  access_mode: "restricted" as const,
  restrict_to_workspace: true,
  effective_access: {
    policy: "managed" as const,
    source: "agent_home" as const,
    grant_root: null,
    exists: true,
    read: true,
    write: true,
    boundary: "folder" as const,
    commands: {
      requested: false,
      available: false,
      backend: "none" as const,
      reason: null,
    },
  },
};

export const folderAccessSessionUpdatedFixture: Extract<InboundEvent, { event: "session_updated" }> = {
  event: "session_updated",
  chat_id: "chat-title",
  scope: "metadata",
  workspace_scope: workspaceScope,
  workspace_access_request: folderAccessRequestFixture,
};

export const folderAccessThreadFixture: WebuiThreadPersistedPayload = {
  schemaVersion: 1,
  sessionKey: "websocket:chat-title",
  messages: [{
    id: "message-needed",
    role: "assistant",
    content: "Folder access is needed to continue this task. Choose a folder or use an allowed project.",
    createdAt: 1_798_228_800_000,
  }],
  workspace_scope: workspaceScope,
  workspace_access_request: folderAccessRequestFixture,
};

export const folderAccessActionResultFixture: FolderAccessActionResult = {
  workspace_access_request: {
    ...folderAccessRequestFixture,
    revision: 2,
    status: "ready",
    selected_path: "/projects/freelance",
    selected_access: {
      policy: "managed",
      source: "grant",
      grant_root: "/projects/freelance",
      exists: true,
      read: true,
      write: false,
      boundary: "folder",
      commands: {
        requested: false,
        available: false,
        backend: "none",
        reason: null,
      },
    },
    can_continue: true,
  },
  workspace_scope: {
    ...workspaceScope,
    project_path: "/projects/freelance",
    project_name: "freelance",
    effective_access: {
      policy: "managed",
      source: "grant",
      grant_root: "/projects/freelance",
      exists: true,
      read: true,
      write: false,
      boundary: "folder",
      commands: {
        requested: false,
        available: false,
        backend: "none",
        reason: null,
      },
    },
  },
};
