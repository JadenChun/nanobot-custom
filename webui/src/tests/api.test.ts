import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  allowWorkspaceAccessOnce,
  cancelMcpOAuth,
  cancelWorkspaceAccessRequest,
  configureChannel,
  completeMcpOAuth,
  completeProviderOAuth,
  continueWorkspaceAccessRequest,
  createModelConfiguration,
  createProviderSettings,
  deleteSkill,
  deleteModelConfiguration,
  deleteSession,
  discoverRepositories,
  fetchFilePreview,
  fetchFilePreviewAvailability,
  fetchAutomations,
  fetchApiService,
  fetchCliApps,
  fetchInstalledCliApps,
  fetchMcpOAuthStatus,
  fetchMcpPresets,
  fetchMarketplaceSkillTrends,
  fetchNanobotFeatures,
  fetchProviderModels,
  fetchSessionAutomations,
  fetchSettingsUsage,
  fetchSidebarState,
  fetchSkillDetail,
  fetchSkills,
  fetchTrendingMarketplaceSkills,
  fetchWebuiThread,
  fetchWorkspaces,
  forgetRepository,
  importMcpConfig,
  installMarketplaceSkill,
  listSessions,
  listSlashCommands,
  loginProviderOAuth,
  logoutProviderOAuth,
  migrateModelConfigurations,
  disableNanobotFeature,
  enableNanobotFeature,
  runAutomationAction,
  runCliAppAction,
  rememberRepository,
  revokeWorkspaceGrant,
  resolveWorkspaceAccessRequest,
  runMcpPresetAction,
  saveCustomMcpServer,
  searchMarketplaceSkills,
  startApiService,
  startMcpOAuth,
  stopApiService,
  cancelChannelConnect,
  pollChannelConnect,
  startChannelConnect,
  setWorkspaceAccessMode,
  updateAutomation,
  updateSidebarState,
  updateImageGenerationSettings,
  updateModelCallOrder,
  updateModelConfiguration,
  updateMcpServerTools,
  updateNetworkSafetySettings,
  updateProviderSettings,
  updateSkillEnabled,
  updateWebSearchSettings,
  upsertWorkspaceGrant,
  validateChannel,
} from "@/lib/api";
import { folderAccessThreadFixture } from "./fixtures/folder-access-request";
import onceResponseFixture from "./fixtures/permission-v2-once-response.json";
import type { WorkspaceOnceActionResult } from "@/lib/types";

const requestMutation = vi.fn();
const mutationTransport = {
  requestMutation: <T>(
    action: string,
    payload?: Record<string, unknown>,
    timeoutMs?: number,
  ) => requestMutation(action, payload, timeoutMs) as Promise<T>,
};

describe("webui API helpers", () => {
  beforeEach(() => {
    requestMutation.mockReset();
    requestMutation.mockResolvedValue({});
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ deleted: true, key: "websocket:chat-1", messages: [] }),
      }),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("sends exact workspace grant and repository mutation wire payloads", async () => {
    const grant = {
      path: "/projects/a",
      file_access: "read_only" as const,
      commands: false,
    };
    const entry = {
      name: "project-a",
      path: "/projects/a",
      purpose: "Work on project A",
      kind: "project" as const,
      knowledge_repos: ["notes"],
      auto_sync: false,
      sync_paths: [],
    };

    await upsertWorkspaceGrant(mutationTransport, grant, 7);
    await revokeWorkspaceGrant(mutationTransport, "/projects/a", 8);
    await rememberRepository(mutationTransport, entry);
    await forgetRepository(mutationTransport, "project-a");
    await discoverRepositories(mutationTransport, "/projects");

    expect(requestMutation).toHaveBeenNthCalledWith(1, "workspace.grant.upsert", {
      expected_revision: 7,
      path: "/projects/a",
      file_access: "read_only",
      commands: false,
    }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(2, "workspace.grant.revoke", {
      expected_revision: 8,
      path: "/projects/a",
    }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(3, "repository.remember", { entry }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(4, "repository.forget", {
      name: "project-a",
    }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(5, "repository.discover", {
      path: "/projects",
    }, 20_000);
  });

  it("preserves the upsert operation receipt instead of replacing it with a newer global revision", async () => {
    const grant = {
      path: "/projects/freelance-alias",
      file_access: "read_only" as const,
      commands: false,
    };
    const receipt = {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
      grants_revision: 7,
    };
    requestMutation.mockResolvedValueOnce({
      management: { grants_revision: 8, grants: [], repositories: [] },
      grant_receipt: receipt,
    });

    const response = await upsertWorkspaceGrant(mutationTransport, grant, 6);

    expect(response.grant_receipt).toEqual(receipt);
    expect(response.management?.grants_revision).toBe(8);
  });

  it("sends exact folder-access recovery mutation payloads", async () => {
    const requestId = "0123456789abcdef0123456789abcdef";

    await resolveWorkspaceAccessRequest(
      mutationTransport,
      "chat-1",
      requestId,
      4,
      "/projects/freelance",
      7,
    );
    await continueWorkspaceAccessRequest(mutationTransport, "chat-1", requestId, 5);
    await cancelWorkspaceAccessRequest(mutationTransport, "chat-1", requestId, 6);

    expect(requestMutation).toHaveBeenNthCalledWith(1, "workspace.access.resolve", {
      chat_id: "chat-1",
      access_request_id: requestId,
      expected_revision: 4,
      path: "/projects/freelance",
      expected_grants_revision: 7,
    }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(2, "workspace.access.continue", {
      chat_id: "chat-1",
      access_request_id: requestId,
      expected_revision: 5,
    }, 20_000);
    expect(requestMutation).toHaveBeenNthCalledWith(3, "workspace.access.cancel", {
      chat_id: "chat-1",
      access_request_id: requestId,
      expected_revision: 6,
    }, 20_000);
  });

  it("sends the exact Once payload and accepts only a receipt correlated to the owning task", async () => {
    const request = {
      version: 1,
      access_request_id: "0123456789abcdef0123456789abcdef",
      revision: 4,
      chat_id: "chat-1",
      task: { turn_id: "turn-original", excerpt: "Inspect this project." },
      blocked_turn_id: "turn-blocked",
      tool_call_id: "tool-1",
      created_at_ms: 1,
      status: "pending",
      reason: "folder_not_granted",
      operation: "read",
      target: { kind: "directory", path: "/projects/alias" },
      purpose: "Inspect this project.",
      selected_path: null,
      selected_access: null,
      can_continue: false,
      unavailable_reason: null,
      continuation_turn_id: null,
    };
    const effectiveAccess = {
      policy: "managed",
      source: "task",
      grant_root: "/projects/canonical",
      exists: true,
      read: true,
      write: false,
      boundary: "folder",
      commands: { requested: false, available: false, backend: "none", reason: null },
    };
    const result = {
      workspace_access_request: {
        ...request,
        revision: 5,
        status: "ready",
        selected_path: "/projects/canonical",
        selected_access: effectiveAccess,
        can_continue: true,
        selected_duration: "once",
        task_access_state: "active",
      },
      workspace_scope: {
        project_path: "/projects/canonical",
        access_mode: "restricted",
        effective_access: effectiveAccess,
      },
      workspace_permissions: {
        version: 1,
        runtime_id: "a".repeat(32),
        revision: 12,
        mode: "restricted",
        state: "active",
        can_enable_full: true,
        full_access_unavailable_reason: null,
        task_access: {
          task_turn_id: "turn-original",
          state: "active",
          folders: [{ path: "/projects/canonical", file_access: "read_only", commands: false }],
        },
        retirement: null,
      },
      once_receipt: {
        version: 1,
        duration: "once",
        runtime_id: "a".repeat(32),
        permission_revision: 12,
        access_request_id: request.access_request_id,
        request_revision: 5,
        task_turn_id: "turn-original",
        submitted_path: "/projects/alias",
        canonical_path: "/projects/canonical",
        file_access: "read_only",
        commands: false,
      },
    };
    requestMutation.mockResolvedValueOnce(result);

    const response = await allowWorkspaceAccessOnce(mutationTransport, {
      chat_id: "chat-1",
      access_request_id: request.access_request_id,
      expected_revision: 4,
      path: "/projects/alias",
      file_access: "read_only",
      commands: false,
      expected_grants_revision: 8,
      expected_runtime_id: "a".repeat(32),
      expected_permission_revision: 11,
    }, "turn-original");

    expect(requestMutation).toHaveBeenCalledWith("workspace.access.allow_once", {
      chat_id: "chat-1",
      access_request_id: request.access_request_id,
      expected_revision: 4,
      path: "/projects/alias",
      file_access: "read_only",
      commands: false,
      expected_grants_revision: 8,
      expected_runtime_id: "a".repeat(32),
      expected_permission_revision: 11,
    }, 20_000);
    expect(response.once_receipt).toEqual(result.once_receipt);

    requestMutation.mockResolvedValueOnce({
      ...result,
      once_receipt: { ...result.once_receipt, task_turn_id: "another-task" },
    });
    await expect(allowWorkspaceAccessOnce(mutationTransport, {
      chat_id: "chat-1",
      access_request_id: request.access_request_id,
      expected_revision: 4,
      path: "/projects/alias",
      file_access: "read_only",
      commands: false,
      expected_grants_revision: 8,
      expected_runtime_id: "a".repeat(32),
      expected_permission_revision: 11,
    }, "turn-original")).rejects.toThrow(/receipt|response/i);
  });

  it("uses permission-mode revisions and preserves a validated retiring conflict", async () => {
    const input = {
      chat_id: "chat-1",
      mode: "restricted" as const,
      expected_runtime_id: "b".repeat(32),
      expected_permission_revision: 19,
    };
    const projection = {
      version: 1,
      runtime_id: input.expected_runtime_id,
      revision: 20,
      mode: "full",
      state: "retiring",
      can_enable_full: false,
      full_access_unavailable_reason: "permission_retiring",
      task_access: null,
      retirement: {
        reason: "mode_changed",
        active_turn: false,
        active_processes: 1,
        unknown_processes: 0,
      },
    };
    const scope = {
      project_path: "/projects/current",
      access_mode: "full",
    };
    requestMutation.mockRejectedValueOnce(Object.assign(new Error("native_processes_active"), {
      status: 409,
      workspaceModeConflict: {
        error: { status: 409, message: "native_processes_active" },
        workspace_scope: scope,
        workspace_permissions: projection,
      },
    }));

    await expect(setWorkspaceAccessMode(mutationTransport, input)).rejects.toMatchObject({
      status: 409,
      conflict: {
        workspace_scope: scope,
        workspace_permissions: projection,
      },
    });
    expect(requestMutation).toHaveBeenCalledWith("workspace.access.mode.set", input, 20_000);
  });

  it("parses the exact Once response emitted by the backend serializer", async () => {
    // Sanitized from the backend's V2_ONCE_JSON serializer output; synthetic IDs, path, and
    // timestamp preserve the original response fields and their cross-field relationships.
    const emitted = onceResponseFixture as WorkspaceOnceActionResult;
    const request = emitted.workspace_access_request;
    const receipt = emitted.once_receipt;
    requestMutation.mockResolvedValueOnce(emitted);

    const parsed = await allowWorkspaceAccessOnce(mutationTransport, {
      chat_id: request.chat_id,
      access_request_id: request.access_request_id,
      expected_revision: request.revision - 1,
      path: receipt.submitted_path,
      file_access: receipt.file_access,
      commands: receipt.commands,
      expected_grants_revision: 0,
      expected_runtime_id: receipt.runtime_id,
      expected_permission_revision: receipt.permission_revision - 1,
    }, request.task.turn_id);

    expect(parsed.once_receipt).toEqual(receipt);
    expect(parsed.workspace_permissions.task_access?.folders).toContainEqual({
      path: receipt.canonical_path,
      file_access: receipt.file_access,
      commands: receipt.commands,
    });
  });

  it("accepts an exact home-relative submitted path with a canonical Once receipt path", async () => {
    const emitted = onceResponseFixture as WorkspaceOnceActionResult;
    const submittedPath = "~/Documents/Client project";
    const canonicalPath = "/home/jaden/Documents/Client project";
    const taskAccess = emitted.workspace_permissions.task_access;
    if (!taskAccess) throw new Error("Once fixture is missing task access");
    const effectiveAccess = {
      ...emitted.workspace_access_request.selected_access!,
      grant_root: canonicalPath,
    };
    const response: WorkspaceOnceActionResult = {
      ...emitted,
      workspace_access_request: {
        ...emitted.workspace_access_request,
        selected_path: canonicalPath,
        selected_access: effectiveAccess,
      },
      workspace_scope: {
        ...emitted.workspace_scope,
        project_path: canonicalPath,
        effective_access: effectiveAccess,
      },
      workspace_permissions: {
        ...emitted.workspace_permissions,
        task_access: {
          ...taskAccess,
          folders: taskAccess.folders.map((folder) => ({ ...folder, path: canonicalPath })),
        },
      },
      once_receipt: {
        ...emitted.once_receipt,
        submitted_path: submittedPath,
        canonical_path: canonicalPath,
      },
    };
    requestMutation.mockResolvedValueOnce(response);

    const parsed = await allowWorkspaceAccessOnce(mutationTransport, {
      chat_id: response.workspace_access_request.chat_id,
      access_request_id: response.once_receipt.access_request_id,
      expected_revision: response.once_receipt.request_revision - 1,
      path: submittedPath,
      file_access: response.once_receipt.file_access,
      commands: response.once_receipt.commands,
      expected_grants_revision: 0,
      expected_runtime_id: response.once_receipt.runtime_id,
      expected_permission_revision: response.once_receipt.permission_revision - 1,
    }, response.once_receipt.task_turn_id);

    expect(parsed.once_receipt.submitted_path).toBe(submittedPath);
    expect(parsed.once_receipt.canonical_path).toBe(canonicalPath);
    expect(parsed.workspace_access_request.selected_path).toBe(canonicalPath);
  });

  it.each([
    "submitted path", "relative canonical path", "selected path", "scope path", "grant root",
    "task folder", "mode", "runtime", "request revision", "permission revision", "task",
    "file access", "commands", "effective write",
  ])("rejects a Once acknowledgement with mismatched %s", async (field) => {
    const response = structuredClone(onceResponseFixture) as WorkspaceOnceActionResult;
    const request = response.workspace_access_request;
    const receipt = response.once_receipt;
    receipt.submitted_path = "~/projects/alias";
    const input = {
      chat_id: request.chat_id,
      access_request_id: request.access_request_id,
      expected_revision: request.revision - 1,
      path: receipt.submitted_path,
      file_access: receipt.file_access,
      commands: receipt.commands,
      expected_grants_revision: 0,
      expected_runtime_id: receipt.runtime_id,
      expected_permission_revision: receipt.permission_revision - 1,
    };
    const taskTurnId = request.task.turn_id;
    if (!request.selected_access || !response.workspace_permissions.task_access) {
      throw new Error("Once fixture is missing its selected authority");
    }
    switch (field) {
      case "submitted path": receipt.submitted_path = "~/different"; break;
      case "relative canonical path": receipt.canonical_path = "relative/project"; break;
      case "selected path": request.selected_path = "/different"; break;
      case "scope path": response.workspace_scope.project_path = "/different"; break;
      case "grant root": request.selected_access.grant_root = "/different"; break;
      case "task folder": response.workspace_permissions.task_access.folders[0]!.path = "/different"; break;
      case "mode": response.workspace_permissions.mode = "full"; break;
      case "runtime": receipt.runtime_id = "b".repeat(32); break;
      case "request revision": receipt.request_revision = input.expected_revision; break;
      case "permission revision":
        receipt.permission_revision = input.expected_permission_revision;
        response.workspace_permissions.revision = input.expected_permission_revision;
        break;
      case "task": receipt.task_turn_id = "other-task"; break;
      case "file access": receipt.file_access = input.file_access === "read_only" ? "read_write" : "read_only"; break;
      case "commands": receipt.commands = !input.commands; break;
      case "effective write": request.selected_access.write = !request.selected_access.write; break;
    }
    requestMutation.mockResolvedValueOnce(response);
    await expect(allowWorkspaceAccessOnce(mutationTransport, input, taskTurnId)).rejects.toMatchObject({
      status: 502,
      message: "Invalid workspace Once receipt or response",
    });
  });

  it("percent-encodes websocket keys when fetching webui-thread snapshot", async () => {
    await fetchWebuiThread("tok", "websocket:chat-1");

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/websocket%3Achat-1/webui-thread",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
        credentials: "same-origin",
        cache: "no-store",
      }),
    );
  });

  it("hydrates the permission request from the owning webui-thread projection", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "application/json" }),
        json: async () => folderAccessThreadFixture,
      }),
    );

    const thread = await fetchWebuiThread("tok", "websocket:chat-title");

    expect(thread?.workspace_access_request).toEqual(folderAccessThreadFixture.workspace_access_request);
  });

  it("passes pagination params when fetching a WebUI thread page", async () => {
    await fetchWebuiThread("tok", "websocket:chat-1", {
      limit: 120,
      before: "abc+/=",
    });

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/websocket%3Achat-1/webui-thread?limit=120&before=abc%2B%2F%3D",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
        credentials: "same-origin",
      }),
    );
  });

  it("aborts a WebUI thread request when its caller signal is aborted", async () => {
    let requestSignal: AbortSignal | null = null;
    vi.mocked(fetch).mockImplementation((_input, init) => new Promise((_resolve, reject) => {
      requestSignal = init?.signal ?? null;
      requestSignal?.addEventListener("abort", () => {
        reject(new DOMException("Aborted", "AbortError"));
      });
    }));
    const controller = new AbortController();

    const request = fetchWebuiThread("tok", "websocket:chat-1", {
      signal: controller.signal,
    });
    controller.abort();

    await expect(request).rejects.toMatchObject({ name: "AbortError" });
    expect(requestSignal?.aborted).toBe(true);
  });

  it("percent-encodes websocket keys and paths when fetching file previews", async () => {
    await fetchFilePreview("tok", "websocket:chat-1", "/tmp/project/hook.py:12");

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/websocket%3Achat-1/file-preview?path=%2Ftmp%2Fproject%2Fhook.py%3A12",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
        credentials: "same-origin",
      }),
    );
  });

  it("probes file preview availability without requesting contents", async () => {
    await expect(
      fetchFilePreviewAvailability("tok", "websocket:chat-1", "notes/ready.md"),
    ).resolves.toBe(true);

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/websocket%3Achat-1/file-preview?path=notes%2Fready.md&probe=1",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
        credentials: "same-origin",
      }),
    );
  });

  it("returns false when a file preview probe is unavailable", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ available: false }),
    } as Response);

    await expect(
      fetchFilePreviewAvailability("tok", "websocket:chat-1", "notes/missing.md"),
    ).resolves.toBe(false);
  });

  it("percent-encodes websocket keys when fetching session automations", async () => {
    await fetchSessionAutomations("tok", "websocket:chat-1");

    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/websocket%3Achat-1/automations",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("fetches workspace automations", async () => {
    await fetchAutomations("tok");

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/automations",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("validates channel settings with form values", async () => {
    await validateChannel(
      mutationTransport,
      "slack",
      { "channels.slack.botToken": "xoxb-test" },
      { instanceId: "default" },
    );

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.channel.validate",
      {
        name: "slack",
        instance_id: "default",
        values: { "channels.slack.botToken": "xoxb-test" },
      },
      20_000,
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("configures channels through the authenticated WebSocket", async () => {
    await configureChannel(
      mutationTransport,
      "discord",
      { "channels.discord.token": "saved-secret" },
      { enable: true },
    );

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.channel.configure",
      {
        name: "discord",
        enable: true,
        values: { "channels.discord.token": "saved-secret" },
      },
      150_000,
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("serializes channel QR connect request envelopes", async () => {
    await startChannelConnect(mutationTransport, "weixin", { force: true });
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.channel.connect.start",
      { channel: "weixin", force: true },
      150_000,
    );

    await pollChannelConnect(mutationTransport, "weixin", "session+/=");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.channel.connect.poll",
      { channel: "weixin", session_id: "session+/=" },
      150_000,
    );

    await cancelChannelConnect(mutationTransport, "weixin", "session+/=");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.channel.connect.cancel",
      { channel: "weixin", session_id: "session+/=" },
      20_000,
    );
  });

  it("serializes workspace automation actions", async () => {
    await runAutomationAction(mutationTransport, "disable", "job 1/2");

    expect(requestMutation).toHaveBeenCalledWith(
      "automation.disable",
      { id: "job 1/2" },
      20_000,
    );
  });

  it("serializes workspace automation updates", async () => {
    const values = {
      name: "每日测验",
      message: "Ask 今日 quiz",
      schedule: { kind: "cron", expr: "0 9 * * *", tz: "Asia/Shanghai" },
    } as const;
    await updateAutomation(mutationTransport, "job 1/2", values);

    expect(requestMutation).toHaveBeenCalledWith(
      "automation.update",
      { id: "job 1/2", values },
      20_000,
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("fetches the WebUI skill summary", async () => {
    await fetchSkills("tok");

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/skills",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("percent-encodes skill names when fetching skill details", async () => {
    await fetchSkillDetail("tok", "current web");

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/skills/current%20web",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("encodes marketplace search queries and provider", async () => {
    await searchMarketplaceSkills("tok", "React & testing");

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/skills/search?q=React+%26+testing&provider=all",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("fetches a provider marketplace leaderboard", async () => {
    await fetchTrendingMarketplaceSkills("tok", "skillhub");

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/skills/trending?provider=skillhub",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("fetches skills.sh trend history independently", async () => {
    await fetchMarketplaceSkillTrends("tok", [
      "vercel-labs/skills/find-skills",
      "acme/skills/react",
    ]);

    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/skills/trends?id=vercel-labs%2Fskills%2Ffind-skills&id=acme%2Fskills%2Freact",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("sends provider install coordinates without placing them in a URL", async () => {
    await installMarketplaceSkill(
      mutationTransport,
      "skillhub",
      "@tencent/skills",
      "ima-skills",
      "1.1.8",
    );

    expect(requestMutation).toHaveBeenCalledWith(
      "skill.install",
      {
        provider: "skillhub",
        source: "@tencent/skills",
        skill: "ima-skills",
        version: "1.1.8",
      },
      150_000,
    );
  });

  it("updates and deletes installed skills over the WebSocket", async () => {
    await updateSkillEnabled(mutationTransport, "custom skill", false);

    expect(requestMutation).toHaveBeenLastCalledWith(
      "skill.update",
      { name: "custom skill", enabled: false },
      20_000,
    );

    await deleteSkill(mutationTransport, "custom skill");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "skill.delete",
      { name: "custom skill" },
      20_000,
    );
  });

  it("sends the session key in a mutation payload", async () => {
    await deleteSession(mutationTransport, "websocket:chat-1");

    expect(requestMutation).toHaveBeenCalledWith(
      "session.delete",
      { key: "websocket:chat-1" },
      20_000,
    );
  });

  it("passes the automation cascade flag when deleting a session", async () => {
    await deleteSession(mutationTransport, "websocket:chat-1", { deleteAutomations: true });

    expect(requestMutation).toHaveBeenCalledWith(
      "session.delete",
      { key: "websocket:chat-1", delete_automations: true },
      20_000,
    );
  });

  it("fetches token usage through the lightweight settings endpoint", async () => {
    await fetchSettingsUsage("tok");

    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/usage",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("serializes model configuration creation", async () => {
    await createModelConfiguration(mutationTransport, {
      name: "Fast writing",
      provider: "openai",
      model: "openai/gpt-4.1-mini",
      maxTokens: 4096,
      contextWindowTokens: 128000,
      temperature: 0.4,
      reasoningEffort: "high",
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.model_configuration.create",
      {
        name: "Fast writing",
        provider: "openai",
        model: "openai/gpt-4.1-mini",
        max_tokens: 4096,
        context_window_tokens: 128000,
        temperature: 0.4,
        reasoning_effort: "high",
      },
      20_000,
    );
  });

  it("serializes model configuration updates", async () => {
    await updateModelConfiguration(mutationTransport, {
      name: "codex",
      newName: "Codex",
      provider: "openai_codex",
      model: "openai-codex/gpt-5.5",
      maxTokens: 8192,
      contextWindowTokens: 65536,
      temperature: 0,
      reasoningEffort: null,
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.model_configuration.update",
      {
        name: "codex",
        new_name: "Codex",
        provider: "openai_codex",
        model: "openai-codex/gpt-5.5",
        max_tokens: 8192,
        context_window_tokens: 65536,
        temperature: 0,
        reasoning_effort: "",
      },
      20_000,
    );
  });

  it("serializes model preset deletion and migration", async () => {
    await deleteModelConfiguration(mutationTransport, "spare");
    await migrateModelConfigurations(mutationTransport);

    expect(requestMutation).toHaveBeenNthCalledWith(
      1,
      "settings.model_configuration.delete",
      { name: "spare" },
      20_000,
    );
    expect(requestMutation).toHaveBeenNthCalledWith(
      2,
      "settings.model_configuration.migrate",
      {},
      20_000,
    );
  });

  it("serializes model call order as an ordered JSON array", async () => {
    await updateModelCallOrder(mutationTransport, ["backup", "primary"]);

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.model_call_order.update",
      { order: ["backup", "primary"] },
      20_000,
    );
  });

  it("reports HTML API fallbacks as gateway mismatch errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "text/html; charset=utf-8" }),
        text: async () => "<!doctype html><html></html>",
      }),
    );

    await expect(fetchApiService("tok")).rejects.toMatchObject({
      status: 200,
      message: "Gateway returned WebUI HTML instead of JSON. Restart nanobot gateway and try again.",
    });
  });

  it("surfaces correlated WebSocket mutation errors", async () => {
    requestMutation.mockRejectedValueOnce(
      Object.assign(new Error("npm error ENOTEMPTY"), { status: 500 }),
    );

    await expect(
      runCliAppAction(mutationTransport, "install", "hyperframes"),
    ).rejects.toMatchObject({
      status: 500,
      message: "npm error ENOTEMPTY",
    });
  });

  it("extracts user-facing messages from JSON API errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        headers: new Headers({ "content-type": "application/json" }),
        text: async () => JSON.stringify({ error: "Paste the complete callback URL." }),
      }),
    );

    await expect(fetchMcpOAuthStatus("tok", "flow-123")).rejects.toMatchObject({
      status: 400,
      message: "Paste the complete callback URL.",
    });
  });

  it("times out when an API request never responds", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));

    const pending = expect(listSessions("tok")).rejects.toThrow(
      "Request timed out after 20000ms",
    );
    await vi.advanceTimersByTimeAsync(20_000);

    await pending;
  });

  it("keeps provider secrets in the WebSocket payload", async () => {
    await updateProviderSettings(mutationTransport, {
      provider: "openrouter",
      apiKey: "sk-or-test",
      apiBase: "https://openrouter.ai/api/v1",
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.provider.update",
      {
        provider: "openrouter",
        apiKey: "sk-or-test",
        apiBase: "https://openrouter.ai/api/v1",
      },
      20_000,
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("serializes OAuth provider advanced settings", async () => {
    await updateProviderSettings(mutationTransport, {
      provider: "xai_grok",
      proxy: "http://127.0.0.1:7890",
      extraBody: '{"tools":[]}',
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.provider.update",
      {
        provider: "xai_grok",
        proxy: "http://127.0.0.1:7890",
        extraBody: '{"tools":[]}',
      },
      20_000,
    );
  });

  it("serializes custom provider creation with advanced settings", async () => {
    const update = {
      name: "Company Gateway",
      apiKey: "sk-company",
      apiBase: "https://gateway.example/v1",
      extraHeaders: '{"X-Tenant":"engineering"}',
      extraBody: '{"service_tier":"priority"}',
      extraQuery: '{"api-version":"2026-01-01"}',
      proxy: "http://127.0.0.1:7890",
      thinkingStyle: "enable_thinking",
    };
    await createProviderSettings(mutationTransport, update);

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.provider.create",
      update,
      20_000,
    );
  });

  it("fetches provider model lists", async () => {
    await fetchProviderModels("tok", "deepseek");

    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/provider-models?provider=deepseek",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("serializes provider OAuth login and logout actions", async () => {
    await loginProviderOAuth(mutationTransport, "openai_codex");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_login",
      { provider: "openai_codex" },
      20_000,
    );

    await loginProviderOAuth(mutationTransport, "openai_codex", true);
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_login",
      { provider: "openai_codex", remote_browser: true },
      20_000,
    );

    await completeProviderOAuth(mutationTransport, "xai_grok", "flow-123");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_complete",
      { provider: "xai_grok", flow_id: "flow-123" },
      20_000,
    );

    await completeProviderOAuth(
      mutationTransport,
      "xai_grok",
      "flow-123",
      "secret",
    );
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_complete",
      { provider: "xai_grok", flow_id: "flow-123", authorization_response: "secret" },
      20_000,
    );

    await completeProviderOAuth(
      mutationTransport,
      "openai_codex",
      "flow-codex",
      "http://localhost:1455/auth/callback?code=secret&state=test",
    );
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_complete",
      {
        provider: "openai_codex",
        flow_id: "flow-codex",
        authorization_response: "http://localhost:1455/auth/callback?code=secret&state=test",
      },
      20_000,
    );

    await logoutProviderOAuth(mutationTransport, "openai_codex");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.provider.oauth_logout",
      { provider: "openai_codex" },
      20_000,
    );
  });

  it("serializes web search settings updates", async () => {
    await updateWebSearchSettings(mutationTransport, {
      provider: "searxng",
      baseUrl: "https://search.example.com",
      maxResults: 8,
      timeout: 45,
      useJinaReader: false,
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.web_search.update",
      {
        provider: "searxng",
        base_url: "https://search.example.com",
        max_results: 8,
        timeout: 45,
        use_jina_reader: false,
      },
      20_000,
    );
  });

  it("serializes network safety settings updates", async () => {
    await updateNetworkSafetySettings(mutationTransport, {
      webuiAllowLocalServiceAccess: false,
      webuiDefaultAccessMode: "full",
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.network_safety.update",
      {
        webui_allow_local_service_access: false,
        webui_default_access_mode: "full",
      },
      20_000,
    );
  });

  it("serializes image generation settings updates", async () => {
    await updateImageGenerationSettings(mutationTransport, {
      enabled: true,
      provider: "openrouter",
      model: "openai/gpt-5.4-image-2",
      defaultAspectRatio: "16:9",
      defaultImageSize: "2K",
      maxImagesPerTurn: 3,
    });

    expect(requestMutation).toHaveBeenCalledWith(
      "settings.image_generation.update",
      {
        enabled: true,
        provider: "openrouter",
        model: "openai/gpt-5.4-image-2",
        default_aspect_ratio: "16:9",
        default_image_size: "2K",
        max_images_per_turn: 3,
      },
      20_000,
    );
  });

  it("reads CLI Apps catalog and serializes actions", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        apps: [],
        installed_count: 0,
        catalog_updated_at: "2026-04-18",
      }),
    } as Response);

    await expect(fetchCliApps("tok")).resolves.toMatchObject({ apps: [] });
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/cli-apps",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );

    await runCliAppAction(mutationTransport, "install", "gimp");
    expect(requestMutation).toHaveBeenCalledWith(
      "settings.cli_app.install",
      { name: "gimp" },
      150_000,
    );
  });

  it("reads installed CLI Apps without fetching the full catalog", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        apps: [],
        installed_count: 0,
        catalog_updated_at: null,
      }),
    } as Response);

    await expect(fetchInstalledCliApps("tok")).resolves.toMatchObject({ apps: [] });
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/cli-apps?installed_only=1",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("reads and toggles nanobot optional features", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        features: [],
        enabled_count: 0,
      }),
    } as Response);

    await expect(fetchNanobotFeatures("tok")).resolves.toMatchObject({ features: [] });
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/nanobot-features",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );

    await enableNanobotFeature(mutationTransport, "matrix");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.feature.enable",
      { name: "matrix" },
      150_000,
    );

    await disableNanobotFeature(mutationTransport, "matrix");
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.feature.disable",
      { name: "matrix" },
      20_000,
    );
  });

  it("manages the API service capability", async () => {
    await fetchApiService("tok");
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/api-service",
      expect.objectContaining({ headers: { Authorization: "Bearer tok" } }),
    );

    await startApiService(
      mutationTransport,
      { host: "127.0.0.1", port: 8900, timeout: 120 },
    );
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.api_service.start",
      { host: "127.0.0.1", port: 8900, timeout: 120 },
      150_000,
    );

    await startApiService(
      mutationTransport,
      { host: "0.0.0.0", port: 8900, timeout: 120, apiKey: "secret-token" },
    );
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.api_service.start",
      { host: "0.0.0.0", port: 8900, timeout: 120, api_key: "secret-token" },
      150_000,
    );

    await stopApiService(mutationTransport);
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.api_service.stop",
      {},
      20_000,
    );
  });

  it("reads MCP presets and serializes actions", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        presets: [],
        installed_count: 0,
      }),
    } as Response);

    await expect(fetchMcpPresets("tok")).resolves.toMatchObject({ presets: [] });
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/mcp-presets",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );

    await runMcpPresetAction(mutationTransport, "enable", "browserbase", {
      browserbase_api_key: "bb_live_test",
    });
    expect(requestMutation).toHaveBeenCalledWith(
      "settings.mcp.enable",
      { name: "browserbase", browserbase_api_key: "bb_live_test" },
      20_000,
    );

    await startMcpOAuth(mutationTransport, "notion", true);
    expect(requestMutation).toHaveBeenCalledWith(
      "settings.mcp.oauth_start",
      { name: "notion", reset: true },
      30_000,
    );

    await fetchMcpOAuthStatus("tok", "flow-123");
    expect(fetch).toHaveBeenCalledWith(
      "/api/settings/mcp-oauth/status?flow_id=flow-123",
      expect.objectContaining({ headers: { Authorization: "Bearer tok" } }),
    );

    const callbackUrl =
      "http://127.0.0.1:8765/auth/mcp/callback?code=secret&state=state-123";
    await completeMcpOAuth(mutationTransport, "flow-123", callbackUrl);
    expect(requestMutation).toHaveBeenCalledWith(
      "settings.mcp.oauth_complete",
      { flow_id: "flow-123", callback_url: callbackUrl },
      20_000,
    );

    await cancelMcpOAuth(mutationTransport, "flow-123");
    expect(requestMutation).toHaveBeenCalledWith(
      "settings.mcp.oauth_cancel",
      { flow_id: "flow-123" },
      20_000,
    );
  });

  it("serializes custom MCP, mcp.json import, and tool allowlist actions", async () => {
    const custom = {
      name: "docs",
      transport: "stdio",
      command: "npx",
      args: '["-y","docs-mcp"]',
      env: '{"API_KEY":"secret"}',
    };
    await saveCustomMcpServer(mutationTransport, custom);
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.mcp.custom",
      custom,
      20_000,
    );

    const oauthCustom = {
      name: "company-mcp",
      transport: "streamableHttp",
      url: "https://mcp.example.com/mcp",
      auth: "oauth",
    };
    await saveCustomMcpServer(mutationTransport, oauthCustom);
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.mcp.custom",
      oauthCustom,
      20_000,
    );

    await importMcpConfig(
      mutationTransport,
      '{"mcpServers":{"docs":{"command":"npx"}}}',
    );
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.mcp.import",
      { config: '{"mcpServers":{"docs":{"command":"npx"}}}' },
      20_000,
    );

    await updateMcpServerTools(mutationTransport, "docs", ["search", "fetch"]);
    expect(requestMutation).toHaveBeenLastCalledWith(
      "settings.mcp.tools",
      { name: "docs", enabled_tools: ["search", "fetch"] },
      20_000,
    );
  });

  it("reads and writes persisted sidebar state", async () => {
    const state = {
      schema_version: 1,
      pinned_keys: ["websocket:chat-1"],
      archived_keys: ["websocket:old"],
      session_order: ["websocket:chat-1", "websocket:old"],
      title_overrides: { "websocket:chat-1": "Release" },
      project_name_overrides: { "/Users/me/nanobot": "Core" },
      tags_by_key: {},
      collapsed_groups: {},
      view: {
        density: "compact" as const,
        show_previews: false,
        show_timestamps: false,
        show_archived: true,
        sort: "updated_desc" as const,
      },
      updated_at: null,
    };
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => state,
    } as Response);

    await expect(fetchSidebarState("tok")).resolves.toEqual(state);
    expect(fetch).toHaveBeenCalledWith(
      "/api/webui/sidebar-state",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );

    await updateSidebarState(mutationTransport, state);
    expect(requestMutation).toHaveBeenCalledWith(
      "sidebar.update",
      { state },
      20_000,
    );
  });

  it("fetches workspace project state", async () => {
    const payload = {
      schema_version: 1,
      default_access_mode: "default" as const,
      default_scope: {
        project_path: "/tmp/workspace",
        project_name: "workspace",
        access_mode: "restricted" as const,
        restrict_to_workspace: true,
      },
      controls: {
        can_change_project: true,
        can_use_full_access: true,
      },
    };
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => payload,
    } as Response);

    await expect(fetchWorkspaces("tok")).resolves.toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "/api/workspaces",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("maps generated session titles from the sessions list", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        sessions: [
          {
            key: "websocket:chat-1",
            created_at: "2026-05-01T10:00:00",
            updated_at: "2026-05-01T10:01:00",
            title: "优化 WebUI 标题",
            model_preset: "fast",
            run_started_at: 1_700_000_000,
            handle: {
              id: "handle_0123456789abcdef0123456789abcdef",
              name: "mira-0123456789",
            },
          },
        ],
      }),
    } as Response);

    await expect(listSessions("tok")).resolves.toMatchObject([
      {
        key: "websocket:chat-1",
        title: "优化 WebUI 标题",
        preview: "",
        modelPreset: "fast",
        runStartedAt: 1_700_000_000,
        handle: {
          id: "handle_0123456789abcdef0123456789abcdef",
          name: "mira-0123456789",
        },
      },
    ]);
  });

  it("maps slash command metadata from the commands endpoint", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        commands: [
          {
            command: "/stop",
            title: "Stop current task",
            description: "Cancel the active task.",
            icon: "square",
            lifecycle: "stop_active_turn",
            accepts_args: false,
          },
          {
            command: "/restart",
            title: "Restart nanobot",
            description: "Restart the bot process.",
            icon: "rotate-cw",
            lifecycle: "side_channel",
            accepts_args: false,
          },
          {
            command: "/history",
            title: "Show conversation history",
            description: "Print the last N messages.",
            icon: "history",
            arg_hint: "[n]",
            lifecycle: "side_channel",
            accepts_args: true,
          },
          {
            command: "/legacy",
            title: "Legacy row",
            description: "Old metadata should not be guessed.",
            icon: "circle-help",
          },
        ],
      }),
    } as Response);

    await expect(listSlashCommands("tok")).resolves.toEqual([
      {
        command: "/stop",
        title: "Stop current task",
        description: "Cancel the active task.",
        icon: "square",
        argHint: "",
        lifecycle: "stop_active_turn",
        acceptsArgs: false,
      },
      {
        command: "/restart",
        title: "Restart nanobot",
        description: "Restart the bot process.",
        icon: "rotate-cw",
        argHint: "",
        lifecycle: "side_channel",
        acceptsArgs: false,
      },
      {
        command: "/history",
        title: "Show conversation history",
        description: "Print the last N messages.",
        icon: "history",
        argHint: "[n]",
        lifecycle: "side_channel",
        acceptsArgs: true,
      },
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "/api/commands",
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });
});
