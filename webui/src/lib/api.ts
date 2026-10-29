import type {
  ApiServicePayload,
  AutomationsPayload,
  AutomationUpdatePayload,
  ChannelConfigurePayload,
  ChannelConnectPayload,
  ChannelValidationPayload,
  ChatSummary,
  EffectiveWorkspaceAccess,
  CliAppsPayload,
  FilePreviewPayload,
  FolderAccessActionResult,
  FolderAccessRequest,
  FolderGrantInput,
  ImageGenerationSettingsUpdate,
  McpPresetsPayload,
  McpOAuthFlowPayload,
  MarketplaceProvider,
  NanobotFeaturesPayload,
  ModelConfigurationCreate,
  ModelConfigurationUpdate,
  NetworkSafetySettingsUpdate,
  PairingPayload,
  ProviderCreationUpdate,
  ProviderModelsPayload,
  ProviderOAuthCompletionResult,
  ProviderOAuthLoginResult,
  ProviderSettingsUpdate,
  RepositoryDiscoveryPayload,
  RepositoryRecord,
  RecoveryState,
  SessionDeleteResult,
  SessionHandle,
  SessionAutomationsPayload,
  SettingsPayload,
  SidebarStatePayload,
  SkillDetail,
  SkillActionPayload,
  SkillInstallPayload,
  SkillsPayload,
  SkillsSearchPayload,
  SkillsTrendsPayload,
  SkillsTrendingPayload,
  SlashCommand,
  SlashCommandLifecycle,
  TranscriptionSettingsUpdate,
  WebSearchSettingsUpdate,
  WorkspacesPayload,
  WorkspaceGrantReceipt,
  WorkspaceGrantUpsertPayload,
  WorkspaceModeActionInput,
  WorkspaceModeActionResult,
  WorkspaceModeConflict,
  WorkspaceOnceActionInput,
  WorkspaceOnceActionResult,
  WorkspaceOnceReceipt,
  WorkspacePermissionProjection,
  WebuiThreadPersistedPayload,
  WorkspaceScopePayload,
} from "./types";
import { fetchWithTimeout } from "./http";

const API_READ_TIMEOUT_MS = 20_000;
const API_MUTATION_TIMEOUT_MS = 20_000;
const PACKAGE_MUTATION_TIMEOUT_MS = 150_000;
const SLASH_COMMAND_LIFECYCLES = new Set<SlashCommandLifecycle>([
  "side_channel",
  "finalize_active_turn",
  "stop_active_turn",
  "agent_turn",
  "agent_turn_with_args",
]);

function isSlashCommandLifecycle(value: unknown): value is SlashCommandLifecycle {
  return (
    typeof value === "string"
    && SLASH_COMMAND_LIFECYCLES.has(value as SlashCommandLifecycle)
  );
}
export class ApiError extends Error {
  status: number;
  details: unknown;
  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.details = details;
    this.name = "ApiError";
  }
}

export class WorkspaceModeConflictError extends ApiError {
  conflict: WorkspaceModeConflict;

  constructor(message: string, conflict: WorkspaceModeConflict) {
    super(409, message);
    this.conflict = conflict;
    this.name = "WorkspaceModeConflictError";
  }
}

export interface WebUIMutationTransport {
  requestMutation<T>(
    action: string,
    payload?: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<T>;
}

async function request<T>(
  url: string,
  token: string,
  init?: RequestInit,
  timeoutMs: number = 0,
): Promise<T> {
  const res = await fetchWithTimeout(
    url,
    {
      ...(init ?? {}),
      headers: {
        ...(init?.headers ?? {}),
        Authorization: `Bearer ${token}`,
      },
      credentials: "same-origin",
    },
    timeoutMs,
  );
  if (!res.ok) {
    const text = typeof res.text === "function" ? (await res.text()).trim() : "";
    let message = text;
    if (text.startsWith("{")) {
      try {
        const payload: unknown = JSON.parse(text);
        if (payload && typeof payload === "object") {
          const error = (payload as { error?: unknown }).error;
          if (typeof error === "string" && error.trim()) message = error.trim();
        }
      } catch {
        // Preserve non-JSON error bodies exactly as returned by the gateway.
      }
    }
    throw new ApiError(res.status, message || `HTTP ${res.status}`);
  }
  const contentType = res.headers?.get?.("content-type") ?? "";
  if (contentType && !contentType.toLowerCase().includes("application/json")) {
    const text = typeof res.text === "function" ? await res.text() : "";
    const isHtml = text.trimStart().toLowerCase().startsWith("<!doctype");
    throw new ApiError(
      res.status,
      isHtml
        ? "Gateway returned WebUI HTML instead of JSON. Restart nanobot gateway and try again."
        : "Gateway returned a non-JSON response.",
    );
  }
  return (await res.json()) as T;
}

async function mutation<T>(
  transport: WebUIMutationTransport,
  action: string,
  payload: Record<string, unknown> = {},
  timeoutMs: number = API_MUTATION_TIMEOUT_MS,
): Promise<T> {
  try {
    return await transport.requestMutation<T>(action, payload, timeoutMs);
  } catch (reason) {
    if (reason instanceof ApiError) throw reason;
    const status = (
      typeof reason === "object"
      && reason !== null
      && "status" in reason
      && typeof reason.status === "number"
    ) ? reason.status : 500;
    const message = reason instanceof Error ? reason.message : "WebUI mutation failed";
    const workspaceModeConflict = (
      typeof reason === "object"
      && reason !== null
      && "workspaceModeConflict" in reason
    ) ? (reason as { workspaceModeConflict?: unknown }).workspaceModeConflict : undefined;
    throw new ApiError(status, message, workspaceModeConflict === undefined
      ? undefined
      : { workspaceModeConflict });
  }
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isCanonicalAbsolutePath(path: string): boolean {
  return path.startsWith("/")
    || /^[A-Za-z]:[\\/]/.test(path)
    || /^\\\\[^\\]+\\[^\\]+/.test(path);
}

function isValidWorkspaceOnceSubmittedPath(path: unknown, expectedPath: string): path is string {
  if (
    typeof path !== "string"
    || path !== expectedPath
    || !path
    || path.length > 2048
    || path.includes("\0")
  ) return false;
  return path === "~"
    || path.startsWith("~/")
    || path.startsWith("~\\")
    || isCanonicalAbsolutePath(path);
}

function parseWorkspaceGrantReceipt(
  value: unknown,
  grant: FolderGrantInput,
): WorkspaceGrantReceipt | null {
  if (!isObjectRecord(value)) return null;
  const receiptFields = [
    "canonical_path",
    "commands",
    "file_access",
    "grants_revision",
    "submitted_path",
  ];
  const keys = Object.keys(value).sort();
  if (keys.length !== receiptFields.length || keys.some((key, index) => key !== receiptFields[index])) {
    return null;
  }

  const canonicalPath = value.canonical_path;
  const revision = value.grants_revision;
  if (
    value.submitted_path !== grant.path
    || value.file_access !== grant.file_access
    || value.commands !== grant.commands
    || typeof canonicalPath !== "string"
    || !isCanonicalAbsolutePath(canonicalPath)
    || typeof revision !== "number"
    || !Number.isInteger(revision)
    || revision < 0
  ) return null;

  return {
    submitted_path: grant.path,
    canonical_path: canonicalPath,
    file_access: grant.file_access,
    commands: grant.commands,
    grants_revision: revision,
  };
}

const PERMISSION_FULL_REASONS = new Set([
  "controls_unavailable",
  "session_required",
  "configured_sandbox",
  "supervisor_unavailable",
  "permission_state_unavailable",
  "permission_retiring",
]);
const TASK_ACCESS_STATES = new Set(["active", "retiring", "expired"]);
const RETIREMENT_REASONS = new Set([
  "task_ended",
  "task_cancelled",
  "task_replaced",
  "mode_changed",
  "source_changed",
  "session_reset",
  "session_deleted",
  "restart",
]);
const FOLDER_ACCESS_STATUSES = new Set([
  "pending",
  "ready",
  "continuing",
  "continued",
  "cancelled",
  "invalidated",
]);
const FOLDER_ACCESS_REASONS = new Set([
  "folder_not_granted",
  "write_not_granted",
  "commands_not_granted",
  "folder_selection_required",
]);
const FOLDER_ACCESS_OPERATIONS = new Set([
  "read",
  "write",
  "execute",
  "discover",
  "select",
  "locate",
]);
const FOLDER_ACCESS_UNAVAILABLE_REASONS = new Set([
  "task_changed",
  "focus_changed",
  "source_changed",
  "session_reset",
  "session_deleted",
  "access_changed",
  "path_missing",
  "path_changed",
  "context_unavailable",
  "continuation_unknown",
  "task_access_expired",
  "permission_retiring",
]);

function hasExactKeys(value: Record<string, unknown>, fields: readonly string[]): boolean {
  const keys = Object.keys(value).sort();
  const expected = [...fields].sort();
  return keys.length === expected.length && keys.every((key, index) => key === expected[index]);
}

function isNonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function parseEffectiveWorkspaceAccess(value: unknown): EffectiveWorkspaceAccess | null {
  if (!isObjectRecord(value)) return null;
  const { commands } = value;
  if (!isObjectRecord(commands)) return null;
  const commandReasons = new Set([
    "not_granted",
    "unsupported_platform",
    "sandbox_unavailable",
    "managed_repository_policy",
    "path_missing",
    "access_revoked",
    "permission_retiring",
    "supervisor_unavailable",
  ]);
  if (
    (value.policy !== "legacy" && value.policy !== "managed")
    || !["agent_home", "grant", "configured", "none", "task", "full"].includes(String(value.source))
    || (value.grant_root !== null && typeof value.grant_root !== "string")
    || typeof value.exists !== "boolean"
    || typeof value.read !== "boolean"
    || typeof value.write !== "boolean"
    || (value.boundary !== "folder" && value.boundary !== "unrestricted")
    || typeof commands.requested !== "boolean"
    || typeof commands.available !== "boolean"
    || (commands.backend !== "native" && commands.backend !== "bwrap" && commands.backend !== "none")
    || (commands.reason !== null && (typeof commands.reason !== "string" || !commandReasons.has(commands.reason)))
  ) return null;
  return value as unknown as EffectiveWorkspaceAccess;
}

export function normalizeWorkspacePermissionProjection(value: unknown): WorkspacePermissionProjection | null {
  if (!isObjectRecord(value) || !hasExactKeys(value, [
    "version",
    "runtime_id",
    "revision",
    "mode",
    "state",
    "can_enable_full",
    "full_access_unavailable_reason",
    "task_access",
    "retirement",
  ])) return null;
  if (
    value.version !== 1
    || typeof value.runtime_id !== "string"
    || !/^[0-9a-f]{32}$/.test(value.runtime_id)
    || !isNonnegativeInteger(value.revision)
    || (value.mode !== "restricted" && value.mode !== "full")
    || (value.state !== "active" && value.state !== "retiring")
    || typeof value.can_enable_full !== "boolean"
    || (
      value.full_access_unavailable_reason !== null
      && (
        typeof value.full_access_unavailable_reason !== "string"
        || !PERMISSION_FULL_REASONS.has(value.full_access_unavailable_reason)
      )
    )
  ) return null;

  let taskAccess: WorkspacePermissionProjection["task_access"] = null;
  if (value.task_access !== null) {
    if (!isObjectRecord(value.task_access) || !hasExactKeys(value.task_access, ["task_turn_id", "state", "folders"])) return null;
    if (
      typeof value.task_access.task_turn_id !== "string"
      || !value.task_access.task_turn_id
      || typeof value.task_access.state !== "string"
      || !TASK_ACCESS_STATES.has(value.task_access.state)
      || !Array.isArray(value.task_access.folders)
    ) return null;
    const folders: NonNullable<WorkspacePermissionProjection["task_access"]>["folders"] = [];
    for (const candidate of value.task_access.folders) {
      if (
        !isObjectRecord(candidate)
        || !hasExactKeys(candidate, ["path", "file_access", "commands"])
        || typeof candidate.path !== "string"
        || !isCanonicalAbsolutePath(candidate.path)
        || (candidate.file_access !== "read_only" && candidate.file_access !== "read_write")
        || typeof candidate.commands !== "boolean"
      ) return null;
      folders.push({
        path: candidate.path,
        file_access: candidate.file_access,
        commands: candidate.commands,
      });
    }
    taskAccess = {
      task_turn_id: value.task_access.task_turn_id,
      state: value.task_access.state as NonNullable<WorkspacePermissionProjection["task_access"]>["state"],
      folders,
    };
  }

  let retirement: WorkspacePermissionProjection["retirement"] = null;
  if (value.retirement !== null) {
    if (!isObjectRecord(value.retirement) || !hasExactKeys(value.retirement, [
      "reason",
      "active_turn",
      "active_processes",
      "unknown_processes",
    ])) return null;
    if (
      typeof value.retirement.reason !== "string"
      || !RETIREMENT_REASONS.has(value.retirement.reason)
      || typeof value.retirement.active_turn !== "boolean"
      || !isNonnegativeInteger(value.retirement.active_processes)
      || !isNonnegativeInteger(value.retirement.unknown_processes)
    ) return null;
    retirement = {
      reason: value.retirement.reason as NonNullable<WorkspacePermissionProjection["retirement"]>["reason"],
      active_turn: value.retirement.active_turn,
      active_processes: value.retirement.active_processes,
      unknown_processes: value.retirement.unknown_processes,
    };
  }

  return {
    version: 1,
    runtime_id: value.runtime_id,
    revision: value.revision,
    mode: value.mode,
    state: value.state,
    can_enable_full: value.can_enable_full,
    full_access_unavailable_reason: value.full_access_unavailable_reason as WorkspacePermissionProjection["full_access_unavailable_reason"],
    task_access: taskAccess,
    retirement,
  };
}

function parseWorkspaceScope(value: unknown): WorkspaceScopePayload | null {
  if (!isObjectRecord(value) || !isCanonicalAbsolutePath(String(value.project_path ?? ""))) return null;
  if (value.access_mode !== "restricted" && value.access_mode !== "full") return null;
  let effectiveAccess: EffectiveWorkspaceAccess | undefined;
  if (value.effective_access !== undefined) {
    const parsedAccess = parseEffectiveWorkspaceAccess(value.effective_access);
    if (!parsedAccess) return null;
    effectiveAccess = parsedAccess;
  }
  if (value.project_name !== undefined && typeof value.project_name !== "string") return null;
  if (value.restrict_to_workspace !== undefined && typeof value.restrict_to_workspace !== "boolean") return null;
  return {
    ...(value as unknown as WorkspaceScopePayload),
    ...(effectiveAccess ? { effective_access: effectiveAccess } : {}),
  };
}

function parseFolderAccessRequest(value: unknown): FolderAccessRequest | null {
  if (!isObjectRecord(value) || value.version !== 1) return null;
  if (
    typeof value.access_request_id !== "string"
    || !value.access_request_id
    || !isNonnegativeInteger(value.revision)
    || typeof value.chat_id !== "string"
    || !value.chat_id
    || !isObjectRecord(value.task)
    || typeof value.task.turn_id !== "string"
    || !value.task.turn_id
    || typeof value.task.excerpt !== "string"
    || typeof value.blocked_turn_id !== "string"
    || (value.tool_call_id !== null && typeof value.tool_call_id !== "string")
    || typeof value.created_at_ms !== "number"
    || !Number.isFinite(value.created_at_ms)
    || typeof value.status !== "string"
    || !FOLDER_ACCESS_STATUSES.has(value.status)
    || typeof value.reason !== "string"
    || !FOLDER_ACCESS_REASONS.has(value.reason)
    || typeof value.operation !== "string"
    || !FOLDER_ACCESS_OPERATIONS.has(value.operation)
    || !isObjectRecord(value.target)
    || (value.target.kind === "unknown" && value.target.path !== null)
    || ((value.target.kind === "directory" || value.target.kind === "file")
      && (typeof value.target.path !== "string" || !isCanonicalAbsolutePath(value.target.path)))
    || (value.target.kind !== "unknown" && value.target.kind !== "directory" && value.target.kind !== "file")
    || typeof value.purpose !== "string"
    || (value.selected_path !== null && (typeof value.selected_path !== "string" || !isCanonicalAbsolutePath(value.selected_path)))
    || (value.selected_access !== null && !parseEffectiveWorkspaceAccess(value.selected_access))
    || typeof value.can_continue !== "boolean"
    || (value.unavailable_reason !== null && (typeof value.unavailable_reason !== "string" || !FOLDER_ACCESS_UNAVAILABLE_REASONS.has(value.unavailable_reason)))
    || (value.continuation_turn_id !== null && typeof value.continuation_turn_id !== "string")
    || (value.selected_duration !== undefined && value.selected_duration !== null && value.selected_duration !== "once" && value.selected_duration !== "always")
    || (value.task_access_state !== undefined && value.task_access_state !== null && (typeof value.task_access_state !== "string" || !TASK_ACCESS_STATES.has(value.task_access_state)))
  ) return null;
  return {
    ...(value as unknown as FolderAccessRequest),
    ...(value.selected_access === null
      ? { selected_access: null }
      : { selected_access: parseEffectiveWorkspaceAccess(value.selected_access)! }),
  };
}

function parseWorkspaceOnceReceipt(
  value: unknown,
  expectedSubmittedPath: string,
): WorkspaceOnceReceipt | null {
  if (!isObjectRecord(value) || !hasExactKeys(value, [
    "version",
    "duration",
    "runtime_id",
    "permission_revision",
    "access_request_id",
    "request_revision",
    "task_turn_id",
    "submitted_path",
    "canonical_path",
    "file_access",
    "commands",
  ])) return null;
  if (
    value.version !== 1
    || value.duration !== "once"
    || typeof value.runtime_id !== "string"
    || !/^[0-9a-f]{32}$/.test(value.runtime_id)
    || !isNonnegativeInteger(value.permission_revision)
    || typeof value.access_request_id !== "string"
    || typeof value.task_turn_id !== "string"
    || !value.task_turn_id
    || !isValidWorkspaceOnceSubmittedPath(value.submitted_path, expectedSubmittedPath)
    || typeof value.canonical_path !== "string"
    || !isCanonicalAbsolutePath(value.canonical_path)
    || (value.file_access !== "read_only" && value.file_access !== "read_write")
    || typeof value.commands !== "boolean"
    || !isNonnegativeInteger(value.request_revision)
  ) return null;
  return value as unknown as WorkspaceOnceReceipt;
}

function sameEffectiveWorkspaceAccess(
  left: EffectiveWorkspaceAccess,
  right: EffectiveWorkspaceAccess,
): boolean {
  return left.policy === right.policy
    && left.source === right.source
    && left.grant_root === right.grant_root
    && left.exists === right.exists
    && left.read === right.read
    && left.write === right.write
    && left.boundary === right.boundary
    && left.commands.requested === right.commands.requested
    && left.commands.available === right.commands.available
    && left.commands.backend === right.commands.backend
    && left.commands.reason === right.commands.reason;
}

function parseWorkspaceOnceResult(
  value: unknown,
  input: WorkspaceOnceActionInput,
  expectedTaskTurnId: string,
): WorkspaceOnceActionResult | null {
  if (!isObjectRecord(value)) return null;
  const request = parseFolderAccessRequest(value.workspace_access_request);
  const scope = parseWorkspaceScope(value.workspace_scope);
  const permissions = normalizeWorkspacePermissionProjection(value.workspace_permissions);
  const receipt = parseWorkspaceOnceReceipt(value.once_receipt, input.path);
  if (!request || !scope || !permissions || !receipt) return null;
  const selected = request.selected_access;
  const scopeAccess = scope.effective_access;
  const task = permissions.task_access;
  if (
    receipt.runtime_id !== input.expected_runtime_id
    || receipt.permission_revision <= input.expected_permission_revision
    || receipt.permission_revision !== permissions.revision
    || receipt.access_request_id !== input.access_request_id
    || receipt.request_revision !== request.revision
    || receipt.request_revision <= input.expected_revision
    || receipt.task_turn_id !== expectedTaskTurnId
    || receipt.submitted_path !== input.path
    || receipt.file_access !== input.file_access
    || receipt.commands !== input.commands
    || request.chat_id !== input.chat_id
    || request.access_request_id !== input.access_request_id
    || request.revision !== receipt.request_revision
    || request.task.turn_id !== expectedTaskTurnId
    || request.status !== "ready"
    || request.selected_duration !== "once"
    || request.task_access_state !== "active"
    || request.selected_path !== receipt.canonical_path
    || !selected
    || scope.project_path !== receipt.canonical_path
    || scope.access_mode !== "restricted"
    || !scopeAccess
    || selected.policy !== "managed"
    || selected.source !== "task"
    || selected.grant_root !== receipt.canonical_path
    || !selected.exists
    || !selected.read
    || selected.write !== (input.file_access === "read_write")
    || selected.boundary !== "folder"
    || selected.commands.requested !== input.commands
    || !sameEffectiveWorkspaceAccess(selected, scopeAccess)
    || permissions.mode !== "restricted"
    || permissions.state !== "active"
    || !task
    || task.task_turn_id !== expectedTaskTurnId
    || task.state !== "active"
    || !task.folders.some((folder) => (
      folder.path === receipt.canonical_path
      && folder.file_access === input.file_access
      && folder.commands === input.commands
    ))
  ) return null;
  return {
    workspace_access_request: request,
    workspace_scope: scope,
    workspace_permissions: permissions,
    once_receipt: receipt,
  };
}

function parseWorkspaceModeActionResult(
  value: unknown,
  input: WorkspaceModeActionInput,
): WorkspaceModeActionResult | null {
  if (!isObjectRecord(value)) return null;
  const scope = parseWorkspaceScope(value.workspace_scope);
  const permissions = normalizeWorkspacePermissionProjection(value.workspace_permissions);
  if (
    !scope
    || !permissions
    || scope.access_mode !== input.mode
    || permissions.mode !== input.mode
    || permissions.state !== "active"
    || permissions.runtime_id !== input.expected_runtime_id
    || permissions.revision < input.expected_permission_revision
  ) return null;
  return { workspace_scope: scope, workspace_permissions: permissions };
}

function parseWorkspaceModeConflict(value: unknown): WorkspaceModeConflict | null {
  if (!isObjectRecord(value) || !isObjectRecord(value.error)) return null;
  const { error } = value;
  if (
    error.status !== 409
    || (error.message !== "chat_running"
      && error.message !== "native_processes_active"
      && error.message !== "native_processes_unknown")
  ) return null;
  const scope = parseWorkspaceScope(value.workspace_scope);
  const permissions = normalizeWorkspacePermissionProjection(value.workspace_permissions);
  if (
    !scope
    || !permissions
    || scope.access_mode !== "full"
    || permissions.mode !== "full"
    || permissions.state !== "retiring"
    || permissions.retirement === null
  ) return null;
  return {
    error: { status: 409, message: error.message },
    workspace_scope: scope,
    workspace_permissions: permissions,
  };
}

function compactMcpValues(values: Record<string, unknown>): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  Object.entries(values).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) payload[key] = trimmed;
      return;
    }
    payload[key] = value;
  });
  return payload;
}

function splitKey(key: string): { channel: string; chatId: string } {
  const idx = key.indexOf(":");
  if (idx === -1) return { channel: "", chatId: key };
  return { channel: key.slice(0, idx), chatId: key.slice(idx + 1) };
}

function normalizeSessionHandle(value: unknown): SessionHandle | null {
  if (!value || typeof value !== "object") return null;
  const handle = value as Partial<SessionHandle>;
  const id = typeof handle.id === "string" ? handle.id.trim() : "";
  const name = typeof handle.name === "string" ? handle.name.trim() : "";
  if (
    !/^handle_[a-f0-9]{32}$/i.test(id)
    || !name
    || !/^[\p{L}\p{N}_-]+$/u.test(name)
  ) return null;
  return { id, name };
}

export async function listSessions(
  token: string,
  base: string = "",
): Promise<ChatSummary[]> {
  type Row = {
    key: string;
    created_at: string | null;
    updated_at: string | null;
    title?: string;
    preview?: string;
    model_preset?: string | null;
    run_started_at?: number | null;
    recovery_state?: RecoveryState | null;
    workspace_scope?: WorkspaceScopePayload | null;
    handle?: SessionHandle | null;
  };
  const body = await request<{ sessions: Row[] }>(
    `${base}/api/sessions`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
  return body.sessions.map((s) => {
    const handle = normalizeSessionHandle(s.handle);
    return {
      key: s.key,
      ...splitKey(s.key),
      createdAt: s.created_at,
      updatedAt: s.updated_at,
      title: s.title ?? "",
      preview: s.preview ?? "",
      modelPreset: s.model_preset ?? null,
      runStartedAt: s.run_started_at ?? null,
      recoveryState: s.recovery_state ?? null,
      workspaceScope: s.workspace_scope ?? null,
      handle,
    };
  });
}

/** Disk-backed WebUI display thread snapshot (separate from agent session). */
export interface FetchWebuiThreadOptions {
  limit?: number;
  direction?: "latest";
  before?: string | null;
  signal?: AbortSignal;
}

export async function fetchWebuiThread(
  token: string,
  key: string,
  optionsOrBase?: FetchWebuiThreadOptions | string,
  base: string = "",
): Promise<WebuiThreadPersistedPayload | null> {
  const options = typeof optionsOrBase === "string" ? undefined : optionsOrBase;
  const resolvedBase = typeof optionsOrBase === "string" ? optionsOrBase : base;
  const params = new URLSearchParams();
  if (options?.limit !== undefined) params.set("limit", String(options.limit));
  if (options?.direction) params.set("direction", options.direction);
  if (options?.before) params.set("before", options.before);
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  const url = `${resolvedBase}/api/sessions/${encodeURIComponent(key)}/webui-thread${suffix}`;
  const res = await fetchWithTimeout(url, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
    cache: "no-store",
    signal: options?.signal,
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new ApiError(res.status, `HTTP ${res.status}`);
  const payload: unknown = await res.json();
  if (!isObjectRecord(payload)) return null;
  if (payload.workspace_permissions === undefined) {
    return payload as unknown as WebuiThreadPersistedPayload;
  }
  return {
    ...payload,
    workspace_permissions: payload.workspace_permissions === null
      ? null
      : normalizeWorkspacePermissionProjection(payload.workspace_permissions),
  } as unknown as WebuiThreadPersistedPayload;
}

export async function fetchFilePreview(
  token: string,
  key: string,
  path: string,
  base: string = "",
): Promise<FilePreviewPayload> {
  const query = new URLSearchParams();
  query.set("path", path);
  return request<FilePreviewPayload>(
    `${base}/api/sessions/${encodeURIComponent(key)}/file-preview?${query}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchFilePreviewAvailability(
  token: string,
  key: string,
  path: string,
  base: string = "",
): Promise<boolean> {
  const query = new URLSearchParams();
  query.set("path", path);
  query.set("probe", "1");
  const payload = await request<{ available?: boolean }>(
    `${base}/api/sessions/${encodeURIComponent(key)}/file-preview?${query}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
  return payload.available !== false;
}

export async function fetchSessionAutomations(
  token: string,
  key: string,
  base: string = "",
): Promise<SessionAutomationsPayload> {
  return request<SessionAutomationsPayload>(
    `${base}/api/sessions/${encodeURIComponent(key)}/automations`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchAutomations(
  token: string,
  base: string = "",
): Promise<AutomationsPayload> {
  return request<AutomationsPayload>(
    `${base}/api/webui/automations`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function runAutomationAction(
  transport: WebUIMutationTransport,
  action: "enable" | "disable" | "delete" | "run",
  id: string,
): Promise<AutomationsPayload> {
  return mutation<AutomationsPayload>(transport, `automation.${action}`, { id });
}

export async function updateAutomation(
  transport: WebUIMutationTransport,
  id: string,
  values: AutomationUpdatePayload,
): Promise<AutomationsPayload> {
  return mutation<AutomationsPayload>(transport, "automation.update", { id, values });
}

export async function fetchSkills(
  token: string,
  base: string = "",
): Promise<SkillsPayload> {
  return request<SkillsPayload>(
    `${base}/api/webui/skills`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchSkillDetail(
  token: string,
  name: string,
  base: string = "",
): Promise<SkillDetail> {
  return request<SkillDetail>(
    `${base}/api/webui/skills/${encodeURIComponent(name)}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function updateSkillEnabled(
  transport: WebUIMutationTransport,
  name: string,
  enabled: boolean,
): Promise<SkillActionPayload> {
  return mutation<SkillActionPayload>(transport, "skill.update", { name, enabled });
}

export async function deleteSkill(
  transport: WebUIMutationTransport,
  name: string,
): Promise<SkillActionPayload> {
  return mutation<SkillActionPayload>(transport, "skill.delete", { name });
}

export async function searchMarketplaceSkills(
  token: string,
  query: string,
  provider: MarketplaceProvider = "all",
  base: string = "",
): Promise<SkillsSearchPayload> {
  const params = new URLSearchParams({ q: query, provider });
  return request<SkillsSearchPayload>(
    `${base}/api/webui/skills/search?${params}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchTrendingMarketplaceSkills(
  token: string,
  provider: MarketplaceProvider = "all",
  base: string = "",
): Promise<SkillsTrendingPayload> {
  const params = new URLSearchParams({ provider });
  return request<SkillsTrendingPayload>(
    `${base}/api/webui/skills/trending?${params}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchMarketplaceSkillTrends(
  token: string,
  skillIds: string[],
  base: string = "",
): Promise<SkillsTrendsPayload> {
  const params = new URLSearchParams();
  skillIds.forEach((id) => params.append("id", id));
  return request<SkillsTrendsPayload>(
    `${base}/api/webui/skills/trends?${params}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function installMarketplaceSkill(
  transport: WebUIMutationTransport,
  provider: Exclude<MarketplaceProvider, "all">,
  source: string,
  skill: string,
  version: string = "",
): Promise<SkillInstallPayload> {
  return mutation<SkillInstallPayload>(
    transport,
    "skill.install",
    { provider, source, skill, ...(version ? { version } : {}) },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function deleteSession(
  transport: WebUIMutationTransport,
  key: string,
  optionsOrBase?: { deleteAutomations?: boolean } | string,
): Promise<SessionDeleteResult> {
  const options = typeof optionsOrBase === "string" ? undefined : optionsOrBase;
  return mutation<SessionDeleteResult>(
    transport,
    "session.delete",
    {
      key,
      ...(options?.deleteAutomations ? { delete_automations: true } : {}),
    },
  );
}

export async function fetchSettings(
  token: string,
  base: string = "",
): Promise<SettingsPayload> {
  return request<SettingsPayload>(
    `${base}/api/settings`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchSettingsUsage(
  token: string,
  base: string = "",
): Promise<NonNullable<SettingsPayload["usage"]>> {
  return request<NonNullable<SettingsPayload["usage"]>>(
    `${base}/api/settings/usage`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export interface VersionCheckResult {
  updateAvailable: {
    currentVersion: string;
    latestVersion: string;
    pypiUrl?: string;
  } | null;
}

export async function checkVersion(
  token: string,
  base: string = "",
): Promise<VersionCheckResult> {
  return request<VersionCheckResult>(
    `${base}/api/settings/version-check`,
    token,
    undefined,
    10_000,
  );
}

export async function fetchWorkspaces(
  token: string,
  base: string = "",
): Promise<WorkspacesPayload> {
  return request<WorkspacesPayload>(
    `${base}/api/workspaces`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function upsertWorkspaceGrant(
  transport: WebUIMutationTransport,
  grant: FolderGrantInput,
  expectedRevision: number,
): Promise<WorkspaceGrantUpsertPayload> {
  const payload = await mutation<WorkspacesPayload & { grant_receipt?: unknown }>(
    transport,
    "workspace.grant.upsert",
    {
    expected_revision: expectedRevision,
    path: grant.path,
    file_access: grant.file_access,
    commands: grant.commands,
    },
  );
  return {
    ...payload,
    grant_receipt: parseWorkspaceGrantReceipt(payload.grant_receipt, grant),
  };
}

export async function allowWorkspaceAccessOnce(
  transport: WebUIMutationTransport,
  input: WorkspaceOnceActionInput,
  expectedTaskTurnId: string,
): Promise<WorkspaceOnceActionResult> {
  const response = await mutation<unknown>(transport, "workspace.access.allow_once", {
    chat_id: input.chat_id,
    access_request_id: input.access_request_id,
    expected_revision: input.expected_revision,
    path: input.path,
    file_access: input.file_access,
    commands: input.commands,
    expected_grants_revision: input.expected_grants_revision,
    expected_runtime_id: input.expected_runtime_id,
    expected_permission_revision: input.expected_permission_revision,
  });
  const parsed = parseWorkspaceOnceResult(response, input, expectedTaskTurnId);
  if (!parsed) throw new ApiError(502, "Invalid workspace Once receipt or response");
  return parsed;
}

export async function setWorkspaceAccessMode(
  transport: WebUIMutationTransport,
  input: WorkspaceModeActionInput,
): Promise<WorkspaceModeActionResult> {
  try {
    const response = await mutation<unknown>(transport, "workspace.access.mode.set", {
      chat_id: input.chat_id,
      mode: input.mode,
      expected_runtime_id: input.expected_runtime_id,
      expected_permission_revision: input.expected_permission_revision,
    });
    const parsed = parseWorkspaceModeActionResult(response, input);
    if (!parsed) throw new ApiError(502, "Invalid workspace permission mode response");
    return parsed;
  } catch (error) {
    if (error instanceof ApiError && error.status === 409 && isObjectRecord(error.details)) {
      const rawConflict = error.details.workspaceModeConflict;
      const conflict = parseWorkspaceModeConflict(rawConflict);
      if (conflict && conflict.error.message === error.message) {
        throw new WorkspaceModeConflictError(error.message, conflict);
      }
    }
    throw error;
  }
}

export async function prepareWorkspaceFullAccess(
  transport: WebUIMutationTransport,
  chatId: string,
): Promise<WorkspaceModeActionResult> {
  const response = await mutation<unknown>(transport, "workspace.access.prepare_full", {
    chat_id: chatId,
  });
  if (!isObjectRecord(response)) {
    throw new ApiError(502, "Invalid workspace Full preparation response");
  }
  const scope = parseWorkspaceScope(response.workspace_scope);
  const permissions = normalizeWorkspacePermissionProjection(
    response.workspace_permissions,
  );
  if (
    !scope
    || !permissions
    || permissions.mode !== "restricted"
    || permissions.state !== "active"
  ) {
    throw new ApiError(502, "Invalid workspace Full preparation response");
  }
  return { workspace_scope: scope, workspace_permissions: permissions };
}

export async function revokeWorkspaceGrant(
  transport: WebUIMutationTransport,
  path: string,
  expectedRevision: number,
): Promise<WorkspacesPayload> {
  return mutation<WorkspacesPayload>(transport, "workspace.grant.revoke", {
    expected_revision: expectedRevision,
    path,
  });
}

export async function rememberRepository(
  transport: WebUIMutationTransport,
  entry: RepositoryRecord,
): Promise<WorkspacesPayload> {
  return mutation<WorkspacesPayload>(transport, "repository.remember", { entry });
}

export async function forgetRepository(
  transport: WebUIMutationTransport,
  name: string,
): Promise<WorkspacesPayload> {
  return mutation<WorkspacesPayload>(transport, "repository.forget", { name });
}

export async function discoverRepositories(
  transport: WebUIMutationTransport,
  path: string,
): Promise<RepositoryDiscoveryPayload> {
  return mutation<RepositoryDiscoveryPayload>(transport, "repository.discover", { path });
}

export async function resolveWorkspaceAccessRequest(
  transport: WebUIMutationTransport,
  chatId: string,
  accessRequestId: string,
  expectedRevision: number,
  path: string,
  expectedGrantsRevision: number,
): Promise<FolderAccessActionResult> {
  return mutation<FolderAccessActionResult>(transport, "workspace.access.resolve", {
    chat_id: chatId,
    access_request_id: accessRequestId,
    expected_revision: expectedRevision,
    path,
    expected_grants_revision: expectedGrantsRevision,
  });
}

export async function continueWorkspaceAccessRequest(
  transport: WebUIMutationTransport,
  chatId: string,
  accessRequestId: string,
  expectedRevision: number,
): Promise<FolderAccessActionResult> {
  return mutation<FolderAccessActionResult>(transport, "workspace.access.continue", {
    chat_id: chatId,
    access_request_id: accessRequestId,
    expected_revision: expectedRevision,
  });
}

export async function cancelWorkspaceAccessRequest(
  transport: WebUIMutationTransport,
  chatId: string,
  accessRequestId: string,
  expectedRevision: number,
): Promise<FolderAccessActionResult> {
  return mutation<FolderAccessActionResult>(transport, "workspace.access.cancel", {
    chat_id: chatId,
    access_request_id: accessRequestId,
    expected_revision: expectedRevision,
  });
}

export async function fetchCliApps(
  token: string,
  base: string = "",
): Promise<CliAppsPayload> {
  return request<CliAppsPayload>(
    `${base}/api/settings/cli-apps`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchInstalledCliApps(
  token: string,
  base: string = "",
): Promise<CliAppsPayload> {
  return request<CliAppsPayload>(
    `${base}/api/settings/cli-apps?installed_only=1`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchNanobotFeatures(
  token: string,
  base: string = "",
): Promise<NanobotFeaturesPayload> {
  return request<NanobotFeaturesPayload>(
    `${base}/api/settings/nanobot-features`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function fetchApiService(token: string, base: string = ""): Promise<ApiServicePayload> {
  return request<ApiServicePayload>(`${base}/api/settings/api-service`, token);
}

export async function startApiService(
  transport: WebUIMutationTransport,
  values: { host: string; port: number; timeout: number; apiKey?: string },
): Promise<ApiServicePayload> {
  return mutation<ApiServicePayload>(
    transport,
    "settings.api_service.start",
    {
      host: values.host,
      port: values.port,
      timeout: values.timeout,
      ...(values.apiKey !== undefined ? { api_key: values.apiKey } : {}),
    },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function stopApiService(
  transport: WebUIMutationTransport,
): Promise<ApiServicePayload> {
  return mutation<ApiServicePayload>(transport, "settings.api_service.stop");
}

export async function enableNanobotFeature(
  transport: WebUIMutationTransport,
  name: string,
  options: { instanceId?: string } = {},
): Promise<NanobotFeaturesPayload> {
  return mutation<NanobotFeaturesPayload>(
    transport,
    "settings.feature.enable",
    { name, ...(options.instanceId ? { instance_id: options.instanceId } : {}) },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function disableNanobotFeature(
  transport: WebUIMutationTransport,
  name: string,
  options: { instanceId?: string } = {},
): Promise<NanobotFeaturesPayload> {
  return mutation<NanobotFeaturesPayload>(
    transport,
    "settings.feature.disable",
    { name, ...(options.instanceId ? { instance_id: options.instanceId } : {}) },
  );
}

export async function fetchPairingRequests(
  token: string,
  base: string = "",
): Promise<PairingPayload> {
  return request<PairingPayload>(
    `${base}/api/settings/pairing`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function runPairingAction(
  transport: WebUIMutationTransport,
  action: "approve" | "deny",
  code: string,
): Promise<PairingPayload> {
  return mutation<PairingPayload>(transport, `settings.pairing.${action}`, { code });
}

export async function startChannelConnect(
  transport: WebUIMutationTransport,
  channel: string,
  options: {
    domain?: string;
    instanceId?: string;
    mode?: "replace" | "create";
    force?: boolean;
  } = {},
): Promise<ChannelConnectPayload> {
  return mutation<ChannelConnectPayload>(
    transport,
    "settings.channel.connect.start",
    {
      channel,
      ...(options.domain ? { domain: options.domain } : {}),
      ...(options.instanceId ? { instance_id: options.instanceId } : {}),
      ...(options.mode ? { mode: options.mode } : {}),
      ...(options.force ? { force: true } : {}),
    },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function pollChannelConnect(
  transport: WebUIMutationTransport,
  channel: string,
  sessionId: string,
  params: Readonly<Record<string, string>> = {},
): Promise<ChannelConnectPayload> {
  const values = Object.fromEntries(
    Object.entries(params).filter(([key]) => key !== "session_id"),
  );
  return mutation<ChannelConnectPayload>(
    transport,
    "settings.channel.connect.poll",
    { channel, session_id: sessionId, ...values },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function cancelChannelConnect(
  transport: WebUIMutationTransport,
  channel: string,
  sessionId: string,
): Promise<ChannelConnectPayload> {
  return mutation<ChannelConnectPayload>(
    transport,
    "settings.channel.connect.cancel",
    { channel, session_id: sessionId },
  );
}

export async function configureChannel(
  transport: WebUIMutationTransport,
  name: string,
  values: Record<string, string>,
  options: { enable?: boolean; instanceId?: string } = {},
): Promise<ChannelConfigurePayload> {
  return mutation<ChannelConfigurePayload>(
    transport,
    "settings.channel.configure",
    {
      name,
      values,
      ...(options.enable !== undefined ? { enable: options.enable } : {}),
      ...(options.instanceId ? { instance_id: options.instanceId } : {}),
    },
    PACKAGE_MUTATION_TIMEOUT_MS,
  );
}

export async function validateChannel(
  transport: WebUIMutationTransport,
  name: string,
  values: Record<string, string> = {},
  options: { instanceId?: string } = {},
): Promise<ChannelValidationPayload> {
  return mutation<ChannelValidationPayload>(
    transport,
    "settings.channel.validate",
    { name, values, ...(options.instanceId ? { instance_id: options.instanceId } : {}) },
  );
}

export async function runCliAppAction(
  transport: WebUIMutationTransport,
  action: "install" | "update" | "uninstall" | "test",
  name: string,
): Promise<CliAppsPayload> {
  return mutation<CliAppsPayload>(
    transport,
    `settings.cli_app.${action}`,
    { name },
    action === "install" || action === "update"
      ? PACKAGE_MUTATION_TIMEOUT_MS
      : API_MUTATION_TIMEOUT_MS,
  );
}

export async function fetchMcpPresets(
  token: string,
  base: string = "",
): Promise<McpPresetsPayload> {
  return request<McpPresetsPayload>(
    `${base}/api/settings/mcp-presets`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function startMcpOAuth(
  transport: WebUIMutationTransport,
  name: string,
  reset: boolean = false,
): Promise<McpOAuthFlowPayload> {
  return mutation<McpOAuthFlowPayload>(
    transport,
    "settings.mcp.oauth_start",
    { name, ...(reset ? { reset: true } : {}) },
    30_000,
  );
}

export async function fetchMcpOAuthStatus(
  token: string,
  flowId: string,
  base: string = "",
): Promise<McpOAuthFlowPayload> {
  const query = new URLSearchParams({ flow_id: flowId });
  return request<McpOAuthFlowPayload>(
    `${base}/api/settings/mcp-oauth/status?${query}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function completeMcpOAuth(
  transport: WebUIMutationTransport,
  flowId: string,
  callbackUrl: string,
): Promise<McpOAuthFlowPayload> {
  return mutation<McpOAuthFlowPayload>(
    transport,
    "settings.mcp.oauth_complete",
    { flow_id: flowId, callback_url: callbackUrl },
  );
}

export async function cancelMcpOAuth(
  transport: WebUIMutationTransport,
  flowId: string,
): Promise<McpOAuthFlowPayload> {
  return mutation<McpOAuthFlowPayload>(
    transport,
    "settings.mcp.oauth_cancel",
    { flow_id: flowId },
  );
}

export async function fetchProviderModels(
  token: string,
  provider: string,
  base: string = "",
): Promise<ProviderModelsPayload> {
  const query = new URLSearchParams();
  query.set("provider", provider);
  return request<ProviderModelsPayload>(
    `${base}/api/settings/provider-models?${query}`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function runMcpPresetAction(
  transport: WebUIMutationTransport,
  action: "enable" | "disable" | "remove" | "test" | "reconnect",
  name: string,
  values: Record<string, string> = {},
): Promise<McpPresetsPayload> {
  return mutation<McpPresetsPayload>(
    transport,
    `settings.mcp.${action}`,
    { name, ...compactMcpValues(values) },
  );
}

export async function saveCustomMcpServer(
  transport: WebUIMutationTransport,
  values: Record<string, string>,
): Promise<McpPresetsPayload> {
  return mutation<McpPresetsPayload>(
    transport,
    "settings.mcp.custom",
    compactMcpValues(values),
  );
}

export async function importMcpConfig(
  transport: WebUIMutationTransport,
  config: string,
): Promise<McpPresetsPayload> {
  return mutation<McpPresetsPayload>(transport, "settings.mcp.import", { config });
}

export async function updateMcpServerTools(
  transport: WebUIMutationTransport,
  name: string,
  enabledTools: string[],
): Promise<McpPresetsPayload> {
  return mutation<McpPresetsPayload>(
    transport,
    "settings.mcp.tools",
    { name, enabled_tools: enabledTools },
  );
}

export async function listSlashCommands(
  token: string,
  base: string = "",
): Promise<SlashCommand[]> {
  type Row = {
    command: string;
    title: string;
    description: string;
    icon: string;
    arg_hint?: string;
    lifecycle?: unknown;
    accepts_args?: unknown;
  };
  const body = await request<{ commands: Row[] }>(
    `${base}/api/commands`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
  return body.commands
    .flatMap((command) => {
      if (!isSlashCommandLifecycle(command.lifecycle)) return [];
      return [{
        command: command.command,
        title: command.title,
        description: command.description,
        icon: command.icon,
        argHint: command.arg_hint ?? "",
        lifecycle: command.lifecycle,
        acceptsArgs: command.accepts_args === true,
      }];
    });
}

export async function fetchSidebarState(
  token: string,
  base: string = "",
): Promise<SidebarStatePayload> {
  return request<SidebarStatePayload>(
    `${base}/api/webui/sidebar-state`,
    token,
    undefined,
    API_READ_TIMEOUT_MS,
  );
}

export async function updateSidebarState(
  transport: WebUIMutationTransport,
  state: SidebarStatePayload,
): Promise<SidebarStatePayload> {
  return mutation<SidebarStatePayload>(transport, "sidebar.update", { state });
}

function modelGenerationSettingsPayload(
  configuration: Pick<
    ModelConfigurationCreate,
    "maxTokens" | "contextWindowTokens" | "temperature" | "reasoningEffort"
  >,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  if (configuration.maxTokens !== undefined) {
    payload.max_tokens = configuration.maxTokens;
  }
  if (configuration.contextWindowTokens !== undefined) {
    payload.context_window_tokens = configuration.contextWindowTokens;
  }
  if (configuration.temperature !== undefined) {
    payload.temperature = configuration.temperature;
  }
  if (configuration.reasoningEffort !== undefined) {
    payload.reasoning_effort = configuration.reasoningEffort ?? "";
  }
  return payload;
}

export async function createModelConfiguration(
  transport: WebUIMutationTransport,
  configuration: ModelConfigurationCreate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.model_configuration.create",
    {
      name: configuration.name,
      provider: configuration.provider,
      model: configuration.model,
      ...modelGenerationSettingsPayload(configuration),
    },
  );
}

export async function updateModelConfiguration(
  transport: WebUIMutationTransport,
  configuration: ModelConfigurationUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.model_configuration.update",
    {
      name: configuration.name,
      ...(configuration.newName !== undefined ? { new_name: configuration.newName } : {}),
      ...(configuration.provider !== undefined ? { provider: configuration.provider } : {}),
      ...(configuration.model !== undefined ? { model: configuration.model } : {}),
      ...modelGenerationSettingsPayload(configuration),
    },
  );
}

export async function deleteModelConfiguration(
  transport: WebUIMutationTransport,
  name: string,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.model_configuration.delete",
    { name },
  );
}

export async function migrateModelConfigurations(
  transport: WebUIMutationTransport,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(transport, "settings.model_configuration.migrate");
}

export async function updateModelCallOrder(
  transport: WebUIMutationTransport,
  order: string[],
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(transport, "settings.model_call_order.update", { order });
}

export async function updateProviderSettings(
  transport: WebUIMutationTransport,
  update: ProviderSettingsUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(transport, "settings.provider.update", { ...update });
}

export async function createProviderSettings(
  transport: WebUIMutationTransport,
  update: ProviderCreationUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(transport, "settings.provider.create", { ...update });
}

export async function loginProviderOAuth(
  transport: WebUIMutationTransport,
  provider: string,
  remoteBrowserAccess: boolean = false,
): Promise<ProviderOAuthLoginResult> {
  return mutation<ProviderOAuthLoginResult>(
    transport,
    "settings.provider.oauth_login",
    { provider, ...(remoteBrowserAccess ? { remote_browser: true } : {}) },
  );
}

export async function completeProviderOAuth(
  transport: WebUIMutationTransport,
  provider: string,
  flowId: string,
  authorizationResponse?: string,
): Promise<ProviderOAuthCompletionResult> {
  return mutation<ProviderOAuthCompletionResult>(
    transport,
    "settings.provider.oauth_complete",
    {
      provider,
      flow_id: flowId,
      ...(authorizationResponse ? { authorization_response: authorizationResponse } : {}),
    },
  );
}

export async function logoutProviderOAuth(
  transport: WebUIMutationTransport,
  provider: string,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(transport, "settings.provider.oauth_logout", { provider });
}

export async function updateWebSearchSettings(
  transport: WebUIMutationTransport,
  update: WebSearchSettingsUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.web_search.update",
    {
      provider: update.provider,
      ...(update.apiKey !== undefined ? { api_key: update.apiKey } : {}),
      ...(update.baseUrl !== undefined ? { base_url: update.baseUrl } : {}),
      ...(update.maxResults !== undefined ? { max_results: update.maxResults } : {}),
      ...(update.timeout !== undefined ? { timeout: update.timeout } : {}),
      ...(update.useJinaReader !== undefined
        ? { use_jina_reader: update.useJinaReader }
        : {}),
      ...(update.fetchProvider !== undefined
        ? { fetch_provider: update.fetchProvider }
        : {}),
      ...(update.browserCdpUrl !== undefined
        ? { browser_cdp_url: update.browserCdpUrl }
        : {}),
      ...(update.browserPinTab !== undefined
        ? { browser_pin_tab: update.browserPinTab }
        : {}),
      ...(update.browserSession !== undefined
        ? { browser_session: update.browserSession }
        : {}),
      ...(update.browserLaunchCommand !== undefined
        ? { browser_launch_command: update.browserLaunchCommand }
        : {}),
      ...(update.browserLaunchWaitSeconds !== undefined
        ? { browser_launch_wait_seconds: update.browserLaunchWaitSeconds }
        : {}),
    },
  );
}

export async function updateNetworkSafetySettings(
  transport: WebUIMutationTransport,
  update: NetworkSafetySettingsUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.network_safety.update",
    {
      webui_allow_local_service_access: update.webuiAllowLocalServiceAccess,
      webui_default_access_mode: update.webuiDefaultAccessMode,
    },
  );
}

export async function updateImageGenerationSettings(
  transport: WebUIMutationTransport,
  update: ImageGenerationSettingsUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.image_generation.update",
    {
      enabled: update.enabled,
      provider: update.provider,
      model: update.model,
      default_aspect_ratio: update.defaultAspectRatio,
      default_image_size: update.defaultImageSize,
      max_images_per_turn: update.maxImagesPerTurn,
    },
  );
}

export async function updateTranscriptionSettings(
  transport: WebUIMutationTransport,
  update: TranscriptionSettingsUpdate,
): Promise<SettingsPayload> {
  return mutation<SettingsPayload>(
    transport,
    "settings.transcription.update",
    {
      enabled: update.enabled,
      provider: update.provider,
      model: update.model,
      language: update.language,
      max_duration_sec: update.maxDurationSec,
      max_upload_mb: update.maxUploadMb,
    },
  );
}
