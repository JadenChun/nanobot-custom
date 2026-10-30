import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FolderAccessRequestCard } from "@/components/thread/FolderAccessRequestCard";
import { folderAccessActionResultFixture, folderAccessRequestFixture } from "@/tests/fixtures/folder-access-request";
import type {
  FolderAccessRequest,
  WorkspacePermissionProjection,
  WorkspacesPayload,
  WorkspaceScopePayload,
} from "@/lib/types";
import { ApiError } from "@/lib/api";
import { ClientProvider } from "@/providers/ClientProvider";

function workspaces(): WorkspacesPayload {
  return {
    schema_version: 2,
    default_access_mode: "restricted",
    default_scope: {
      project_path: "/agent/home",
      project_name: "Agent home",
      access_mode: "restricted",
      effective_access: {
        policy: "managed",
        source: "agent_home",
        grant_root: null,
        exists: true,
        read: true,
        write: true,
        boundary: "folder",
        commands: { requested: false, available: false, backend: "none", reason: null },
      },
    },
    managed_access: true,
    controls: {
      can_change_project: true,
      can_use_full_access: false,
      can_manage_folders: true,
      can_manage_catalog: true,
      can_discover_repositories: true,
    },
    management: { grants_revision: 12, grants: [], repositories: [
      {
        name: "freelance",
        path: "/projects/freelance",
        purpose: "Current work",
        kind: "project",
        knowledge_repos: [],
        auto_sync: false,
        sync_paths: [],
        exists: true,
        effective_access: {
          policy: "managed",
          source: "grant",
          grant_root: "/projects/freelance",
          exists: true,
          read: true,
          write: false,
          boundary: "folder",
          commands: { requested: false, available: false, backend: "none", reason: null },
        },
      },
    ] },
  };
}

function restrictedPermissionProjection(): WorkspacePermissionProjection {
  return {
    version: 1,
    runtime_id: "a".repeat(32),
    revision: 4,
    mode: "restricted",
    state: "active",
    can_enable_full: true,
    full_access_unavailable_reason: null,
    task_access: null,
    retirement: null,
  };
}

function onceActionResult(
  request: FolderAccessRequest,
  path: string,
  fileAccess: "read_only" | "read_write",
  commands: boolean,
  canonicalPath = path,
) {
  const access = {
    policy: "managed" as const,
    source: "task" as const,
    grant_root: canonicalPath,
    exists: true,
    read: true,
    write: fileAccess === "read_write",
    boundary: "folder" as const,
    commands: {
      requested: commands,
      available: commands,
      backend: commands ? "native" as const : "none" as const,
      reason: null,
    },
  };
  const revision = request.revision + 1;
  return {
    workspace_access_request: {
      ...request,
      revision,
      status: "ready" as const,
      selected_path: canonicalPath,
      selected_access: access,
      can_continue: true,
      selected_duration: "once" as const,
      task_access_state: "active" as const,
    },
    workspace_scope: {
      ...folderAccessActionResultFixture.workspace_scope,
      project_path: canonicalPath,
      project_name: canonicalPath.split(/[\\/]/).filter(Boolean).at(-1) ?? canonicalPath,
      access_mode: "restricted" as const,
      effective_access: access,
    },
    workspace_permissions: {
      ...restrictedPermissionProjection(),
      revision: 5,
      task_access: {
        task_turn_id: request.task.turn_id,
        state: "active" as const,
        folders: [{ path: canonicalPath, file_access: fileAccess, commands }],
      },
    },
    once_receipt: {
      version: 1 as const,
      duration: "once" as const,
      runtime_id: "a".repeat(32),
      permission_revision: 5,
      access_request_id: request.access_request_id,
      request_revision: revision,
      task_turn_id: request.task.turn_id,
      submitted_path: path,
      canonical_path: canonicalPath,
      file_access: fileAccess,
      commands,
    },
  };
}

function renderCard({
  request = folderAccessRequestFixture,
  requestMutation = vi.fn(async () => folderAccessActionResultFixture),
  workspacesPayload = workspaces(),
  chatId = "chat-title",
  requestLoading = false,
  requestRefreshFailed = false,
  workspacePermissions = null,
  onRefreshRequest,
  onApplyWorkspaceScope,
  onPickFolder,
}: {
  request?: FolderAccessRequest | null;
  requestMutation?: ReturnType<typeof vi.fn>;
  workspacesPayload?: WorkspacesPayload | null;
  chatId?: string | null;
  requestLoading?: boolean;
  requestRefreshFailed?: boolean;
  workspacePermissions?: WorkspacePermissionProjection | null;
  onRefreshRequest?: () => Promise<FolderAccessRequest | null>;
  onApplyWorkspaceScope?: (scope: WorkspaceScopePayload) => void;
  onPickFolder?: () => Promise<string | null>;
} = {}) {
  const onUpdateRequest = vi.fn();
  const refreshRequest = onRefreshRequest ?? vi.fn(async () => folderAccessRequestFixture);
  const applyWorkspaceScope = onApplyWorkspaceScope ?? vi.fn();
  const onRefreshWorkspaces = vi.fn(async () => {});
  const renderCard = (overrides: Partial<{
    chatId: string | null;
    request: FolderAccessRequest | null;
    requestLoading: boolean;
    requestRefreshFailed: boolean;
  }> = {}) => (
    <ClientProvider client={{ requestMutation } as never} token="test-token">
      <FolderAccessRequestCard
        chatId={overrides.chatId ?? chatId}
        request={overrides.request === undefined ? request : overrides.request}
        requestLoading={overrides.requestLoading ?? requestLoading}
        requestRefreshFailed={overrides.requestRefreshFailed ?? requestRefreshFailed}
        busy={false}
        workspaces={workspacesPayload}
        workspacePermissions={workspacePermissions}
        onUpdateRequest={onUpdateRequest}
        onRefreshRequest={refreshRequest}
        onApplyWorkspaceScope={applyWorkspaceScope}
        onRefreshWorkspaces={onRefreshWorkspaces}
        onPickFolder={onPickFolder}
        onRetryRequestRefresh={async () => {
          await Promise.all([refreshRequest(), onRefreshWorkspaces()]);
        }}
      />
    </ClientProvider>
  );
  const ui = render(renderCard());
  return {
    requestMutation,
    onUpdateRequest,
    onRefreshRequest: refreshRequest,
    onApplyWorkspaceScope: applyWorkspaceScope,
    onRefreshWorkspaces,
    unmount: () => ui.unmount(),
    rerenderCard: (overrides: Parameters<typeof renderCard>[0]) => ui.rerender(renderCard(overrides)),
  };
}

describe("FolderAccessRequestCard", () => {
  it("accepts a Once receipt whose submitted home path resolves to its canonical project path", async () => {
    const user = userEvent.setup();
    const submittedPath = "~/Documents/Client project";
    const canonicalPath = "/home/jaden/Documents/Client project";
    const request = { ...folderAccessRequestFixture };
    const mutation = vi.fn(async () => onceActionResult(
      request,
      submittedPath,
      "read_only",
      false,
      canonicalPath,
    ));
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const { onUpdateRequest } = renderCard({
      request,
      requestMutation: mutation,
      workspacesPayload: v2Workspaces,
      workspacePermissions: restrictedPermissionProjection(),
    });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.type(screen.getByLabelText("Folder path"), submittedPath);
    await user.click(screen.getByRole("button", { name: "Allow once and use folder" }));

    await waitFor(() => expect(onUpdateRequest).toHaveBeenCalledWith(expect.objectContaining({ selected_path: canonicalPath })));
    expect(screen.queryByText("The folder action did not finish. Refresh the task and try again.")).not.toBeInTheDocument();
    expect(mutation).toHaveBeenCalledWith("workspace.access.allow_once", expect.objectContaining({
      path: submittedPath,
    }), 20_000);
    expect(onUpdateRequest).toHaveBeenCalledWith(expect.objectContaining({ selected_path: canonicalPath }));
  });

  it("describes draft command access as requested until backend availability is known", () => {
    renderCard({ request: { ...folderAccessRequestFixture, operation: "execute" } });

    expect(screen.getByText("Selected access: Read only · Commands follow Sandbox file rules. Read-only folders stay read-only for programs and their child processes.")).toBeInTheDocument();
    expect(screen.queryByText("Selected access: Read only · Commands available")).not.toBeInTheDocument();
  });

  it("keeps an adequate saved grant when directly choosing Always for its known folder", async () => {
    const user = userEvent.setup();
    const request: FolderAccessRequest = {
      ...folderAccessRequestFixture,
      target: { kind: "directory", path: "/projects/freelance" },
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    v2Workspaces.management!.grants = [{
      path: "/projects/freelance",
      file_access: "read_write",
      commands: true,
      exists: true,
      effective_access: {
        policy: "managed",
        source: "grant",
        grant_root: "/projects/freelance",
        exists: true,
        read: true,
        write: true,
        boundary: "folder",
        commands: { requested: true, available: true, backend: "native", reason: null },
      },
    }];
    const mutation = vi.fn(async () => folderAccessActionResultFixture);
    renderCard({
      request,
      requestMutation: mutation,
      workspacesPayload: v2Workspaces,
      workspacePermissions: restrictedPermissionProjection(),
    });

    await waitFor(() => expect(screen.getByText(/Selected access: Read and write/)).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Always allow" }));

    await waitFor(() => expect(mutation).toHaveBeenCalledWith("workspace.access.resolve", expect.objectContaining({
      path: "/projects/freelance",
    }), 20_000));
    expect(mutation).not.toHaveBeenCalledWith("workspace.grant.upsert", expect.anything(), expect.anything());
    expect(mutation.mock.calls.map(([action]) => action)).toEqual(["workspace.access.resolve"]);
  });

  it("does not inherit an existing grant permission while typing past its path prefix", async () => {
    const user = userEvent.setup();
    const v2Workspaces = workspaces();
    v2Workspaces.management!.grants = [{
      path: "/projects/foo",
      file_access: "read_write",
      commands: true,
      exists: true,
      effective_access: {
        policy: "managed",
        source: "grant",
        grant_root: "/projects/foo",
        exists: true,
        read: true,
        write: true,
        boundary: "folder",
        commands: { requested: true, available: true, backend: "native", reason: null },
      },
    }];
    renderCard({
      request: { ...folderAccessRequestFixture, operation: "read" },
      workspacesPayload: v2Workspaces,
    });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/foobar");

    expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Read only" })).toBeChecked();
  });

  it("shows the insufficient-capability guidance once for a ready task", () => {
    const request: FolderAccessRequest = {
      ...folderAccessActionResultFixture.workspace_access_request,
      operation: "execute",
      selected_access: {
        ...folderAccessActionResultFixture.workspace_access_request.selected_access!,
        commands: {
          requested: true,
          available: false,
          backend: "none",
          reason: "not_granted",
        },
      },
    };
    renderCard({ request });
    const guidance = "The selected access for /projects/freelance cannot satisfy this request. Choose another folder or review its access.";

    expect(screen.getAllByText(guidance)).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent(guidance);
  });

  it("shows a known folder and purpose inline and allows it directly with the required write access", async () => {
    const user = userEvent.setup();
    const request: FolderAccessRequest = {
      ...folderAccessRequestFixture,
      operation: "write",
      target: { kind: "directory", path: "/projects/freelance" },
      purpose: "Save the generated inventory report.",
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const mutation = vi.fn(async () => onceActionResult(request, "/projects/freelance", "read_write", false));
    renderCard({
      request,
      requestMutation: mutation,
      workspacesPayload: v2Workspaces,
      workspacePermissions: restrictedPermissionProjection(),
    });

    expect(screen.getByLabelText("Requested folder")).toHaveTextContent("/projects/freelance");
    expect(screen.getByText("Purpose: Save the generated inventory report.")).toBeInTheDocument();
    expect(screen.getByText(/Required: read and write file access/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Allow for this task" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Always allow" })).toBeEnabled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Allow for this task" }));

    await waitFor(() => expect(mutation).toHaveBeenCalledWith("workspace.access.allow_once", {
      chat_id: "chat-title",
      access_request_id: request.access_request_id,
      expected_revision: request.revision,
      path: "/projects/freelance",
      file_access: "read_write",
      commands: false,
      expected_grants_revision: 12,
      expected_runtime_id: "a".repeat(32),
      expected_permission_revision: 4,
    }, 20_000));
    expect(mutation).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
  });

  it.each([
    ["folder_access_required", "Nanobot cannot use that folder with the current access settings."],
    ["folder grants cannot overlap private runtime storage", "This folder includes Nanobot's private runtime files."],
  ])("keeps a safe explanation for direct approval error %s", async (message, expected) => {
    const user = userEvent.setup();
    const request: FolderAccessRequest = {
      ...folderAccessRequestFixture,
      target: { kind: "directory", path: "/projects/freelance" },
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const mutation = vi.fn(async () => {
      throw new ApiError(message === "folder_access_required" ? 403 : 400, message);
    });
    renderCard({
      request,
      requestMutation: mutation,
      workspacesPayload: v2Workspaces,
      workspacePermissions: restrictedPermissionProjection(),
    });

    await user.click(screen.getByRole("button", { name: "Allow for this task" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(expected);
    expect(screen.getByLabelText("Requested folder")).toHaveTextContent("/projects/freelance");
    if (message === "folder grants cannot overlap private runtime storage") {
      expect(screen.getByText(/includes Nanobot's private runtime files/)).toBeInTheDocument();
    } else {
      expect(screen.queryByText(/includes Nanobot's private runtime files/)).not.toBeInTheDocument();
    }
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
  });

  it("keeps the selected path and request after an invalid Once path error in the alternative form", async () => {
    const user = userEvent.setup();
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const mutation = vi.fn(async () => {
      throw new ApiError(400, "invalid_access_request");
    });
    renderCard({
      requestMutation: mutation,
      workspacesPayload: v2Workspaces,
      workspacePermissions: restrictedPermissionProjection(),
    });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/freelance");
    await user.click(screen.getByRole("button", { name: "Allow once and use folder" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Nanobot could not verify that folder. Choose an existing folder and try again.",
    );
    expect(screen.getByLabelText("Folder path")).toHaveValue("/projects/freelance");
    expect(screen.getByRole("dialog", { name: "Review folder access" })).toBeInTheDocument();
    expect(screen.getByText("Check the website and inventory projects in Freelance.")).toBeInTheDocument();
  });

  it.each([
    ["read", "Read only", false],
    ["write", "Read and write", false],
    ["execute", "Read only", true],
  ] as const)("initializes %s access to the minimum requested capabilities", async (operation, fileAccess, commands) => {
    const user = userEvent.setup();
    renderCard({ request: { ...folderAccessRequestFixture, operation } });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));

    expect(screen.getByRole("radio", { name: fileAccess })).toBeChecked();
    if (commands) {
      expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();
    } else {
      expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();
    }
  });

  it.each([
    ["/projects/freelance/README.md", "/projects/freelance"],
    ["/README.md", "/"],
    ["C:\\projects\\freelance\\README.md", "C:\\projects\\freelance"],
    ["C:\\README.md", "C:\\"],
    ["\\\\server\\share\\README.md", "\\\\server\\share"],
  ])("shows the file %s and opens review with its containing folder %s", async (filePath, folderPath) => {
    const user = userEvent.setup();
    renderCard({ request: {
      ...folderAccessRequestFixture,
      target: { kind: "file", path: filePath },
    } });

    expect(screen.getByLabelText("Requested file")).toHaveTextContent(filePath);
    expect(screen.getByLabelText("Folder to authorize")).toHaveTextContent(folderPath);
    await user.click(screen.getByRole("button", { name: "Review access" }));
    expect(screen.getByLabelText("Folder path")).toHaveValue(folderPath);
  });

  it("defaults review to Allow once and exposes Always allow as the other duration", async () => {
    const user = userEvent.setup();
    const legacyWorkspaces = workspaces();
    renderCard({
      workspacesPayload: {
        ...legacyWorkspaces,
        controls: { ...legacyWorkspaces.controls, permission_v2: true },
      },
      workspacePermissions: restrictedPermissionProjection(),
    });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));

    const once = await screen.findByRole("radio", { name: /Allow once/ });
    const always = screen.getByRole("radio", { name: /Always allow/ });
    expect(once).toBeChecked();
    expect(always).not.toBeChecked();
    expect(screen.queryByRole("radio", { name: /chat duration|this chat/i })).not.toBeInTheDocument();
  });

  it("uses the correlated Once action and receipt without saving a reusable grant", async () => {
    const user = userEvent.setup();
    const permissions: WorkspacePermissionProjection = {
      ...restrictedPermissionProjection(),
      revision: 5,
      task_access: {
        task_turn_id: folderAccessRequestFixture.task.turn_id,
        state: "active",
        folders: [{ path: "/projects/freelance", file_access: "read_only", commands: false }],
      },
    };
    const selectedAccess = {
      policy: "managed" as const,
      source: "task" as const,
      grant_root: "/projects/freelance",
      exists: true,
      read: true,
      write: false,
      boundary: "folder" as const,
      commands: { requested: false, available: false, backend: "none" as const, reason: null },
    };
    const onceResult = {
      workspace_access_request: {
        ...folderAccessRequestFixture,
        revision: 2,
        status: "ready" as const,
        selected_path: "/projects/freelance",
        selected_access: selectedAccess,
        can_continue: true,
        selected_duration: "once" as const,
        task_access_state: "active" as const,
      },
      workspace_scope: {
        project_path: "/projects/freelance",
        project_name: "freelance",
        access_mode: "restricted" as const,
        restrict_to_workspace: true,
        effective_access: selectedAccess,
      },
      workspace_permissions: permissions,
      once_receipt: {
        version: 1 as const,
        duration: "once" as const,
        runtime_id: "a".repeat(32),
        permission_revision: 5,
        access_request_id: folderAccessRequestFixture.access_request_id,
        request_revision: 2,
        task_turn_id: folderAccessRequestFixture.task.turn_id,
        submitted_path: "/projects/freelance",
        canonical_path: "/projects/freelance",
        file_access: "read_only" as const,
        commands: false,
      },
    };
    const mutation = vi.fn(async () => onceResult);
    const legacyWorkspaces = workspaces();
    const { onApplyWorkspaceScope, onUpdateRequest } = renderCard({
      requestMutation: mutation,
      workspacesPayload: {
        ...legacyWorkspaces,
        controls: { ...legacyWorkspaces.controls, permission_v2: true },
      },
      workspacePermissions: restrictedPermissionProjection(),
    });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/freelance");
    expect(screen.getByRole("radio", { name: /Allow once/ })).toBeChecked();
    await user.click(screen.getByRole("button", { name: "Allow once and use folder" }));

    await waitFor(() => expect(mutation).toHaveBeenCalledTimes(1));
    expect(mutation).toHaveBeenCalledWith("workspace.access.allow_once", {
      chat_id: "chat-title",
      access_request_id: folderAccessRequestFixture.access_request_id,
      expected_revision: 1,
      path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
      expected_grants_revision: 12,
      expected_runtime_id: "a".repeat(32),
      expected_permission_revision: 4,
    }, 20_000);
    expect(onApplyWorkspaceScope).toHaveBeenCalledWith(onceResult.workspace_scope);
    expect(onUpdateRequest).toHaveBeenCalledWith(onceResult.workspace_access_request);
    expect(mutation.mock.calls[0]?.[0]).not.toBe("workspace.grant.upsert");
  });

  it("keeps an unknown request pending until the user reviews and explicitly saves access", async () => {
    const user = userEvent.setup();
    const mutation = vi.fn(async (action: string) => action === "workspace.grant.upsert"
      ? {
          management: {
            grants_revision: 13,
            grants: [{
              path: "/projects/freelance",
              file_access: "read_only",
              commands: false,
              exists: true,
              effective_access: folderAccessActionResultFixture.workspace_access_request.selected_access!,
            }],
            repositories: [],
          },
          grant_receipt: {
            submitted_path: "/projects/freelance",
            canonical_path: "/projects/freelance",
            file_access: "read_only",
            commands: false,
            grants_revision: 13,
          },
        }
      : folderAccessActionResultFixture);
    const { onUpdateRequest, onApplyWorkspaceScope, onRefreshWorkspaces } = renderCard({ requestMutation: mutation });

    expect(screen.getByText("Check the website and inventory projects in Freelance.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    expect(screen.getByRole("radio", { name: "Read only" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Read and write" })).not.toBeChecked();
    expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();
    expect(screen.getByText(/normal chats in this Nanobot instance/i)).toBeInTheDocument();
    await user.type(screen.getByLabelText("Folder path"), "/projects/freelance");
    expect(screen.getByLabelText("Selected folder path")).toHaveTextContent("/projects/freelance");
    await user.click(screen.getByRole("button", { name: "Save access and use folder" }));

    await waitFor(() => expect(mutation).toHaveBeenNthCalledWith(1, "workspace.grant.upsert", {
      expected_revision: 12,
      path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
    }, 20_000));
    await waitFor(() => expect(mutation).toHaveBeenNthCalledWith(2, "workspace.access.resolve", {
      chat_id: "chat-title",
      access_request_id: folderAccessRequestFixture.access_request_id,
      expected_revision: folderAccessRequestFixture.revision,
      path: "/projects/freelance",
      expected_grants_revision: 13,
    }, 20_000));
    expect(onApplyWorkspaceScope).toHaveBeenCalledWith(folderAccessActionResultFixture.workspace_scope);
    expect(onUpdateRequest).toHaveBeenCalledWith(folderAccessActionResultFixture.workspace_access_request);
    expect(onRefreshWorkspaces).toHaveBeenCalled();
    expect(onUpdateRequest.mock.calls.at(-1)?.[0]).toMatchObject({ status: "ready", can_continue: true });
    expect(mutation).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["missing receipt", undefined],
    ["wrong submitted path", {
      submitted_path: "/projects/other-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
      grants_revision: 13,
    }],
    ["wrong file access", {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_write",
      commands: false,
      grants_revision: 13,
    }],
    ["wrong command request", {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_only",
      commands: true,
      grants_revision: 13,
    }],
    ["non-absolute canonical path", {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "projects/freelance",
      file_access: "read_only",
      commands: false,
      grants_revision: 13,
    }],
    ["invalid committed revision", {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
      grants_revision: -1,
    }],
    ["extra receipt field", {
      submitted_path: "/projects/freelance-alias",
      canonical_path: "/projects/freelance",
      file_access: "read_only",
      commands: false,
      grants_revision: 13,
      inferred_from_global_rows: true,
    }],
  ])("keeps saved access unresolved for a %s", async (_caseName, receipt) => {
    const user = userEvent.setup();
    const mutation = vi.fn(async (action: string) => action === "workspace.grant.upsert"
      ? {
          management: {
            grants_revision: 14,
            grants: [{
              path: "/projects/freelance",
              file_access: "read_only",
              commands: false,
              exists: true,
              effective_access: folderAccessActionResultFixture.workspace_access_request.selected_access!,
            }],
            repositories: [],
          },
          ...(receipt === undefined ? {} : { grant_receipt: receipt }),
        }
      : folderAccessActionResultFixture);
    renderCard({ requestMutation: mutation });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/freelance-alias");
    await user.click(screen.getByRole("button", { name: "Save access and use folder" }));

    expect(await screen.findByText(
      "Folder access was saved, but this task is still waiting for you to use the folder.",
    )).toBeInTheDocument();
    expect(screen.getByText(
      "Nanobot could not confirm which folder this task should use. Refresh the task before trying again.",
    )).toBeInTheDocument();
    expect(mutation).toHaveBeenCalledTimes(1);
  });

  it("shows a single ready action set and opens review for a replacement folder", async () => {
    const user = userEvent.setup();
    const ready = { ...folderAccessActionResultFixture.workspace_access_request };
    const { requestMutation, onApplyWorkspaceScope, onUpdateRequest } = renderCard({ request: ready });

    expect(screen.getByRole("heading", { name: "Ready to continue" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue task" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose another folder" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Cancel task" })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Choose folder" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Use an allowed project")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Choose another folder" }));
    expect(screen.getByRole("dialog", { name: "Review folder access" })).toBeInTheDocument();
    expect(screen.getByLabelText("Folder path")).toHaveValue("");
    await user.click(screen.getByRole("button", { name: "Cancel", exact: true }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(requestMutation).not.toHaveBeenCalled();
    expect(onApplyWorkspaceScope).not.toHaveBeenCalled();
    expect(onUpdateRequest).not.toHaveBeenCalled();
  });

  it.each([
    ["once", "Access allowed for this task. Ready to continue using /projects/freelance."],
    ["always", "Saved access. Ready to continue using /projects/freelance."],
  ] as const)("shows the canonical selected folder without its old unknown-target helper for Ready %s", (duration, status) => {
    const ready: FolderAccessRequest = {
      ...folderAccessActionResultFixture.workspace_access_request,
      selected_duration: duration,
      task_access_state: duration === "once" ? "active" : null,
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const permissions = duration === "once"
      ? {
          ...restrictedPermissionProjection(),
          task_access: {
            task_turn_id: ready.task.turn_id,
            state: "active" as const,
            folders: [{ path: ready.selected_path!, file_access: "read_only" as const, commands: false }],
          },
        }
      : null;
    renderCard({ request: ready, workspacesPayload: v2Workspaces, workspacePermissions: permissions });

    expect(screen.getByRole("heading", { name: "Ready to continue" })).toBeInTheDocument();
    expect(screen.queryByText(/Nanobot has not identified the exact folder yet/)).not.toBeInTheDocument();
    expect(screen.getByText(status)).toBeInTheDocument();
    expect(screen.getByText("Read only · Commands off")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue task" })).toBeEnabled();
  });

  it("keeps the unknown-target explanation while the request is pending", () => {
    renderCard();

    expect(screen.getByRole("heading", { name: "Folder access needed" })).toBeInTheDocument();
    expect(screen.getByText(/Nanobot has not identified the exact folder yet/)).toBeInTheDocument();
  });

  it("shows expired Once after continuation without offering another access action", () => {
    const continuedOnce: FolderAccessRequest = {
      ...folderAccessActionResultFixture.workspace_access_request,
      status: "continued",
      selected_duration: "once",
      task_access_state: "expired",
      selected_access: null,
      can_continue: false,
      unavailable_reason: "task_access_expired",
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const permissions: WorkspacePermissionProjection = {
      ...restrictedPermissionProjection(),
      task_access: {
        task_turn_id: continuedOnce.task.turn_id,
        state: "expired",
        folders: [],
      },
    };

    renderCard({ request: continuedOnce, workspacesPayload: v2Workspaces, workspacePermissions: permissions });

    expect(screen.getByRole("heading", { name: "Task continued" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/One-time access has ended/);
    expect(screen.queryByText(/Nanobot has not identified the exact folder yet/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose another folder" })).not.toBeInTheDocument();
  });

  it.each(["continuing", "continued", "cancelled", "invalidated"] as const)(
    "shows One-time access is ending for a matching retiring Once task in %s state",
    (status) => {
      const retiringOnce: FolderAccessRequest = {
        ...folderAccessActionResultFixture.workspace_access_request,
        status,
        selected_duration: "once",
        task_access_state: "retiring",
        selected_access: null,
        can_continue: false,
      };
      const v2Workspaces = workspaces();
      v2Workspaces.controls.permission_v2 = true;
      const permissions: WorkspacePermissionProjection = {
        ...restrictedPermissionProjection(),
        task_access: {
          task_turn_id: retiringOnce.task.turn_id,
          state: "retiring",
          folders: [],
        },
        retirement: {
          reason: "task_ended",
          active_turn: false,
          active_processes: 0,
          unknown_processes: 0,
        },
      };

      renderCard({ request: retiringOnce, workspacesPayload: v2Workspaces, workspacePermissions: permissions });

      expect(screen.getByRole("heading")).toHaveTextContent(
        status === "continuing" ? "Continuing task"
          : status === "continued" ? "Task continued"
            : status === "cancelled" ? "Task cancelled"
              : "Task can't continue",
      );
      expect(screen.getByRole("status")).toHaveTextContent(/One-time access.*ending/i);
      expect(screen.queryByText(/Nanobot has not identified the exact folder yet/)).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Choose folder" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Choose another folder" })).not.toBeInTheDocument();
    },
  );

  it("does not use another task's expired Once projection for a new pending unknown request", () => {
    const pending = {
      ...folderAccessRequestFixture,
      task: { ...folderAccessRequestFixture.task, turn_id: "turn-new-task" },
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const stalePermissions: WorkspacePermissionProjection = {
      ...restrictedPermissionProjection(),
      task_access: {
        task_turn_id: "turn-previous-task",
        state: "expired",
        folders: [],
      },
    };

    renderCard({ request: pending, workspacesPayload: v2Workspaces, workspacePermissions: stalePermissions });

    expect(screen.getByRole("heading", { name: "Folder access needed" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Choose a folder to continue this task.");
    expect(screen.getByText(/Nanobot has not identified the exact folder yet/)).toBeInTheDocument();
    expect(screen.queryByText(/One-time access has ended|One-time access.*ending/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose folder" })).toBeEnabled();
  });

  it("does not treat another task's expired projection as proof for a completed Once request", () => {
    const continuedOnce: FolderAccessRequest = {
      ...folderAccessActionResultFixture.workspace_access_request,
      status: "continued",
      selected_duration: "once",
      task_access_state: null,
      selected_access: null,
      can_continue: false,
    };
    const v2Workspaces = workspaces();
    v2Workspaces.controls.permission_v2 = true;
    const stalePermissions: WorkspacePermissionProjection = {
      ...restrictedPermissionProjection(),
      task_access: {
        task_turn_id: "turn-previous-task",
        state: "expired",
        folders: [],
      },
    };

    renderCard({ request: continuedOnce, workspacesPayload: v2Workspaces, workspacePermissions: stalePermissions });

    expect(screen.getByRole("heading", { name: "Task continued" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("This task has been continued.");
    expect(screen.getByRole("status")).not.toHaveTextContent(/One-time access has ended|One-time access.*ending/i);
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose folder" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose another folder" })).not.toBeInTheDocument();
  });

  it("opens an accessible access review dialog and cancel leaves task access unchanged", async () => {
    const user = userEvent.setup();
    const mutation = vi.fn(async () => folderAccessActionResultFixture);
    const { onApplyWorkspaceScope, onUpdateRequest } = renderCard({ requestMutation: mutation });

    const chooseFolderButton = screen.getByRole("button", { name: "Choose folder" });
    await user.click(chooseFolderButton);
    const dialog = screen.getByRole("dialog", { name: "Review folder access" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Read only" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Read and write" })).not.toBeChecked();
    expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(chooseFolderButton).toHaveFocus();
    await user.click(chooseFolderButton);
    await user.click(screen.getByRole("button", { name: "Cancel", exact: true }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(mutation).not.toHaveBeenCalled();
    expect(onApplyWorkspaceScope).not.toHaveBeenCalled();
    expect(onUpdateRequest).not.toHaveBeenCalled();
  });

  it.each([
    ["continuing", "Continuing task"],
    ["continued", "Task continued"],
    ["cancelled", "Task cancelled"],
    ["invalidated", "Task can't continue"],
  ] as const)("uses the %s heading for terminal recovery status", (status, heading) => {
    renderCard({ request: { ...folderAccessRequestFixture, status, can_continue: false } });
    expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
  });

  it("continues only through the dedicated action and hides actions for invalidated requests", async () => {
    const user = userEvent.setup();
    const mutation = vi.fn(async () => ({ ...folderAccessActionResultFixture,
      workspace_access_request: { ...folderAccessActionResultFixture.workspace_access_request, status: "continuing" },
    }));
    const ready = { ...folderAccessActionResultFixture.workspace_access_request };
    renderCard({ request: ready, requestMutation: mutation });
    await user.click(screen.getByRole("button", { name: "Continue task" }));
    await waitFor(() => expect(mutation).toHaveBeenCalledWith("workspace.access.continue", {
      chat_id: "chat-title",
      access_request_id: ready.access_request_id,
      expected_revision: ready.revision,
    }, 20_000));
    expect(mutation.mock.calls.some(([action]) => action === "message")).toBe(false);

    cleanup();
    renderCard({ request: { ...ready, status: "invalidated", can_continue: false } });
    expect(screen.queryByRole("button", { name: "Continue task" })).not.toBeInTheDocument();
  });

  it("does not apply an old chat's delayed continuation after switching chats", async () => {
    const user = userEvent.setup();
    let resolveMutation!: (value: typeof folderAccessActionResultFixture) => void;
    const delayedMutation = new Promise<typeof folderAccessActionResultFixture>((resolve) => {
      resolveMutation = resolve;
    });
    const mutation = vi.fn(() => delayedMutation);
    const requestA = { ...folderAccessActionResultFixture.workspace_access_request, chat_id: "chat-title" };
    const requestB = {
      ...requestA,
      access_request_id: "fedcba9876543210fedcba9876543210",
      chat_id: "chat-next",
    };
    const { onApplyWorkspaceScope, rerenderCard } = renderCard({ request: requestA, requestMutation: mutation });

    await user.click(screen.getByRole("button", { name: "Continue task" }));
    rerenderCard({ chatId: "chat-next", request: requestB });
    resolveMutation(folderAccessActionResultFixture);

    await waitFor(() => expect(mutation).toHaveBeenCalledTimes(1));
    expect(onApplyWorkspaceScope).not.toHaveBeenCalled();
  });

  it("keeps recovery actions disabled when refreshing task status fails, with an explicit retry", async () => {
    const user = userEvent.setup();
    const mutation = vi.fn(async () => { throw new ApiError(409, "changed"); });
    const failedRefresh = vi.fn(async () => { throw new Error("offline"); });
    const { rerenderCard } = renderCard({
      request: folderAccessActionResultFixture.workspace_access_request,
      requestMutation: mutation,
      onRefreshRequest: failedRefresh,
    });

    await user.click(screen.getByRole("button", { name: "Continue task" }));
    expect(await screen.findByText(/Task status could not be refreshed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue task" })).toBeDisabled();
    expect(screen.getAllByRole("button", { name: "Cancel task" }).every((button) => button.hasAttribute("disabled"))).toBe(true);
    expect(screen.getByRole("button", { name: "Refresh task status" })).toBeEnabled();

    rerenderCard({ requestRefreshFailed: true });
    expect(screen.getByRole("button", { name: "Continue task" })).toBeDisabled();
  });

  it("ignores a late picker selection after the active chat and request change", async () => {
    const user = userEvent.setup();
    let resolvePicker!: (path: string | null) => void;
    const delayedPicker = new Promise<string | null>((resolve) => {
      resolvePicker = resolve;
    });
    const onPickFolder = vi.fn(() => delayedPicker);
    const { rerenderCard } = renderCard({ onPickFolder });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.click(screen.getByRole("button", { name: "Browse…" }));
    expect(onPickFolder).toHaveBeenCalledTimes(1);

    const nextRequest = {
      ...folderAccessRequestFixture,
      access_request_id: "fedcba9876543210fedcba9876543210",
      chat_id: "chat-next",
    };
    rerenderCard({ chatId: "chat-next", request: nextRequest });
    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    expect(screen.getByLabelText("Folder path")).toHaveValue("");

    await act(async () => {
      resolvePicker("/stale/request-folder");
      await delayedPicker;
    });
    expect(screen.getByLabelText("Folder path")).toHaveValue("");
  });

  it("does not show a late picker failure in a replacement request", async () => {
    const user = userEvent.setup();
    let rejectPicker!: (reason: Error) => void;
    const delayedPicker = new Promise<string | null>((_resolve, reject) => {
      rejectPicker = reject;
    });
    const onPickFolder = vi.fn(() => delayedPicker);
    const { rerenderCard } = renderCard({ onPickFolder });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.click(screen.getByRole("button", { name: "Browse…" }));
    rerenderCard({
      chatId: "chat-next",
      request: {
        ...folderAccessRequestFixture,
        access_request_id: "fedcba9876543210fedcba9876543210",
        chat_id: "chat-next",
      },
    });
    await user.click(screen.getByRole("button", { name: "Choose folder" }));

    await act(async () => {
      rejectPicker(new Error("native picker failed"));
      await delayedPicker.catch(() => {});
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.queryByText("Folder browsing is unavailable. Enter an absolute path instead.")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Folder path")).toHaveValue("");
  });

  it("keeps successful and cancelled picker results scoped to the same request and ignores unmounted results", async () => {
    const user = userEvent.setup();
    let resolvePicker!: (path: string | null) => void;
    const delayedPicker = new Promise<string | null>((resolve) => {
      resolvePicker = resolve;
    });
    const onPickFolder = vi.fn()
      .mockResolvedValueOnce("/picked/folder")
      .mockResolvedValueOnce(null)
      .mockImplementationOnce(() => delayedPicker);
    const { unmount } = renderCard({ onPickFolder });

    await user.click(screen.getByRole("button", { name: "Choose folder" }));
    await user.click(screen.getByRole("button", { name: "Browse…" }));
    await waitFor(() => expect(screen.getByLabelText("Folder path")).toHaveValue("/picked/folder"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Browse…" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Browse…" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Browse…" })).toBeEnabled());
    expect(screen.getByLabelText("Folder path")).toHaveValue("/picked/folder");
    expect(screen.queryByText("Folder browsing is unavailable. Enter an absolute path instead.")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Browse…" }));
    expect(onPickFolder).toHaveBeenCalledTimes(3);
    unmount();
    await act(async () => {
      resolvePicker("/after/unmount");
      await delayedPicker;
    });
  });
});

describe("folder proposals", () => {
  const proposed: FolderAccessRequest = {
    ...folderAccessRequestFixture,
    folder_proposal: {
      entry: { name: "reference-documents", path: "/projects/freelance", purpose: "Reference documents",
        kind: "project", knowledge_repos: [], sync_paths: [], auto_sync: false, agent_map: true },
      requested_access: "read_write",
    },
  };

  it("lets the user choose read-only access and map visibility in one confirmation", async () => {
    const user = userEvent.setup();
    const mutation = vi.fn(async () => folderAccessActionResultFixture);
    const { onUpdateRequest } = renderCard({ request: proposed, requestMutation: mutation });
    expect(screen.getByText("Reference documents")).toBeInTheDocument();
    expect(screen.getByText(/Read-only by default/)).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Sandbox access"), "read_only");
    await user.click(screen.getByLabelText("Include in Agent map"));
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await waitFor(() => expect(mutation).toHaveBeenCalledWith("workspace.access.add_folder", {
      chat_id: proposed.chat_id, access_request_id: proposed.access_request_id,
      expected_revision: proposed.revision, expected_grants_revision: 12, expected_catalog_revision: 0,
      file_access: "read_only", agent_map: false,
    }, 20_000));
    expect(onUpdateRequest).toHaveBeenCalled();
  });

  it("shows the writable agent workspace as current write access", () => {
    const workspaceProposal: FolderAccessRequest = {
      ...proposed,
      folder_proposal: {
        ...proposed.folder_proposal!,
        entry: { ...proposed.folder_proposal!.entry, path: "/agent/home/reference" },
      },
    };
    renderCard({ request: workspaceProposal });
    expect(screen.getByText(/Read and write · Agent workspace/)).toBeInTheDocument();
    expect(screen.queryByText(/Read-only by default/)).not.toBeInTheDocument();
  });

  it("prefers the writable workspace over a read-only ancestor grant", () => {
    const payload = workspaces();
    payload.management!.grants.push({
      path: "/agent",
      file_access: "read_only",
      commands: false,
      exists: true,
      effective_access: {
        policy: "managed",
        source: "grant",
        grant_root: "/agent",
        exists: true,
        read: true,
        write: false,
        boundary: "folder",
        commands: { requested: false, available: false, backend: "none", reason: null },
      },
    });
    const workspaceProposal: FolderAccessRequest = {
      ...proposed,
      folder_proposal: {
        ...proposed.folder_proposal!,
        entry: { ...proposed.folder_proposal!.entry, path: "/agent/home/reference" },
      },
    };
    renderCard({ request: workspaceProposal, workspacesPayload: payload });
    expect(screen.getByText(/Read and write · Agent workspace/)).toBeInTheDocument();
    expect(screen.queryByText(/Read-only · \/agent$/)).not.toBeInTheDocument();
  });

  it("declines without saving a grant or metadata", async () => {
    const user = userEvent.setup(); const mutation = vi.fn(async () => folderAccessActionResultFixture);
    renderCard({ request: proposed, requestMutation: mutation });
    await user.click(screen.getByRole("button", { name: "Decline" }));
    await waitFor(() => expect(mutation).toHaveBeenCalledTimes(1));
    expect(mutation.mock.calls[0]![0]).toBe("workspace.access.cancel");
  });

  it("does not apply an approval reply to a different task", async () => {
    const user = userEvent.setup();
    let finish!: (value: typeof folderAccessActionResultFixture) => void;
    const mutation = vi.fn(() => new Promise<typeof folderAccessActionResultFixture>((resolve) => { finish = resolve; }));
    const { rerenderCard, onUpdateRequest } = renderCard({ request: proposed, requestMutation: mutation });
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    rerenderCard({ chatId: "another-chat", request: null });
    await act(async () => { finish(folderAccessActionResultFixture); });
    expect(onUpdateRequest).not.toHaveBeenCalled();
  });
});
