import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FoldersProjectsManager } from "@/components/settings/folders/FoldersProjectsManager";
import { ApiError } from "@/lib/api";
import type {
  EffectiveWorkspaceAccess,
  WorkspacesPayload,
} from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

const managedAccess: EffectiveWorkspaceAccess = {
  policy: "managed",
  source: "grant",
  grant_root: "/projects/parent",
  exists: true,
  read: true,
  write: false,
  commands: {
    requested: true,
    available: false,
    backend: "none",
    reason: "sandbox_unavailable",
  },
};

function workspaces(): WorkspacesPayload {
  return {
    schema_version: 2,
    default_access_mode: "restricted",
    default_scope: {
      project_path: "/agent/home",
      project_name: "home",
      access_mode: "restricted",
      restrict_to_workspace: true,
      effective_access: {
        ...managedAccess,
        source: "agent_home",
        grant_root: null,
        commands: {
          requested: false,
          available: false,
          backend: "none",
          reason: null,
        },
      },
    },
    agent_home: "/agent/home",
    managed_access: true,
    controls: {
      can_change_project: true,
      can_use_full_access: false,
      can_manage_folders: true,
      can_manage_catalog: true,
      can_discover_repositories: true,
    },
    management: {
      catalog_revision: 4,
      grants_revision: 12,
      grants: [
        {
          path: "/projects/child",
          file_access: "read_only",
          commands: true,
          exists: true,
          effective_access: managedAccess,
        },
        {
          path: "/projects/missing",
          file_access: "read_write",
          commands: false,
          exists: false,
          effective_access: {
            ...managedAccess,
            exists: false,
            read: false,
            write: false,
            commands: {
              requested: false,
              available: false,
              backend: "none",
              reason: "path_missing",
            },
          },
        },
      ],
      repositories: [
        {
          name: "client-app",
          path: "/projects/client-app",
          purpose: "The current application",
          kind: "project",
          knowledge_repos: ["notes"],
          auto_sync: false,
          sync_paths: ["decisions/**"],
          exists: true,
          effective_access: managedAccess,
        },
      ],
    },
  };
}

function renderManager(
  payload: WorkspacesPayload | null = workspaces(),
  requestMutation = vi.fn(async () => ({})),
  onPickFolder?: () => Promise<string | null>,
) {
  const onRefreshWorkspaces = vi.fn(async () => {});
  render(
    <ClientProvider client={{ requestMutation } as never} token="test-token">
      <FoldersProjectsManager
        workspaces={payload}
        onRefreshWorkspaces={onRefreshWorkspaces}
        onPickFolder={onPickFolder}
      />
    </ClientProvider>,
  );
  return { onRefreshWorkspaces, requestMutation };
}

describe("FoldersProjectsManager", () => {
  it("shows one folder list, inherited access, missing folders and stored Sandbox wording", () => {
    renderManager();
    expect(screen.getByText("/projects/child")).toBeInTheDocument();
    expect(screen.getByText(/Inherited from \/projects\/parent/)).toBeInTheDocument();
    expect(screen.getByText("Folder missing or moved")).toBeInTheDocument();
    expect(screen.getByText(/Stored Sandbox rules control writes/)).toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  });

  it("saves metadata, map visibility and write access in one user action", async () => {
    const user = userEvent.setup();
    const { requestMutation, onRefreshWorkspaces } = renderManager();
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    expect(screen.getByLabelText("Sandbox access")).toHaveValue("inherit");
    expect(screen.queryByRole("checkbox", { name: "Allow commands" })).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("Folder path"), "/projects/new");
    await user.type(screen.getByLabelText("Name"), "Reference documents");
    await user.type(screen.getByLabelText("Purpose"), "Project records");
    await user.selectOptions(screen.getByLabelText("Sandbox access"), "read_write");
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("workspace.folder.edit", expect.objectContaining({
      expected_grants_revision: 12, expected_catalog_revision: 4, path: "/projects/new", access: "read_write",
      operation_id: expect.stringMatching(/^[a-f0-9]{32}$/),
      entry: expect.objectContaining({ name: "reference-documents", label: "Reference documents", purpose: "Project records", agent_map: true }),
    }), 20_000));
    expect(onRefreshWorkspaces).toHaveBeenCalledTimes(1);
  });

  it("can save access-only metadata without adding it to the model map", async () => {
    const user = userEvent.setup(); const { requestMutation } = renderManager();
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/documents");
    await user.click(screen.getByLabelText("Include in Agent map"));
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("workspace.folder.edit", expect.objectContaining({
      entry: expect.objectContaining({ agent_map: false }), access: "inherit",
    }), 20_000));
  });

  it("groups only the nearest saved ancestors and keeps independent roots separate", () => {
    const payload = workspaces(); const management = payload.management!;
    management.grants.push({ ...management.grants[0]!, path: "/projects" });
    management.repositories.push({ ...management.repositories[0]!, name: "nested", path: "/projects/client-app/docs" },
      { ...management.repositories[0]!, name: "unrelated", path: "/projects-extra/client" },
      { ...management.repositories[0]!, name: "remote-root", path: "/srv/reference" },
      { ...management.repositories[0]!, name: "child-map", path: "/projects/child" });
    renderManager(payload);
    const row = (path: string) => screen.getByText(path).closest("[data-folder-path]");
    expect(row("/projects")).toHaveAttribute("data-depth", "0");
    expect(row("/projects/client-app")).toHaveAttribute("data-depth", "1");
    expect(row("/projects/client-app/docs")).toHaveAttribute("data-depth", "2");
    expect(row("/projects-extra/client")).toHaveAttribute("data-depth", "0");
    expect(row("/srv/reference")).toHaveAttribute("data-depth", "0");
    expect(screen.getAllByText("/projects/child")).toHaveLength(1);
  });

  it("previews removal and inherited fallback before sending a combined edit", async () => {
    const user = userEvent.setup(); const { requestMutation } = renderManager();
    await user.click(screen.getByRole("button", { name: "Remove client-app" }));
    expect(screen.getByText(/Parent rules remain/)).toBeInTheDocument();
    expect(requestMutation).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Remove folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("workspace.folder.edit", expect.objectContaining({
      path: "/projects/client-app", entry: null, access: "inherit",
    }), 20_000));
  });

  it("describes the writable agent workspace as inherited write access", async () => {
    const user = userEvent.setup();
    renderManager();
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/agent/home/project");
    expect(screen.getByText(/Read and write · Agent workspace/)).toBeInTheDocument();
    expect(screen.queryByText(/Read-only by default/)).not.toBeInTheDocument();
  });

  it("prefers the more-specific writable workspace over a read-only ancestor grant", async () => {
    const user = userEvent.setup();
    const payload = workspaces();
    payload.management!.grants.push({
      ...payload.management!.grants[0]!,
      path: "/agent",
      file_access: "read_only",
      effective_access: {
        ...managedAccess,
        grant_root: "/agent",
        write: false,
      },
    });
    renderManager(payload);
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/agent/home/project");
    expect(screen.getByText(/Read and write · Agent workspace/)).toBeInTheDocument();
    expect(screen.queryByText(/Read-only · \/agent$/)).not.toBeInTheDocument();
  });

  it("keeps a more-specific read-only child rule inside the workspace", async () => {
    const user = userEvent.setup();
    const payload = workspaces();
    payload.management!.grants.push({
      ...payload.management!.grants[0]!,
      path: "/agent/home/restricted",
      file_access: "read_only",
      effective_access: {
        ...managedAccess,
        grant_root: "/agent/home/restricted",
        write: false,
      },
    });
    renderManager(payload);
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/agent/home/restricted/child");
    expect(screen.getByText(/Read-only · \/agent\/home\/restricted$/)).toBeInTheDocument();
  });

  it("shows the agent workspace default for a saved workspace folder", () => {
    const payload = workspaces();
    payload.management!.repositories.push({
      ...payload.management!.repositories[0]!,
      name: "workspace-project",
      path: "/agent/home/project",
      effective_access: {
        ...managedAccess,
        source: "agent_home",
        grant_root: null,
        write: true,
      },
    });
    renderManager(payload);
    expect(screen.getByText("Read and write · Agent workspace default")).toBeInTheDocument();
  });

  it("retries the identical Settings payload after a partial save", async () => {
    const user = userEvent.setup();
    const requestMutation = vi.fn()
      .mockRejectedValueOnce(new ApiError(409, "folder_metadata_saved_access_pending"))
      .mockResolvedValue({});
    renderManager(workspaces(), requestMutation);
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/new");
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/metadata was saved/i);
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledTimes(2));
    expect(requestMutation.mock.calls[1]![1]).toEqual(requestMutation.mock.calls[0]![1]);
  });

  it("keeps an operation id for a retry after a persistence failure", async () => {
    const user = userEvent.setup(); const requestMutation = vi.fn().mockRejectedValueOnce(new ApiError(500, "storage unavailable")).mockResolvedValue({});
    renderManager(workspaces(), requestMutation);
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.type(screen.getByLabelText("Folder path"), "/documents");
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await screen.findByRole("alert");
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledTimes(2));
    expect(requestMutation.mock.calls[1]![1]).toEqual(requestMutation.mock.calls[0]![1]);
  });

  it("preserves knowledge links and explicit Git sync settings while editing", async () => {
    const user = userEvent.setup(); const payload = workspaces();
    payload.management!.repositories[0]!.kind = "knowledge";
    payload.management!.repositories[0]!.knowledge_repos = [];
    payload.management!.repositories[0]!.auto_sync = true;
    const { requestMutation } = renderManager(payload);
    await user.click(screen.getByRole("button", { name: "Edit folder client-app" }));
    await user.click(screen.getByRole("button", { name: "Save folder" }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("workspace.folder.edit", expect.objectContaining({
      entry: expect.objectContaining({ kind: "knowledge", auto_sync: true, sync_paths: ["decisions/**"] }),
    }), 20_000));
  });

  it("uses the supplied local folder picker", async () => {
    const user = userEvent.setup(); const picker = vi.fn().mockResolvedValue("/documents");
    renderManager(workspaces(), vi.fn(async () => ({})), picker);
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await user.click(screen.getByRole("button", { name: "Browse…" }));
    expect(screen.getByLabelText("Folder path")).toHaveValue("/documents");
  });

  it("does not show remote folder management data", () => {
    const payload = workspaces(); payload.management = null; payload.controls.can_manage_folders = false; payload.controls.can_manage_catalog = false;
    renderManager(payload);
    expect(screen.getByRole("status")).toHaveTextContent("Folder management is available only in the local app.");
    expect(screen.queryByText("/projects/child")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add folder" })).toBeDisabled();
  });
});
