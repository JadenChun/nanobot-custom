import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FoldersProjectsManager } from "@/components/settings/folders/FoldersProjectsManager";
import { ApiError } from "@/lib/api";
import type {
  EffectiveWorkspaceAccess,
  RepositoryDiscoveryPayload,
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
          auto_sync: true,
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
  it("renders inherited access, unavailable commands, and missing roots", () => {
    renderManager();

    expect(screen.getByText("/projects/child")).toBeInTheDocument();
    expect(screen.getAllByText(/Effective access inherited from \/projects\/parent/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Commands unavailable: sandbox unavailable/)).toBeInTheDocument();
    expect(screen.getByText("Folder missing")).toBeInTheDocument();
    expect(screen.getByText(/Already running confined commands keep their launch access/)).toBeInTheDocument();
  });

  it("shows available native commands for a legacy access row in a v2 payload", () => {
    const payload = workspaces();
    const grant = payload.management!.grants[0]!;
    grant.effective_access = {
      ...grant.effective_access,
      policy: "legacy",
      source: "configured",
      grant_root: null,
      commands: {
        requested: true,
        available: true,
        backend: "native",
        reason: null,
      },
    };
    renderManager(payload);

    expect(screen.getByText("Read only · Commands available")).toBeInTheDocument();
    expect(screen.queryByText(/Read only · Commands unavailable/)).not.toBeInTheDocument();
  });

  it("saves read/write and command choices with the current grant revision", async () => {
    const user = userEvent.setup();
    const { requestMutation, onRefreshWorkspaces } = renderManager();

    await user.click(screen.getByRole("button", { name: "Add folder access" }));
    expect(screen.getByLabelText("File access")).toHaveValue("read_only");
    expect(screen.getByRole("checkbox", { name: "Allow commands" })).not.toBeChecked();
    await user.type(screen.getByLabelText("Folder path"), "/projects/new");
    await user.selectOptions(screen.getByLabelText("File access"), "read_write");
    await user.click(screen.getByRole("checkbox", { name: "Allow commands" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("workspace.grant.upsert", {
      expected_revision: 12,
      path: "/projects/new",
      file_access: "read_write",
      commands: true,
    }, 20_000));
    expect(onRefreshWorkspaces).toHaveBeenCalledTimes(1);
  });

  it("refreshes stale grants and asks for review without resubmitting", async () => {
    const user = userEvent.setup();
    const requestMutation = vi.fn(async () => {
      throw new ApiError(409, "grants_changed");
    });
    const { onRefreshWorkspaces } = renderManager(workspaces(), requestMutation);

    await user.click(screen.getByRole("button", { name: "Add folder access" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/new");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Folder access changed elsewhere. Review the updated values before trying again.",
    );
    expect(requestMutation).toHaveBeenCalledTimes(1);
    expect(onRefreshWorkspaces).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["native_processes_active", /Wait for it to finish/],
    ["native_processes_unknown", /Restarting Nanobot will not clear this state; you may need to restart this device/],
  ])("explains %s safely without presenting it as a stale grants conflict", async (code, expectedCopy) => {
    const user = userEvent.setup();
    const requestMutation = vi.fn(async () => {
      throw new ApiError(409, code);
    });
    const { onRefreshWorkspaces } = renderManager(workspaces(), requestMutation);

    await user.click(screen.getByRole("button", { name: "Add folder access" }));
    await user.type(screen.getByLabelText("Folder path"), "/projects/new");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(expectedCopy);
    expect(screen.queryByText(/Folder access changed elsewhere/)).not.toBeInTheDocument();
    expect(onRefreshWorkspaces).not.toHaveBeenCalled();
  });

  it("uses the supplied local folder chooser in the Settings manager", async () => {
    const user = userEvent.setup();
    const onPickFolder = vi.fn().mockResolvedValue("/projects/from-picker");
    renderManager(workspaces(), vi.fn(async () => ({})), onPickFolder);

    await user.click(screen.getByRole("button", { name: "Add folder access" }));
    await user.click(screen.getByRole("button", { name: "Browse…" }));

    expect(onPickFolder).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByLabelText("Folder path")).toHaveValue(
      "/projects/from-picker",
    ));
  });

  it("preserves knowledge sync settings when repository metadata is edited", async () => {
    const user = userEvent.setup();
    const requestMutation = vi.fn(async () => ({}));
    renderManager(workspaces(), requestMutation);

    await user.click(screen.getByRole("tab", { name: "Known repositories" }));
    await user.click(screen.getByRole("button", { name: "Edit client-app" }));
    await user.clear(screen.getByLabelText("Purpose"));
    await user.type(screen.getByLabelText("Purpose"), "Updated application notes");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("repository.remember", {
      entry: {
        name: "client-app",
        path: "/projects/client-app",
        purpose: "Updated application notes",
        kind: "project",
        knowledge_repos: ["notes"],
        auto_sync: true,
        sync_paths: ["decisions/**"],
      },
    }, 20_000));
  });

  it("keeps a linked repository when forgetting returns repository_in_use", async () => {
    const user = userEvent.setup();
    const requestMutation = vi.fn(async () => {
      throw new ApiError(409, "repository_in_use");
    });
    renderManager(workspaces(), requestMutation);

    await user.click(screen.getByRole("tab", { name: "Known repositories" }));
    await user.click(screen.getByRole("button", { name: "Forget client-app" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This repository is linked from a project. Edit that project's knowledge associations before forgetting it.",
    );
    expect(requestMutation).toHaveBeenCalledTimes(1);
  });

  it("shows bounded discovery candidates and remembers them without granting access", async () => {
    const user = userEvent.setup();
    const candidateAccess = { ...managedAccess, commands: { ...managedAccess.commands, requested: false, reason: null } };
    const discovery: RepositoryDiscoveryPayload = {
      root: "/projects",
      visited_directories: 4,
      truncated: true,
      candidates: [{
        path: "/projects/candidate",
        suggested_name: "candidate",
        known_name: null,
        effective_access: candidateAccess,
      }],
    };
    const requestMutation = vi.fn(async (action: string) => {
      if (action === "repository.discover") return discovery;
      return {};
    });
    renderManager(workspaces(), requestMutation);

    await user.click(screen.getByRole("tab", { name: "Known repositories" }));
    await user.type(screen.getByLabelText("Discover repositories under a folder"), "/projects");
    await user.click(screen.getByRole("button", { name: "Discover" }));

    expect(await screen.findByText(/Results were limited/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remember" }));
    await user.type(screen.getByLabelText("Purpose"), "Candidate project");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith("repository.remember", {
      entry: {
        name: "candidate",
        path: "/projects/candidate",
        purpose: "Candidate project",
        kind: "project",
        knowledge_repos: [],
        auto_sync: false,
        sync_paths: [],
      },
    }, 20_000));
    expect(requestMutation.mock.calls.some(([action]) => action === "workspace.grant.upsert")).toBe(false);
  });

  it("hides remote management controls and data", () => {
    const remote = workspaces();
    remote.controls.can_manage_folders = false;
    remote.controls.can_manage_catalog = false;
    remote.controls.can_discover_repositories = false;
    remote.management = null;
    renderManager(remote);

    expect(screen.getByRole("status")).toHaveTextContent(
      "Folder and repository management is available only from a trusted local WebUI connection.",
    );
    expect(screen.queryByText("/projects/child")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add folder access" })).not.toBeInTheDocument();
  });
});
