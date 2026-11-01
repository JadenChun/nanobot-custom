import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

const requestMutation = vi.fn(async () => ({
  root_id: "projects",
  relative_path: "alpha",
  display_path: "/srv/projects/alpha",
  directory_id: "dir-1",
  directory_revision: "rev-1",
  policy_revision: "policy-1",
  expires_at: 999,
  commit_token: "commit-token",
}));

vi.mock("@/providers/ClientProvider", () => ({
  useClient: () => ({ client: { requestMutation } }),
}));

import {
  parseListing,
  parseRoots,
  RemoteFolderBrowserDialog,
} from "@/components/settings/folders/RemoteFolderBrowserDialog";

const roots = {
  roots: [{ id: "projects", label: "Projects", display_path: "/srv/projects" }],
  policy_revision: "policy-1",
  empty: false,
};

const listing = {
  root_id: "projects",
  relative_path: "",
  entries: [
    { name: "alpha", directory_id: "id-alpha", symlink: false, accessible: true },
    { name: "escape", directory_id: "id-escape", symlink: true, accessible: false },
  ],
  next_cursor: null,
  policy_revision: "policy-1",
};

afterEach(() => {
  vi.unstubAllGlobals();
  requestMutation.mockClear();
});

describe("RemoteFolderBrowserDialog", () => {
  it("validates roots and listings", () => {
    expect(parseRoots(roots)?.roots[0].id).toBe("projects");
    expect(parseRoots({ roots: "bad" })).toBeNull();
    expect(parseListing(listing)?.entries[0].name).toBe("alpha");
    expect(parseListing({ ...listing, entries: "bad" })).toBeNull();
  });

  it("lists configured roots, navigates and resolves a selected folder", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.startsWith("/api/workspace-directories/roots")) {
        return new Response(JSON.stringify(roots), { status: 200 });
      }
      if (path.startsWith("/api/workspace-directories/list")) {
        return new Response(JSON.stringify(listing), { status: 200 });
      }
      return new Response(JSON.stringify({ error: "unexpected" }), { status: 500 });
    }));
    const onSelected = vi.fn();
    render(
      <RemoteFolderBrowserDialog open onOpenChange={() => undefined} onSelected={onSelected} />,
    );

    await waitFor(() => expect(screen.getByRole("button", { name: /Projects/ })).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /Projects/ }));
    await waitFor(() => expect(screen.getByText("alpha")).toBeInTheDocument());
    expect(screen.getByText("escape").closest("button")).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: /Use this folder/ }));
    await waitFor(() => expect(requestMutation).toHaveBeenCalledWith(
      "workspace.directories.resolve",
      { root: "projects", path: "", policy_revision: "policy-1" },
    ));
    await waitFor(() => expect(onSelected).toHaveBeenCalledWith({
      root_id: "projects",
      relative_path: "",
      display_path: "/srv/projects/alpha",
      policy_revision: "policy-1",
      commit_token: "commit-token",
    }));
  });

  it("shows the explicit empty-roots state", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      roots: [],
      policy_revision: "empty",
      empty: true,
    }), { status: 200 })));
    render(
      <RemoteFolderBrowserDialog open onOpenChange={() => undefined} onSelected={() => undefined} />,
    );
    await waitFor(() => expect(screen.getByText(/No server folder roots/)).toBeInTheDocument());
  });
});

describe("owner grant commit payload", () => {
  it("uses the token-bound relative path and keeps the commit fields typed", async () => {
    const calls: Array<{ action: string; payload: Record<string, unknown> }> = [];
    const transport = {
      requestMutation: async (action: string, payload: Record<string, unknown>) => {
        calls.push({ action, payload });
        return {
          schema_version: 2,
          controls: { can_manage_folders: true },
          management: { grants_revision: 2, grants: [], repositories: [], catalog_revision: 0 },
        };
      },
    };
    const { upsertWorkspaceGrant } = await import("@/lib/api");
    await upsertWorkspaceGrant(
      transport as never,
      { path: "/srv/projects/alpha", file_access: "read_only", commands: false },
      1,
      { commit_token: "token", root: "projects", extent: "read", relative_path: "alpha" },
    );
    expect(calls[0].action).toBe("workspace.grant.upsert");
    expect(calls[0].payload).toMatchObject({
      expected_revision: 1,
      path: "alpha",
      file_access: "read_only",
      commands: false,
      commit_token: "token",
      root: "projects",
      extent: "read",
    });
    expect(calls[0].payload).not.toHaveProperty("relative_path");
  });
});
