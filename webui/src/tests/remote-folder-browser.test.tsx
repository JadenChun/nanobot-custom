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
