import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RemoteAccessSettings } from "@/components/settings/RemoteAccessSettings";

const status = {
  enabled: true,
  installation_id: "server-1",
  devices: [
    {
      installation_id: "install-1",
      label: "Phone",
      scopes: ["chat.read"],
      status: "active" as const,
      created_at: 1,
      last_seen_at: 2,
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

function mockFetch() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.endsWith("/enable")) return new Response(JSON.stringify({ enabled: false }), { status: 200 });
    if (path.endsWith("/revoke")) return new Response(JSON.stringify({ revoked: true }), { status: 200 });
    if (path.endsWith("/pairing/start")) {
      return new Response(JSON.stringify({ pairing_id: "pair-1", comparison_code: "123456" }), { status: 200 });
    }
    if (path.endsWith("/pairing/approve")) return new Response(JSON.stringify({ ok: true }), { status: 200 });
    return new Response(JSON.stringify(status), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("RemoteAccessSettings", () => {
  it("renders status and revokes a device", async () => {
    mockFetch();
    render(<RemoteAccessSettings token="token" initialStatus={status} />);
    expect(screen.getByText("Phone")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Revoke" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/settings/remote-access/revoke",
      expect.objectContaining({ method: "POST" }),
    ));
  });

  it("toggles remote access and starts a pairing challenge", async () => {
    mockFetch();
    render(<RemoteAccessSettings token="token" initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Disable" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/settings/remote-access/enable",
      expect.objectContaining({ method: "POST" }),
    ));
    await userEvent.type(screen.getByLabelText("Exact HTTPS origin"), "https://fixture.example");
    await userEvent.click(screen.getByRole("button", { name: "Create pairing challenge" }));
    await waitFor(() => expect(screen.getByText("123456")).toBeInTheDocument());
  });
});
