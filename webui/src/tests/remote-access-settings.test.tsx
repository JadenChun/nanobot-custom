import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("qrcode", () => ({
  default: {
    toDataURL: vi.fn(async () => "data:image/png;base64,qr"),
  },
}));

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
      return new Response(JSON.stringify({
        pairing_id: "pair-1",
        comparison_code: "123456",
        nonce: "nonce-value-000000000000",
        collector_secret: "secret-value-000000000000",
        expires_at: 999,
      }), { status: 200 });
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
    expect(screen.getByRole("alertdialog", { name: "Revoke Phone" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Confirm revoke" }));
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
    await userEvent.type(screen.getByLabelText("Phone installation ID"), "install-phone-1");
    await userEvent.click(screen.getByRole("button", { name: "Create pairing challenge" }));
    await waitFor(() => expect(screen.getByText("123456")).toBeInTheDocument());
    expect(await screen.findByAltText("Phone pairing QR code")).toBeInTheDocument();
    const link = screen.getByLabelText("Phone pairing link") as HTMLTextAreaElement;
    expect(link.value).toContain("nanobot://pair?");
    expect(link.value).toContain("installation=install-phone-1");
    expect(link.value).toContain("challenge=pair-1.nonce-value-000000000000.secret-value-000000000000");
    await userEvent.type(screen.getByLabelText("Approval comparison code"), "123456");
    await userEvent.click(screen.getByRole("button", { name: "Approve pairing" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      "/api/settings/remote-access/pairing/approve",
      expect.objectContaining({ method: "POST" }),
    ));
  });
});


it("hides local-only enable and pairing controls on a mobile owner", () => {
  mockFetch();
  render(
    <RemoteAccessSettings token="token" initialStatus={status} localSetupAvailable={false} />,
  );
  expect(screen.queryByRole("button", { name: "Disable" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Create pairing challenge" })).not.toBeInTheDocument();
  expect(screen.getByText(/New-device pairing and remote on\/off stay/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Revoke" })).toBeInTheDocument();
});
