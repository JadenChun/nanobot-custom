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

function mutationTransport() {
  let approvalScopes = ["chat.read", "chat.write", "media.read", "attachment.write", "owner"];
  return {
    requestMutation: vi.fn(async (action: string, payload: Record<string, unknown>) => {
      if (action === "settings.remote_access.enable") return { enabled: false };
      if (action === "settings.remote_access.revoke") return { revoked: true };
      if (action === "settings.remote_access.pairing.start") {
        approvalScopes = payload.fullControlConsent === true
          ? ["chat.read", "chat.write", "media.read", "attachment.write", "owner"]
          : ["chat.read", "chat.write", "media.read", "attachment.write"];
        return {
          pairing_id: "pair-1",
          comparison_code: "123456",
          nonce: "nonce-value-000000000000",
          collector_secret: "secret-value-000000000000",
          expires_at: 999,
        };
      }
      if (action === "settings.remote_access.pairing.approve") {
        return { pairing_id: "pair-1", installation_id: "install-1", scopes: approvalScopes };
      }
      return {};
    }),
  };
}

describe("RemoteAccessSettings", () => {
  it("renders status and revokes a device after confirmation", async () => {
    const mutate = mutationTransport();
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    expect(screen.getByText("Phone")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Revoke" }));
    expect(screen.getByRole("alertdialog", { name: "Revoke Phone" })).toBeInTheDocument();
    expect(mutate.requestMutation).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Confirm revoke" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.revoke",
      { installation_id: "install-1" },
    ));
  });

  it("toggles remote access and starts an owner pairing challenge", async () => {
    const mutate = mutationTransport();
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Disable" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.enable",
      { enabled: false },
    ));

    await userEvent.click(screen.getByRole("button", { name: "Set up remote access" }));
    await userEvent.type(screen.getByLabelText("Server address"), "https://fixture.example");
    await userEvent.click(screen.getByRole("button", { name: "Get started" }));
    await waitFor(() => expect(screen.getByText("123456")).toBeInTheDocument());
    expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.start",
      expect.objectContaining({
        origin: "https://fixture.example",
        label: "phone",
        chatConsent: true,
        fullControlConsent: true,
        sharedHistoryConsent: true,
        mediaConsent: true,
      }),
    );
    expect(await screen.findByAltText("Pairing QR code")).toBeInTheDocument();
    const link = screen.getByLabelText("Phone pairing link") as HTMLTextAreaElement;
    expect(link.value).toContain("nanobot://pair?");
    expect(link.value).toContain("installation=");
    expect(link.value).toContain("challenge=pair-1.nonce-value-000000000000.secret-value-000000000000");
    await userEvent.click(screen.getByRole("button", { name: "Approve phone" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.approve",
      { pairingId: "pair-1", comparisonCode: "123456" },
    ));
    expect(await screen.findByText("You're connected")).toBeInTheDocument();
  });

  it("describes chat-and-media pairing when full owner control is off", async () => {
    const mutate = mutationTransport();
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Set up remote access" }));
    await userEvent.type(screen.getByLabelText("Server address"), "https://fixture.example");
    await userEvent.click(screen.getByLabelText("Allow full owner control"));
    expect(screen.getByText(/Full owner control is off/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Get started" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.start",
      expect.objectContaining({
        fullControlConsent: false,
        sharedHistoryConsent: false,
        chatConsent: true,
        mediaConsent: true,
      }),
    ));
    await userEvent.click(screen.getByRole("button", { name: "Approve phone" }));
    expect(await screen.findByText(/paired for chats and media only/)).toBeInTheDocument();
    expect(screen.queryByText(/paired with full owner control/)).not.toBeInTheDocument();
  });

  it("fetches status with bearer authorization", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => status,
    }));
    vi.stubGlobal("fetch", fetchMock);
    render(<RemoteAccessSettings token="api-token" mutate={mutationTransport()} />);
    expect(await screen.findByText("Phone")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer api-token");
  });

  it("hides local-only enable and pairing controls on a mobile owner", () => {
    const mutate = mutationTransport();
    render(
      <RemoteAccessSettings
        token="token"
        mutate={mutate}
        initialStatus={status}
        localSetupAvailable={false}
      />,
    );
    expect(screen.queryByRole("button", { name: "Disable" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set up remote access" })).not.toBeInTheDocument();
    expect(screen.getByText(/New-device pairing and remote on\/off stay/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Revoke" })).toBeInTheDocument();
  });
});
