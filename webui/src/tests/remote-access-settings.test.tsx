import { render, screen, waitFor, within } from "@testing-library/react";
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
      scopes: ["chat.read", "chat.write", "media.read", "attachment.write", "owner"],
      status: "active" as const,
      created_at: 1,
      last_seen_at: 2,
    },
  ],
};

const pending = {
  pairing_id: "pair-1",
  state: "pending_approval" as const,
  expires_at: Math.floor(Date.now() / 1000) + 300,
  device: { installation_id: "phone-1", label: "Pixel 10", platform: "Android" },
  scopes: ["chat.read", "chat.write", "media.read", "attachment.write", "owner"],
};

afterEach(() => vi.unstubAllGlobals());

function mutationTransport(onStart?: () => void) {
  return {
    requestMutation: vi.fn(async (action: string) => {
      if (action === "settings.remote_access.enable") return { enabled: false };
      if (action === "settings.remote_access.revoke") return { revoked: true };
      if (action === "settings.remote_access.pairing.start") {
        onStart?.();
        return {
          pairing_id: "pair-1",
          comparison_code: "12345678",
          nonce: "nonce-value-000000000000",
          collector_secret: "secret-value-000000000000",
          expires_at: Math.floor(Date.now() / 1000) + 300,
          transport: "tailscale",
        };
      }
      if (action === "settings.remote_access.pairing.approve") {
        return { pairing_id: "pair-1", installation_id: "phone-1", scopes: ["owner"] };
      }
      if (action === "settings.remote_access.pairing.deny") return {};
      if (action === "settings.remote_access.pairing.renew") {
        return {
          ...pending,
          expires_at: Math.floor(Date.now() / 1000) + 300,
          renewable: true,
        };
      }
      return {};
    }),
  };
}

function pendingFetch(state: { collected: boolean } = { collected: false }) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/tailscale")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          available: true,
          connected: true,
          serve_mapped: true,
          dns_name: "fixture.example",
          origin: "https://fixture.example",
          error: "",
        }),
      };
    }
    return {
      ok: true,
      status: 200,
      json: async () =>
        url.includes("/pairing/pending") ? { pending: state.collected ? [pending] : [] } : status,
    };
  });
}

describe("RemoteAccessSettings", () => {
  it("renders owner status and revokes after confirmation", async () => {
    const mutate = mutationTransport();
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    expect(screen.getByText("Phone")).toBeInTheDocument();
    expect(screen.getByText("Owner")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Revoke" }));
    expect(screen.getByRole("alertdialog", { name: "Revoke Phone" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Confirm revoke" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.revoke",
      { installation_id: "install-1" },
    ));
  });

  it("starts an owner-only pairing and approves the scanned phone without a code", async () => {
    const collection = { collected: false };
    const mutate = mutationTransport(() => {
      collection.collected = true;
    });
    vi.stubGlobal("fetch", pendingFetch(collection));
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Pair owner phone" }));
    await waitFor(() => expect(screen.getByText("1234 5678")).toBeInTheDocument());
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
    expect(screen.getByRole("button", { name: "Copy link" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Pixel 10 wants owner access/)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Approve owner access" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.approve",
      { pairingId: "pair-1", comparisonCode: "12345678" },
    ));
    expect(await screen.findByText("Owner phone connected")).toBeInTheDocument();
  });

  it("denies a pending owner request", async () => {
    const collection = { collected: false };
    const mutate = mutationTransport(() => {
      collection.collected = true;
    });
    vi.stubGlobal("fetch", pendingFetch(collection));
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Pair owner phone" }));
    await waitFor(() => expect(screen.getByText(/Pixel 10 wants owner access/)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Deny" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.deny",
      { pairingId: "pair-1" },
    ));
  });

  it("keeps the pending setup across closing and reopening the dialog", async () => {
    const collection = { collected: false };
    const mutate = mutationTransport(() => {
      collection.collected = true;
    });
    vi.stubGlobal("fetch", pendingFetch(collection));
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Pair owner phone" }));
    await waitFor(() => expect(screen.getByText("1234 5678")).toBeInTheDocument());
    const dialog = screen.getByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: /close/i }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Pair owner phone" }));
    await waitFor(() => expect(screen.getByText("1234 5678")).toBeInTheDocument());
    expect(
      mutate.requestMutation.mock.calls.filter(
        ([action]) => action === "settings.remote_access.pairing.start",
      ),
    ).toHaveLength(1);
  });

  it("renews an expired offering without rescanning", async () => {
    const mutate = mutationTransport();
    const renewPending = {
      ...pending,
      state: "expired" as const,
      expires_at: Math.floor(Date.now() / 1000) - 30,
      renewable: true,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/tailscale")) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              available: true,
              connected: true,
              serve_mapped: true,
              origin: "https://fixture.example",
              error: "",
            }),
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () =>
            url.includes("/pairing/pending")
              ? { pending: [renewPending] }
              : status,
        };
      }),
    );
    render(<RemoteAccessSettings token="token" mutate={mutate} initialStatus={status} />);
    await userEvent.click(screen.getByRole("button", { name: "Pair owner phone" }));
    const renewButton = await screen.findByRole("button", { name: "Renew setup" });
    await userEvent.click(renewButton);
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.renew",
      { pairingId: "pair-1" },
    ));
  });

  it("restores a phone-scanned request after reopening settings", async () => {
    const mutate = mutationTransport();
    vi.stubGlobal("fetch", pendingFetch({ collected: true }));
    render(<RemoteAccessSettings token="token" mutate={mutate} />);
    expect(
      await screen.findByText("Pixel 10 is waiting for owner approval"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Approve owner access" }));
    await waitFor(() => expect(mutate.requestMutation).toHaveBeenCalledWith(
      "settings.remote_access.pairing.approve",
      { pairingId: "pair-1" },
    ));
  });

  it("hides local-only enable and pairing controls on a mobile owner", () => {
    render(
      <RemoteAccessSettings
        token="token"
        mutate={mutationTransport()}
        initialStatus={status}
        localSetupAvailable={false}
      />,
    );
    expect(screen.queryByRole("button", { name: "Disable" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Pair owner phone" })).not.toBeInTheDocument();
    expect(screen.getByText(/New-phone pairing and remote on\/off stay/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Revoke" })).toBeInTheDocument();
  });
});
