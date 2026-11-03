import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchRemoteAccessStatus,
  revokeRemoteDevice,
} from "@/lib/remote-access-api";
import {
  installMobileRuntime,
  resetMobileRuntime,
  setMobileOwnerCapability,
  type MobileRuntime,
} from "@/lib/mobile-runtime";

function ownerRuntime() {
  const remoteStatus = vi.fn().mockResolvedValue({
    ok: true,
    enabled: true,
    installation_id: "install-owner",
    scopes: ["chat.read", "owner"],
  });
  const remoteDevices = vi.fn().mockResolvedValue({
    ok: true,
    devices: [{
      installation_id: "install-owner",
      label: "phone",
      scopes: ["chat.read", "owner"],
      status: "active",
      created_at: 1,
      last_seen_at: 2,
    }],
  });
  const remoteRevoke = vi.fn().mockResolvedValue({ ok: true });
  const refresh = vi.fn().mockResolvedValue(true);
  const runtime: MobileRuntime = {
    transport: {
      remoteStatus,
      remoteDevices,
      remoteRevoke,
      refresh,
    } as unknown as MobileRuntime["transport"],
    request: vi.fn(),
    handshake: vi.fn(),
    read: vi.fn(),
    onEvent: vi.fn(),
    createWebSocket: vi.fn(),
    recover: vi.fn(),
  };
  return { runtime, remoteStatus, remoteDevices, remoteRevoke, refresh };
}

afterEach(() => resetMobileRuntime());

describe("mobile owner remote access", () => {
  it("reads status and devices through the independent owner transport", async () => {
    const { runtime, remoteStatus, remoteDevices } = ownerRuntime();
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);

    await expect(fetchRemoteAccessStatus("token")).resolves.toEqual({
      enabled: true,
      installation_id: "install-owner",
      devices: [{
        installation_id: "install-owner",
        label: "phone",
        scopes: ["chat.read", "owner"],
        status: "active",
        created_at: 1,
        last_seen_at: 2,
      }],
      owner_available: true,
    });
    expect(remoteStatus).toHaveBeenCalledOnce();
    expect(remoteDevices).toHaveBeenCalledOnce();
  });

  it("refreshes once and retries an owner read after 401", async () => {
    const { runtime, remoteStatus, remoteDevices, refresh } = ownerRuntime();
    remoteStatus
      .mockRejectedValueOnce(new Error("HTTP 401 token expired"))
      .mockResolvedValue({
        ok: true,
        enabled: true,
        installation_id: "install-owner",
        scopes: ["chat.read", "owner"],
      });
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);

    await expect(fetchRemoteAccessStatus("token")).resolves.toMatchObject({
      enabled: true,
      installation_id: "install-owner",
    });
    expect(refresh).toHaveBeenCalledOnce();
    expect(remoteDevices).toHaveBeenCalledOnce();
  });

  it("falls back to the owner transport when the gateway is unavailable", async () => {
    const { runtime, remoteRevoke } = ownerRuntime();
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);
    const transport = {
      requestMutation: vi.fn().mockRejectedValue(new Error("socket is not open")),
    };

    await revokeRemoteDevice(transport, "install-victim");

    expect(transport.requestMutation).toHaveBeenCalledOnce();
    expect(remoteRevoke).toHaveBeenCalledWith("install-victim");
  });

  it("uses the gateway revoke first while it is reachable", async () => {
    const { runtime, remoteRevoke } = ownerRuntime();
    installMobileRuntime(runtime);
    setMobileOwnerCapability(true);
    const transport = { requestMutation: vi.fn().mockResolvedValue({ devices: [] }) };

    await revokeRemoteDevice(transport, "install-victim");

    expect(transport.requestMutation).toHaveBeenCalledOnce();
    expect(remoteRevoke).not.toHaveBeenCalled();
  });
});
