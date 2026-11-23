import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useSidebarState } from "@/hooks/useSidebarState";
import type { KohakuClient } from "@/lib/kohaku-client";
import type { SidebarStatePayload } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

describe("useSidebarState", () => {
  it("serializes full-state writes so an older request cannot overwrite a newer update", async () => {
    let resolveFirstWrite: (() => void) | null = null;
    let sidebarStateUpdateHandler: ((state: SidebarStatePayload) => void) | null = null;
    const setSidebarState = vi.fn()
      .mockImplementationOnce((state: SidebarStatePayload) => new Promise<SidebarStatePayload>(
        (resolve) => {
          resolveFirstWrite = () => resolve(state);
        },
      ))
      .mockImplementation(async (state: SidebarStatePayload) => state);
    const client = {
      status: "open" as const,
      onStatus: () => () => {},
      onSidebarStateUpdate: (handler: (state: SidebarStatePayload) => void) => {
        sidebarStateUpdateHandler = handler;
        return () => {
          sidebarStateUpdateHandler = null;
        };
      },
      setSidebarState,
    } as unknown as KohakuClient;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ClientProvider client={client} token="token">
        {children}
      </ClientProvider>
    );
    const { result } = renderHook(() => useSidebarState([], false), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "First" },
      }));
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "Second" },
      }));
    });

    expect(setSidebarState).toHaveBeenCalledTimes(1);
    act(() => {
      sidebarStateUpdateHandler?.(setSidebarState.mock.calls[0]?.[0]);
    });
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Second",
    });
    act(() => resolveFirstWrite?.());
    await waitFor(() => expect(setSidebarState).toHaveBeenCalledTimes(2));
    expect(setSidebarState.mock.calls[1]?.[0]).toEqual(expect.objectContaining({
      title_overrides: { "websocket:a": "Second" },
    }));
    // C3: the older response must not regress the newer accepted edit.
    await waitFor(() => expect(result.current.saveState).toBe("saved"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Second",
    });
  });
});

describe("useSidebarState save status", () => {
  function setup(setSidebarState: ReturnType<typeof vi.fn>) {
    const client = {
      status: "open" as const,
      onStatus: () => () => {},
      onSidebarStateUpdate: () => () => {},
      setSidebarState,
    } as unknown as KohakuClient;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ClientProvider client={client} token="token">
        {children}
      </ClientProvider>
    );
    return renderHook(() => useSidebarState([], false), { wrapper });
  }

  it("reports a failed save, keeps the optimistic state, and sends a changed payload under a new request id", async () => {
    const setSidebarState = vi.fn()
      .mockRejectedValueOnce(new Error("save failed"))
      .mockImplementation(async (state: SidebarStatePayload) => state);
    const { result } = setup(setSidebarState);
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "Optimistic" },
      }));
    });

    await waitFor(() => expect(result.current.saveState).toBe("failed"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Optimistic",
    });
    expect(setSidebarState).toHaveBeenCalledTimes(1);
    const requestId = setSidebarState.mock.calls[0]?.[1];
    expect(requestId).toEqual(expect.any(String));

    act(() => {
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "Retried" },
      }));
    });

    await waitFor(() => expect(setSidebarState).toHaveBeenCalledTimes(2));
    // C2: a changed payload must not reuse the failed request identity.
    const changedRequestId = setSidebarState.mock.calls[1]?.[1];
    expect(changedRequestId).toEqual(expect.any(String));
    expect(changedRequestId).not.toBe(requestId);
    await waitFor(() => expect(result.current.saveState).toBe("saved"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Retried",
    });
  });

  it("retries an unchanged failed save with the same request id and identical payload", async () => {
    const setSidebarState = vi.fn()
      .mockRejectedValueOnce(new Error("save failed"))
      .mockImplementation(async (state: SidebarStatePayload) => state);
    const { result } = setup(setSidebarState);
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "Same" },
      }));
    });

    await waitFor(() => expect(result.current.saveState).toBe("failed"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Same",
    });
    expect(setSidebarState).toHaveBeenCalledTimes(1);

    // C1: the automatic retry keeps one identity and one byte-identical payload.
    await waitFor(() => expect(setSidebarState).toHaveBeenCalledTimes(2), {
      timeout: 5_000,
    });
    expect(setSidebarState.mock.calls[1]?.[1]).toBe(setSidebarState.mock.calls[0]?.[1]);
    expect(setSidebarState.mock.calls[1]?.[0]).toEqual(setSidebarState.mock.calls[0]?.[0]);

    await waitFor(() => expect(result.current.saveState).toBe("saved"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Same",
    });
  });

  it("reports saved after a successful save", async () => {
    let resolveSave: (() => void) | null = null;
    const setSidebarState = vi.fn(
      (state: SidebarStatePayload) => new Promise<SidebarStatePayload>((resolve) => {
        resolveSave = () => resolve(state);
      }),
    );
    const { result } = setup(setSidebarState);
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      void result.current.update((current) => ({
        ...current,
        title_overrides: { "websocket:a": "Saved" },
      }));
    });

    await waitFor(() => expect(setSidebarState).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(result.current.saveState).toBe("pending"));
    act(() => resolveSave?.());

    await waitFor(() => expect(result.current.saveState).toBe("saved"));
    expect(result.current.state.title_overrides).toEqual({
      "websocket:a": "Saved",
    });
  });
});
