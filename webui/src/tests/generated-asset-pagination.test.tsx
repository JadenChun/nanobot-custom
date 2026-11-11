import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useGeneratedAssetLibrary } from "@/hooks/useGeneratedAssetLibrary";
import { useSessionGeneratedAssets } from "@/hooks/useSessionGeneratedAssets";
import { fetchAllGeneratedAssets, fetchSessionGeneratedAssets } from "@/lib/api";
import type { SessionGeneratedAssetsPayload } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  fetchAllGeneratedAssets: vi.fn(),
  fetchSessionGeneratedAssets: vi.fn(),
}));

function deferred() {
  let resolve!: (value: SessionGeneratedAssetsPayload) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<SessionGeneratedAssetsPayload>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function page(id: string, next: string | null = null): SessionGeneratedAssetsPayload {
  return {
    schemaVersion: 1,
    sessionKey: "websocket:a",
    total: 2,
    next,
    groups: [{
      id, turnId: id, toolCallId: id, toolName: "generate_image", sequence: 1,
      arguments: {}, artifacts: [],
    }],
  };
}

beforeEach(() => vi.resetAllMocks());

for (const surface of ["session", "library"] as const) {
  describe(`${surface} media pagination`, () => {
    function setup() {
      const fetchPage = surface === "session"
        ? vi.mocked(fetchSessionGeneratedAssets)
        : vi.mocked(fetchAllGeneratedAssets);
      const more = deferred();
      fetchPage
        .mockResolvedValueOnce(page("first", "cursor"))
        .mockReturnValueOnce(more.promise);
      const hook = renderHook(
        ({ active, token, refreshKey }) => surface === "session"
          ? useSessionGeneratedAssets(active, token, "websocket:a", refreshKey)
          : useGeneratedAssetLibrary(active, token, refreshKey),
        { initialProps: { active: true, token: "token", refreshKey: 0 } },
      );
      return { ...hook, fetchPage, more };
    }

    it("ignores an old page after refresh", async () => {
      const { result, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.groups[0]?.id).toBe("first"));
      act(() => result.current.loadMore());
      fetchPage.mockResolvedValueOnce(page("refreshed"));
      act(() => result.current.refresh());
      await waitFor(() => expect(result.current.groups[0]?.id).toBe("refreshed"));
      await act(async () => more.resolve(page("stale", "old-next")));
      expect(result.current.groups.map((group) => group.id)).toEqual(["refreshed"]);
      expect(result.current.next).toBeNull();
    });

    it("requests a page once when load more is clicked twice before rendering", async () => {
      const { result, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.next).toBe("cursor"));
      act(() => {
        result.current.loadMore();
        result.current.loadMore();
      });
      expect(fetchPage).toHaveBeenCalledTimes(2);
      await act(async () => more.resolve(page("second")));
      expect(result.current.groups.map((group) => group.id)).toEqual(["first", "second"]);
    });

    it("retries a failed page with its original cursor", async () => {
      const { result, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.next).toBe("cursor"));
      act(() => result.current.loadMore());
      await act(async () => more.reject(new Error("offline")));
      expect(result.current.groups.map((group) => group.id)).toEqual(["first"]);
      expect(result.current.next).toBe("cursor");
      fetchPage.mockResolvedValueOnce(page("retried"));
      act(() => result.current.loadMore());
      await waitFor(() => expect(result.current.next).toBeNull());
      expect(result.current.groups.map((group) => group.id)).toEqual(["first", "retried"]);
      expect(fetchPage.mock.calls.at(-1)?.at(-1)).toEqual({ cursor: "cursor", limit: 50 });
    });

    it("discards old credentials' data and pending pages", async () => {
      const { result, rerender, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.next).toBe("cursor"));
      act(() => result.current.loadMore());
      const replacement = deferred();
      fetchPage.mockReturnValueOnce(replacement.promise);
      rerender({ active: true, token: "new-token", refreshKey: 0 });
      expect(result.current.groups).toEqual([]);
      await act(async () => more.resolve(page("old-private-page")));
      expect(result.current.groups).toEqual([]);
      expect(result.current.loading).toBe(true);
      await act(async () => replacement.resolve(page("new-private-page")));
      expect(result.current.groups.map((group) => group.id)).toEqual(["new-private-page"]);
    });

    it("ignores a closed view's page and allows pagination after reopening", async () => {
      const { result, rerender, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.next).toBe("cursor"));
      act(() => result.current.loadMore());
      rerender({ active: false, token: "token", refreshKey: 0 });
      await act(async () => more.resolve(page("closed-page")));
      expect(result.current.groups.map((group) => group.id)).not.toContain("closed-page");
      fetchPage.mockResolvedValueOnce(page("reopened", "new-cursor"));
      rerender({ active: true, token: "token", refreshKey: 0 });
      await waitFor(() => expect(result.current.groups[0]?.id).toBe("reopened"));
      fetchPage.mockResolvedValueOnce(page("reopened-more"));
      act(() => result.current.loadMore());
      await waitFor(() => expect(result.current.next).toBeNull());
      expect(result.current.groups.map((group) => group.id)).toEqual(["reopened", "reopened-more"]);
    });

    it("does not let a stale finalizer clear a newer page's loading flag", async () => {
      const { result, fetchPage, more } = setup();
      await waitFor(() => expect(result.current.next).toBe("cursor"));
      act(() => result.current.loadMore());
      fetchPage.mockResolvedValueOnce(page("refreshed", "new-cursor"));
      act(() => result.current.refresh());
      await waitFor(() => expect(result.current.groups[0]?.id).toBe("refreshed"));
      const newPage = deferred();
      fetchPage.mockReturnValueOnce(newPage.promise);
      act(() => result.current.loadMore());
      expect(result.current.loadingMore).toBe(true);
      await act(async () => more.resolve(page("stale")));
      expect(result.current.loadingMore).toBe(true);
      await act(async () => newPage.resolve(page("new-page")));
      expect(result.current.loadingMore).toBe(false);
      expect(result.current.groups.map((group) => group.id)).toEqual(["refreshed", "new-page"]);
    });
  });
}

it("keeps a page from chat A out of chat B", async () => {
  const oldPage = deferred();
  vi.mocked(fetchSessionGeneratedAssets)
    .mockResolvedValueOnce(page("a-first", "cursor"))
    .mockReturnValueOnce(oldPage.promise)
    .mockResolvedValueOnce(page("b-first"));
  const { result, rerender } = renderHook(
    ({ session }) => useSessionGeneratedAssets(true, "token", session),
    { initialProps: { session: "websocket:a" } },
  );
  await waitFor(() => expect(result.current.next).toBe("cursor"));
  act(() => result.current.loadMore());
  rerender({ session: "websocket:b" });
  await waitFor(() => expect(result.current.groups[0]?.id).toBe("b-first"));
  await act(async () => oldPage.resolve(page("a-more")));
  expect(result.current.groups.map((group) => group.id)).toEqual(["b-first"]);
});
