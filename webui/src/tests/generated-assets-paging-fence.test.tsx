import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSessionGeneratedAssets } from "@/hooks/useSessionGeneratedAssets";

const fetchSessionGeneratedAssetsMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchSessionGeneratedAssets: fetchSessionGeneratedAssetsMock,
  };
});

type AssetGroup = { sessionKey: string; turnId: string };
type AssetPage = { groups: AssetGroup[]; total: number; next: string | null };

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function AssetProbe({ sessionKey }: { sessionKey: string }) {
  const { groups, next, loadMore } = useSessionGeneratedAssets(true, "token", sessionKey);
  return (
    <div>
      <div data-testid="groups">
        {groups.map((group) => group.turnId ?? "unknown").join(",")}
      </div>
      <div data-testid="next">{next ?? "none"}</div>
      <button
        type="button"
        onClick={() => {
          void loadMore();
        }}
      >
        load-more
      </button>
    </div>
  );
}

function requestedPage(sessionKey: string, cursor?: string): boolean {
  return fetchSessionGeneratedAssetsMock.mock.calls.some(
    ([, calledSessionKey, , options]) =>
      calledSessionKey === sessionKey
      && (options as { cursor?: string } | undefined)?.cursor === cursor,
  );
}

describe("useSessionGeneratedAssets asset-page fence", () => {
  beforeEach(() => {
    fetchSessionGeneratedAssetsMock.mockReset();
  });

  it("discards a late asset page after the session changes", async () => {
    const firstA = deferred<AssetPage>();
    const moreA = deferred<AssetPage>();
    const pageB: AssetPage = {
      groups: [{ sessionKey: "websocket:b", turnId: "turn-b" }],
      total: 1,
      next: null,
    };

    fetchSessionGeneratedAssetsMock.mockImplementation((...args: unknown[]) => {
      const sessionKey = args[1];
      const options = args[3] as { cursor?: string; limit?: number } | undefined;
      if (sessionKey === "websocket:a") {
        return options?.cursor === "cursor-a" ? moreA.promise : firstA.promise;
      }
      if (sessionKey === "websocket:b" && options?.cursor === undefined) {
        return Promise.resolve(pageB);
      }
      return Promise.reject(new Error(`unexpected asset request for ${String(sessionKey)}`));
    });

    const { rerender } = render(<AssetProbe sessionKey="websocket:a" />);

    await waitFor(() => {
      expect(fetchSessionGeneratedAssetsMock).toHaveBeenCalled();
    });
    expect(fetchSessionGeneratedAssetsMock.mock.calls[0]?.[1]).toBe("websocket:a");
    expect(fetchSessionGeneratedAssetsMock.mock.calls[0]?.[3]).toEqual({ limit: 50 });

    await act(async () => {
      firstA.resolve({
        groups: [{ sessionKey: "websocket:a", turnId: "turn-a" }],
        total: 2,
        next: "cursor-a",
      });
      await firstA.promise;
    });

    await waitFor(() => {
      expect(screen.getByTestId("groups")).toHaveTextContent("turn-a");
    });
    expect(screen.getByTestId("next")).toHaveTextContent("cursor-a");

    fireEvent.click(screen.getByRole("button", { name: "load-more" }));

    await waitFor(() => {
      expect(requestedPage("websocket:a", "cursor-a")).toBe(true);
    });

    rerender(<AssetProbe sessionKey="websocket:b" />);

    await waitFor(() => {
      expect(requestedPage("websocket:b")).toBe(true);
      const groupsText = screen.getByTestId("groups").textContent ?? "";
      expect(groupsText).toContain("turn-b");
      expect(groupsText).not.toContain("turn-a");
    });

    const groupsBeforeLatePage = screen.getByTestId("groups").textContent;
    expect(groupsBeforeLatePage?.split(",").filter(Boolean)).toEqual(["turn-b"]);

    await act(async () => {
      moreA.resolve({
        groups: [{ sessionKey: "websocket:a", turnId: "turn-a-2" }],
        total: 2,
        next: null,
      });
      await moreA.promise;
    });

    await waitFor(() => {
      const renderedGroups = (screen.getByTestId("groups").textContent ?? "")
        .split(",")
        .filter(Boolean);
      expect(renderedGroups).toEqual(["turn-b"]);
      expect(renderedGroups).not.toContain("turn-a");
      expect(renderedGroups).not.toContain("turn-a-2");
    });
    expect(screen.getByTestId("groups").textContent).toBe(groupsBeforeLatePage);
  });
});
