import { lazy, Suspense } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ErrorBoundary, isChunkLoadError } from "@/components/ErrorBoundary";

describe("ErrorBoundary", () => {
  it("detects dynamic import and chunk load errors", () => {
    expect(
      isChunkLoadError(new TypeError("Failed to fetch dynamically imported module: /assets/SettingsView-abc.js")),
    ).toBe(true);
    expect(isChunkLoadError(new Error("Importing a module script failed."))).toBe(true);
    expect(isChunkLoadError(new Error("Loading chunk 42 failed"))).toBe(true);
    expect(isChunkLoadError(new Error("Normal render error"))).toBe(false);
  });

  it("renders a recoverable fallback instead of unmounting when a lazy view chunk fails", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const onBack = vi.fn();
    const onReload = vi.fn();
    const BrokenLazyView = lazy(() =>
      Promise.reject(
        new TypeError("Failed to fetch dynamically imported module: /assets/SettingsView-old.js"),
      ),
    );

    try {
      render(
        <ErrorBoundary onBack={onBack} onReload={onReload} backLabel="Back to chat">
          <Suspense fallback={<div>Loading...</div>}>
            <BrokenLazyView />
          </Suspense>
        </ErrorBoundary>,
      );

      await waitFor(() => {
        expect(screen.getByRole("alert")).toBeInTheDocument();
      });
      expect(screen.getByText("This view could not be loaded")).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Back to chat" }));
      expect(onBack).toHaveBeenCalledTimes(1);

      fireEvent.click(screen.getByRole("button", { name: "Reload" }));
      expect(onReload).toHaveBeenCalledTimes(1);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("resets error state automatically when resetKeys change", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    function Bomb({ shouldThrow }: { shouldThrow: boolean }) {
      if (shouldThrow) throw new Error("boom");
      return <div>Healthy view</div>;
    }

    try {
      const { rerender } = render(
        <ErrorBoundary resetKeys={["settings"]}>
          <Bomb shouldThrow />
        </ErrorBoundary>,
      );

      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText("boom")).toBeInTheDocument();

      rerender(
        <ErrorBoundary resetKeys={["chat"]}>
          <Bomb shouldThrow={false} />
        </ErrorBoundary>,
      );

      expect(screen.getByText("Healthy view")).toBeInTheDocument();
    } finally {
      consoleError.mockRestore();
    }
  });
});
