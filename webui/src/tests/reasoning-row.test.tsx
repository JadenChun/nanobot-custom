import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { preloadMarkdownText } from "@/components/MarkdownText";
import { ReasoningRow } from "@/components/thread/activity/ReasoningRow";

describe("ReasoningRow", () => {
  it("discloses complete reasoning Markdown and removes its body when collapsed", async () => {
    await act(async () => {
      await preloadMarkdownText();
    });

    const source = [
      "BEGIN_REASONING_SOURCE",
      "**MIDDLE EMPHASIS**",
      "middle detail ".repeat(190),
      "[REFERENCE](https://example.com/guide)",
      "END_REASONING_SOURCE",
    ].join("\n\n");
    expect(source.length).toBeGreaterThan(2_000);

    render(<ReasoningRow text={source} streaming={false} />);

    const disclosure = screen.getByRole("button", { name: "Reasoning" });
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(disclosure).toHaveAttribute("aria-controls");
    expect(disclosure).not.toHaveTextContent("BEGIN_REASONING_SOURCE");

    const body = document.getElementById(disclosure.getAttribute("aria-controls")!);
    expect(body).toBeInTheDocument();
    expect(body).toHaveAttribute("role", "region");
    expect(body).toHaveAttribute("hidden");
    expect(body).toHaveAttribute("aria-hidden", "true");
    expect(body).toBeEmptyDOMElement();

    await userEvent.click(disclosure);

    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(body).not.toHaveAttribute("hidden");
    await within(body!).findByText("MIDDLE EMPHASIS", { selector: "strong" });
    expect(within(body!).getByText("BEGIN_REASONING_SOURCE")).toBeInTheDocument();
    expect(within(body!).getByRole("link", { name: "REFERENCE" })).toHaveAttribute(
      "href",
      "https://example.com/guide",
    );
    expect(within(body!).getByText("END_REASONING_SOURCE")).toBeInTheDocument();

    await userEvent.click(disclosure);

    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(body).toHaveAttribute("hidden");
    expect(body).toBeEmptyDOMElement();
  });

  it("uses independent native disclosures and omits controls for empty text", async () => {
    render(
      <>
        <ReasoningRow text="first thought" streaming={false} />
        <ReasoningRow text="second thought" streaming={false} />
        <ReasoningRow text={" \n  "} streaming />
      </>,
    );

    const rows = screen.getAllByTestId("activity-step");
    const disclosures = rows.flatMap((row) => {
      const disclosure = within(row).queryByRole("button", { name: "Reasoning" });
      return disclosure ? [disclosure] : [];
    });
    expect(disclosures).toHaveLength(2);
    const emptyRow = screen.getByText("Thinking…").closest('[data-testid="activity-step"]');
    expect(emptyRow).not.toBeNull();
    expect(within(emptyRow!).queryByRole("button")).not.toBeInTheDocument();
    const ids = disclosures.map((disclosure) => disclosure.getAttribute("aria-controls"));
    expect(new Set(ids).size).toBe(2);
    for (const [index, disclosure] of disclosures.entries()) {
      expect(disclosure).toHaveAttribute("aria-expanded", "false");
      expect(document.getElementById(ids[index]!)).toBeInTheDocument();
    }
    expect(screen.getByText("Thinking…")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.tab();
    expect(disclosures[0]).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(disclosures[0]).toHaveAttribute("aria-expanded", "true");

    await user.tab();
    expect(disclosures[1]).toHaveFocus();
    await user.keyboard(" ");
    expect(disclosures[1]).toHaveAttribute("aria-expanded", "true");
    expect(disclosures[0]).toHaveAttribute("aria-expanded", "true");
  });

  it("streams multi-line reasoning verbosely in a bounded viewport and auto-collapses when done", () => {
    const { rerender } = render(
      <ReasoningRow text={"checking imports\n\nreading viewport"} streaming />,
    );
    const disclosure = screen.getByRole("button", { name: "Reasoning" });
    const body = document.getElementById(disclosure.getAttribute("aria-controls")!)!;

    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("reasoning-live-stream")).toBeInTheDocument();
    expect(body).toHaveTextContent("checking imports");
    expect(body).toHaveTextContent("reading viewport");

    rerender(
      <ReasoningRow text={"checking imports\n\nreading viewport\n\nverifying layout"} streaming />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(body).toHaveTextContent("verifying layout");

    rerender(
      <ReasoningRow
        text={"checking imports\n\nreading viewport\n\nverifying layout"}
        streaming={false}
      />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(body).toHaveAttribute("hidden");
    expect(body).toBeEmptyDOMElement();
    expect(screen.getByTestId("activity-reasoning-marker")).toHaveAttribute("data-state", "done");
  });

  it("preserves the user's manual open or closed choice as streaming reasoning updates", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ReasoningRow text={"initial thought\n\nsecond line"} streaming />,
    );
    const disclosure = screen.getByRole("button", { name: "Reasoning" });
    const body = document.getElementById(disclosure.getAttribute("aria-controls")!)!;

    // Multi-line streaming starts expanded; clicking collapses it and keeps it collapsed.
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    await user.click(disclosure);
    rerender(<ReasoningRow text={"initial thought\n\nsecond line\n\nappended thought"} streaming />);
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(body).toHaveAttribute("hidden");
    expect(body).toBeEmptyDOMElement();

    // Clicking again pins it open across streaming updates and completion.
    await user.click(disclosure);
    rerender(
      <ReasoningRow
        text={"initial thought\n\nsecond line\n\nappended thought"}
        streaming={false}
      />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("activity-reasoning-marker")).toHaveAttribute("data-state", "done");
    expect(body).toHaveTextContent("appended thought");

    await user.click(disclosure);
    rerender(
      <ReasoningRow
        text={"initial thought\n\nsecond line\n\nappended thought\n\nclosed update"}
        streaming={false}
      />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(body).toHaveAttribute("hidden");
    expect(body).toBeEmptyDOMElement();
    expect(screen.getByTestId("activity-reasoning-marker")).toHaveAttribute("data-state", "done");
  });

  it("uses compact Thinking… / Thought / Thought for Xs headers without truncated previews", () => {
    const { rerender } = render(<ReasoningRow text="short streaming note" streaming />);

    const disclosure = screen.getByRole("button", { name: "Reasoning" });
    expect(disclosure).toHaveTextContent("Thinking…");
    expect(disclosure).not.toHaveTextContent("short streaming note");
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("reasoning-live-stream")).toHaveTextContent("short streaming note");

    rerender(<ReasoningRow text="short streaming note" streaming={false} durationMs={16_000} />);
    expect(disclosure).toHaveTextContent("Thought for 16s");
    expect(disclosure).toHaveAttribute("aria-expanded", "false");

    rerender(<ReasoningRow text="short streaming note" streaming nestedInScrollViewport />);
    expect(screen.getByTestId("reasoning-live-stream")).not.toHaveClass("overflow-y-auto");

    rerender(<ReasoningRow text="short streaming note" streaming={false} />);
    expect(disclosure).toHaveTextContent("Thought");
    expect(disclosure).not.toHaveTextContent("Thought for");
  });

  it("keeps auto-following the latest reasoning line across rapid content growth until the user scrolls up", () => {
    const { rerender } = render(<ReasoningRow text="line 1" streaming />);
    const viewport = screen.getByTestId("reasoning-live-stream");

    let scrollHeight = 200;
    Object.defineProperty(viewport, "clientHeight", { configurable: true, get: () => 100 });
    Object.defineProperty(viewport, "scrollHeight", { configurable: true, get: () => scrollHeight });

    rerender(<ReasoningRow text={"line 1\n\nline 2"} streaming />);
    expect(viewport.scrollTop).toBe(100);

    // Content grows faster than a queued scroll event without the user scrolling up.
    scrollHeight = 360;
    fireEvent.scroll(viewport);

    rerender(<ReasoningRow text={"line 1\n\nline 2\n\nline 3"} streaming />);
    expect(viewport.scrollTop).toBe(260);

    // User actually scrolls upward -> auto-follow pauses.
    viewport.scrollTop = 40;
    fireEvent.scroll(viewport);

    scrollHeight = 440;
    rerender(<ReasoningRow text={"line 1\n\nline 2\n\nline 3\n\nline 4"} streaming />);
    expect(viewport.scrollTop).toBe(40);

    // User scrolls back near the bottom -> auto-follow resumes.
    viewport.scrollTop = 335;
    fireEvent.scroll(viewport);

    scrollHeight = 500;
    rerender(<ReasoningRow text={"line 1\n\nline 2\n\nline 3\n\nline 4\n\nline 5"} streaming />);
    expect(viewport.scrollTop).toBe(400);
  });
});
