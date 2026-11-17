import { act, render, screen, within } from "@testing-library/react";
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

  it("preserves the open or closed choice as streaming reasoning updates", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ReasoningRow text="initial thought" streaming />,
    );
    const disclosure = screen.getByRole("button", { name: "Reasoning" });
    const body = document.getElementById(disclosure.getAttribute("aria-controls")!)!;

    await user.click(disclosure);
    rerender(<ReasoningRow text={"initial thought\n\nappended thought"} streaming />);
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(body).toHaveTextContent("appended thought");

    rerender(
      <ReasoningRow text={"initial thought\n\nappended thought"} streaming={false} />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("activity-reasoning-marker")).toHaveAttribute("data-state", "done");
    expect(body).toHaveTextContent("appended thought");

    await user.click(disclosure);
    rerender(
      <ReasoningRow
        text={"initial thought\n\nappended thought\n\nclosed update"}
        streaming={false}
      />,
    );
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(body).toHaveAttribute("hidden");
    expect(body).toBeEmptyDOMElement();
    expect(screen.getByTestId("activity-reasoning-marker")).toHaveAttribute("data-state", "done");
  });
});
