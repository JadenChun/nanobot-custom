import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HarnessControl } from "@/components/thread/HarnessControl";

describe("HarnessControl", () => {
  it("displays only the selected value and switches between Kohaku and Claude", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { rerender } = render(
      <HarnessControl value="native" claudeAvailable onChange={onChange} />,
    );

    const trigger = screen.getByRole("button", { name: "Harness: Kohaku" });
    expect(trigger).toHaveTextContent("Kohaku");
    expect(trigger).not.toHaveTextContent("Harness");

    await user.click(trigger);
    await user.click(await screen.findByRole("menuitem", { name: "Claude" }));
    expect(onChange).toHaveBeenCalledWith("claude_sdk");

    rerender(<HarnessControl value="claude_sdk" claudeAvailable onChange={onChange} />);
    const claudeTrigger = screen.getByRole("button", { name: "Harness: Claude" });
    expect(claudeTrigger).toHaveTextContent("Claude");
    expect(claudeTrigger).not.toHaveTextContent("Claude Code");
  });

  it("disables Claude with an explanation for temporary chats", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<HarnessControl value="native" claudeAvailable temporary onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Harness: Kohaku" }));
    const claudeItem = await screen.findByRole("menuitem", { name: "Claude" });
    expect(claudeItem).toHaveAttribute("aria-disabled", "true");
    expect(screen.getAllByText("Claude requires a saved chat.").length).toBeGreaterThan(0);
    expect(onChange).not.toHaveBeenCalled();
  });
});
