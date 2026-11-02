import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApprovalRequestCard } from "@/components/thread/ApprovalRequestCard";

afterEach(cleanup);

describe("ApprovalRequestCard", () => {
  it("submits the affirmative reply through the existing action path", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn();
    const onCancel = vi.fn();
    render(
      <ApprovalRequestCard
        summary="rewrite most of the matched block in LEADERBOARD.md"
        onApprove={onApprove}
        onCancel={onCancel}
      />,
    );

    expect(screen.getByText("Approval required")).toBeInTheDocument();
    expect(
      screen.getByText("rewrite most of the matched block in LEADERBOARD.md"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Approve and continue" }));

    expect(onApprove).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("submits the negative reply through the existing action path", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn();
    const onCancel = vi.fn();
    render(
      <ApprovalRequestCard
        summary="rewrite the roadmap"
        onApprove={onApprove}
        onCancel={onCancel}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onApprove).not.toHaveBeenCalled();
  });

  it("disables both actions while the thread is busy", () => {
    render(
      <ApprovalRequestCard
        summary="rewrite the roadmap"
        busy
        onApprove={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve and continue" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });
});
