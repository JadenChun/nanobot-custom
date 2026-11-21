import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApprovalRequestCard } from "@/components/thread/ApprovalRequestCard";

const APPROVAL_ID = "0123456789abcdef0123456789abcdef";

afterEach(cleanup);

describe("ApprovalRequestCard", () => {
  it("submits the affirmative reply through the existing action path", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn();
    const onCancel = vi.fn();
    render(
      <ApprovalRequestCard
        summary="rewrite most of the matched block in LEADERBOARD.md"
        approvalId={APPROVAL_ID}
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
        approvalId={APPROVAL_ID}
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
        approvalId={APPROVAL_ID}
        busy
        onApprove={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve and continue" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });

  it("disables both actions and shows the update-required message when the approval id is null", () => {
    render(
      <ApprovalRequestCard
        summary="rewrite the roadmap"
        approvalId={null}
        onApprove={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve and continue" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(
      screen.getByText(
        "This approval request needs an updated client. Chat text cannot grant it.",
      ),
    ).toBeInTheDocument();
  });

  it("disables both actions and shows the update-required message when the approval id is missing", () => {
    render(
      <ApprovalRequestCard
        summary="rewrite the roadmap"
        onApprove={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve and continue" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(
      screen.getByText(
        "This approval request needs an updated client. Chat text cannot grant it.",
      ),
    ).toBeInTheDocument();
  });
});
