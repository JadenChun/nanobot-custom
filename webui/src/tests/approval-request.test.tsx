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
    expect(screen.getAllByRole("button")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /Allow & save/ })).not.toBeInTheDocument();
    expect(
      screen.getByText("rewrite most of the matched block in LEADERBOARD.md"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Approve and continue" }));

    expect(onApprove).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("offers three choices when the approval can save a Sandbox exclusion", async () => {
    const user = userEvent.setup();
    const onApproveAndSave = vi.fn();
    render(
      <ApprovalRequestCard
        summary="run outside Sandbox"
        approvalId={APPROVAL_ID}
        saveExclusionCommand="printf"
        onApprove={vi.fn()}
        onApproveAndSave={onApproveAndSave}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getAllByRole("button")).toHaveLength(3);
    expect(screen.getByRole("button", { name: "Allow once" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Deny" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Allow & save printf" }));
    expect(onApproveAndSave).toHaveBeenCalledOnce();
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

  it("clears the pending state when resetSignal increments after a rejected send", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn();
    const onCancel = vi.fn();
    const { rerender } = render(
      <ApprovalRequestCard
        summary="overwrite storyboard_v3.md"
        approvalId={APPROVAL_ID}
        resetSignal={0}
        onApprove={onApprove}
        onCancel={onCancel}
      />,
    );

    const approveButton = screen.getByRole("button", { name: "Approve and continue" });
    await user.click(approveButton);
    expect(approveButton).toBeDisabled();

    rerender(
      <ApprovalRequestCard
        summary="overwrite storyboard_v3.md"
        approvalId={APPROVAL_ID}
        resetSignal={1}
        onApprove={onApprove}
        onCancel={onCancel}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve and continue" })).toBeEnabled();
  });
});
