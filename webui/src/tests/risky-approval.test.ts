import { describe, expect, it } from "vitest";

import { parseRiskyApprovalPrompt } from "@/lib/risky-approval";

const APPROVAL_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90";
const PROMPT = `I'm about to take a risky action and want your approval first. Planned action: rewrite most of the matched block in LEADERBOARD.md. Approval ID: ${APPROVAL_ID}. Reply with the approval control to continue or deny.`;

describe("parseRiskyApprovalPrompt", () => {
  it("extracts the planned action and approval id from the policy prompt", () => {
    expect(parseRiskyApprovalPrompt(PROMPT)).toEqual({
      summary: "rewrite most of the matched block in LEADERBOARD.md",
      approvalId: APPROVAL_ID,
    });
  });

  it("tolerates surrounding whitespace", () => {
    expect(parseRiskyApprovalPrompt(`\n  ${PROMPT}  \n`)).toEqual({
      summary: "rewrite most of the matched block in LEADERBOARD.md",
      approvalId: APPROVAL_ID,
    });
  });

  it("returns a null approval id when the prompt has no approval id segment", () => {
    expect(
      parseRiskyApprovalPrompt(
        "I'm about to take a risky action and want your approval first. Planned action: rewrite most of the matched block in LEADERBOARD.md. Reply with the approval control to continue or deny.",
      ),
    ).toEqual({
      summary: "rewrite most of the matched block in LEADERBOARD.md",
      approvalId: null,
    });
  });

  it("does not treat ordinary assistant text as an approval request", () => {
    expect(parseRiskyApprovalPrompt("I will rewrite the leaderboard now.")).toBeNull();
    expect(parseRiskyApprovalPrompt("I'm about to take a risky action.")).toBeNull();
    expect(parseRiskyApprovalPrompt(`${PROMPT} extra`)).toBeNull();
  });
});
