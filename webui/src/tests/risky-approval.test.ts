import { describe, expect, it } from "vitest";

import { parseRiskyApprovalPrompt } from "@/lib/risky-approval";

const PROMPT =
  "I'm about to take a risky action and want your approval first. Planned action: rewrite most of the matched block in LEADERBOARD.md. Reply `yes` to continue or `no` to cancel.";

describe("parseRiskyApprovalPrompt", () => {
  it("extracts the planned action from the policy prompt", () => {
    expect(parseRiskyApprovalPrompt(PROMPT)).toEqual({
      summary: "rewrite most of the matched block in LEADERBOARD.md",
    });
  });

  it("tolerates surrounding whitespace", () => {
    expect(parseRiskyApprovalPrompt(`\n  ${PROMPT}  \n`)).toEqual({
      summary: "rewrite most of the matched block in LEADERBOARD.md",
    });
  });

  it("does not treat ordinary assistant text as an approval request", () => {
    expect(parseRiskyApprovalPrompt("I will rewrite the leaderboard now.")).toBeNull();
    expect(parseRiskyApprovalPrompt("I'm about to take a risky action.")).toBeNull();
    expect(parseRiskyApprovalPrompt(`${PROMPT} extra`)).toBeNull();
  });
});
