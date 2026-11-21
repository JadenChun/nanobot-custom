const APPROVAL_PREFIX =
  "I'm about to take a risky action and want your approval first. Planned action: ";
const APPROVAL_SUFFIX =
  ". Reply with the approval control to continue or deny.";
const APPROVAL_ID_PATTERN = /Approval ID:\s*([0-9a-f]{32})/i;

export interface RiskyApprovalPrompt {
  summary: string;
  approvalId: string | null;
}

/**
 * Recognize the current risky-action approval prompt emitted by
 * ``RiskyActionPolicy``. The ID identifies the server-owned request; it is
 * not authority. The typed approval control sends the decision.
 */
export function parseRiskyApprovalPrompt(content: string): RiskyApprovalPrompt | null {
  const text = content.trim();
  if (!text.startsWith(APPROVAL_PREFIX) || !text.endsWith(APPROVAL_SUFFIX)) {
    return null;
  }
  const body = text.slice(APPROVAL_PREFIX.length, -APPROVAL_SUFFIX.length).trim();
  const match = APPROVAL_ID_PATTERN.exec(body);
  const summary = body
    .replace(APPROVAL_ID_PATTERN, "")
    .replace(/\.\s*$/, "")
    .trim();
  return { summary: summary || body, approvalId: match?.[1] ?? null };
}
