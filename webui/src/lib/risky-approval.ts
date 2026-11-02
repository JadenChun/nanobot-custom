const APPROVAL_PREFIX =
  "I'm about to take a risky action and want your approval first. Planned action: ";
const APPROVAL_SUFFIX = ". Reply `yes` to continue or `no` to cancel.";

export interface RiskyApprovalPrompt {
  summary: string;
}

/**
 * Recognize the current risky-action approval prompt emitted by
 * ``RiskyActionPolicy``. This is presentation-only: the backend still owns the
 * pending approval and consumes the ordinary ``yes``/``no`` reply.
 */
export function parseRiskyApprovalPrompt(content: string): RiskyApprovalPrompt | null {
  const text = content.trim();
  if (!text.startsWith(APPROVAL_PREFIX) || !text.endsWith(APPROVAL_SUFFIX)) {
    return null;
  }
  const summary = text.slice(APPROVAL_PREFIX.length, -APPROVAL_SUFFIX.length).trim();
  return summary ? { summary } : null;
}
