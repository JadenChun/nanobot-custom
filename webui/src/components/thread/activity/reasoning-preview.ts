const PREVIEW_SOURCE_TAIL_CHARS = 2_000;
const PREVIEW_MAX_CHARS = 240;

/**
 * Compact a reasoning stream into a single-line preview.
 *
 * Reasoning can grow to tens of thousands of characters. Only the recent tail
 * is shown in the activity row, so normalize that slice instead of rescanning
 * the whole string on every stream frame.
 */
export function compactReasoningPreview(
  value: string,
  maxLength = PREVIEW_MAX_CHARS,
): string {
  const source = value.length > PREVIEW_SOURCE_TAIL_CHARS
    ? value.slice(-PREVIEW_SOURCE_TAIL_CHARS)
    : value;
  const compacted = source
    .replace(/\[([^\]]+)]\([^)]+\)/g, "$1")
    .replace(/[*_#`~]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return compacted.length > maxLength
    ? `${compacted.slice(0, Math.max(0, maxLength - 3))}...`
    : compacted;
}
