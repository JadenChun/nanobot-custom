import type { SkillsPayload } from "@/lib/types";

export const SKILLS_CHANGED_EVENT = "nanobot:skills-changed";
const eventScopes = new WeakMap<Event, string>();

export function skillsEventMatchesScope(event: Event, token: string): boolean | undefined {
  const scope = eventScopes.get(event);
  return scope === undefined ? undefined : scope === token;
}

export function isSkillsPayload(value: unknown): value is SkillsPayload {
  return (
    !!value
    && typeof value === "object"
    && Array.isArray((value as { skills?: unknown }).skills)
  );
}

export function notifySkillsChanged(payload: SkillsPayload, token: string): void {
  if (typeof window === "undefined") return;
  const event = new CustomEvent<SkillsPayload>(SKILLS_CHANGED_EVENT, {
    detail: payload,
  });
  eventScopes.set(event, token);
  window.dispatchEvent(event);
}
