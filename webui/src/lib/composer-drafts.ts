/**
 * Chat-scoped composer drafts.
 *
 * - In-memory store: full draft (text, quote, mentions, ready attachments,
 *   caret/focus) so A -> B -> A restores everything without flashing B.
 * - Device-local persistence (localStorage) for saved chats only: text, quote
 *   and mentions plus attachment *names* (never payloads). After relaunch the
 *   names drive an explicit "reattach" notice.
 * - Keys are bound to the authenticated gateway/client namespace, the session
 *   key and (when known) the session incarnation, so a replaced session never
 *   inherits an older draft.
 */

export interface DraftAttachment {
  dataUrl: string;
  name?: string;
  kind?: "image" | "file";
}

export interface ComposerDraft {
  text: string;
  quotedContext: string | null;
  mentions: unknown[];
  attachments: DraftAttachment[];
  /** Attachment names that could not be restored (relaunch). */
  reattachNames: string[];
  selection: { start: number; end: number } | null;
  focused: boolean;
  /** Session incarnation this draft was written under, when known. */
  incarnation?: string | null;
}

export interface DraftScope {
  namespace: string;
  key: string;
  incarnation?: string | null;
  /** Only saved chats persist across relaunch; temporary / new-chat do not. */
  persist: boolean;
}

export const NEW_CHAT_DRAFT_KEY = "\u0000new-chat";
const STORAGE_KEY = "nanobot.webui.composerDrafts.v1";
const MAX_MEMORY_ENTRIES = 50;
const MAX_PERSISTED_ENTRIES = 30;
const MAX_TEXT_CHARS = 16_000;
const MAX_QUOTE_CHARS = 8_000;
const MAX_MENTIONS = 8;
const MAX_NAMES = 8;

const memory = new Map<string, ComposerDraft>();
/** Deleted sessions never accept a late save (e.g. from a switch-away). */
const deletedKeys = new Map<string, string | null>();

function isDeletedScope(scope: Pick<DraftScope, "key" | "incarnation">): boolean {
  if (!deletedKeys.has(scope.key)) return false;
  const deletedInc = deletedKeys.get(scope.key) ?? null;
  if (scope.incarnation && scope.incarnation !== deletedInc) {
    deletedKeys.delete(scope.key);
    return false;
  }
  return true;
}
const memoryScopes = new Map<string, { namespace: string; key: string }>();

/** Short non-reversible namespace so the bearer token never lands in storage. */
export function draftNamespaceFor(token: string | null | undefined): string {
  const input = token ?? "";
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < input.length; i += 1) {
    const c = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + c, 0x85ebca6b) >>> 0;
  }
  return `${h1.toString(36)}${h2.toString(36)}`;
}

/** Identity of the editor target. Incarnation is checked separately: it is
 * unknown until history loads and must not look like a chat switch. */
export function draftScopeId(scope: Pick<DraftScope, "namespace" | "key">): string {
  return [scope.namespace, scope.key].join("\u0000");
}

function incarnationConflicts(stored: string | null | undefined, current: string | null | undefined) {
  return !!stored && !!current && stored !== current;
}

export function isDraftEmpty(draft: ComposerDraft): boolean {
  return !draft.text.trim()
    && !draft.quotedContext
    && draft.mentions.length === 0
    && draft.attachments.length === 0
    && draft.reattachNames.length === 0;
}

interface PersistedEntry {
  ns: string;
  key: string;
  inc: string;
  text: string;
  quote: string | null;
  mentions: unknown[];
  names: string[];
  at: number;
}

function readPersisted(): PersistedEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { entries?: unknown };
    if (!Array.isArray(parsed.entries)) return [];
    return parsed.entries.filter((e): e is PersistedEntry => (
      !!e && typeof e === "object"
      && typeof (e as PersistedEntry).ns === "string"
      && typeof (e as PersistedEntry).key === "string"
      && typeof (e as PersistedEntry).text === "string"
    ));
  } catch {
    return [];
  }
}

function writePersisted(entries: PersistedEntry[]): void {
  if (typeof window === "undefined") return;
  try {
    if (entries.length === 0) window.localStorage.removeItem(STORAGE_KEY);
    else {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ version: 1, entries: entries.slice(-MAX_PERSISTED_ENTRIES) }),
      );
    }
  } catch {
    // Storage is best-effort for drafts.
  }
}

function sameSession(e: PersistedEntry, scope: { namespace: string; key: string }): boolean {
  return e.ns === scope.namespace && e.key === scope.key;
}

export function saveComposerDraft(scope: DraftScope, draft: ComposerDraft): void {
  const id = draftScopeId(scope);
  if (isDeletedScope(scope)) return;
  const persisted = scope.persist ? readPersisted() : [];
  const priorInc = memory.get(id)?.incarnation
    ?? persisted.find((e) => sameSession(e, scope))?.inc
    ?? null;
  // Incarnation sticks once known; a later null (history not loaded) keeps it.
  const incarnation = scope.incarnation ?? priorInc ?? null;
  const stored: ComposerDraft = { ...draft, incarnation };
  if (isDraftEmpty(stored)) {
    memory.delete(id);
    memoryScopes.delete(id);
  } else {
    memory.delete(id);
    memory.set(id, stored);
    memoryScopes.set(id, { namespace: scope.namespace, key: scope.key });
    while (memory.size > MAX_MEMORY_ENTRIES) {
      const oldest = memory.keys().next().value as string;
      memory.delete(oldest);
      memoryScopes.delete(oldest);
    }
  }
  if (!scope.persist) return;
  const others = persisted.filter((e) => !sameSession(e, scope));
  if (!isDraftEmpty(stored)) {
    const names = [
      ...stored.attachments.map((a) => a.name || "attachment"),
      ...stored.reattachNames,
    ].slice(0, MAX_NAMES);
    others.push({
      ns: scope.namespace,
      key: scope.key,
      inc: incarnation ?? "",
      text: stored.text.slice(0, MAX_TEXT_CHARS),
      quote: stored.quotedContext ? stored.quotedContext.slice(0, MAX_QUOTE_CHARS) : null,
      mentions: stored.mentions.slice(0, MAX_MENTIONS),
      names,
      at: Date.now(),
    });
  }
  writePersisted(others);
}

export function loadComposerDraft(scope: DraftScope): ComposerDraft | null {
  const id = draftScopeId(scope);
  if (isDeletedScope(scope)) return null;
  const live = memory.get(id);
  if (live) {
    if (!incarnationConflicts(live.incarnation, scope.incarnation)) return live;
    // Replacement / reset: the remembered draft belongs to another incarnation.
    clearComposerDraftsForSession(scope.namespace, scope.key);
    return null;
  }
  if (!scope.persist) return null;
  const entry = readPersisted().find((e) => sameSession(e, scope));
  if (!entry) return null;
  if (incarnationConflicts(entry.inc, scope.incarnation)) {
    clearComposerDraftsForSession(scope.namespace, scope.key);
    return null;
  }
  return {
    text: entry.text,
    quotedContext: entry.quote ?? null,
    mentions: Array.isArray(entry.mentions) ? entry.mentions : [],
    attachments: [],
    reattachNames: Array.isArray(entry.names) ? entry.names.filter((n) => typeof n === "string") : [],
    selection: null,
    focused: false,
    incarnation: entry.inc || null,
  };
}

/** Drop drafts for one session (delete, reset, revoked access). */
export function clearComposerDraftsForSession(
  namespace: string | null | undefined,
  key: string,
  options: { deleted?: boolean; incarnation?: string | null } = {},
): void {
  let priorInc: string | null = options.incarnation ?? null;
  for (const [id, scope] of memoryScopes) {
    if (scope.key !== key || (namespace != null && scope.namespace !== namespace)) continue;
    priorInc = priorInc ?? memory.get(id)?.incarnation ?? null;
    memory.delete(id);
    memoryScopes.delete(id);
  }
  const persisted = readPersisted();
  for (const entry of persisted) {
    if (entry.key === key && (namespace == null || entry.ns === namespace)) {
      priorInc = priorInc ?? (entry.inc || null);
    }
  }
  if (options.deleted) {
    deletedKeys.set(key, priorInc);
  }
  writePersisted(persisted.filter((e) => !(
    e.key === key && (namespace == null || e.ns === namespace)
  )));
}

/** True when a known stored incarnation (memory or persisted) differs from the live one. */
export function draftIncarnationReplaced(
  scope: Pick<DraftScope, "namespace" | "key">,
  incarnation: string | null | undefined,
): boolean {
  const stored =
    memory.get(draftScopeId(scope))?.incarnation
    ?? readPersisted().find((e) => sameSession(e, scope))?.inc
    ?? null;
  return incarnationConflicts(stored, incarnation);
}

export function updateComposerDraftSelection(
  scope: Pick<DraftScope, "namespace" | "key">,
  selection: { start: number; end: number } | null,
  focused: boolean,
): void {
  const existing = memory.get(draftScopeId(scope));
  if (!existing) return;
  existing.selection = selection;
  existing.focused = focused;
}

export function handoffComposerDraft(
  namespace: string,
  targetKey: string,
  snapshot: ComposerDraft | null,
  fallbackText?: string | null,
): void {
  const currentNew = loadComposerDraft({
    namespace,
    key: NEW_CHAT_DRAFT_KEY,
    persist: false,
  });
  const source = snapshot ?? currentNew;
  const text = fallbackText ?? source?.text ?? "";
  const draft: ComposerDraft = source
    ? { ...source, text }
    : {
        text,
        quotedContext: null,
        mentions: [],
        attachments: [],
        reattachNames: [],
        selection: null,
        focused: true,
      };
  if (!isDraftEmpty(draft)) {
    saveComposerDraft({ namespace, key: targetKey, persist: true }, draft);
  }
  if (!currentNew || currentNew.text === (snapshot?.text ?? fallbackText ?? "")) {
    clearComposerDraftsForSession(namespace, NEW_CHAT_DRAFT_KEY);
  }
}

export function resetComposerDraftsForTests(): void {
  memory.clear();
  memoryScopes.clear();
  deletedKeys.clear();
  if (typeof window !== "undefined") {
    try { window.localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }
}
