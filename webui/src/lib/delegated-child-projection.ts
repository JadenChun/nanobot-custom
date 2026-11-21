import type {
  DelegatedChildReference,
  DelegatedChildPayload,
  DelegatedChildSummaryPayload,
  SourceOrigin,
  StopExpectation,
  StopExpectationChild,
  UIMessage,
} from "@/lib/types";

export type DelegatedChildProjectionMode = "live" | "hydration";

export interface DelegatedChildProjectionOptions {
  expectedRootKey: string;
  expectedRoot?: SourceOrigin | null;
  mode: DelegatedChildProjectionMode;
  allowLifetimeReplacement?: boolean;
}

export interface DelegatedChildProjectionResult {
  messages: UIMessage[];
  refreshRequired: boolean;
}

const CHILD_FACT_KEYS = [
  "parent_session_key",
  "child_session_key",
  "child_id",
  "profile",
  "task_label",
  "task_label_truncated",
  "state",
  "outcome",
  "runtime_generation",
  "parent_run_id",
  "child_run_id",
  "parent_store_identity",
  "parent_incarnation",
  "parent_generation",
  "root_key",
  "root_store_identity",
  "root_incarnation",
  "root_generation",
  "child_store_identity",
  "child_incarnation",
  "child_generation",
  "run_sequence",
  "state_sequence",
  "control_epoch",
  "holds_capacity",
  "can_stop",
  "owned_work",
  "pending_approval_id",
  "blocked_run_id",
  "result_disposition",
  "result_summary",
  "result_summary_truncated",
] as const satisfies readonly (keyof DelegatedChildPayload)[];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOwn(record: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function nonEmptyString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function codePointLength(value: string): number {
  return Array.from(value).length;
}

function nullableString(record: Record<string, unknown>, key: string): string | null | undefined {
  if (!hasOwn(record, key)) return undefined;
  const value = record[key];
  return value === null || typeof value === "string" ? value : undefined;
}

function nonNegativeInteger(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : null;
}

function booleanField(record: Record<string, unknown>, key: string): boolean | null {
  return typeof record[key] === "boolean" ? record[key] as boolean : null;
}

function requiredNullableString(
  record: Record<string, unknown>,
  key: string,
): string | null | undefined {
  return nullableString(record, key);
}

/** Validate a current canonical child row at the live/history JSON boundary. */
export function parseDelegatedChildPayload(value: unknown): DelegatedChildPayload | null {
  if (!isRecord(value)) return null;
  if (value.history_kind !== undefined && value.history_kind !== "canonical") return null;
  const parentSessionKey = nonEmptyString(value, "parent_session_key");
  const childSessionKey = nonEmptyString(value, "child_session_key");
  const childId = nonEmptyString(value, "child_id");
  const profile = nonEmptyString(value, "profile");
  const state = nonEmptyString(value, "state");
  const taskLabel = value.task_label;
  const taskLabelTruncated = booleanField(value, "task_label_truncated");
  const resultSummary = requiredNullableString(value, "result_summary");
  const resultSummaryTruncated = booleanField(value, "result_summary_truncated");
  const runtimeGeneration = nonNegativeInteger(value, "runtime_generation");
  const parentStoreIdentity = nonEmptyString(value, "parent_store_identity");
  const parentIncarnation = nonEmptyString(value, "parent_incarnation");
  const parentGeneration = nonNegativeInteger(value, "parent_generation");
  const rootKey = nonEmptyString(value, "root_key");
  const rootStoreIdentity = nonEmptyString(value, "root_store_identity");
  const rootIncarnation = nonEmptyString(value, "root_incarnation");
  const rootGeneration = nonNegativeInteger(value, "root_generation");
  const childStoreIdentity = nonEmptyString(value, "child_store_identity");
  const childIncarnation = nonEmptyString(value, "child_incarnation");
  const childGeneration = nonNegativeInteger(value, "child_generation");
  const runSequence = nonNegativeInteger(value, "run_sequence");
  const stateSequence = nonNegativeInteger(value, "state_sequence");
  const controlEpoch = nonNegativeInteger(value, "control_epoch");
  const holdsCapacity = booleanField(value, "holds_capacity");
  const canStop = booleanField(value, "can_stop");
  const ownedWork = booleanField(value, "owned_work");
  const outcome = requiredNullableString(value, "outcome");
  const turnId = nullableString(value, "turn_id");
  const parentRunId = requiredNullableString(value, "parent_run_id");
  const childRunId = requiredNullableString(value, "child_run_id");
  const pendingApprovalId = requiredNullableString(value, "pending_approval_id");
  const blockedRunId = requiredNullableString(value, "blocked_run_id");
  const resultDisposition = requiredNullableString(value, "result_disposition");

  if (
    !parentSessionKey
    || !childSessionKey
    || !childId
    || !profile
    || !state
    || typeof taskLabel !== "string"
    || codePointLength(taskLabel) > 160
    || taskLabelTruncated === null
    || resultSummary === undefined
    || (resultSummary !== null && codePointLength(resultSummary) > 1000)
    || resultSummaryTruncated === null
    || runtimeGeneration === null
    || !parentStoreIdentity
    || !parentIncarnation
    || parentGeneration === null
    || !rootKey
    || !rootStoreIdentity
    || !rootIncarnation
    || rootGeneration === null
    || !childStoreIdentity
    || !childIncarnation
    || childGeneration === null
    || runSequence === null
    || stateSequence === null
    || controlEpoch === null
    || holdsCapacity === null
    || canStop === null
    || ownedWork === null
    || outcome === undefined
    || (hasOwn(value, "turn_id") && turnId === undefined)
    || parentRunId === undefined
    || childRunId === undefined
    || pendingApprovalId === undefined
    || blockedRunId === undefined
    || resultDisposition === undefined
  ) return null;

  return {
    parent_session_key: parentSessionKey,
    child_session_key: childSessionKey,
    child_id: childId,
    profile,
    task_label: taskLabel,
    task_label_truncated: taskLabelTruncated,
    result_summary: resultSummary,
    result_summary_truncated: resultSummaryTruncated,
    state,
    outcome,
    runtime_generation: runtimeGeneration,
    turn_id: turnId ?? null,
    parent_run_id: parentRunId,
    child_run_id: childRunId,
    parent_store_identity: parentStoreIdentity,
    parent_incarnation: parentIncarnation,
    parent_generation: parentGeneration,
    root_key: rootKey,
    root_store_identity: rootStoreIdentity,
    root_incarnation: rootIncarnation,
    root_generation: rootGeneration,
    child_store_identity: childStoreIdentity,
    child_incarnation: childIncarnation,
    child_generation: childGeneration,
    run_sequence: runSequence,
    state_sequence: stateSequence,
    control_epoch: controlEpoch,
    holds_capacity: holdsCapacity,
    can_stop: canStop,
    owned_work: ownedWork,
    pending_approval_id: pendingApprovalId,
    blocked_run_id: blockedRunId,
    result_disposition: resultDisposition,
  };
}

function nullableNonNegativeInteger(record: Record<string, unknown>, key: string): number | null | undefined {
  if (!hasOwn(record, key)) return undefined;
  const value = record[key];
  if (value === null) return null;
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

function requiredNullableOriginField(
  record: Record<string, unknown>,
  key: string,
): string | null | undefined {
  return nullableString(record, key);
}

/** Validate an explicit read-only source-less summary without canonical facts. */
export function parseDelegatedChildSummaryPayload(
  value: unknown,
): DelegatedChildSummaryPayload | null {
  if (!isRecord(value) || value.history_kind !== "summary") return null;
  const parentSessionKey = nonEmptyString(value, "parent_session_key");
  const childSessionKey = nonEmptyString(value, "child_session_key");
  const childId = nonEmptyString(value, "child_id");
  const profile = nonEmptyString(value, "profile");
  const state = nonEmptyString(value, "state");
  const taskLabel = nullableString(value, "task_label");
  const taskLabelTruncated = booleanField(value, "task_label_truncated");
  const resultSummary = nullableString(value, "result_summary");
  const resultSummaryTruncated = booleanField(value, "result_summary_truncated");
  const outcome = nullableString(value, "outcome");
  const runtimeGeneration = nullableNonNegativeInteger(value, "runtime_generation");
  const turnId = nullableString(value, "turn_id");
  const parentRunId = nullableString(value, "parent_run_id");
  const childRunId = nullableString(value, "child_run_id");
  const parentStoreIdentity = requiredNullableOriginField(value, "parent_store_identity");
  const parentIncarnation = requiredNullableOriginField(value, "parent_incarnation");
  const parentGeneration = nullableNonNegativeInteger(value, "parent_generation");
  const rootKey = nonEmptyString(value, "root_key");
  const rootStoreIdentity = requiredNullableOriginField(value, "root_store_identity");
  const rootIncarnation = requiredNullableOriginField(value, "root_incarnation");
  const rootGeneration = nullableNonNegativeInteger(value, "root_generation");
  const childStoreIdentity = requiredNullableOriginField(value, "child_store_identity");
  const childIncarnation = requiredNullableOriginField(value, "child_incarnation");
  const childGeneration = nullableNonNegativeInteger(value, "child_generation");
  const runSequence = nullableNonNegativeInteger(value, "run_sequence");
  const stateSequence = nullableNonNegativeInteger(value, "state_sequence");
  const controlEpoch = nullableNonNegativeInteger(value, "control_epoch");
  if (
    !parentSessionKey
    || !childSessionKey
    || !childId
    || !profile
    || !state
    || taskLabel === undefined
    || (taskLabel !== null && codePointLength(taskLabel) > 160)
    || taskLabelTruncated === null
    || resultSummary === undefined
    || (resultSummary !== null && codePointLength(resultSummary) > 1000)
    || resultSummaryTruncated === null
    || outcome === undefined
    || runtimeGeneration === undefined
    || (hasOwn(value, "turn_id") && turnId === undefined)
    || parentRunId === undefined
    || childRunId === undefined
    || parentStoreIdentity === undefined
    || parentIncarnation === undefined
    || parentGeneration === undefined
    || !rootKey
    || rootStoreIdentity === undefined
    || rootIncarnation === undefined
    || rootGeneration === undefined
    || childStoreIdentity !== null
    || childIncarnation !== null
    || childGeneration !== null
    || runSequence === undefined
    || stateSequence === undefined
    || controlEpoch === undefined
    || value.can_stop !== false
  ) return null;
  return {
    history_kind: "summary",
    parent_session_key: parentSessionKey,
    child_session_key: childSessionKey,
    child_id: childId,
    profile,
    state,
    task_label: taskLabel,
    task_label_truncated: taskLabelTruncated,
    result_summary: resultSummary,
    result_summary_truncated: resultSummaryTruncated,
    outcome,
    runtime_generation: runtimeGeneration,
    ...(hasOwn(value, "turn_id") ? { turn_id: turnId as string | null } : {}),
    parent_run_id: parentRunId,
    child_run_id: childRunId,
    parent_store_identity: parentStoreIdentity,
    parent_incarnation: parentIncarnation,
    parent_generation: parentGeneration,
    root_key: rootKey,
    root_store_identity: rootStoreIdentity,
    root_incarnation: rootIncarnation,
    root_generation: rootGeneration,
    child_store_identity: childStoreIdentity,
    child_incarnation: childIncarnation,
    child_generation: childGeneration,
    run_sequence: runSequence,
    state_sequence: stateSequence,
    control_epoch: controlEpoch,
    can_stop: false,
  };
}

export function parseStopExpectation(value: unknown): StopExpectation | null {
  if (!isRecord(value)) return null;
  if (value.version !== 1 || !hasOnlyKeys(value, [
    "version",
    "source",
    "control_epoch",
    "active_root_present",
    "active_turn_id",
    "active_root_run_ids",
    "children",
    "pending_work_digest",
  ])) return null;
  const source = parseSourceOrigin(value.source);
  const controlEpoch = nonNegativeInteger(value, "control_epoch");
  const activeRootPresent = booleanField(value, "active_root_present");
  const activeTurnId = value.active_turn_id;
  const rootRunIds = stringArray(value.active_root_run_ids);
  const children = Array.isArray(value.children)
    ? value.children.map(parseStopExpectationChild)
    : null;
  const digest = value.pending_work_digest;
  if (
    !source
    || controlEpoch === null
    || activeRootPresent === null
    || !(activeTurnId === null || (typeof activeTurnId === "string" && activeTurnId.length > 0))
    || !rootRunIds
    || !isSortedUnique(rootRunIds)
    || !children
    || children.some((child) => child === null)
    || typeof digest !== "string"
    || !/^[0-9a-f]{64}$/.test(digest)
  ) return null;
  const validChildren = children.filter((child): child is StopExpectationChild => child !== null);
  if (
    !isSortedUnique(validChildren.map((child) => child.child_id))
    || (activeRootPresent === false && (activeTurnId !== null || rootRunIds.length > 0))
    || (activeRootPresent === true && rootRunIds.length === 0)
  ) return null;
  return value as unknown as StopExpectation;
}

function parseStopExpectationChild(value: unknown): StopExpectationChild | null {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "child_id",
    "source",
    "child_run_id",
    "run_sequence",
    "control_epoch",
  ])) return null;
  const childId = nonEmptyString(value, "child_id");
  const source = parseSourceOrigin(value.source);
  const childRunId = value.child_run_id;
  const runSequence = nonNegativeInteger(value, "run_sequence");
  const controlEpoch = nonNegativeInteger(value, "control_epoch");
  if (
    !childId
    || !source
    || !(childRunId === null || (typeof childRunId === "string" && childRunId.length > 0))
    || runSequence === null
    || controlEpoch === null
  ) return null;
  return { child_id: childId, source, child_run_id: childRunId, run_sequence: runSequence, control_epoch: controlEpoch };
}

export function parseSourceOrigin(value: unknown): SourceOrigin | null {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "store_identity",
    "session_key",
    "incarnation",
    "generation",
  ])) return null;
  const storeIdentity = nonEmptyString(value, "store_identity");
  const sessionKey = nonEmptyString(value, "session_key");
  const incarnation = nonEmptyString(value, "incarnation");
  const generation = nonNegativeInteger(value, "generation");
  if (!storeIdentity || !sessionKey || !incarnation || generation === null) return null;
  return {
    store_identity: storeIdentity,
    session_key: sessionKey,
    incarnation,
    generation,
  };
}

function hasOnlyKeys(record: Record<string, unknown>, keys: string[]): boolean {
  const expected = new Set(keys);
  return Object.keys(record).every((key) => expected.has(key))
    && keys.every((key) => hasOwn(record, key));
}

function stringArray(value: unknown): string[] | null {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || !item)) {
    return null;
  }
  return value as string[];
}

function isSortedUnique(values: string[]): boolean {
  return values.every((value, index) => (
    index === 0 || values[index - 1]! < value
  ));
}

function sourceFromChild(payload: DelegatedChildPayload, prefix: "parent" | "root" | "child"): SourceOrigin {
  if (prefix === "parent") {
    return {
      store_identity: payload.parent_store_identity!,
      session_key: payload.parent_session_key!,
      incarnation: payload.parent_incarnation!,
      generation: payload.parent_generation!,
    };
  }
  if (prefix === "root") {
    return {
      store_identity: payload.root_store_identity!,
      session_key: payload.root_key!,
      incarnation: payload.root_incarnation!,
      generation: payload.root_generation!,
    };
  }
  return {
    store_identity: payload.child_store_identity!,
    session_key: payload.child_session_key,
    incarnation: payload.child_incarnation!,
    generation: payload.child_generation!,
  };
}

export function sameSourceOrigin(a: SourceOrigin, b: SourceOrigin): boolean {
  return a.store_identity === b.store_identity
    && a.session_key === b.session_key
    && a.incarnation === b.incarnation
    && a.generation === b.generation;
}

export function delegatedChildMatchesRoot(
  value: unknown,
  root: SourceOrigin,
): boolean {
  const payload = parseDelegatedChildPayload(value);
  return payload !== null
    && payload.root_key === root.session_key
    && sameSourceOrigin(sourceFromChild(payload, "root"), root);
}

function sameLifetime(a: DelegatedChildPayload, b: DelegatedChildPayload): boolean {
  return sameSourceOrigin(sourceFromChild(a, "parent"), sourceFromChild(b, "parent"))
    && sameSourceOrigin(sourceFromChild(a, "root"), sourceFromChild(b, "root"))
    && sameSourceOrigin(sourceFromChild(a, "child"), sourceFromChild(b, "child"));
}

function compareVersion(a: DelegatedChildPayload, b: DelegatedChildPayload): number {
  if (a.run_sequence !== b.run_sequence) return a.run_sequence! < b.run_sequence! ? -1 : 1;
  if (a.state_sequence !== b.state_sequence) {
    return a.state_sequence! < b.state_sequence! ? -1 : 1;
  }
  return 0;
}

function canonicalFactsEqual(a: DelegatedChildPayload, b: DelegatedChildPayload): boolean {
  return CHILD_FACT_KEYS.every((key) => a[key] === b[key]);
}

function sourceIsAuthorized(
  candidate: DelegatedChildPayload,
  rows: DelegatedChildPayload[],
  options: DelegatedChildProjectionOptions,
): boolean {
  if (candidate.root_key !== options.expectedRootKey) return false;
  const candidateRoot = sourceFromChild(candidate, "root");
  if (candidateRoot.session_key !== options.expectedRootKey) return false;
  if (options.expectedRoot && !sameSourceOrigin(candidateRoot, options.expectedRoot)) return false;
  const parent = sourceFromChild(candidate, "parent");
  if (candidate.parent_session_key === options.expectedRootKey) {
    return sameSourceOrigin(parent, candidateRoot);
  }
  return rows.some((row) => (
    row.child_session_key === candidate.parent_session_key
    && row.root_key === candidate.root_key
    && sameSourceOrigin(sourceFromChild(row, "root"), candidateRoot)
    && sameSourceOrigin(sourceFromChild(row, "child"), parent)
  ));
}

function messageForChild(payload: DelegatedChildPayload): UIMessage {
  return {
    id: `delegated-child-${payload.child_id}`,
    role: "assistant",
    kind: "delegated_child",
    content: "",
    createdAt: Date.now(),
    delegatedChild: payload,
    ...(payload.turn_id ? { turnId: payload.turn_id } : {}),
  };
}

function messageForSummary(payload: DelegatedChildSummaryPayload): UIMessage {
  return {
    id: `delegated-summary-${payload.child_id}`,
    role: "assistant",
    kind: "delegated_child",
    content: "",
    createdAt: 0,
    timestampAvailable: false,
    delegatedChild: payload,
  };
}

function childRowIndex(messages: UIMessage[], childId: string): number {
  return messages.findIndex((message) => (
    message.kind === "delegated_child"
    && message.delegatedChild?.history_kind !== "summary"
    && message.delegatedChild?.child_id === childId
  ));
}

function summaryChildRowIndex(messages: UIMessage[], childId: string): number {
  return messages.findIndex((message) => (
    message.kind === "delegated_child"
    && message.delegatedChild?.history_kind === "summary"
    && message.delegatedChild.child_id === childId
  ));
}

/**
 * The single version/lifetime reducer shared by live events and history pages.
 * Hydration can prove a changed source lifetime; a live event cannot.
 */
export function mergeDelegatedChildRows(
  currentMessages: UIMessage[],
  incomingValues: unknown[],
  options: DelegatedChildProjectionOptions,
): DelegatedChildProjectionResult {
  const summaryIncoming = incomingValues.map((value) => (
    isRecord(value) && value.history_kind === "summary"
      ? parseDelegatedChildSummaryPayload(value)
      : null
  ));
  const canonicalValues = incomingValues.filter((value) => (
    !isRecord(value) || value.history_kind !== "summary"
  ));
  const incoming = canonicalValues.map(parseDelegatedChildPayload);
  let refreshRequired = incoming.some((candidate) => candidate === null)
    || summaryIncoming.some((candidate, index) => (
      isRecord(incomingValues[index])
      && incomingValues[index].history_kind === "summary"
      && candidate === null
    ));
  if (options.mode === "live" && summaryIncoming.some((candidate) => candidate !== null)) {
    refreshRequired = true;
  }
  const validIncoming = incoming.filter(
    (candidate): candidate is DelegatedChildPayload => candidate !== null,
  );
  const canonicalChildIds = new Set([
    ...delegatedChildRows(currentMessages).map((candidate) => candidate.child_id),
    ...validIncoming.map((candidate) => candidate.child_id),
  ]);
  const seenSummaryIds = new Set<string>();
  const knownRows = [
    ...currentMessages.flatMap((message) => (
      message.kind === "delegated_child" && message.delegatedChild
        ? [parseDelegatedChildPayload(message.delegatedChild)].filter(
            (candidate): candidate is DelegatedChildPayload => candidate !== null,
          )
        : []
    )),
    ...validIncoming,
  ];
  let messages = currentMessages;

  if (options.mode === "hydration") {
    for (const summary of summaryIncoming) {
      if (!summary) continue;
      if (canonicalChildIds.has(summary.child_id) || seenSummaryIds.has(summary.child_id)) {
        refreshRequired = true;
        continue;
      }
      seenSummaryIds.add(summary.child_id);
      const index = summaryChildRowIndex(messages, summary.child_id);
      if (index < 0) {
        messages = [...messages, messageForSummary(summary)];
      } else {
        const current = messages[index]?.delegatedChild;
        if (current?.history_kind !== "summary") {
          refreshRequired = true;
          continue;
        }
        if (JSON.stringify(current) !== JSON.stringify(summary)) {
          const next = messages.slice();
          next[index] = { ...next[index]!, delegatedChild: summary };
          messages = next;
        }
      }
    }
  }

  for (const candidate of validIncoming) {
    if (!sourceIsAuthorized(candidate, knownRows, options)) {
      refreshRequired = true;
      continue;
    }
    const index = childRowIndex(messages, candidate.child_id);
    if (index < 0) {
      messages = [...messages, messageForChild(candidate)];
      continue;
    }
    const current = messages[index]?.delegatedChild;
    const parsedCurrent = current ? parseDelegatedChildPayload(current) : null;
    if (!parsedCurrent) {
      refreshRequired = true;
      continue;
    }
    if (!sameLifetime(parsedCurrent, candidate)) {
      if (
        options.mode !== "hydration"
        || options.allowLifetimeReplacement === false
      ) {
        refreshRequired = true;
        continue;
      }
      const next = messages.slice();
      next[index] = {
        ...next[index]!,
        delegatedChild: candidate,
        ...(candidate.turn_id ? { turnId: candidate.turn_id } : {}),
      };
      messages = next;
      continue;
    }
    const order = compareVersion(parsedCurrent, candidate);
    if (order > 0) continue;
    if (order === 0) {
      if (!canonicalFactsEqual(parsedCurrent, candidate)) refreshRequired = true;
      continue;
    }
    const next = messages.slice();
    next[index] = {
      ...next[index]!,
      delegatedChild: candidate,
      ...(candidate.turn_id ? { turnId: candidate.turn_id } : {}),
    };
    messages = next;
  }

  return { messages, refreshRequired };
}

export function delegatedChildRows(messages: UIMessage[]): DelegatedChildPayload[] {
  return messages.flatMap((message) => (
    message.kind === "delegated_child" && message.delegatedChild
      ? [parseDelegatedChildPayload(message.delegatedChild)].filter(
          (candidate): candidate is DelegatedChildPayload => candidate !== null,
        )
      : []
  ));
}

export function delegatedChildSummaryRows(
  messages: UIMessage[],
): DelegatedChildSummaryPayload[] {
  return messages.flatMap((message) => (
    message.kind === "delegated_child" && message.delegatedChild
      ? [parseDelegatedChildSummaryPayload(message.delegatedChild)].filter(
          (candidate): candidate is DelegatedChildSummaryPayload => candidate !== null,
        )
      : []
  ));
}

export function delegatedChildReferences(messages: UIMessage[]): DelegatedChildReference[] {
  return [...delegatedChildRows(messages), ...delegatedChildSummaryRows(messages)];
}
