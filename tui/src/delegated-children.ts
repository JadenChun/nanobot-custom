export interface SourceOrigin {
  store_identity: string
  session_key: string
  incarnation: string
  generation: number
}

export interface StopExpectationChild {
  child_id: string
  source: SourceOrigin
  child_run_id: string | null
  run_sequence: number
  control_epoch: number
}

export interface StopExpectation {
  version: 1
  source: SourceOrigin
  control_epoch: number
  active_root_present: boolean
  active_turn_id: string | null
  active_root_run_ids: string[]
  children: StopExpectationChild[]
  pending_work_digest: string
}

/** One current canonical projection row for a delegated child. */
export interface DelegatedChildActivity {
  history_kind?: "canonical"
  parent_session_key: string
  child_session_key: string
  child_id: string
  profile: string
  state: string
  task_label: string
  task_label_truncated: boolean
  result_summary: string | null
  result_summary_truncated: boolean
  outcome: string | null
  runtime_generation: number
  turn_id: string | null
  parent_run_id: string | null
  child_run_id: string | null
  parent_store_identity: string
  parent_incarnation: string
  parent_generation: number
  root_key: string
  root_store_identity: string
  root_incarnation: string
  root_generation: number
  child_store_identity: string
  child_incarnation: string
  child_generation: number
  run_sequence: number
  state_sequence: number
  control_epoch: number
  holds_capacity: boolean
  can_stop: boolean
  owned_work: boolean
  pending_approval_id: string | null
  blocked_run_id: string | null
  result_disposition: string | null
}

/** Explicit, source-less history summary. It carries display data only. */
export interface DelegatedChildSummary {
  history_kind: "summary"
  parent_session_key: string
  child_session_key: string
  child_id: string
  profile: string
  state: string
  task_label: string | null
  task_label_truncated: boolean
  result_summary: string | null
  result_summary_truncated: boolean
  outcome: string | null
  runtime_generation: number | null
  turn_id?: string | null
  parent_run_id: string | null
  child_run_id: string | null
  parent_store_identity: string | null
  parent_incarnation: string | null
  parent_generation: number | null
  root_key: string
  root_store_identity: string | null
  root_incarnation: string | null
  root_generation: number | null
  child_store_identity: null
  child_incarnation: null
  child_generation: null
  run_sequence: number | null
  state_sequence: number | null
  control_epoch: number | null
  can_stop: false
}

export type DelegatedChildReference = DelegatedChildActivity | DelegatedChildSummary

export type DelegatedChildApplyResult =
  | "inserted"
  | "updated"
  | "stale"
  | "duplicate"
  | "conflict"
  | "lifetime_mismatch"
  | "invalid"

const CHILD_FACT_KEYS = [
  "parent_session_key",
  "child_session_key",
  "child_id",
  "profile",
  "state",
  "task_label",
  "task_label_truncated",
  "result_summary",
  "result_summary_truncated",
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
] as const satisfies readonly (keyof DelegatedChildActivity)[]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function hasOwn(record: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 2048
}

function codePointLength(value: string): number {
  return Array.from(value).length
}

function nonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
}

function nullableString(record: Record<string, unknown>, key: string): string | null | undefined {
  if (!hasOwn(record, key)) return undefined
  const value = record[key]
  return value === null || nonEmptyString(value) ? value : undefined
}

function nullableInteger(record: Record<string, unknown>, key: string): number | null | undefined {
  if (!hasOwn(record, key)) return undefined
  const value = record[key]
  if (value === null) return null
  return nonNegativeInteger(value) ? value : undefined
}

export function parseSourceOrigin(value: unknown): SourceOrigin | null {
  if (!isRecord(value)) return null
  if (!exactKeys(value, ["store_identity", "session_key", "incarnation", "generation"])) {
    return null
  }
  if (
    !nonEmptyString(value.store_identity)
    || !nonEmptyString(value.session_key)
    || !nonEmptyString(value.incarnation)
    || !nonNegativeInteger(value.generation)
  ) return null
  return {
    store_identity: value.store_identity,
    session_key: value.session_key,
    incarnation: value.incarnation,
    generation: value.generation,
  }
}

export function parseStopExpectation(value: unknown): StopExpectation | null {
  if (!isRecord(value) || !exactKeys(value, [
    "version",
    "source",
    "control_epoch",
    "active_root_present",
    "active_turn_id",
    "active_root_run_ids",
    "children",
    "pending_work_digest",
  ])) return null
  const source = parseSourceOrigin(value.source)
  const rootRunIds = stringArray(value.active_root_run_ids)
  const children = Array.isArray(value.children)
    ? value.children.map(parseStopExpectationChild)
    : null
  if (
    value.version !== 1
    || !source
    || !nonNegativeInteger(value.control_epoch)
    || typeof value.active_root_present !== "boolean"
    || !(value.active_turn_id === null || nonEmptyString(value.active_turn_id))
    || !rootRunIds
    || !isSortedUnique(rootRunIds)
    || !children
    || children.some((child) => child === null)
    || typeof value.pending_work_digest !== "string"
    || !/^[0-9a-f]{64}$/u.test(value.pending_work_digest)
  ) return null
  const validChildren = children.filter(
    (child): child is StopExpectationChild => child !== null,
  )
  if (
    !isSortedUnique(validChildren.map((child) => child.child_id))
    || validChildren.some((child) => child.source.session_key === source.session_key)
    || (value.active_root_present === false
      && (value.active_turn_id !== null || rootRunIds.length > 0))
    || (value.active_root_present === true && rootRunIds.length === 0)
  ) return null
  return {
    version: 1,
    source,
    control_epoch: value.control_epoch,
    active_root_present: value.active_root_present,
    active_turn_id: value.active_turn_id,
    active_root_run_ids: rootRunIds,
    children: validChildren,
    pending_work_digest: value.pending_work_digest,
  }
}

function parseStopExpectationChild(value: unknown): StopExpectationChild | null {
  if (!isRecord(value) || !exactKeys(value, [
    "child_id",
    "source",
    "child_run_id",
    "run_sequence",
    "control_epoch",
  ])) return null
  const source = parseSourceOrigin(value.source)
  if (
    !nonEmptyString(value.child_id)
    || !source
    || !(value.child_run_id === null || nonEmptyString(value.child_run_id))
    || !nonNegativeInteger(value.run_sequence)
    || !nonNegativeInteger(value.control_epoch)
  ) return null
  return {
    child_id: value.child_id,
    source,
    child_run_id: value.child_run_id,
    run_sequence: value.run_sequence,
    control_epoch: value.control_epoch,
  }
}

/** Validate the additive child event and the equivalent history/hydration row. */
export function parseDelegatedChildActivity(value: unknown): DelegatedChildActivity | null {
  if (!isRecord(value)) return null
  if (value.history_kind !== undefined && value.history_kind !== "canonical") return null
  const nullableFields = [
    "result_summary",
    "outcome",
    "parent_run_id",
    "child_run_id",
    "pending_approval_id",
    "blocked_run_id",
    "result_disposition",
  ] as const
  const nullableValues = nullableFields.map((key) => nullableString(value, key))
  const taskLabel = value.task_label
  const resultSummary = nullableValues[0]
  if (
    !nonEmptyString(value.parent_session_key)
    || !nonEmptyString(value.child_session_key)
    || !nonEmptyString(value.child_id)
    || !nonEmptyString(value.profile)
    || !nonEmptyString(value.state)
    || typeof taskLabel !== "string"
    || codePointLength(taskLabel) > 160
    || typeof value.task_label_truncated !== "boolean"
    || resultSummary === undefined
    || (resultSummary !== null && codePointLength(resultSummary) > 1000)
    || typeof value.result_summary_truncated !== "boolean"
    || nullableValues.some((item) => item === undefined)
    || !nonNegativeInteger(value.runtime_generation)
    || !nonEmptyString(value.parent_store_identity)
    || !nonEmptyString(value.parent_incarnation)
    || !nonNegativeInteger(value.parent_generation)
    || !nonEmptyString(value.root_key)
    || !nonEmptyString(value.root_store_identity)
    || !nonEmptyString(value.root_incarnation)
    || !nonNegativeInteger(value.root_generation)
    || !nonEmptyString(value.child_store_identity)
    || !nonEmptyString(value.child_incarnation)
    || !nonNegativeInteger(value.child_generation)
    || !nonNegativeInteger(value.run_sequence)
    || !nonNegativeInteger(value.state_sequence)
    || !nonNegativeInteger(value.control_epoch)
    || typeof value.holds_capacity !== "boolean"
    || typeof value.can_stop !== "boolean"
    || typeof value.owned_work !== "boolean"
    || (hasOwn(value, "turn_id") && !(value.turn_id === null || nonEmptyString(value.turn_id)))
  ) return null

  return {
    parent_session_key: value.parent_session_key,
    child_session_key: value.child_session_key,
    child_id: value.child_id,
    profile: value.profile,
    state: value.state,
    task_label: taskLabel,
    task_label_truncated: value.task_label_truncated,
    result_summary: resultSummary,
    result_summary_truncated: value.result_summary_truncated,
    outcome: nullableValues[1] as string | null,
    runtime_generation: value.runtime_generation,
    // P1 history omits delivery turn_id. Null records absence; it is excluded
    // from canonical equality and never used as current control authority.
    turn_id: hasOwn(value, "turn_id") ? value.turn_id as string | null : null,
    parent_run_id: nullableValues[2] as string | null,
    child_run_id: nullableValues[3] as string | null,
    parent_store_identity: value.parent_store_identity,
    parent_incarnation: value.parent_incarnation,
    parent_generation: value.parent_generation,
    root_key: value.root_key,
    root_store_identity: value.root_store_identity,
    root_incarnation: value.root_incarnation,
    root_generation: value.root_generation,
    child_store_identity: value.child_store_identity,
    child_incarnation: value.child_incarnation,
    child_generation: value.child_generation,
    run_sequence: value.run_sequence,
    state_sequence: value.state_sequence,
    control_epoch: value.control_epoch,
    holds_capacity: value.holds_capacity,
    can_stop: value.can_stop,
    owned_work: value.owned_work,
    pending_approval_id: nullableValues[4] as string | null,
    blocked_run_id: nullableValues[5] as string | null,
    result_disposition: nullableValues[6] as string | null,
  }
}

/** Parse source-less summaries only at an authoritative child hydration boundary. */
export function parseDelegatedChildSummary(value: unknown): DelegatedChildSummary | null {
  if (!isRecord(value) || value.history_kind !== "summary") return null
  const nullableFields = [
    "result_summary",
    "outcome",
    "turn_id",
    "parent_run_id",
    "child_run_id",
    "parent_store_identity",
    "parent_incarnation",
    "root_store_identity",
    "root_incarnation",
  ] as const
  const nullableValues = nullableFields.map((key) => nullableString(value, key))
  const taskLabel = nullableString(value, "task_label")
  const resultSummary = nullableValues[0]
  const parentGeneration = nullableInteger(value, "parent_generation")
  const rootGeneration = nullableInteger(value, "root_generation")
  const runtimeGeneration = nullableInteger(value, "runtime_generation")
  const runSequence = nullableInteger(value, "run_sequence")
  const stateSequence = nullableInteger(value, "state_sequence")
  const controlEpoch = nullableInteger(value, "control_epoch")
  if (
    !nonEmptyString(value.parent_session_key)
    || !nonEmptyString(value.child_session_key)
    || !nonEmptyString(value.child_id)
    || !nonEmptyString(value.profile)
    || !nonEmptyString(value.state)
    || taskLabel === undefined
    || (taskLabel !== null && codePointLength(taskLabel) > 160)
    || typeof value.task_label_truncated !== "boolean"
    || resultSummary === undefined
    || (resultSummary !== null && codePointLength(resultSummary) > 1000)
    || typeof value.result_summary_truncated !== "boolean"
    || nullableValues.some((item, index) => index !== 2 && item === undefined)
    || (hasOwn(value, "turn_id") && nullableValues[2] === undefined)
    || nullableInteger(value, "parent_generation") === undefined
    || nullableInteger(value, "root_generation") === undefined
    || !nonEmptyString(value.root_key)
    || value.child_store_identity !== null
    || value.child_incarnation !== null
    || value.child_generation !== null
    || runtimeGeneration === undefined
    || runSequence === undefined
    || stateSequence === undefined
    || controlEpoch === undefined
    || value.can_stop !== false
  ) return null

  return {
    history_kind: "summary",
    parent_session_key: value.parent_session_key,
    child_session_key: value.child_session_key,
    child_id: value.child_id,
    profile: value.profile,
    state: value.state,
    task_label: taskLabel,
    task_label_truncated: value.task_label_truncated,
    result_summary: resultSummary,
    result_summary_truncated: value.result_summary_truncated,
    outcome: nullableValues[1] as string | null,
    runtime_generation: runtimeGeneration,
    ...(hasOwn(value, "turn_id") ? { turn_id: nullableValues[2] as string | null } : {}),
    parent_run_id: nullableValues[3] as string | null,
    child_run_id: nullableValues[4] as string | null,
    parent_store_identity: nullableValues[5] as string | null,
    parent_incarnation: nullableValues[6] as string | null,
    parent_generation: parentGeneration as number | null,
    root_key: value.root_key,
    root_store_identity: nullableValues[7] as string | null,
    root_incarnation: nullableValues[8] as string | null,
    root_generation: rootGeneration as number | null,
    child_store_identity: null,
    child_incarnation: null,
    child_generation: null,
    run_sequence: runSequence,
    state_sequence: stateSequence,
    control_epoch: controlEpoch,
    can_stop: false,
  }
}

function stringArray(value: unknown): string[] | null {
  if (!Array.isArray(value) || !value.every(nonEmptyString)) return null
  return value as string[]
}

function isSortedUnique(values: readonly string[]): boolean {
  return values.every((value, index) => index === 0 || values[index - 1]! < value)
}

function exactKeys(record: Record<string, unknown>, keys: readonly string[]): boolean {
  const expected = new Set(keys)
  return Object.keys(record).length === expected.size && Object.keys(record).every((key) => expected.has(key))
}

function sourceForChild(
  child: DelegatedChildActivity,
  prefix: "parent" | "root" | "child",
): SourceOrigin {
  return {
    store_identity: child[`${prefix}_store_identity`],
    session_key: prefix === "parent"
      ? child.parent_session_key
      : prefix === "root" ? child.root_key : child.child_session_key,
    incarnation: child[`${prefix}_incarnation`],
    generation: child[`${prefix}_generation`],
  }
}

function sameSource(left: SourceOrigin, right: SourceOrigin): boolean {
  return left.store_identity === right.store_identity
    && left.session_key === right.session_key
    && left.incarnation === right.incarnation
    && left.generation === right.generation
}

function sameLifetime(left: DelegatedChildActivity, right: DelegatedChildActivity): boolean {
  return left.child_id === right.child_id
    && sameSource(sourceForChild(left, "parent"), sourceForChild(right, "parent"))
    && sameSource(sourceForChild(left, "root"), sourceForChild(right, "root"))
    && sameSource(sourceForChild(left, "child"), sourceForChild(right, "child"))
}

function compareVersion(left: DelegatedChildActivity, right: DelegatedChildActivity): number {
  return left.run_sequence - right.run_sequence
    || left.state_sequence - right.state_sequence
}

function sameFacts(left: DelegatedChildActivity, right: DelegatedChildActivity): boolean {
  return CHILD_FACT_KEYS.every((key) => left[key] === right[key])
}

/** One reducer for socket events, transcript rows, and paged live hydration. */
export class DelegatedChildProjection {
  private readonly rowsById = new Map<string, DelegatedChildActivity>()
  private readonly summariesById = new Map<string, DelegatedChildSummary>()
  private readonly revisions = new Map<string, number>()
  private nextRevision = 0
  private rootSource: SourceOrigin | null = null
  refreshRequired = false

  get rows(): DelegatedChildActivity[] {
    return [...this.rowsById.values()]
  }

  get summaryRows(): DelegatedChildSummary[] {
    return [...this.summariesById.values()]
  }

  get references(): DelegatedChildReference[] {
    return [...this.rowsById.values(), ...this.summariesById.values()]
  }

  getReference(childId: string): DelegatedChildReference | undefined {
    return this.rowsById.get(childId) ?? this.summariesById.get(childId)
  }

  get(childId: string): DelegatedChildActivity | undefined {
    return this.rowsById.get(childId)
  }

  revisionSnapshot(): Map<string, number> {
    return new Map(this.revisions)
  }

  applyLive(value: unknown, expectedRootKey: string): DelegatedChildApplyResult {
    const candidate = parseDelegatedChildActivity(value)
    if (!candidate || candidate.root_key !== expectedRootKey) return "invalid"
    const candidateRoot = sourceForChild(candidate, "root")
    if (this.rootSource && !sameSource(this.rootSource, candidateRoot)) {
      this.refreshRequired = true
      return "lifetime_mismatch"
    }
    const current = this.rowsById.get(candidate.child_id)
    if (!current) {
      this.put(candidate)
      this.summariesById.delete(candidate.child_id)
      this.rootSource ||= candidateRoot
      return "inserted"
    }
    if (!sameLifetime(current, candidate)) {
      this.refreshRequired = true
      return "lifetime_mismatch"
    }
    return this.reduceCurrent(current, candidate)
  }

  applyHistory(values: readonly unknown[], expectedRootKey: string): void {
    for (const value of values) {
      const candidate = parseDelegatedChildActivity(value)
      if (!candidate || candidate.root_key !== expectedRootKey) continue
      const candidateRoot = sourceForChild(candidate, "root")
      if (this.rootSource && !sameSource(this.rootSource, candidateRoot)) {
        this.refreshRequired = true
        continue
      }
      const current = this.rowsById.get(candidate.child_id)
      if (!current) {
        this.put(candidate)
        this.summariesById.delete(candidate.child_id)
        this.rootSource ||= candidateRoot
      } else if (sameLifetime(current, candidate)) {
        this.reduceCurrent(current, candidate)
      } else {
        this.refreshRequired = true
      }
    }
  }

  /** Replace/reconcile only after every page of the authoritative endpoint loads. */
  applyHydration(
    values: readonly unknown[],
    expectedRootKey: string,
    target: StopExpectation | null,
    startedAt: ReadonlyMap<string, number>,
  ): boolean {
    const candidates: DelegatedChildActivity[] = []
    const summaries: DelegatedChildSummary[] = []
    const ids = new Set<string>()
    let rootSource = target?.source ?? null
    for (const value of values) {
      if (isRecord(value) && value.history_kind === "summary") {
        const summary = parseDelegatedChildSummary(value)
        if (!summary || summary.root_key !== expectedRootKey || ids.has(summary.child_id)) {
          this.refreshRequired = true
          return false
        }
        ids.add(summary.child_id)
        summaries.push(summary)
        continue
      }
      const candidate = parseDelegatedChildActivity(value)
      if (!candidate || candidate.root_key !== expectedRootKey || ids.has(candidate.child_id)) {
        this.refreshRequired = true
        return false
      }
      const candidateRoot = sourceForChild(candidate, "root")
      if (rootSource && !sameSource(rootSource, candidateRoot)) {
        this.refreshRequired = true
        return false
      }
      rootSource = candidateRoot
      ids.add(candidate.child_id)
      candidates.push(candidate)
    }
    if (target && target.source.session_key !== expectedRootKey) {
      this.refreshRequired = true
      return false
    }

    const nextRows = new Map(this.rowsById)
    const nextSummaries = new Map<string, DelegatedChildSummary>()
    const changed = new Set<string>()
    let conflict = false
    for (const candidate of candidates) {
      nextSummaries.delete(candidate.child_id)
      const current = nextRows.get(candidate.child_id)
      if (!current) {
        nextRows.set(candidate.child_id, candidate)
        changed.add(candidate.child_id)
        continue
      }
      if (sameLifetime(current, candidate)) {
        const order = compareVersion(candidate, current)
        if (order < 0) continue
        if (order === 0) {
          if (!sameFacts(current, candidate)) conflict = true
          continue
        }
        nextRows.set(candidate.child_id, candidate)
        changed.add(candidate.child_id)
        continue
      }
      if (startedAt.get(candidate.child_id) !== this.revisions.get(candidate.child_id)) {
        conflict = true
        continue
      }
      nextRows.set(candidate.child_id, candidate)
      changed.add(candidate.child_id)
    }

    for (const [childId] of this.rowsById) {
      if (ids.has(childId)) continue
      if (startedAt.get(childId) !== this.revisions.get(childId)) {
        conflict = true
        continue
      }
      nextRows.delete(childId)
      changed.add(childId)
    }

    for (const summary of summaries) {
      // A stronger live canonical row always wins over a source-less summary.
      if (nextRows.has(summary.child_id)) {
        conflict = true
        continue
      }
      nextSummaries.set(summary.child_id, summary)
    }
    this.rowsById.clear()
    for (const [childId, row] of nextRows) this.rowsById.set(childId, row)
    this.summariesById.clear()
    for (const [childId, row] of nextSummaries) this.summariesById.set(childId, row)
    for (const childId of changed) this.revisions.set(childId, ++this.nextRevision)
    if (rootSource) this.rootSource = rootSource
    this.refreshRequired = conflict
    return true
  }

  clear(): void {
    this.rowsById.clear()
    this.summariesById.clear()
    this.revisions.clear()
    this.rootSource = null
    this.refreshRequired = false
  }

  private reduceCurrent(
    current: DelegatedChildActivity,
    candidate: DelegatedChildActivity,
  ): DelegatedChildApplyResult {
    const order = compareVersion(candidate, current)
    if (order < 0) return "stale"
    if (order === 0) {
      if (sameFacts(current, candidate)) return "duplicate"
      this.refreshRequired = true
      return "conflict"
    }
    this.put(candidate)
    return "updated"
  }

  private put(candidate: DelegatedChildActivity): void {
    this.rowsById.set(candidate.child_id, candidate)
    this.revisions.set(candidate.child_id, ++this.nextRevision)
  }
}
