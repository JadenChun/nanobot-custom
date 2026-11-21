/**
 * Ordering for next-turn setting changes against new-turn admission.
 *
 * A setting mutation (folder, model/effort, permission mode) is acknowledged by
 * the gateway after the preference owner commits it. A new turn must be
 * admitted only after the acknowledgement for the *latest* choice of each kind
 * settles, so a send never runs under the previous folder/model/mode. Stop and
 * Steer never use this ordering: they address the already admitted turn.
 */

export type NextTurnSettingKind = "workspace" | "model" | "reasoning";

interface InFlightSetting {
  sequence: number;
  /** Resolves with the acknowledgement; rejects when the owner refuses it. */
  promise: Promise<unknown>;
}

/**
 * Per-chat record of the latest in-flight choice for each setting kind.
 *
 * Only the latest operation of a kind can be stale-checked: an acknowledgement
 * for a superseded choice must never replace a newer local selection.
 */
export class NextTurnSettingsTracker {
  private sequence = 0;
  private inFlight = new Map<string, Map<NextTurnSettingKind, InFlightSetting>>();
  private latestSequence = new Map<string, Map<NextTurnSettingKind, number>>();
  private incarnations = new Map<string, string | null>();

  /** Bind the tracker to a chat lifetime; a replacement invalidates waiters. */
  bindIncarnation(chatId: string, incarnation: string | null): void {
    const previous = this.incarnations.get(chatId);
    if (previous === incarnation) return;
    if (incarnation === null) return;
    this.incarnations.set(chatId, incarnation);
    if (previous !== undefined && previous !== null && previous !== incarnation) {
      this.clear(chatId);
    }
  }

  /** Start one setting operation and return its local sequence number. */
  begin(
    chatId: string,
    kind: NextTurnSettingKind,
    operation: Promise<unknown>,
  ): number {
    const sequence = ++this.sequence;
    let byKind = this.inFlight.get(chatId);
    if (!byKind) {
      byKind = new Map();
      this.inFlight.set(chatId, byKind);
    }
    let latestByKind = this.latestSequence.get(chatId);
    if (!latestByKind) {
      latestByKind = new Map();
      this.latestSequence.set(chatId, latestByKind);
    }
    latestByKind.set(kind, sequence);
    const entry: InFlightSetting = { sequence, promise: operation };
    byKind.set(kind, entry);
    const settle = () => {
      const current = this.inFlight.get(chatId)?.get(kind);
      if (current !== entry) return;
      this.inFlight.get(chatId)?.delete(kind);
      if (this.inFlight.get(chatId)?.size === 0) this.inFlight.delete(chatId);
    };
    // Do not create an unhandled rejection when the caller relies on the
    // admission waiter instead of this returned promise.
    void operation.then(settle, settle);
    return sequence;
  }

  /** True while this exact choice is still the newest one for its kind (even after settling). */
  isCurrent(chatId: string, kind: NextTurnSettingKind, sequence: number): boolean {
    return this.latestSequence.get(chatId)?.get(kind) === sequence;
  }

  hasPending(chatId: string): boolean {
    return (this.inFlight.get(chatId)?.size ?? 0) > 0;
  }

  pendingKinds(chatId: string): Set<NextTurnSettingKind> {
    return new Set(this.inFlight.get(chatId)?.keys() ?? []);
  }

  /**
   * Settle after the newest setting change of each kind at call time.
   *
   * A superseded acknowledgement is not admission evidence: only the latest
   * choice describes what the next turn would run with. Rejects when one of
   * those choices was refused so the caller keeps the draft or queued row.
   */
  async pending(chatId: string): Promise<void> {
    const entries = [...(this.inFlight.get(chatId)?.values() ?? [])];
    if (entries.length === 0) return;
    await Promise.all(entries.map((entry) => entry.promise));
  }

  clear(chatId: string): void {
    this.inFlight.delete(chatId);
    this.latestSequence.delete(chatId);
  }
}
