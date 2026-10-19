# Offline cron delivery reconciliation

This is a dormant, operator-only recovery plane for E2 delivery attempts that
are `claimed`, `sending`, or `uncertain`. It is deliberately usable while the
global runtime migration guard is closed. It does not choose a production
outbox path, discover stores, load provider credentials, construct channels,
claim work, dispatch messages, arm a timer, or enable cron runtime behavior.

Every invocation requires three explicit authority selectors: an absolute
non-root `--outbox-root`, the exact `--namespace`, and a separate absolute
non-root `--artifacts-root`. The roots may not overlap, contain `..`, traverse
links, or change identity. Operators must select the exact stopped/offline E2
store; this protocol is cooperative local-host fencing, not cross-host proof.

## Operator sequence

1. Stop all delivery workers and inspect without creating anything:

   ```console
   nanobot cron-delivery inspect \
     --outbox-root /absolute/e2-outbox \
     --namespace exact-namespace \
     --artifacts-root /absolute/private-reconciliation --json
   ```

   Exit 0 is clean. Exit 2 means one or more redacted entries need action.
   Inspection is existing-only: an absent outbox is clean and neither absent
   nor existing inspection creates a directory, lock, artifact, repair, or
   fsync. Unknown entries, temporary residue, unsafe permissions, malformed
   records, namespace mismatch, or in-call drift fail closed.

2. Import evidence for exactly one reported predecessor:

   ```console
   nanobot cron-delivery evidence import \
     --outbox-root /absolute/e2-outbox \
     --namespace exact-namespace \
     --artifacts-root /absolute/private-reconciliation \
     --outbox-id OUTBOX --destination-id DESTINATION \
     --outcome confirmed_not_sent
   ```

   `confirmed_delivered` additionally requires `--receipt-file` containing one
   strict complete `nanobot.channel_text_receipt.v1` Telegram receipt. Provider
   message/chat IDs supplied manually are operator attestations unless they
   came from trusted retained evidence. A Telegram acknowledgement means the
   provider accepted the message; it does **not** prove that a user read it.
   Stored artifacts replace the raw target with a domain-separated target
   digest. Message text, tokens, and the raw target are never stored there.

3. Create and review an immutable plan:

   ```console
   nanobot cron-delivery plan ... --evidence-ref sha256:DIGEST
   nanobot cron-delivery plan-show ... --plan-id PLAN_ID --json
   ```

   Plan creation rereads the complete strict E2 store and requires the exact
   predecessor revision, attempt, owner, state, action, request digest,
   namespace, destination, and authority inode bindings. The plan ID is
   `SHA256("nanobot-cron-delivery-reconciliation-plan-v1\\0" + canonical-plan)`.
   Evidence, plans, and application receipts are canonical strict JSON,
   content-addressed, private, exclusively created, fsynced, and immutable.

4. Apply only after reviewing the plan. Both confirmations are mandatory:

   ```console
   nanobot cron-delivery apply ... --plan-id PLAN_ID \
     --confirm-plan PLAN_ID --confirmation CONFIRM-NOT-SENT
   ```

   The outcome literals are `CONFIRM-NOT-SENT`, `CONFIRM-DELIVERED`, and
   `CONFIRM-ABANDON`. A missing or wrong literal changes neither E2 nor the
   artifact store. Apply reopens all evidence and authority bindings, then
   calls E2's exact internally fenced compare-and-swap reconciliation API.
   It never claims or sends a successor. `pending` means only retry-eligible.

## Safe state table

| Predecessor | Permitted resolution | Evidence meaning | Result |
| --- | --- | --- | --- |
| `claimed` | `confirmed_not_sent` | Exact `claimed_never_sending` operator evidence plus the successful apply-time E2 destination fence | `pending` |
| `sending` / `uncertain` | `confirmed_delivered` | Complete request-bound strict Telegram acknowledgement covering every stored message | `delivered` |
| `sending` / `uncertain` | `abandon` | Explicit conservative operator attestation; no retry | `failed` |

`sending`/`uncertain` can never be declared not sent in this stage. A claimed
attempt cannot be delivered or abandoned. Compensation is unsupported. An
abandon decision may classify an actually delivered message failed, but it
prevents an automatic duplicate retry.

## Crash and replay behavior

E2 acquires the permanent destination send fence before reading the
predecessor and retains it through durable publication. A live sender maps to
exit 3 with no transition or application receipt. Stale/conflicting state is
exit 4; malformed/unsafe evidence or storage is exit 5; other configuration or
operational failure is exit 1.

After E2 changes state, apply publishes a separate content-free immutable
application receipt. If publication or its acknowledgement is lost, rerun the
exact same plan and confirmations. E2 first reproves the retained exact
resolution idempotently, without a second transition, then the tool publishes
or reproves the same application receipt. Never create a replacement plan for
an already applied action, overwrite an artifact, remove residue, or interpret
a free lock as evidence that no detached Telegram effect occurred.

Human and `--json` output are bounded and redacted. JSON is one stable object
and may contain IDs, digests, channel, state, owner, attempt/revision, permitted
outcomes, and receipt schema/part count. It never contains message content,
raw target, artifact bytes, tokens, provider objects, or raw exception text.

