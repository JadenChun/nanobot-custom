# Q1 portable/native Responses ID dispositions — 2026-10-05

This is a bounded provider-ID implementation record, not Q1/P6 acceptance or
runtime authorization. The dated source checkpoint and initial diagnostic
inventory remain unchanged. The parent contract records explicit user approval
for replacing two obsolete normalized-native expectations with exact native
identity coverage; portable bounded-ID expectations are retained, not obsolete.

## Source and assertion provenance

Original source: `tests/providers/test_openai_codex_provider_fork.py`, SHA256
`f722144089b2a91c05cdfb9c83e9a386189a32c02dea3b6b6220586e6e7fde7b`.
The immutable approved before-image is
`/private/tmp/nanobot-p6-q1.xQHcyt/q1-contract-approved-before-source/`;
`q1-contract-approved-before.json` contains its source hashes. The generated
`q1-id-assertion-proof.json` in that QA directory records every original
assertion's exact text, exact `ast.dump(..., include_attributes=False)`, text
SHA256 and AST SHA256, original/current node, disposition, and replacement.
`q1-id-proof.py` reproduces this extraction directly from the retained source.
The original collection failure remains in `q1-full-collection-current.json`.
No assertion is skipped, deleted without disposition, or relabeled an unknown
bug. All ten retained assertions have identical current text and AST.

| Original node / line | Exact original assertion | Disposition / current positive assertion |
| --- | --- | --- |
| `test_convert_messages_normalizes_long_tool_call_ids` / 38 | `assert fn_call["type"] == "function_call"` | RETAIN, same node and assertion |
| same / 39 | `assert len(fn_call["call_id"]) <= 64` | RETAIN, same node and assertion |
| same / 40 | `assert len(fn_call["id"]) <= 64` | RETAIN, same node and assertion |
| same / 41 | `assert fn_output["type"] == "function_call_output"` | RETAIN, same node and assertion |
| same / 42 | `assert fn_output["call_id"] == fn_call["call_id"]` | RETAIN, same node and assertion |
| `test_normalize_codex_id_hashes_unsafe_values` / 48 | `assert normalized.startswith("call_")` | RETAIN, same node and assertion |
| same / 49 | `assert len(normalized) <= 64` | RETAIN, same node and assertion |
| same / 50 | `assert all(ch.isalnum() or ch in "_-" for ch in normalized)` | RETAIN, same node and assertion; new ASCII-only checks additionally reject Unicode alphanumeric IDs |
| `test_consume_sse_path_stores_safe_ids` / 96 | `assert finish_reason == "stop"` | RETAIN at `test_consume_sse_path_stores_exact_native_ids` |
| same / 97 | `assert len(tool_calls) == 1` | RETAIN at that current node |
| same / 99 | `assert len(call_id) <= 64` | APPROVED_REPLACE: `assert call_id == raw_call` at that current node |
| same / 100 | `assert len(item_id) <= 64` | APPROVED_REPLACE: `assert item_id == raw_item` at that current node |

The portable converter owner is now
`nanobot.providers.openai_responses.converters.convert_messages`, not removed
Codex private helpers. The unsafe test obtains its normalized value from an
actual converted output, preserving the original prefix/bound/charset semantic
assertions at the owning path. The current normalizer is also tested directly.
The native owner is `parsing.consume_sse`, whose supported return is a three
tuple, not the removed private four tuple. The two replacements are necessary
because rewriting a server-native call/item identity would make retained native
items disagree with pending tool outputs. Stop and single-tool invariants remain.

## Behavior and positive coverage

Portable conversion defaults to normalization. Nonempty ASCII letters, digits,
underscore and hyphen IDs of at most64 characters remain stable; other strings
receive a deterministic SHA256-derived `call_`, `fc_`, or `msg_` ID. The typed
normalizer validates the string-only boundary and supported prefix. Request-local
mapping pairs declarations and outputs, including outputs before declarations
and repeated raw IDs. Safe raw call IDs are reserved ahead of hash allocation;
generated collisions get bounded unique suffixes. Duplicate item allocation
retains `rs_same_2` and `fc_0_2`; a64-character base is trimmed before suffixing.
The existing replay argument serializer is unchanged and input objects are not
mutated.

Compatible native state uses exact prior items and explicitly passes
`normalize_ids=False` for pending deltas. This mode does not normalize or
duplicate-rewrite server-native function item IDs. Native IDs are opaque
identities, not portable identifiers to allocate. Parsing, transport, persisted
schema, error policy and provider classes are unchanged; no dual persistence or
global parser normalization is introduced. Absent or mismatched state instead
uses the portable full-transcript fallback.

New current-owner nodes in `tests/providers/test_portable_responses_ids.py`:

- `test_portable_paired_ids_are_ascii_bounded_and_nonmutating`: long, unsafe,
  Unicode IDs; pairing before declaration and repeated raw IDs; deterministic
  output; unique item IDs; unchanged messages/arguments.
- `test_duplicate_item_ids_are_bounded`: safe duplicate semantics,64-character
  suffix bounds and collision with a later literal suffixed ID.
- `test_normalizer_preserves_safe_and_hashes_unsafe_strings`: stable safe IDs,
  deterministic prefixed hashes, ASCII bounds, validated nonstring/prefix edges.
- `test_generated_call_hash_cannot_steal_reserved_safe_raw_id`: forced hash
  collisions between distinct raw IDs and a safe raw ID equal to the generated
  candidate; exact declaration/output pairing with the safe ID unchanged.
- `test_sse_capture_state_and_pending_replay_keep_exact_native_ids`: raw SSE
  call/item equality, completed capture exact items, state exact items, pending
  tool output matches retained native call, unchanged event/state inputs.
- `test_sdk_native_replay_vs_portable_fallback`: SDK raw compound identity and
  exact persisted payload; compatible native replay versus mismatched/absent
  portable reconstruction; state unchanged.

## Executed evidence

All commands use the unchanged audited no-home/no-network runner
`python3 /private/tmp/nanobot-p6-q1.xQHcyt/run.py LABEL MODE TARGETS`.
Exact argv, source inventory, sandbox profile, stdout/stderr and exit codes are
retained in the QA directory's corresponding JSON/stdout/stderr records.

| Label | Mode and targets | Outcome |
| --- | --- | --- |
| `q1-id-red` | test: fork + portable modules |9 failed,4 passed; intended portable normalization/bound gaps before production edit |
| `q1-id-green` | test: same two modules |16 passed |
| `q1-id-broader` | test: those two modules + `tests/providers/test_openai_responses.py`, `tests/providers/test_openai_codex_provider.py`, `tests/providers/test_conversation_state.py` |203 passed |
| `q1-id-lint` | lint: converter, state, fork, portable files | one import-order failure, corrected |
| `q1-id-lint-final` | lint: same four files | PASS |
| `q1-id-full-type` | type: full unchanged policy |3,881 errors; one owned unnecessary string `isinstance` error, corrected by validated `object` boundary (no cast/suppression) |
| `q1-id-final-focus` | test: fork + portable modules, after boundary validation correction |16 passed |
| `q1-id-final-lint` | lint: same four owned Python files | PASS |
| `q1-id-final-type` | type: full unchanged policy |3,880 errors,463 analyzed files; converter/state have zero diagnostics; full-source strict remains FAIL |

The runner rejects targets in type mode; the attempted owned-file type command
stopped at its existing `assert not targets`, with no policy edits. Final focused,
lint and unchanged full-type evidence is recorded as `q1-id-final-focus`,
`q1-id-final-lint`, and `q1-id-final-type`. Full-source qualification remains the
parent’s final fan-in gate, and unrelated provider-suite failures are not claimed
fixed by this package.

`q1-id-strict-delta.json` compares full diagnostic signature multiplicities
(relative file, rule, message) against the retained final checkpoint: zero
introduced, zero resolved, zero owned converter/state diagnostics. This is not
strict-green acceptance. `q1-id-owned.diff` contains the complete five-file
before/after diff including additions, and `q1-id-parser-proof.json` confirms
unchanged native parser bytes against the approved before-source hash.
