# P4 memory receipt recovery boundary

This package establishes the startup boundary for the legacy staging receipt
`memory/.consolidation.pending.json`. It validates and detaches a receipt, then
blocks `MemoryStore` construction while the receipt remains present. Applying,
preserving, or deleting the receipt belongs to the follow-on recovery package.

## Constructor ordering

`MemoryStore.__init__` preflights the receipt before creating `memory/` and
before any legacy `HISTORY.md` migration action. A missing receipt is the only
pass-through condition. A valid, malformed, insufficiently evidenced, or
unsafe receipt raises the fixed, content-free message `memory recovery
required`. The preflight performs no writes, renames, chmod operations, or
receipt deletion.

## Receipt schema

The JSON root must be an object with these emitted fields:

| Field | Requirement |
| --- | --- |
| `version` | The exact integer `1`. |
| `chunk_digest` | A lowercase 64-character SHA-256 digest. |
| `history_marker_id` | A nonblank string. |
| `history_entry` | A nonblank string. |
| `memory_output` | A string, including its exact UTF-8 representation. |
| `base_memory_digest` | A lowercase 64-character SHA-256 digest. |
| `output_memory_digest` | A lowercase 64-character SHA-256 digest of `memory_output`. |
| `reference_only` | The exact JSON boolean type. |
| `created_at` | A nonblank string retained verbatim. |

The optional fields are `session_key`, `start_offset`, `end_offset`, and
`session_generation`. Session identity is a nonblank string when present.
Counters are exact, nonnegative JSON integers no greater than `2**63 - 1`; when
both offsets are present, `start_offset <= end_offset`. JSON booleans do not
count as integers.

The accepted aliases are `memory_update` for `memory_output`,
`memory_base_digest` for `base_memory_digest`, `memory_output_digest` for
`output_memory_digest`, and `generation` for `session_generation`. Canonical
fields remain required where the schema requires them. If a canonical field
and its alias are both present, they must have exactly equal values. A
contradictory alias rejects the receipt.

Every JSON object, including unknown metadata, is checked for duplicate keys.
Nonstandard JSON constants such as `NaN` are rejected. Unknown fields and the
exact original receipt bytes are retained on the detached `MemoryReceipt` for
the later recovery package. Parsing never rewrites the file.

## File binding

The receipt must be an owner-readable regular file owned by the current user
with no group or other permission bits and with one directory entry (no hard
link ambiguity). Its parent path and receipt entry are checked without
following symlinks. The file is opened with `O_NOFOLLOW` and its identity is
compared before and after reading, including size and available modification
and change times. A symlink, nonregular entry, ownership or privacy mismatch,
replacement race, or unavailable no-follow primitive fails closed. The
preflight never changes permissions on a user path.

The descriptor walk binds every existing ancestor before opening its child, so
renaming `memory/` and replacing it with a symlink at the leaf-open boundary
cannot redirect the read. Once an entry has been observed, disappearance or a
special-file replacement also fails closed; the leaf is opened nonblocking
where the platform supports it. For compatibility with macOS-style `/tmp` or
`/var` aliases, only the workspace path is resolved before the descriptor walk;
the `memory/` and receipt components remain no-follow.

There is one deliberate startup boundary: if the descriptor walk observes no
receipt entry, the constructor passes through to the existing `ensure_dir` and
legacy migration behavior. A receipt can therefore be created or an absent
ancestor can be replaced after the read-only absence check and before those
legacy writes. Package B must synchronize recovery, receipt creation, and
constructor migration if it needs to close that absence/create race.

## Deliberate non-goals

This package does not apply history or memory output, advance session offsets,
delete receipts, change session state, or add a `MemoryConsolidator` alias.
Digestless legacy receipts remain blocked until a later package can establish
the required evidence without overwriting the original receipt.
