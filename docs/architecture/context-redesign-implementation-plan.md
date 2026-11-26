# Kohaku Context Module Redesign — Detailed Implementation Plan

**Status**: Ready for Execution  
**Owner**: Implementing Agent (stage-by-stage execution) + Reviewing Agent (per-WP audit)  
**Target Package**: `kohaku/context/` (with `kohaku/agent/context.py` retained as a thin compatibility facade)  
**Reference Patterns**: OpenAI Codex (`codex-rs`) progressive tool loading (`ToolExposure` + BM25 `tool_search` / `defer_loading`) and two-phase long-term memory (`rollout_summaries/` + `raw_memories.md` -> `memory_summary.md` + `MEMORY.md`).

---

## 0. Ground Rules for the Implementing Agent

Follow every rule below strictly across all Work Packages (`WP-1` … `WP-5`):

1. **Repository Constraints** (`AGENTS.md`, `.agent/design.md`, `.agent/gotchas.md`):
   - Python 3.11+, `asyncio` throughout, line length 100.
   - Use `ruff check kohaku/` ONLY. **NEVER run `ruff format`** (destroys git blame history).
   - Preserve Windows compatibility (`pathlib.Path`, UTF-8 explicit encoding, cross-platform file locking).
   - Preserve atomic durable writes (temp file + `fsync` + `os.replace` + directory `fsync`) for all memory and session metadata writes.
   - Type dynamic boundaries at the edge (`TypedDict` / `dataclass` / narrow `Protocol`). Never add unvalidated `typing.cast` solely to silence BasedPyright.
2. **Strict Type-Regression Gate**:
   - Before editing each WP, freeze a baseline full-source `basedpyright` report.
   - After editing each WP, run `scripts/check_type_regressions.py` against the frozen before/after reports: **zero newly introduced diagnostic signatures across ALL files** (lower total count alone does not pass).
3. **Scope & Commit Discipline**:
   - Execute **one Work Package at a time** in strict numerical order (`WP-1` -> `WP-2` -> `WP-3` -> `WP-4` -> `WP-5`).
   - Do **not** bundle changes from a later WP into an earlier WP.
   - At the end of each WP, stop and present a structured **Handoff Report** containing:
     1. Files created / modified / deleted (with LOC delta).
     2. Commands executed + exact pass/fail counts (`pytest`, `ruff check`, `basedpyright` regression diff).
     3. Quantitative telemetry check results (where applicable).
     4. Any deviations, edge cases encountered, or open questions for the reviewer.

---

## 1. Architecture Overview & Quantitative Targets

### 1.1 Measured Baseline vs. Target Metrics

| Metric | Current Baseline | Target After WP-5 |
|---|---:|---:|
| Empty-workspace system prompt | 14,732 chars (~3,683 tok) | **≤ 4,800 chars (~1,200 tok)** |
| Default Turn-1 tool definitions | 29–31 tools (~6,262–6,595 tok) | **12–14 `DIRECT` tools (~2,000–2,400 tok)** |
| Combined Turn-1 fixed overhead (no MCP/memory) | ~10,000–10,200 tok | **≤ 3,600 tok (~64% reduction)** |
| 50-tool + 50-resource MCP server Turn-1 overhead | ~25,000–40,000 tok | **≤ 600 tok (namespace index only; loaded on demand)** |
| Effective context window on Anthropic / Bedrock | Crashes at ~25–30% (1 byte = 1 tok bug) | **Full 90%+ usable with working overflow recovery** |
| System prompt prefix stability across normal turns | Busted by memory/run-state/tools/skills churn | **Byte-identical Tier 0/1/2 within session epoch** |
| Dead/dormant legacy code in context/memory/compaction | ~27,000+ LOC | **0 LOC (removed in WP-1)** |

### 1.2 Target Package Layout (`kohaku/context/`)

```text
kohaku/context/
├── __init__.py
├── types.py              # PromptSection, PromptTier, RetentionPolicy, ToolExposure, TurnContextEnvelope
├── compiler.py           # Declarative PromptCompiler (replaces build_system_prompt + _filter_tool_guidance)
├── project_docs.py       # Codex-style hierarchical AGENTS.md / AGENTS.override.md discovery (32 KiB cap)
├── tool_surface.py       # DIRECT vs DEFERRED ToolSurface, BM25 tool_search, namespace index, native defer bridge
├── tool_result_gate.py   # Universal artifact-backed tool-result normalizer for ALL tools (built-in + MCP)
├── budget.py             # Unified calibrated TokenMeter (unwrapped LocalUsageAnchor + thinking_blocks)
├── compaction.py         # Single CompactionEngine (inline pressure, /compact, subagents, thin idle runner)
└── memory_pipeline.py    # Codex-style Stage-1 rollout extractor + Phase-2 Dream consolidator + FTS5 memory_search
```

`kohaku/agent/context.py` remains as a thin compatibility facade exporting `ContextBuilder`, `TranscriptInput`, `PersistedPromptContextResolver`, `session_extra`, and `handle_runtime_control`.

---

## WP-1 — Legacy & Dead Code Removal (~27,000+ LOC)

**Goal**: Eliminate dead, disabled, and dormant subsystems before building the new `kohaku/context/` package so every subsequent WP operates on a single clean codepath.

### WP-1.1 Extract Minimal Workspace Lock Helper

Before removing `memory_recovery_storage.py`, preserve the single active file lock shared by `kohaku/utils/dream_operation.py:103`, `kohaku/agent/memory.py:360`, and `kohaku/attachments/_media.py:594`.

- **Create** `kohaku/utils/workspace_lock.py` (~50 LOC):
  - Define `class WorkspaceLockError(OSError): ...`
  - Define `@contextmanager def open_workspace_memory_lock(workspace: Path) -> Iterator[None]`
  - **CRITICAL INVARIANT**: Must lock `workspace / ".memory-recovery.lock"` (same lock file path used by `kohaku/attachments/_media.py:594` so media-retention fencing stays mutually exclusive with Dream / `MemoryStore` writes).
  - Support both POSIX (`fcntl.flock`) and Windows (`msvcrt.locking`) cleanly, matching existing cross-platform lock semantics.
- **Update callers**:
  - `kohaku/utils/dream_operation.py:99-113`: import `open_workspace_memory_lock`, `WorkspaceLockError` from `kohaku.utils.workspace_lock` (catch `WorkspaceLockError` -> raise `DreamOperationStorageError`).
  - `kohaku/agent/memory.py:360-364`: import `open_workspace_memory_lock` from `kohaku.utils.workspace_lock`.

### WP-1.2 Remove Dead Request Budgeting & Unused Config/Memory Helpers

- **Delete**:
  - `kohaku/agent/context_budget.py`
  - `tests/agent/test_context_budget.py`
  - `tests/agent/test_context_budget_memory_boundary.py`
- **Edit**:
  - `kohaku/utils/prompt_budget.py`: remove `reduce_messages_to_budget`, alias `trim_messages_to_budget`, and the dead `"[Past Knowledge]"` entry at line 317 (keep `PromptBudget`, `PromptBudgetExceeded`, `measure_prompt`, and any helper still imported elsewhere — verify with `grep`).
  - `kohaku/config/schema.py:432-434`: remove dead fields `tool_result_clearing_keep`, `consolidation_trigger_ratio`, `consolidation_target_ratio` from `AgentDefaults` (keep `extra="ignore"` or alias tolerance if needed so existing user `config.json` files do not fail validation — verify `BaseModel` config in `schema.py`!).
  - `kohaku/agent/memory.py:298-300`: remove unused `MemoryStore.get_memory_context()` and update `tests/agent/test_memory_store.py` if referenced.

### WP-1.3 Remove Disabled `AutoCompact` & Dormant I1–I3e Idle Compaction Stack (~7,800 LOC)

- **Delete production modules**:
  - `kohaku/agent/autocompact.py`
  - `kohaku/agent/idle_compaction.py`
  - `kohaku/agent/idle_lifecycle.py`
  - `kohaku/agent/idle_runtime.py`
  - `kohaku/agent/idle_scanner.py`
  - `kohaku/agent/idle_summary.py`
  - `kohaku/session/idle_checkpoint.py`
  - `kohaku/session/idle_compaction.py`
- **Delete dormant test suites**:
  - `tests/agent/test_auto_compact.py`
  - `tests/agent/test_autocompact_unit.py`
  - `tests/agent/test_governed_idle_compaction.py`
  - `tests/agent/test_idle_*.py` (all 8 files)
  - `tests/session/test_idle_*.py` (all 8 files)
  - `tests/config/test_idle_compaction_config.py`
- **Clean up external references**:
  - `kohaku/agent/loop.py`: remove `AutoCompact` import (`:36`), `IdleLifecycleIntegration` import (`:52`), `enable_legacy_auto_compact` param (`:834`), `idle_integration` param (`:833`), `self.auto_compact` instantiation (`:1052-1064`), and all `self.auto_compact` / `self._idle_integration` hooks (`:3323-3379`, `:4471-4484`, `:7225-7238`). Preserve persisted summary replay on legacy sessions if still tested until WP-5 (`:7456`).
  - `kohaku/cli/agent.py:227`, `kohaku/cli/gateway_runtime.py:489`, `kohaku/runtime_composition.py:74, 199, 214, 297`: remove `enable_legacy_auto_compact=False` and unused `idle_runtime_enabled` wiring.
  - `kohaku/session/source_store.py:34, 93, 599, 816-1230`: remove dormant `capture_idle_compaction`, `publish_idle_checkpoint`, `publish_idle_checkpoint_in_execution_transaction`, `commit_idle_checkpoint_unlocked`, `validate_idle_checkpoint_admission` and imports from `idle_checkpoint` / `idle_compaction`.
  - `kohaku/session/artifact_verification.py:6-7`: update module docstring removing the reference to `kohaku.session.idle_checkpoint`.
  - `kohaku/config/schema.py`: retain a minimal `IdleCompactionConfig` (e.g., `enabled: bool = False`, `idle_minutes: int = 30`, `min_messages: int = 8`) only if referenced by `_migration_preflight.py:364` or for WP-5's thin idle runner; remove obsolete I1–I3e-only tuning knobs.

### WP-1.4 Remove Retired `.consolidation.pending.json` `memory_recovery_*` Stack (~17,800 LOC)

- **Delete production modules** (27 files):
  - All 24 `kohaku/agent/memory_recovery*.py` files
  - `kohaku/session/checked_source.py` (1,889 LOC — zero callers outside `memory_recovery_*`)
  - `kohaku/session/legacy_receipt_source.py` (366 LOC — zero callers outside `memory_recovery_*`)
  - `kohaku/session/recovery_locator.py` (zero callers outside `memory_recovery_*` / `checked_source.py`)
- **Delete retired test suites**:
  - `tests/agent/test_memory_recovery_*.py`
  - `tests/agent/test_memory_receipt_*.py`
  - `tests/agent/test_memory_evidence_interface.py`
  - `tests/agent/test_memory_lock_storage_interface.py`
  - `tests/agent/test_memory_parsing_interface.py`
  - `tests/agent/test_memory_target_read_interface.py`
  - `tests/agent/test_memory_store_lock_gate.py`
  - Any `tests/session/test_checked_source*.py` or `test_legacy_receipt_source*.py` / `test_recovery_locator*.py` (verify with `grep`).
- **Clean up external references**:
  - `kohaku/agent/memory.py:27-35, 103-109, 135-136`: remove `require_startup_receipt_admission`, `admitted_startup_mutation`, `require_memory_receipt_recovery`. Replace with a simple 8-line check in `MemoryStore.__init__`: if `(self.memory_dir / ".consolidation.pending.json").exists()`, log a clear warning directing the user to inspect/remove the retired legacy receipt file.
  - `kohaku/agent/loop.py:54` (and its call sites): remove `require_startup_receipt_admission`.
  - `kohaku/session/manager.py:37, 1104`: remove `MEMORY_RECEIPT_FILENAME`, `require_memory_receipt_recovery`, and `memory_recovery_startup_admission` calls.
  - `kohaku/runtime_composition.py:189`, `kohaku/cli/agent.py:17`, `kohaku/cli/commands.py:132, 307, 532`, `kohaku/cli/gateway.py:14`, `kohaku/cli/gateway_runtime.py:20`, `kohaku/cli/webui.py:104`: remove `require_startup_receipt_admission` imports and calls.
  - `kohaku/attachments/_inventory.py:299`: inspect `.consolidation.pending.json` reference; keep a plain string constant if `_inventory.py` simply ignores/recognizes that filename on disk.

### WP-1 Acceptance Criteria & Verification

- `grep -rn "memory_recovery\|context_budget\|autocompact\|idle_scanner\|idle_summary\|idle_lifecycle\|checked_source\|legacy_receipt_source" kohaku/` returns **0 matches** (except the lock filename `".memory-recovery.lock"` in `workspace_lock.py` and `attachments/_media.py`).
- `ruff check kohaku/` exits `0`.
- `basedpyright` regression check (`scripts/check_type_regressions.py`) reports **0 newly introduced diagnostic signatures**.
- Full `pytest` passes (excluding only documented pre-existing unrelated failures if any, which must be reported explicitly).

---

## WP-2 — P0 Correctness, Universal Tool-Result Gate & Provider Prompt Caching

**Goal**: Fix every P0 production failure in token budgeting, overflow recovery, tool-result bounding, and provider prompt caching.

### WP-2.1 Unified Token Metering & Universal Overflow Recovery (`kohaku/context/budget.py`, `request_budget.py`, `context_governance.py`)

1. **Create `kohaku/context/budget.py`**:
   - Define `TokenMeter` owning:
     - `estimate_messages_tokens(provider, model, messages, tools) -> tuple[int, str]`
     - `unwrap_anchor_provider(provider) -> LLMProvider` — unwraps `FallbackProvider` and `PooledProvider` (via a narrow `Protocol` or attribute check, never `type(p) is OpenAICompatProvider`) so `LocalUsageAnchor` calibration remains active when fallback/pooled providers are configured.
     - `compute_summary_reserve(budget: int, max_tool_result_chars: int, generation_allowance: int, envelope_tokens: int) -> int`:
       - **Fix unit mismatch**: convert `max_tool_result_chars` to tokens via `max_tool_tokens = max(0, max_tool_result_chars // 4)` before computing `growth = min(max_tool_tokens, max(1, budget // 8))`.
2. **Fix `kohaku/utils/helpers.py:758-900`**:
   - Include `thinking_blocks` (Anthropic/Bedrock structured thinking blocks: `block.get("thinking")` / `block.get("text")`) in both `_estimate_prompt_tokens_with_source()` and `estimate_message_tokens()`.
3. **Fix `kohaku/providers/request_budget.py:1035-1056`**:
   - When `known_tiktoken_encoding(model)` returns `None` under `policy == "strict"` (Anthropic, Bedrock, Gemini, xAI, Qwen, DeepSeek, etc.):
     - **STOP returning `len(encoded)` (raw UTF-8 bytes as 1 byte = 1 token!)**.
     - Instead compute calibrated token count: `math.ceil(len(cl100k_base.encode(serialized_text, disallowed_special=())) * 1.15)` (matching `estimated_compat`), while still enforcing the hard payload byte ceiling (`MAX_REQUEST_PAYLOAD_BYTES`, e.g., 64 MB) separately via `RequestPayloadBytesExceeded`.
   - Record `source = "strict_cl100k_calibrated"` so telemetry clearly distinguishes exact GPT encodings from calibrated non-GPT encodings.
4. **Fix `kohaku/agent/context_governance.py:1435-1551`**:
   - Update `_is_finalized_overflow()` and `recover_finalized_request()` to accept overflow errors from **all** providers and transports (`anthropic`, `bedrock`, `openai_responses`, `openai_chat_completions`, `azure`, `codex`, `xai`, `gemini`, etc.), not only `openai_chat_completions` with `estimated_compat`.
   - Replace inline `_summary_reserve` calculation (`:779-795`) with `TokenMeter.compute_summary_reserve(...)`.
5. **Fix `kohaku/agent/runner.py:2963-2986` & `kohaku/providers/fallback_provider.py:247, 296`**:
   - Use `TokenMeter.unwrap_anchor_provider(...)` when capturing and validating `LocalUsageAnchor` (scoped to the actual selected route identity `(provider_name, model)` so switching fallback routes invalidates cleanly without disabling anchoring on the primary route).

### WP-2.2 Fix All Provider Prompt-Caching Bugs

1. **OpenRouter (`kohaku/providers/openai_compat_provider.py:1141-1165`)**:
   - Move `_apply_cache_control(...)` so it executes **after** `_sanitize_empty_content(...)`.
   - Ensure `_mark` handles tool-calling assistant turns (`content is None` or `""` with `tool_calls`): when `messages[-2]` has no text content block to annotate, walk backward to the nearest cacheable non-empty message block (or annotate the tool-result block) instead of creating an empty text block that gets stripped.
2. **AWS Bedrock (`kohaku/providers/bedrock_provider.py:420-497`)**:
   - Add `_supports_cache_points(model: str) -> bool` (True for Anthropic Claude 3.5+/3.7+/4+ and Amazon Nova models on Bedrock).
   - When supported, emit `{"cachePoint": {"type": "default"}}` at:
     1. End of `system` blocks (if non-empty and not already ending in a `cachePoint`).
     2. End of `toolConfig["tools"]` (if non-empty; also mark builtin/MCP boundary using `_tool_cache_marker_indices` when within Bedrock's 4-cachePoint limit).
     3. End of the second-to-last message's `content` list (`len(messages) >= 3`).
3. **xAI Grok (`kohaku/providers/xai_grok_provider.py:358-375`)**:
   - Stop generating random `uuid.uuid4()` per HTTP request for `x-grok-conv-id` and `x-grok-session-id`.
   - Derive both deterministically from `session_id` (e.g., `str(uuid.uuid5(uuid.NAMESPACE_URL, f"kohaku:grok:{session_id}"))`) when `session_id` is available, falling back to a per-provider-instance UUID only when stateless.
4. **OpenAI Direct & Azure (`openai_compat_provider.py`, `azure_openai_provider.py`)**:
   - When `session_id` is present on direct OpenAI / Azure Responses or Chat Completions requests (where the backend accepts `prompt_cache_key`), set `body["prompt_cache_key"] = hashlib.sha256(session_id.encode()).hexdigest()` (matching `openai_codex_provider.py:130-131`). Guard by provider capability flag so third-party OpenAI-compat endpoints that reject unknown fields never fail.
5. **Delegated Child System Prompt (`kohaku/agent/delegation.py:1205-1228`)**:
   - Remove `Current Time: {datetime.now().astimezone().isoformat(timespec='seconds')}` from line 1 of `_base_prompt()`. Inject coarse time via Turn Tail runtime context instead.

### WP-2.3 Universal Tool-Result Gate (`kohaku/context/tool_result_gate.py`)

Fix the canonical source mode gap where MCP tools and `web_fetch` enter transcripts completely unbounded.

1. **Enable artifacts by default**: Change `ArtifactsConfig.enabled` default from `False` to `True` in `kohaku/config/schema.py:983`.
2. **Create `kohaku/context/tool_result_gate.py`**:
   - Define `ToolResultGate`:
     - Applies uniformly to **all** tools (`read_file`, `grep`, `exec`, `web_fetch`, `MCP*`, plugins) in **both** canonical source mode and non-source mode.
     - Respects `config.max_tool_result_chars` (default `16_000`; hard floor `2_000`).
     - Exempts `read_artifact` (to prevent offload -> read -> offload loops) and optionally `read_file` when already within its own bound.
     - When text output exceeds `max_tool_result_chars`:
       - If a session artifact store is available, persist the full UTF-8 output as an artifact blob (extending artifact producer support beyond `FilesystemReadReceipt` via a generic `ToolOutputArtifactOrigin(tool_name, tool_call_id, sha256, byte_length)`), and return a structured preview (`head 800 chars` + `... [N chars / M lines omitted; call read_artifact(artifact_id="...") to inspect] ...` + `tail 400 chars`).
       - If no artifact store is available (ephemeral/stateless call), fall back cleanly to `maybe_persist_tool_result` (if workspace exists) or deterministic head+tail truncation.
     - For multimodal `list[dict]` outputs: apply the text gate to each `"text"` block while preserving `"image_url"` blocks intact.
3. **Preserve Append-Only Image History (Stop Mid-Turn Prefix Busting)**:
   - Update `ContextGovernor.project_source_images()` (`context_governance.py:2179-2212`) so images are NOT rewritten in-place on every 2nd image during normal turns.
   - Instead, keep images append-only while under budget pressure (`< 75%` of `input_budget`), and prune older images to the latest batch only during budget-pressure reduction or compaction. Keep `max_tool_images` cap applied at ingestion time for a single tool batch.
4. **Fix Registration Order & `build_messages` Bug**:
   - In `kohaku/agent/loop.py:4303-4310` / `runner.py:746-779`: ensure `read_artifact` is registered on `effective_tools` **before** `build_source_transcript(..., tool_names=...)` is called.
   - In `kohaku/agent/context.py:543-555`: forward `tool_names=tool_names` from `build_messages()` to `self.build_transcript(...)`.

### WP-2 Acceptance Criteria & Verification

- New unit tests verifying:
  - Anthropic & Bedrock 60K-token prompts pass `validate_finalized_request` (no 240K byte blowup) and trigger `recover_finalized_request` when genuinely over window.
  - `_summary_reserve` computes `max_tool_result_chars // 4` tokens (not raw chars).
  - `FallbackProvider` preserves `LocalUsageAnchor` on primary route and resets cleanly on route switch.
  - `estimate_prompt_tokens_chain` counts `thinking_blocks`.
  - OpenRouter tool-calling assistant turn retains `cache_control` after `_sanitize_empty_content`.
  - Bedrock emits valid `cachePoint` blocks on system/tools/messages for Claude/Nova models.
  - xAI Grok emits identical `x-grok-conv-id` / `x-grok-session-id` across consecutive iterations of the same session.
  - Canonical mode offloads a 50K-char MCP tool output and a 50K-char `web_fetch` output into an artifact with bounded preview + working `read_artifact` retrieval.
  - Second image in canonical turn does not mutate historical message 0..N-1 while under 75% budget.
- `ruff check kohaku/`, `basedpyright` regression check (0 new errors), and all affected `pytest` suites pass.

---

## WP-3 — Standardized `PromptCompiler`, Hierarchical `AGENTS.md` & Runtime Context

**Goal**: Replace monolithic string assembly and destructive regex filtering with a declarative, cache-frozen 3-tier prompt compiler and explicit runtime-context retention semantics.

### WP-3.1 Core Data Types (`kohaku/context/types.py`)

```python
PromptTier = Literal["static", "session_snapshot", "turn_delta"]
RetentionPolicy = Literal["turn_only", "persist_compact", "ephemeral_body"]

@dataclass(frozen=True, slots=True)
class PromptSection:
    id: str
    tier: PromptTier
    content: str
    priority: int = 100
    max_tokens: int | None = None
    requires_all_tools: frozenset[str] = frozenset()
    requires_any_tools: frozenset[str] = frozenset()

@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    static_text: str             # Tier 1 (Cache Breakpoint A)
    session_snapshot_text: str   # Tier 2 (Cache Breakpoint B)
    section_tokens: dict[str, int]  # Telemetry per section id

    @property
    def full_system_text(self) -> str:
        return "\n\n---\n\n".join(p for p in (self.static_text, self.session_snapshot_text) if p)

    def system_blocks(self) -> list[dict[str, Any]]:
        """Return multi-block system representation for providers supporting split cache breakpoints."""
        ...
```

Extend `RuntimeContextBlock` in `kohaku/runtime_context.py` with:
- `retention: RetentionPolicy = "turn_only"`
- `compact_summary: str | None = None` (used when `retention == "ephemeral_body"` to leave a 1-line breadcrumb like `[Used skill: webapp-testing]` in history, or when `retention == "persist_compact"` to retain a bounded quote/mention across follow-up turns).

### WP-3.2 Declarative `PromptCompiler` (`kohaku/context/compiler.py`)

1. **Delete `_filter_tool_guidance` and `_render_tool_contract` regex surgery completely.**
   - Every prompt rule group is declared as a `PromptSection` with explicit `requires_all_tools` / `requires_any_tools` predicates.
   - `untrusted_content` section uses `requires_any_tools=frozenset({"web_fetch", "web_search", "read_file"})` — never deleted when one web tool is absent.
   - Sandbox/folder policy sections use structured predicates rather than fragile line-splitting.
2. **Deterministic Per-Section Token Caps**:
   - Apply `truncate_section_to_tokens(text, max_tokens, hint=...)` using `TokenMeter`:
     - `SOUL.md`: `400` tokens
     - `USER.md`: `500` tokens
     - `memory_summary.md` (or legacy `MEMORY.md` fallback until WP-5): `1,500` tokens
     - Project `AGENTS.md` hierarchy: `32 KiB` (~8,000 tokens max, matching Codex)
     - MCP server instructions total: `1,500` tokens (`500` tokens per server)
     - Skills summary index: `1,200` tokens
     - Active `always` skills total: `2,000` tokens
   - When truncated, append `\n\n[Truncated to fit {max_tokens}-token budget; use read_file on the source file if full text is needed.]`.
3. **Template Cleanup & Deduplication (`kohaku/templates/`)**:
   - Add `"TOOLS.md"` and `"IDENTITY.md"` to `_SKIPPABLE_DEFAULTS` (alongside `"AGENTS.md"` and `"USER.md"`).
   - Consolidate overlapping persona text between `SOUL.md` and `IDENTITY.md`.
   - Rewrite `kohaku/templates/agent/tool_contract.md` into concise universal operating principles (**target ≤ 550 tokens** instead of 1,990 tokens), organized as declarative conditional blocks:
     - Remove hardcoded video-editing MCP timeline preview rule (`tool_contract.md:97` — move to a dedicated skill if needed).
     - Remove redundant per-tool usage prose that duplicates tool JSON schemas.
   - Remove hardcoded personal Telegram ID `8281248569` from `kohaku/templates/AGENTS.md:19`.
   - Fix heading hierarchy (`identity.md` starts at `# Runtime & Workspace` or consistent `##` under `# System`).
   - Stop injecting unrelated context repositories and volatile `runs/summaries.md`, `runs/current.json`, `workspace/current/summary_index.md`, `workspace/current/plan.json` into the global system prompt.
   - Ensure `include_memory=False` suppresses **all** personal/memory sections (`USER.md`, `MEMORY.md` / `memory_summary.md`, context repo memory).
4. **Codex-Style Hierarchical `AGENTS.md` Discovery (`kohaku/context/project_docs.py`)**:
   - Implement `discover_project_instructions(workspace: Path, project_path: Path, *, max_bytes: int = 32_768, validator=...) -> str`:
     - Walks from git root (or `workspace` if outside git) down to `project_path`.
     - At each directory level, prefers `AGENTS.override.md` over `AGENTS.md` (matching Codex).
     - Validates every file through `ResourceAccessPolicy` in managed mode (preserving symlink/boundary containment checks).
     - Concatenates root -> leaf with clear path headers, stopping cleanly at `32 KiB`.
5. **Deterministic Skills Ordering (`kohaku/agent/skills.py:159`)**:
   - Sort `base.iterdir()` by `entry.name` (`sorted(base.iterdir(), key=lambda p: p.name)`).
   - Keep display root invariant where possible (always use canonical `skills/<name>/SKILL.md` for bundled/agent-home skills so workspace switches don't churn the Tier-1 skill index!).
6. **Tier-2 Session Epoch Snapshot**:
   - Attach a `SessionPromptSnapshot` object to the active session state (created on first turn of the session, invalidated **only** on explicit compaction, `/new`, `/dream` manual refresh request, or workspace switch).
   - Tier-2 content (`SOUL.md`, `USER.md`, `memory_summary.md`, active `AGENTS.md`, MCP server `instructions`) is read from the session snapshot so concurrent disk edits (`save_knowledge`, background Dream) never bust prompt caches mid-session.
   - Pass multi-block `system_blocks()` to Anthropic and Bedrock providers so Tier 1 and Tier 2 get independent cache boundaries when supported!

### WP-3.3 Standardized Tier-3 Runtime Context (`TurnContextEnvelope`)

1. **Restore Missing Routing & Coarse Time Metadata**:
   - Inject coarse time (`Current Time: YYYY-MM-DD HH:00 (Day, TZ)` — hourly granularity or turn-tail only, NEVER in system prompt!) + `Channel` + `Chat ID` (when available) as a `retention="turn_only"` Tier-3 block on normal user turns.
2. **Enforce Retention Policies Consistently Across Source & Legacy Sessions**:
   - `turn_only` (time, channel, compact active goal status pointer, `@cli_app`, `@mcp_preset`): attached to the model request for this turn ONLY; never persisted into session history.
   - `persist_compact` (`webui_quote`, `session_mentions`): persisted as compact bounded user-turn context in **both** canonical source events and legacy sessions so follow-up turns retain quote/mention context without replaying 7 KB goal prompts.
   - `ephemeral_body` (explicit `$skill-name` full markdown): attached in full for the active turn ONLY; history retains only `[Used skill: <name>]` (eliminating 19 KB/turn replay bloat).
3. **Compact Active Goal Pointer**:
   - When a sustained goal is active, inject `goal_runtime.md` (2.9 KB) **only on the `/goal` creation turn** (as `ephemeral_body`), and on subsequent turns inject only a compact 4–6 line `Active Goal` status block (`turn_only`: goal id, objective summary ≤500 chars, current step).
4. **Fix Duplicate & Subagent `$skill` Injection**:
   - Remove the duplicate call to `build_explicit_skill_runtime_context` in `kohaku/agent/loop.py:2562`; resolve explicit `$skill` references in exactly one place (`PromptCompiler.build_turn_envelope`), gated strictly on `turn_kind is TurnKind.USER and include_skill_context`.
   - Fix `loop.py:2497-2503, 7502-7506` so `TurnKind.SYSTEM` / subagent announcements never pass `include_skill_context=True`.

### WP-3 Acceptance Criteria & Verification

- Empty-workspace assembled system prompt drops from **14,732 chars (~3,683 tok)** to **≤ 4,800 chars (~1,200 tok)**.
- Omitting `web_fetch` while keeping `web_search` (or omitting `message` / `repository_catalog` / `select_workspace`) **preserves** `untrusted_content.md`, `MEMORY.md`, and sandbox rules with zero broken half-sentences.
- Calling `compile_system_prompt()` twice across two turns after modifying `MEMORY.md` on disk produces **byte-identical** Tier 1 and Tier 2 output within the same session epoch.
- `webui_quote` survives Turn N+1 in canonical source sessions; `$skill-creator` (19 KB) is NOT replayed in full on Turn N+1.
- `ruff check kohaku/`, `basedpyright` regression check (0 new errors), and full affected `pytest` suites pass.

---

## WP-4 — Codex-Style Tool Loading (`ToolExposure` + `tool_search`) & MCP Gateway

**Goal**: Adopt Codex's `DIRECT` vs. `DEFERRED` tool exposure model with a BM25 `tool_search` handler (plus native `defer_loading: true` bridge), cutting Turn-1 tool tokens by ~65% and making large MCP servers reliable and cache-friendly.

### WP-4.1 `ToolExposure` & `ToolSurface` (`kohaku/context/tool_surface.py`, `kohaku/agent/tools/base.py`, `registry.py`)

1. **Extend `Tool` base class (`kohaku/agent/tools/base.py`)**:
   - Add class/property defaults:
     - `exposure: ToolExposure = ToolExposure.DIRECT` (`class ToolExposure(str, Enum): DIRECT = "direct"; DEFERRED = "deferred"`)
     - `namespace: str = "core"`
     - `search_keywords: tuple[str, ...] = ()`
2. **Classify Built-In Tools (`fork_factories.py` & individual tool classes)**:
   - **`DIRECT` (~12–14 Core Tools — always in initial schema list)**:
     - Filesystem & search: `read_file`, `edit_file`, `write_file`, `find_files`, `grep`, `list_dir`
     - Execution: `exec`, `exec_session`
     - Web: `web_search`, `web_fetch`
     - Context & retrieval: `read_artifact`, `memory_search` (added in WP-5; `search_sessions`/`read_session` deferred or aliased), `tool_search`
     - Messaging: `message` (when channel supports it)
   - **`DEFERRED` (withheld until discovered or auto-activated)**:
     - Automation / media / UI: `agent_browser`, `agent_device`, `desktop_use`, `generate_image`, `cron`, `run_cli_app`
     - Workspace / goals / sessions / meta: `create_goal`, `update_goal`, `my`, `repository_catalog`, `select_workspace`, `request_folder_access`, `list_exec_sessions`, `list_sessions`, `search_sessions`, `read_session`, `save_knowledge`, `delegate_task`, `plan_task`, `review_work`, `explore`, `followup_child`
     - **All MCP server tools & meta-tools** (configurable per server: `MCPServerConfig.defer_loading: bool = True`, defaulting to `True` so connecting 100 MCP tools adds ~0 upfront schema tokens; users can opt small critical servers into `defer_loading=False`).
3. **Implement `ToolSurface` & BM25 `tool_search` (`kohaku/context/tool_surface.py`)**:
   - **BM25 Index**: Pure-Python lightweight BM25 ranker (~60 LOC, zero external dependencies) indexing each deferred tool's `(name, namespace, description, parameter_names, search_keywords)`.
   - **`ToolSearchTool` (`name = "tool_search"`, `exposure = DIRECT`)**:
     - Parameters: `query: str` (natural language keywords or `"namespace:<server>"`), `namespace: str | None = None`, `limit: int = 8` (clamped `1..16`).
     - Behavior:
       1. Ranks matching `DEFERRED` tools allowed by the session's policy.
       2. Marks matched tool names as **activated** in the session's `ToolSurfaceState` (`activated_tools: list[str]`, append-only insertion order within the session!).
       3. Returns a compact structured response listing the newly activated tool names and brief summaries (the actual JSON schemas are appended cleanly to the **deferred tail** of `tools` on the next LLM call — preserving the frozen `DIRECT` tool cache breakpoint!).
     - Also supports direct lookup by exact name (`query="cron"` or `query="mcp_github_create_issue"`).
   - **Namespace Index Block (injected in Tier 1/2)**:
     - Compact 1-line summary per deferred namespace (e.g., `- browser_and_device: agent_browser, agent_device (2 tools) — use tool_search to load`, `- mcp:github: GitHub API server (42 tools) — use tool_search(namespace="mcp:github", query="...")`).
   - **Auto-Activation Hooks**:
     - `/goal` or active goal -> auto-activate `create_goal`, `update_goal`.
     - `@cli_app` attachment/mention -> auto-activate `run_cli_app`.
     - `@mcp_preset` mention/attachment (`metadata["mcp_presets"]`) -> auto-activate the matching MCP server's tools (up to cap, e.g., 16 tools; if larger, inject its namespace hint into Tier 3 so `tool_search` targets it immediately).
     - `CliAppsTool.description`: make static (move installed app list to Tier-3 runtime context instead of mutating schema description).
   - **Provider-Native `defer_loading` Bridge**:
     - Add `ToolSurface.definitions_for_provider(provider, model)` which:
       - For providers supporting OpenAI Responses native `tool_search` + `defer_loading: true` (opt-in capability flag `supports_native_tool_search: bool`), emits deferred tool definitions with `"defer_loading": True` + native `{"type": "tool_search"}`.
       - For all other providers (Anthropic, Bedrock, Gemini, OpenAI Chat Completions, local models), uses Kohaku's client-side `ToolSearchTool` + append-only activated tail after the `DIRECT` cache marker (`_tool_cache_marker_indices` places marker 1 at end of `DIRECT` tools and marker 2 at end of activated tail!).
   - **Invariant Tool Prefix Across USER vs. SYSTEM Turns**:
     - Stop adding/removing `select_workspace` and `request_folder_access` between `TurnKind.USER` and `TurnKind.SYSTEM` turns in `loop.py:7150-7177` (they are now `DEFERRED` anyway, and policy checks enforce `TurnKind.USER` at execution time without mutating the cached tool prefix).

### WP-4.2 MCP Gateway Hardening (`kohaku/agent/tools/mcp.py`)

1. **Replace Eager Per-Resource / Per-Prompt Tool Explosion with 2 Meta-Tools**:
   - Remove per-resource `MCPResourceWrapper` and per-prompt `MCPPromptWrapper` eager registration loops (`mcp.py:1966-1995`).
   - Register **2 deferred meta-tools** whenever at least one MCP server exposes resources or prompts:
     - `mcp_resources(action: "list" | "read", server: str, uri: str | None = None)`
     - `mcp_prompts(action: "list" | "get", server: str, name: str | None = None, arguments: dict[str, str] | None = None)`
   - Both return `ToolResult.error(...)` on every timeout, cancellation, or exception (fixing the `"status=ok"` error-accounting bug).
2. **Normalize Schema Before Validating (`mcp.py:511-582, 1914`)**:
   - Run `_normalize_schema_for_openai(tool_def.inputSchema)` **before** `_validate_mcp_schema(...)`.
   - Expand `_SUPPORTED_MCP_SCHEMA_KEYS` to allow `"$defs"`, `"oneOf"`, `"allOf"`, `"format"`, `"pattern"`, `"const"`, `"examples"`, `"default"`, `"title"` (stripping only genuinely provider-hostile keywords during normalization instead of silently dropping Pydantic v2 MCP tools).
3. **Parallel Connection & Live Registry Delegation**:
   - Parallelize `connect_mcp_servers()` (`mcp.py:2105-2113`) using `asyncio.gather(..., return_exceptions=True)` with bounded concurrency so N unhealthy servers fail in `max(timeout)` instead of `sum(timeouts)`.
   - Fix the filtered per-turn `spec.tools` revocation bug (`loop.py:7150-7177`, `runner.py:761-814`):
     - Replace copied `tools.filtered(...)` overlays with a `PolicyToolView(root_registry, allow_predicate, session_surface)` that snapshots schemas per turn for cache stability, while resolving live execution handles from `root_registry` (so `_MCPReadinessHook.before_run` new connections and `_refresh_terminated_server()` mid-turn reconnects are immediately executable instead of failing with `"its captured registration was revoked"`).
4. **Pagination, Bounds, Collisions & `InitializeResult.instructions`**:
   - Follow `nextCursor` pagination loops (with safety cap, e.g., max 20 pages) for `list_tools`, `list_resources`, and `list_prompts`.
   - Cap each MCP tool's `description` at `512` chars and cap normalized `inputSchema` JSON at `8,192` chars (falling back to a bounded description + simplified schema warning if exceeded).
   - Resolve sanitized 64-char name collisions deterministically by appending a short hash suffix instead of raising `ValueError` and rolling back the entire server.
   - Capture `init_result = await session.initialize()` -> store `init_result.instructions` (stripped & bounded to 500 tokens/server) on the server handle so `PromptCompiler` injects connected server instructions into Tier 2.

### WP-4 Acceptance Criteria & Verification

- Measured Turn-1 default tool definitions drop from **~6,262–6,595 tokens** to **≤ 2,500 tokens**.
- Connecting a mock MCP server with 80 tools + 50 resources + 20 prompts adds **≤ 300 tokens** to Turn 1; calling `tool_search(query=...)` activates only the top-K matching tools at the deferred tail without altering the `DIRECT` prefix hash.
- Pydantic v2 MCP tool schemas (`$defs`, `oneOf`, `format`, `pattern`) register cleanly instead of being silently dropped.
- Paginated `list_tools` (`nextCursor`) discovers all pages; duplicate sanitized names disambiguate cleanly; parallel connect of 3 failing servers finishes in ~1× timeout (not 3×); mid-turn MCP reconnect succeeds through `PolicyToolView`.
- `ruff check kohaku/`, `basedpyright` regression check (0 new errors), and all `tests/tools/test_mcp*.py` + new `tests/context/test_tool_surface.py` pass.

---

## WP-5 — Codex-Style Two-Phase Memory & Unified `CompactionEngine`

**Goal**: Unify all compaction paths (inline, `/compact`, subagents, thin idle runner) into one clean engine, and replace the disconnected Dream/history setup with Codex's working two-phase memory pipeline + FTS5 `memory_search`.

### WP-5.1 Unified `CompactionEngine` (`kohaku/context/compaction.py`)

1. **Single Canonical Engine**:
   - Consolidate compaction logic from `context_governance.py` (`compact_source_request`, `_compact_request_history`, `snip_history`) into `kohaku/context/compaction.py`.
   - Default thresholds: `trigger_ratio = 0.80`, `target_ratio = 0.50`, `overall_timeout_seconds = 120.0`, `idle_timeout_seconds = 45.0`.
   - Add optional `compaction_model: str | None = None` in `AgentDefaults` (`kohaku/config/schema.py`) so users can route compaction to a fast/cheap model (defaults to session model).
   - Always force `reasoning_effort = "none"` during compaction calls so reasoning tokens never starve summary output.
2. **Structured 5-Section Compaction Prompt (`kohaku/templates/agent/compaction_summary.md`)**:
   - Enforce an explicit structured template:
     ```markdown
     ## 1. Active Objective & Constraints
     ## 2. Files Inspected / Modified (exact paths & symbols)
     ## 3. Key Decisions, Findings & Verified Facts
     ## 4. Open Errors, Blockers & Unresolved Questions
     ## 5. Next Concrete Steps & Artifact Handles
     ```
   - When updating an existing checkpoint, explicitly instruct the model to merge and prune resolved items while preserving still-active constraints and exact file/artifact references.
3. **Performance, Role & Tokenizer Fixes**:
   - Replace O(B) sequential preflight + reverse-search deepcopy loops with **O(log B) binary search** over closed eligible boundaries using `TokenMeter`.
   - Validate summary length against a model-consistent budget (`len(summary) <= retained_tokens * 5` or calibrated `TokenMeter` check with 10% tolerance) instead of discarding valid non-OpenAI summaries on `cl100k_base` +1 token drift.
   - Emit `CheckpointNode.message()` as a structured `role: "user"` context block (`<conversation_summary>\n...\n</conversation_summary>`) — or merge cleanly with a continuation cue — **never** as a bare leading `role: "assistant"` message right after `system`.
   - Wire `/compact` on canonical source sessions (`kohaku/agent/loop.py:7351-7352`) directly to `CompactionEngine.compact_now(session_key)`.
   - Replace subagent blind `snip_history` with `CompactionEngine` (preserving the initial user task message as pinned anchor if emergency truncation is ever needed).
   - Add a **thin background idle runner** (~150 LOC in `kohaku/context/compaction.py`, wired cleanly in `gateway_runtime.py` when `idle_compaction.enabled=True`) that runs `CompactionEngine` on idle canonical sessions and triggers Stage-1 memory extraction.
   - Retire legacy H/delta `ContextCompactionState` and `MemoryArchiver` once all callers/tests run through `CompactionEngine`.

### WP-5.2 Codex-Style Two-Phase Long-Term Memory (`kohaku/context/memory_pipeline.py`)

Directly modeled on `codex-rs/core/src/memories/`:

1. **Workspace Memory Layout (`<workspace>/memory/`)**:
   ```text
   <workspace>/memory/
   ├── state.json                    # Watermarks: processed session revisions, last_phase2_at, pending_rollouts
   ├── rollout_summaries/
   │   └── <YYYYMMDD>-<session_slug>.md  # Stage-1 output: concise per-session summary (secrets stripped)
   ├── raw_memories.md               # Stage-1 output: chronological append of extracted memory bullets
   ├── memory_summary.md             # Phase-2 output: v1 navigational index (<= 1,500 tokens — ONLY file injected in Tier 2!)
   ├── MEMORY.md                     # Phase-2 output: detailed searchable handbook (ON-DEMAND only — NOT injected in full!)
   └── memory_index.sqlite           # SQLite FTS5 index rebuilt incrementally for memory_search
   ```
   - Backward compatibility: if `memory_summary.md` does not yet exist on an existing workspace that has a customized `MEMORY.md`, `PromptCompiler` falls back to injecting `MEMORY.md` (hard-capped at 1,500 tokens) until the first Phase-2 consolidation creates `memory_summary.md`.
2. **Stage 1 — Async Per-Session Rollout Extraction (`extract_session_rollout`)**:
   - **Triggers**:
     1. Canonical `CompactionEngine` commits a checkpoint (emits a lightweight async extraction job).
     2. User executes `/new` (session close).
     3. Background idle runner marks a canonical session idle (`>= idle_minutes`).
   - **Behavior**:
     - Skips ephemeral/temporary chats (`persist=False`) and internal `dream:` sessions.
     - Strips secrets/credentials (reusing Kohaku's existing secret scanner from `repositories.py`) and tool noise.
     - Calls the configured memory/compaction model (`reasoning_effort="none"`) with `kohaku/templates/agent/memory_stage1_extract.md` to produce:
       - `rollout_summaries/<YYYYMMDD>-<slug>.md` (concise task outcome, lessons learned, reusable workflows, key file/repo refs; capped at 4,000 chars).
       - Appends compact extracted bullet points to `raw_memories.md` under lock (`open_workspace_memory_lock`), updating `state.json`.
     - Bounded retention: keeps latest `max_rollout_summaries = 200` files (pruning oldest after Phase-2 consolidation).
3. **Phase 2 — Global Consolidation / Dream v2 (`consolidate_memories`)**:
   - **Triggers**:
     1. Automatic background trigger when `pending_rollouts >= phase2_threshold` (default e.g. `5`, configurable via `DreamConfig`).
     2. Manual `/dream` slash command (runs immediately and reports diff summary).
   - **Behavior**:
     - Runs under `dream_workspace_lock` + Git-tracked baseline (`GitStore`) with restricted write tools scoped to `memory/memory_summary.md`, `memory/MEMORY.md`, `USER.md`, `SOUL.md`, and `skills/`.
     - Reads `raw_memories.md` + newly added `rollout_summaries/` + current `memory_summary.md` / `MEMORY.md` / `USER.md` / `SOUL.md`.
     - Updates:
       - **`memory_summary.md`** (must start with `<!-- kohaku-memory-summary: v1 -->`, strictly ≤ 1,500 tokens, dense navigational index pointing to sections of `MEMORY.md` and `rollout_summaries/`).
       - **`MEMORY.md`** (detailed durable handbook organized by project/domain/preference).
       - **`USER.md`** (personal profile/communication preferences, ≤ 500 tokens) and **`SOUL.md`** (interaction/style lessons, ≤ 400 tokens).
     - Archives/clears processed section of `raw_memories.md`, commits Git snapshot, and triggers incremental FTS5 re-index.
4. **Unify Memory Templates**:
   - Align `kohaku/templates/SOUL.md`, `USER.md`, `memory/MEMORY.md`, `agent/dream.md`, `agent/memory_stage1_extract.md`, and `kohaku/skills/memory/SKILL.md` to one consistent MECE contract. Remove stale `[Archived Context Summary]` and undefined `[skip]`/`[SKILL]` references.

### WP-5.3 Unified `memory_search` Tool (SQLite FTS5 / BM25)

- **Create `MemorySearchTool` (`name = "memory_search"`, `exposure = ToolExposure.DIRECT`)** in `kohaku/context/memory_pipeline.py` (registered in `fork_factories.py`):
  - Parameters:
    - `query: str` (natural language or keywords — tokenized FTS5 BM25 query, NOT literal exact substring!)
    - `scope: Literal["all", "memory", "rollouts", "sessions", "knowledge"] = "all"`
    - `limit: int = 8` (clamped `1..20`)
  - Indexes (incrementally by `(path, mtime_ns, size)` in `<workspace>/memory/memory_index.sqlite`):
    1. `<workspace>/memory/MEMORY.md` (chunked by markdown `##`/`###` sections)
    2. `<workspace>/memory/rollout_summaries/*.md`
    3. Linked knowledge repositories (`RepositoryCatalog` entries with `kind == "knowledge"` readable under `ResourceAccessPolicy`)
    4. Persisted session transcripts (visible turns + compaction checkpoints)
  - Returns ranked snippets (capped at 400 chars/hit, total response ≤ 2,000 tokens) with exact source paths / session keys so the model can follow up with `read_file` or `read_session`.
  - Respects `ResourceAccessPolicy` in managed mode (never returns snippets from unauthorized paths).

### WP-5 Acceptance Criteria & Verification

- `/compact` works on canonical source sessions and produces a structured 5-section `<conversation_summary>` user block (no leading `role: "assistant"`).
- Stage-1 extraction runs on canonical session `/new` or compaction checkpoint, writing `rollout_summaries/<slug>.md` + `raw_memories.md` with secrets stripped.
- Phase-2 consolidation updates `memory_summary.md` (`v1`, strictly ≤ 1,500 tokens) and `MEMORY.md`, and `PromptCompiler` injects ONLY `memory_summary.md` into Tier 2.
- `memory_search(query="auth redirect bug")` finds relevant chunks across `MEMORY.md`, `rollout_summaries/`, sessions, and linked knowledge repos even when the words are non-contiguous.
- `ruff check kohaku/`, `basedpyright` regression check (0 new errors), and full `pytest` suite pass.

---

## 2. Reviewer Checklist (Used to Audit Each Completed WP)

When reviewing an implementing agent's completed Work Package, verify every item below before approving the next WP:

1. **Diff & Scope Hygiene**:
   - Only files belonging to the current WP were modified/deleted.
   - No `ruff format` churn; git diff is minimal and readable.
   - No unvalidated `typing.cast` added; no `Any` leakage on first-party interfaces.
2. **Hard Gates**:
   - `ruff check kohaku/` passes with 0 warnings/errors.
   - `scripts/check_type_regressions.py` confirms **0 new basedpyright diagnostics across ALL files**.
   - Relevant + full `pytest` suites pass with zero regressions.
3. **Architectural Invariants**:
   - **WP-1**: `.memory-recovery.lock` coordination with `attachments/_media.py` preserved; zero remaining references to deleted modules.
   - **WP-2**: Non-GPT strict providers no longer count UTF-8 bytes as tokens; overflow recovery works on Anthropic/Bedrock; MCP/`web_fetch` outputs are artifact-bounded in canonical mode; OpenRouter/Bedrock/xAI/OpenAI caching verified by unit tests.
   - **WP-3**: Zero regex post-filtering over system prompt; `untrusted_content.md` cannot be stripped; Tier 1/2 prefix byte-stable across turns; `webui_quote` persists in canonical follow-ups while `$skill` does not replay full body.
   - **WP-4**: Turn-1 default tool schema tokens ≤ 2,500; BM25 `tool_search` activates deferred tools at tail without invalidating `DIRECT` cache prefix; Pydantic v2 MCP schemas + pagination + parallel connect + mid-turn reconnect verified.
   - **WP-5**: `/compact` works on canonical sessions; structured 5-section checkpoint emitted as valid non-assistant role; Stage-1 -> Phase-2 produces bounded `memory_summary.md` (≤ 1,500 tok); FTS5 `memory_search` retrieves non-contiguous multi-word queries cleanly.
