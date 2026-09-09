# Remaining adapter contracts from bounded migration analysis

These are implementation requirements, not completion evidence. Original migration plan/audit remain authoritative.

## Next: canonical image compatibility

Keep only `tools/image_generation.py::ImageGenerationTool` registered. Add a persisted configuration-source selector: omitted upstream imageGeneration section selects legacy tools.image (including Config()); explicit upstream section honors upstream enablement. Normalize before defaults obscure input presence and preserve selection through save/reload. Use one effective-settings helper for enabled/create/reload. Default legacy Codex exposes the tool even when executable is absent; missing executable is an invocation error. Codex CLI aliases remain distinct from upstream OAuth openai_codex. Legacy OpenRouter key/model/base precedence remains tools.image then environment.

Canonical schema gains optional output_path and retains refs/count/image_size. Validate count and output_path/count ambiguity before backend calls. Preserve exact stripped legacy prompt and 16:9 default. CLI edits without supported reference arguments fail explicitly. Add a dedicated owned-process Codex provider returning validated image data URLs. Keep argv invocation, configured command/model/timeout/home. Capture only fresh invocation-associated images under configured generation directory; serialize capture window and reject stale/unrelated/ambiguous files. Never accept arbitrary printed paths or unchanged output. Extend existing OpenRouter parser to legacy content image blocks/direct data URLs as well as upstream message.images; retain upstream validation/network protections and a narrow legacy payload mode.

All images pass upstream artifact storage. Compatibility output mapping: relative -> workspace/generated_images/basename; workspace absolute -> generated basename unless already in generated dir; external absolute only if filesystem policy authorizes. Validate before generation; recheck under shared write lock before atomic replacement. Return legacy path/size alongside artifacts, and bounded CLI provenance. Publication failure retains prior output and identifies persisted artifact without another provider attempt. Empty responses stop after one call. Reload validates replacement without weakening duplicate protection.

Acceptance: default/legacy/explicit disabled/save-reload modes; exact one canonical class; aliases and credential precedence; old+new schemas; no backend on invalid count/path; three OpenRouter response forms; fake CLI argv/provenance/timeout/cancel; path mapping/protected/symlink/authorized external destinations; artifacts usable as references; empty response and publication failure; reload.

## P3 provider adapters

Actual retained upstream files are fallback_provider.py and factory.py; key-pool coordination still needs implementation. Do not copy whole fork transports. Configuration retains custom fields but selectors/factory still use scalar api_key and only upstream fallback_models.

Credential availability/get_api_key must use effective_keys while preserving OAuth/local exemptions. api_keys list wins, including nonempty list of empty entries not resurrecting scalar. Keep both fallback forms with explicit policy selection; reject simultaneous nonempty legacy/upstream lists rather than silently discard one. Factory signatures must include effective ordered keys, rate_limit, timeout and complete fallback settings.

Use immutable per-key upstream OpenAI-compatible leaves with a shared coordinator. Preserve healthy preferred winner, rotating healthy remainder, cooling keys by expiry; cooling deprioritizes rather than excludes. Snapshot order per request, success promotes/clears cooldown. Rotate only fork quota/rate/availability cases (429/503/text); do not rotate timeouts/auth/generic connection errors. Quota/rate/overload cooldowns 1800/60/15 seconds. After exhausted pool, preserve structured last response and prevent outer retry from restarting pool. Anthropic/Azure only take first effective key as before.

Pacing: shared lock prune/check/reserve, release before sleeping/network, recheck after wake; 60-second window and one-second buffer. Cancellation before admission must not reserve capacity.

Legacy fallback policy: configured ready order then cooling order, quota/rate only, stop on nonquota error from any candidate, per-candidate five-minute cooldown, clear only successful candidate, all-cooling probes. Caller model override primary-only; explicit generation overrides remain effective. Cache candidate leaves. Keep upstream fallback policy separately for upstream presets.

Preserve chat/chat_stream/context-aware surfaces, generation/runtime, retry notifications and recovery policy. Any externally delivered delta prevents replay at leaf/base/pool/fallback layers. Bind continuation state to nonsecret credential identity; incompatible credential/provider/model state rebuilds from portable history. Preserve session/events and native compaction metadata. State ownership must identify actual leaf, never shared mutable current-provider state. Observe each actual leaf attempt once, not wrapper plus leaf.

Explicit provider timeout must bound request and stream idle wait while a continuously active stream can exceed it; close on timeout/cancel. Keep SDK max_retries=0, proxy/extra body/query/API type.

Tests: fake client/clock ordering, concurrent key isolation and RPM waiters, cancellation, rotation eligibility/exhaustion, structured metadata, legacy secondary nonquota stop/cooldowns/overrides, partial-delta no-replay at every layer, credential/state isolation and compaction preservation, observer counts, factory cache invalidation and timeout cleanup. Existing authentication/API/cron/media P3 audit gates remain additional work.
