# Local context-budget correction

Date: 2026-10-31; branch: `fix/local-context-budget`

## Historical failure

Before this correction, one finalized local request containing three inline PNGs
was rejected at `1,567,570` measured units against a `156,672` input limit. The
captured request log attributed the result to `canonical_json_utf8_bytes`. The
same investigation recorded that a prior iteration used 36,441 prompt tokens,
which made the final-body measurement inconsistent with the logical estimator.
Those values describe the earlier live observation; they are not a measurement
of the code in this patch.

The existing image-history change also retains contiguous batches of image
tool results. That behavior and its tests are outside this correction and stay
unchanged.

## Measurement behavior

The final request guard first recognizes media at the selected transport's
actual content-block positions. It replaces only the payload value with a
content-free marker, keeps the rest of the body intact, and adds a model-aware
image allowance. Tool inputs, tool JSON, schemas, defaults, metadata, text, and
nested lookalikes remain text. OpenAI Chat image URL fields are recognized
only at message content blocks; unrelated fields such as `file_id` remain
text. Responses `file_id` references are recognized only on actual
`input_image` blocks. Bedrock image bytes count only beneath actual image
blocks or image blocks in tool results; redacted reasoning bytes count only at
their documented direct block path. Other byte values fail closed.

Data URI headers use a bounded search and a single parameter pass before the
existing bounded prefix decode. PNG, GIF, WebP, and JPEG dimension readers do
not decode full image pixels.

The logical estimator uses the known encoding for a selected provider/model
identity when one is available. Unknown model names use the historical
`cl100k_base` compatibility estimate, reported as `tiktoken_compatibility`;
this label does not claim an authoritative tokenizer for that model. A
tokenizer loading or encoding failure uses UTF-8 bytes. Without a selected
model, the logical estimator retains its historical `cl100k_base` default.

For other providers, the final request guard continues to use exact installed
model mappings, listed canonical families, valid dated snapshots, and
provider-owned Codex/Copilot routing prefixes. Unknown aliases and unavailable
encodings retain the strict UTF-8 byte-based behavior. The estimated
compatibility policy is declared by a provider preset or by the
`requestAccounting` config field; it is never inferred from a provider name,
URL, model-name fragment, or wrapper. Local Inference Manager is one preset
that declares it. Other local providers remain strict until they opt in.

For that route, the same finalized effective Chat body is used at
pressure, auxiliary preflight, post-summary fit, preview, and dispatch. Ordinary
Chat fields and additive configured extras are merged with the installed SDK's
top-level `extra_body` precedence; client-only timeout/header/query controls
are excluded from the JSON body. Conflicting governed model or message
overrides are rejected before transport. Text uses the explicitly named
`estimated_compat` policy with the existing 15% margin and the existing
selected-model image allowance. This is a compatibility estimate, not a Qwen
tokenizer or a guarantee of backend headroom. If the encoding is unavailable,
the measurement identifies the UTF-8 byte-ceiling fallback rather than claiming
a token estimate.

The local request policy also applies an independent 64 MiB ceiling to the full
serialized effective JSON, including inline base64 media and metadata. It
reports `payload_bytes` with byte counts and does not turn a byte overflow into
context recovery. Positive configured capacity, the selected leaf's narrower
capacity, actual output reserve, and safety buffer remain in the token-budget
calculation; those units are still estimates.

A reported-usage floor is available only to an attributed direct local leaf
after a successful response is validated, recorded, and its source ticket is
settled. It requires a single request with reported prompt usage and unchanged
provider/client, credential, endpoint, model, tools, generation, media and
projection scope. Reuse is limited to verified append-only converted input;
cache read/write counts are subsets of prompt input and are not added again.
Changed or uncertain scope falls back to a full estimate. Wrappers, failed or
estimated usage, auxiliary summaries, malformed responses and unrecorded
responses cannot install an anchor.

## Image policies

- The exact `local_manager` model `Qwen3.8-27B-UD-Q6_K_XL` requires known image
  dimensions and uses `max(128, ceil(width/28)*ceil(height/28),
  ceil(width*height/750)) + 32` tokens per image. It has no universal 16,384
  cap. Low, high, and automatic detail use the same conservative allowance.
- `gpt-4o` and `gpt-4.1` use an 85-token base and 170 tokens per 512-pixel tile;
  `gpt-4o-mini` uses a 2,833-token base and 5,667 per tile. Low detail uses the
  base. High and automatic detail apply the documented 2,048-pixel bound,
  768-pixel short edge, and 512-pixel tiles. Unknown dimensions and file/URL
  references use an eight-tile upper bound.
- Registered `gpt-5.6`, `gpt-5.5`, `gpt-5.4` and `gpt-5.4-mini` wire models use
  conservative per-detail maxima. The 5.6 and 5.5 low-detail bound covers a
  512-by-512 image; high and automatic/original detail retain the documented
  family-specific ceilings. The 5.4 low bound intentionally exceeds its high
  bound. `gpt-4.1-mini` and `gpt-4.1-nano` use their documented conservative
  patch ceilings.
- Claude 4.5/4.6 Haiku, Sonnet, and Opus use a 1,568-token visual ceiling plus
  framing. Claude Opus 4.7 uses its high-resolution 4,784-token ceiling plus
  framing for automatic/default and high detail. The default Bedrock ID
  `global.anthropic.claude-opus-4-7` is recognized.

Unsupported detail values, unknown image models, unknown dimensions for the
local Qwen model, and actual audio/video/document/file blocks fail closed.
Logical estimates keep their generic helper defaults; the final transport
guard does not use them as universal model limits.

## Source-backed recovery and summaries

Foreground source compaction defaults to a 300-second stream idle bound and a
900-second absolute operation deadline shared across preflight, up to four
folds, waits and publication. Each incoming SDK chunk, including reasoning or
usage-only progress, resets the idle interval. These settings apply to the
foreground path and do not change general main-request timeouts. Direct legacy
canonical summary helpers retain their existing 60-second default.

The values are configured under `agents.defaults.sourceCompaction`:

```json
{
  "agents": {
    "defaults": {
      "sourceCompaction": {
        "idleTimeoutSeconds": 300.0,
        "overallTimeoutSeconds": 900.0
      }
    }
  }
}
```

Summary evidence remains tied to acknowledged, closed source boundaries. When
the direct known local route can prove that a converted historical source
prefix has the same SDK message/tool prefix and compatible request envelope as
the main request, the first fold may use that prefix. Any conversion or
envelope mismatch falls back to the canonical source representation. Later
folds consume disjoint source intervals and include the prior summary once.
Unacknowledged or pending inputs, transient reasoning, and expanded
request-only image/context text are not promoted to source evidence. An
acknowledged canonical current-user event may contribute when it is eligible
at a closed source boundary; its pinned source row remains retained by the
source rebuild. Prefixes are conditional opportunities for reuse; no cache
hit or universal prefix match is promised.

The once-streaming auxiliary path validates complete nonempty stop responses,
rejects tool calls and truncation, closes its SDK stream on success and
failure, and does not use framework retries, fallback construction or pool
rotation. Cancellation closes the client stream where supported, but does not
prove that a remote backend stopped processing immediately. Diagnostics retain
the original recovery cause plus a specific sanitized failure reason and safe
budget, measurement, prefix and call metadata. Prompts, summaries, images,
credentials, private reasoning, and full endpoint query strings are not logged.

## Offline verification

The correction is verified with focused unit and mocked-dispatch tests plus
repository Ruff checks. These offline checks do not call a live provider or
modify a gateway, desktop configuration, session, or model service. They do not
establish exact Qwen token counts, guarantee that every local deployment shares
the 64 MiB payload setting, establish universal cache behavior, or prove
immediate backend termination after cancellation. The historical live
observation above is separate from the offline test results.
