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

The final request guard uses exact installed model mappings, listed canonical
families, valid dated snapshots, and provider-owned Codex/Copilot routing
prefixes. It uses the selected model encoding with a 15% estimate margin;
unknown aliases, local model identities, and tokenizer failures use UTF-8 byte
counts. This keeps the logical estimator compatible with existing context
governors while the final guard stays conservative at dispatch.

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

## Offline verification

The correction is verified with focused unit and mocked-dispatch tests plus
repository Ruff checks. These offline checks do not call a live provider or
modify a gateway, desktop configuration, session, or model service. The
historical live observation above is separate from the offline test results.
