# Local context-budget failure: diagnosis and fix

Date: 2026-10-31
Branch: `fix/local-context-budget` (based on `staging`)
Environment: staging gateway, `local_manager` provider, Qwen3.8 27B GGUF at `n_ctx=165888`

## Symptom

The loop rejected a live request after three PNG reads:

```text
Local request budget rejected provider=local_manager model=Qwen3.8-27B-UD-Q6_K_XL
transport=openai_chat_completions_stream measured_units=1567570
input_limit=156672 over_by=1410898 completion_reserve=8192
total_limit=165888 measurement_source=canonical_json_utf8_bytes
```

The previous iteration had only used 36,441 real prompt tokens, so the model was
not remotely close to its 165,888-token window.

## Root causes

1. **Base64 images were measured as text.** The final transport guard counted
   every UTF-8 byte of an inline data URI as one token. One 1024x1024 PNG was
   ~1.44 MB of base64, so it alone measured ~1.44M "tokens". llama-server reports
   1,043 prompt tokens for the same image.
2. **Text was also measured as bytes.** For ordinary prose the guard required
   ~3.4 bytes per real token. A 165,888-token local window therefore behaved like
   a ~46k-token window before any image was involved.
3. **The logical estimator ignored images.** `estimate_prompt_tokens` skipped
   non-text blocks, while the final gate counted all of their bytes. Budgeting
   and compaction could not see the same cost the final gate enforced.
4. **Parallel image tool results were silently truncated.** The newest-image
   window kept only the newest image-bearing *message*. Three parallel
   `read_file` calls produce three contiguous tool messages, so only the last
   image reached the model. This both corrupted the visual comparison and made
   the request look like one image instead of three.

## Fix

- `nanobot/utils/media_tokens.py`
  - Reads image dimensions from PNG/GIF/WebP/JPEG headers without decoding pixels.
  - Uses Qwen-style 28x28 patch accounting and Claude-style area/750 accounting,
    capped at the documented 16,384 visual-token maximum.
  - Applies bounded allowances to remote media and unknown containers.
- `nanobot/providers/request_budget.py`
  - Replaces inline media with a content-free marker and adds the media token
    allowance instead of the base64 wire length.
  - Measures text with `cl100k_base` plus a 15% safety ratio; falls back to the
    previous byte ceiling if tokenization is unavailable.
  - Handles OpenAI Chat/Responses, Anthropic base64/url sources, Bedrock binary
    media, and audio/video blocks by field shape.
  - Keeps data URIs embedded in ordinary text on the text-token path.
- `nanobot/utils/helpers.py`
  - Counts image/media blocks in `estimate_prompt_tokens` and
    `estimate_message_tokens` so logical pressure matches the final guard.
- `nanobot/providers/image_history.py`
  - Keeps a contiguous parallel tool-result batch together instead of dropping
    all but the newest image.
- `nanobot/utils/runtime.py`
  - Error text now tells users to reduce attached images as well as tool output.

## Evidence

Measured with the production provider body shape and the three benchmark PNGs:

| Case | Old byte units | New token units | llama-server actual |
|---|---:|---:|---:|
| One 1024x1024 PNG | 1,442,420 | 1,506 | 1,043 |
| Three 1024x1024 PNGs | 5,205,180 | 4,414 | 3,094 |
| Persisted 80 KB text session | 80,084 | 28,172 | ~24k tiktoken |

At `total_tokens=165888`, `completion_reserve=8192`, `safety_buffer=1024`, the
three-image request now measures 4,414 against a 156,672-token input limit and
fits. The old gate rejected the one-image request at 1,567,570 units.

## Verification

```bash
pytest tests/providers -q          # 1750 passed, 3 skipped
pytest tests/agent -q              # 5107 passed, 1 skipped
pytest tests/session tests/cli tests/utils -q  # 3118 passed, 2 skipped
ruff check nanobot tests
basedpyright --pythonpath <venv>/bin/python \
  nanobot/utils/media_tokens.py nanobot/providers/request_budget.py \
  nanobot/providers/image_history.py nanobot/agent/runner.py
```

Desktop adapter tests in a worktree without built WebUI assets fail with
`webui_assets_unavailable`; that is an environment prerequisite, not this change.

## Live verification (2026-10-31)

A desktop-managed gateway on the isolated branch ran a three-image turn against
the live `Qwen3.8-27B-UD-Q6_K_XL` server. The agent issued three parallel
`read_file` calls, the model saw all three images, and the turn completed:

| Iteration | Prompt tokens | Meaning |
|---|---:|---|
| 0 | 9,211 | text + tools, before image results |
| 1 | 12,528 | +3,317 tokens = three 1024x1024 images |
| 2 | 12,803 | final answer |

No `Local request budget rejected` line was emitted. The final answer correctly
identified all three images as square 1024x1024 compositions.

The staging desktop app and gateway now run from an isolated interpreter at
`/home/jaden/.codex/worktrees/local-context-budget/venv`. Its `.pth` points at
this worktree and reuses the main venv's dependencies; the main
`nanobot-custom/.venv` was not modified. The desktop preference backup is at
`/tmp/opencode/nanobot-desktop-preferences.json.bak`.

Python WebSocket clients must raise `max_size`: tool-event frames that carry
inline image results can exceed the library's 1 MiB default. The WebUI already
uses the gateway's configured 37 MiB limit.

## Rollout notes

- The running staging gateway must be restarted on the merged branch; the live
  process still has the old code.
- Persisted image payloads are intentionally replaced by placeholders. After a
  restart, the model must re-read an image rather than replaying an old turn.
- Verify a new turn logs `measurement_source=canonical_json_tiktoken+inline_image`
  and `measured_tokens` in the low thousands for a few images.
- Keep model presets' `contextWindowTokens` aligned with the local server's
  actual `n_ctx`. The 27B preset uses 165,888, matching `/props`; the
  Flash-Next preset uses 262,144, matching its catalog entry.
