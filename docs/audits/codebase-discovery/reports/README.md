# Discovery reports and exact read lists

Read the [current map](../README.md), [completion record](../completion-2026-11-06.md),
then the relevant reports below. Each Markdown report has its original input hash and
publication notice. Coverage JSON remains byte-identical to the retained exact list;
[report-manifest.json](../provenance/report-manifest.json) records input/publication hashes.
Exact lists include supplementary reads where originally reported; they are not primary
assignment denominators. The final ledger conservatively credits only assigned/inspected
completion paths, excluding incidental completion context paths.

## Historical nine-family reports

Pin `4e2bf4d8d0b34f133c01591b67826ca956fed384`, tree
`d5cc262acda5fb8051ffac9996ee9ab189c361cd`. These are readable editions of coordinator-condensed
historical notes, not re-created analyst transcripts or new source review. Only exact
report membership plus identical previous/current SHA256 carries inspection. Changed-path
claims are superseded by the refresh. Prior commits are local-only relative to current
staging; the portable previous hashes and current identical bytes establish inheritance
without requiring those objects on the remote. Original fine-grained/deferred limits remain
visible; completion closes the current mapped-only assignments separately.

| Report | Exact list | Listed inspected paths at original pin |
| --- | --- | ---: |
| [Entrypoints and turns](prior-2026-11-05/01-entrypoints-turns.md) | [JSON](prior-2026-11-05/01-entrypoints-turns-coverage.json) | 52 |
| [State and context](prior-2026-11-05/02-state-context.md) | [JSON](prior-2026-11-05/02-state-context-coverage.json) | 67 |
| [Sandbox and authority](prior-2026-11-05/03-sandbox-authority.md) | [JSON](prior-2026-11-05/03-sandbox-authority-coverage.json) | 51 |
| [Providers and media](prior-2026-11-05/04-providers-media.md) | [JSON](prior-2026-11-05/04-providers-media-coverage.json) | 81 |
| [Tools and automation](prior-2026-11-05/05-tools-automation.md) | [JSON](prior-2026-11-05/05-tools-automation-coverage.json) | 108 |
| [Channels and extensions](prior-2026-11-05/06-channels-extensions.md) | [JSON](prior-2026-11-05/06-channels-extensions-coverage.json) | 130 |
| [Backend, remote and mobile](prior-2026-11-05/07-backend-remote.md) | [JSON](prior-2026-11-05/07-backend-remote-coverage.json) | 185 |
| [Browser, desktop and terminal clients](prior-2026-11-05/08-clients.md) | [JSON](prior-2026-11-05/08-clients-coverage.json) | 287 |
| [Tooling and efficiency](prior-2026-11-05/09-tooling-efficiency.md) | [JSON](prior-2026-11-05/09-tooling-efficiency-coverage.json) | 136 |

## Refresh reports

Pin `27adced0aacd55d61791bb88a7240d7ad7bd5b75`, tree
`d49aca97c9a34a2d17eae53d672fa591652457f1`. These reports govern changed behavior over the
historical reports. Their unique current union is 93 paths/85 source targets. The
[refresh checkpoint](../refresh-2026-11-06.md) retains all 47 changed responsibilities and
its original 85 current/695 carried/139 mapped-only source counts.

| Report | Exact list | Listed inspected paths | Changed implementation paths |
| --- | --- | ---: | ---: |
| [Core and automation](refresh-2026-11-06/01-core-automation.md) | [JSON](refresh-2026-11-06/01-core-automation-coverage.json) | 41 | 13 |
| [Backend and mobile](refresh-2026-11-06/02-backend-mobile.md) | [JSON](refresh-2026-11-06/02-backend-mobile-coverage.json) | 37 | 13 |
| [WebUI](refresh-2026-11-06/03-webui.md) | [JSON](refresh-2026-11-06/03-webui-coverage.json) | 33 | 21 |

## Completion reports

Same current pin/tree. These three independently accepted source readers close exactly 139
assigned/inspected paths and provide 139 distinct responsibility rows. The prose is retained
with publication notices and portable source links. Their extra context lists are separate
from new coverage. No source comparison, architecture choice or runtime qualification follows
from complete module/relevant-body coverage.

| Report | Exact list | Assigned/inspected paths |
| --- | --- | ---: |
| [Startup/core helpers](completion-2026-11-06/01-startup-core.md) | [JSON](completion-2026-11-06/01-startup-core-coverage.json) | 59 |
| [State/recovery helpers](completion-2026-11-06/02-state-recovery.md) | [JSON](completion-2026-11-06/02-state-recovery-coverage.json) | 35 |
| [Knowledge/runtime instructions](completion-2026-11-06/03-knowledge-instructions.md) | [JSON](completion-2026-11-06/03-knowledge-instructions-coverage.json) | 45 |
