# Automations

<!-- Meta description: Create, run, and manage supported scheduled automations and local triggers, and inspect inactive periodic features. -->

Automations are agent turns that run later. Use them
when nanobot should do work without someone actively typing: reminders,
recurring checks, nightly summaries, CI follow-ups, local script reports, or
webhook-driven events.

The gateway currently supports two explicitly enabled execution paths:
`gateway.cron.enabled: true` admits fresh scheduled turns with Telegram
receipt delivery; `gateway.localTriggers: true` delivers queued local messages
to their bound chat sessions. Both default to off. Session-bound scheduled jobs,
WebSocket scheduled delivery, periodic Heartbeat, and periodic Dream are not
installed by the normal gateway.

## Choose an Automation Type

| Type | Starts from | Best for | Created with |
|---|---|---|---|
| Scheduled automation | Time, interval, or cron expression | Recurring reminders, scheduled summaries, one-time future tasks | Ask nanobot in the target topic to schedule it with the `cron` tool |
| Local trigger | A local `nanobot trigger ...` command | CI jobs, webhooks, shell scripts, generated reports | `/trigger <name>` in the target topic |
| Heartbeat | Unavailable in the normal gateway | Retained configuration and workspace task file | Periodic execution is inactive |

The two user-created automation types are scheduled automations and local
triggers. Retained system definitions remain protected from normal automation
edits; their presence does not mean a background worker is installed.

## Before You Create One

Enable the execution path you need, then keep `nanobot gateway` running:

```json
{
  "gateway": {
    "cron": { "enabled": true },
    "localTriggers": true
  }
}
```

Only enable cron after migrating existing enabled jobs to its fresh-turn
contract. Unsupported enabled definitions stop activation with an explicit
error. Enabling cron does not activate Heartbeat or Dream.

Use the same workspace and config for the gateway and any process that sends
local trigger messages. If you run multiple nanobot instances, pass the matching
`--config` or `--workspace` option to `nanobot trigger`.

Create a local trigger from the target chat or WebUI topic. For scheduled
execution, use a supported Telegram chat. Other scheduled origins can retain
definitions while execution remains unavailable. An automation without a linked
origin cannot be enabled or run from the WebUI.

## Scheduled Automations

Scheduled automations are created by the agent's `cron` tool. For the installed
scheduler, ask nanobot from the target Telegram chat:

```text
Every weekday at 9am, check open pull requests and summarize blockers here.
```

or:

```text
Tomorrow at 4pm, remind me to send the release notes.
```

The cron tool supports interval schedules, cron expressions, and one-time
scheduled tasks. Cron expressions can include an IANA timezone such as
`America/Vancouver`; otherwise nanobot uses the runtime default timezone.

The installed scheduler executes each run with fresh context and a dedicated
`cron:<job-id>` session. Supported Telegram destinations receive a durable
receipt-backed result. It does not resume the history of the chat that created
the job. Existing session-bound jobs require explicit migration before execution.
Other delivery transports are unsupported by this scheduler.

## Local Triggers

Local triggers let a local script or external service send a message into a
specific nanobot session later.

Set `gateway.localTriggers` to `true` and restart the gateway to activate
delivery. Creation and CLI enqueue remain available while delivery is inactive;
messages stay in the queue, and the command reports the activation requirement.

Create the trigger from the chat or WebUI topic where future messages should
arrive:

```text
/trigger PR review
```

nanobot replies with a trigger ID and a command shaped like:

```bash
nanobot trigger trg_8K4P2Q9X "Review PR #4502"
```

Replace the quoted text with the message nanobot should receive. For generated
or longer content, pipe stdin:

```bash
generate-report | nanobot trigger trg_8K4P2Q9X
```

For multiple instances, use the same config or workspace selector as the
gateway:

```bash
nanobot trigger --config ./bot-a/config.json trg_8K4P2Q9X "Nightly report"
nanobot trigger --workspace ./bot-a/workspace trg_8K4P2Q9X "Nightly report"
```

nanobot does not provide a built-in public webhook receiver for local triggers.
If GitHub, CI, or another external system should wake nanobot, run your own
small webhook service and have it call `nanobot trigger` after it builds the
final message.

## Heartbeat

Periodic Heartbeat is inactive in the normal gateway, including when the
compatibility `gateway.heartbeat.enabled` flag is true. Editing `HEARTBEAT.md`
does not schedule execution. Settings report configured intent separately from
effective support. See
[`configuration.md#gateway-heartbeat`](./configuration.md#gateway-heartbeat).

## Manage Automations

Use the WebUI Automations view to:

- filter by all, active, paused, needs-attention, or system jobs;
- search by task name, message, trigger command, linked topic, schedule, or
  status;
- sort by next run, last run, updated time, or name;
- run supported scheduled automations now when the gateway advertises admission;
- pause or resume, rename, or delete user-created automations;
- copy the CLI command for local triggers;
- inspect protected system automations without changing them.

Local triggers do not have a WebUI "Run now" action because each run needs a
message. Copy the `nanobot trigger ...` command from the WebUI and replace
`"message"` with the content that should be delivered.

Unsupported or inactive execution paths display a reason. The UI reads
next-run, latest occurrence outcome, and delivery status from the installed
runtime; definition-only storage cannot supply that state.

## Delivery and Reliability

Automation delivery is workspace-local. Scheduled jobs and local trigger
deliveries use the same workspace as the gateway.

Local trigger messages are written to a durable queue. If the gateway is not
running yet, the message waits in that workspace. If the linked topic is
already running a turn, the trigger waits until the session becomes idle instead
of being injected into the active turn.

The local trigger queue is at-least-once, not exactly-once. If the gateway exits
after claiming a delivery but before the linked turn completes, the next gateway
start requeues that delivery. External scripts should make repeated trigger
messages safe. If the delivery reaches the agent and the turn fails, the
delivery is marked failed instead of retrying forever.

Each local trigger delivery writes an audit record under
`<workspace>/triggers/runs`. Run one gateway consumer per workspace; the local
queue is not a distributed multi-consumer queue.

## Common Patterns

For a nightly scheduled report, ask from the target Telegram chat:

```text
Every night at 9pm, review today's workspace changes and summarize anything I should handle tomorrow.
```

For a CI follow-up, create a trigger once:

```text
/trigger CI follow-up
```

Then have your CI or webhook adapter call:

```bash
nanobot trigger <trigger-id> "Build failed on main. Inspect the logs and suggest the next fix."
```

For a local report script:

```bash
generate-report | nanobot trigger <trigger-id>
```

## Troubleshooting

If an automation does not run, check its execution reason in Automations, the
gateway's activation flags, and whether its execution mode and delivery
transport are supported.

If a local trigger waits forever, confirm the command uses the same workspace or
config as the gateway and that `gateway.localTriggers` is enabled.

If a trigger message appears twice after a restart, treat it as expected
at-least-once delivery and make the external message idempotent.

If you need to edit, pause, resume, rename, delete, or inspect automations, use
the WebUI Automations view.

## Related Docs

- [`webui.md#automations`](./webui.md#automations) for the browser management view
- [`chat-commands.md#local-triggers`](./chat-commands.md#local-triggers) for `/trigger`
- [`cli-reference.md#local-triggers`](./cli-reference.md#local-triggers) for `nanobot trigger`
- [`configuration.md#gateway-heartbeat`](./configuration.md#gateway-heartbeat) for heartbeat settings
- [`guides/long-running-ai-agent.md`](./guides/long-running-ai-agent.md) for long-running agent work
