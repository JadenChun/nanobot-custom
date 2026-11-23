# Build a WhatsApp AI Agent with Kohaku

This guide connects Kohaku to WhatsApp through the `whatsapp` channel. The
channel links as a WhatsApp device and uses the same kohaku agent runtime,
tools, memory, and workspace as the CLI and WebUI.

## What this guide builds

- WhatsApp optional dependencies installed
- a linked WhatsApp device session
- the `whatsapp` channel enabled in `config.json`
- one pairing-approved WhatsApp sender

## Prerequisites

- A working local Kohaku reply:

```bash
kohaku agent -m "Hello!"
```

- A WhatsApp account that can link a new device.
- A machine that can keep `kohaku gateway` running.

## Install Kohaku

```bash
# Run from the root of a clone of this repository
python -m pip install -e .
kohaku onboard --wizard
```

## Enable the WhatsApp channel

Install the optional channel dependency:

```bash
kohaku plugins enable whatsapp
```

Link WhatsApp as a device:

```bash
kohaku channels login whatsapp
```

Scan the QR code from WhatsApp -> Settings -> Linked Devices.

Merge this snippet into `~/.kohaku/config.json`:

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "groupPolicy": "mention"
    }
  }
}
```

Omitting `allowFrom` enables pairing-only mode for private chats. `groupPolicy`
defaults to `"open"` in the channel, but `"mention"` is safer for a first
deployment.

## Run kohaku gateway

```bash
kohaku channels status
kohaku gateway
```

## Test a message

Send the bot a private WhatsApp message. It should return a pairing code.
Approve it from a trusted local surface:

```bash
kohaku agent -m "/pairing approve ABCD-EFGH"
```

Send the message again after approval. The reply should use the same model and
workspace as your local CLI check.

## Security notes

- Treat the WhatsApp session database as account access.
- Prefer pairing-only mode for first setup. Add `allowFrom` only when you want a
  static allowlist.
- Keep `groupPolicy` as `"mention"` before adding the bot to groups.
- Avoid `allowFrom: ["*"]` unless the bot is intentionally public or isolated.

## Troubleshooting

- If QR linking fails, rerun `kohaku channels login whatsapp`.
- If you are migrating from the old bridge, remove `bridgeUrl` and
  `bridgeToken`, then re-login.
- If a sender appears as a LID instead of a phone number, let Kohaku learn the
  mapping at runtime or use `lidMappings` in the full reference.
- If a first private message returns a pairing code, approve it before testing
  normal replies.

## Next: memory, automations, MCP tools

- [Chat Apps reference](../chat-apps.md)
- [Pairing](../configuration.md#pairing)
- [Secure local AI agent](./secure-local-ai-agent.md)
- [Deployment](../deployment.md)
