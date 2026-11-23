# Kohaku Security Policy

## Report a Vulnerability

Do not report a security vulnerability in a public issue. Create a private report through the [Kohaku Security Advisories page](https://github.com/JadenChun/nanobot-custom/security/advisories/new).

Include the Kohaku version, operating system, affected feature, and steps that reproduce the problem. Remove API keys, bot tokens, passwords, private keys, and personal data from the report.

## Protect Credentials

- Do not commit API keys, bot tokens, passwords, or private keys.
- Store credentials in a protected secret manager or environment variable.
- Use an environment reference in `~/.kohaku/config.json` when a setting accepts one, such as `${PROVIDER_API_KEY}`.
- Restrict access to config files that contain credentials. On macOS and Linux, use `chmod 600 ~/.kohaku/config.json`.
- Rotate a credential after you expose it in a log, issue, screenshot, or chat.

See [Configuration](./docs/configuration.md) for environment variables and secret settings.

## Limit Access

- Set channel allowlists or pairing controls before you connect a public chat.
- Do not use `allowFrom: ["*"]` unless every sender can use the agent.
- Keep test bots and production bots in separate configs and workspaces.
- Review chat access and pairing requests before you approve them.

See [Chat Apps](./docs/chat-apps.md) for channel access settings.

## Limit Tool Access

Kohaku tools can read and write files, run commands, and contact external services. Give the agent only the files and permissions that the task needs.

- Use a dedicated operating-system account with limited permissions.
- Use workspace restrictions or the supported shell sandbox when the host supports it.
- Do not run Kohaku as root or as an administrator.
- Review commands and file changes before you approve them.
- Keep credentials outside the agent workspace.

Read [Concepts](./docs/concepts.md) and [Sandbox Folders](./docs/sandbox-folders.md) before you enable broad file or shell access.

## Protect Network Endpoints

- Keep local WebUI and API endpoints bound to loopback unless remote access is required.
- Require the supported authentication settings before you expose an endpoint to a network.
- Use HTTPS at a trusted reverse proxy for public access.
- Restrict firewall access to the required clients.

The OpenAI-compatible API requires an API key when it binds to a non-loopback address. Read [WebSocket](./docs/websocket.md) and [OpenAI-Compatible API](./docs/openai-api.md) before you expose either service.

## Protect Local Data

Kohaku stores config, sessions, logs, memory, and workspace files on the local system. New default data lives under `~/.kohaku/`.

- Restrict access to the config, workspace, session, and log directories.
- Protect backups with the same controls as the original data.
- Review logs before you share them. Logs can contain prompts, file paths, and service responses.
- Keep model-provider privacy and retention policies in mind when you send prompts.

If you move from Nanobot to Kohaku, follow [Upgrade to Kohaku](./docs/upgrade-to-kohaku.md). The guide explains how to protect snapshots and preserve the original paths.

## Update Kohaku

Install and update Kohaku from a clone of this repository until a Kohaku package release is available. Use only shell and wheel builds that passed the same release checks. Do not mix files from different releases.

For an existing profile, read [Upgrade to Kohaku](./docs/upgrade-to-kohaku.md) before you update. Keep a protected snapshot and know how to restore it to the original paths.
