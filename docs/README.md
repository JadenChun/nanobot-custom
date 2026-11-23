# Kohaku Documentation

Use these docs to install Kohaku from a clone of this repository and get one reply. Then open a task guide when you need another capability.

These guides describe the current source tree. For a fresh install, follow [Install and Quick Start](./quick-start.md). If you already use Nanobot, read [Upgrade to Kohaku](./upgrade-to-kohaku.md) first.

## Start Here

| Your situation | Read this | You are done when... |
|---|---|---|
| Terminals, Python, or API keys are new to you | [Beginner walkthrough](./start-without-technical-background.md) | The browser can send `Hello!` and receive a reply |
| You are comfortable running commands | [Install and Quick Start](./quick-start.md) | `kohaku status` is healthy and the CLI can get one reply |
| Something already failed | [Troubleshooting](./troubleshooting.md) | You have isolated the problem to install, config, model, gateway, channel, or tool access |

The recommended first-run path is:

1. Clone this repository and install Kohaku in a virtual environment.
2. Configure a provider and model with `kohaku onboard --wizard`.
3. Send `Hello!` with `kohaku -m "Hello!"`.
4. Build the WebUI assets before you run `kohaku webui` from an editable install.

Most people do not need to edit JSON for the first run. The setup wizard configures a provider and model. After you open the WebUI, use **Settings → Channels** for chat apps and **Apps** for tool integrations.

## Add One Capability

Pick the row that matches what you want to accomplish next:

| Goal | Guide |
|---|---|
| Learn the browser workbench | [WebUI](./webui.md) |
| Connect Telegram, Discord, Slack, Feishu, WeChat, Email, or another chat app | [Chat Apps](./chat-apps.md) |
| Choose a hosted, OAuth, company, or local model | [Provider Cookbook](./provider-cookbook.md) |
| Add model fallbacks | [Configure Model Fallback](./guides/configure-model-fallback.md) |
| Enable web search | [Configure Web Search](./guides/configure-web-search.md) |
| Manage Agent Plugins, CLI Apps, or MCP integrations | [WebUI Apps](./webui.md#apps) |
| Add an MCP tool server | [Configure MCP Tools](./guides/configure-mcp-tools.md) |
| Generate images | [Image Generation](./image-generation.md) |
| Schedule work or create a local trigger | [Automations](./automations.md) |
| Understand and manage long-term memory | [Memory](./memory.md) |
| Run Kohaku continuously | [Deployment](./deployment.md) |
| Run separate bots or workspaces | [Multiple Instances](./multiple-instances.md) |
| Call Kohaku from Python | [Python SDK](./python-sdk.md) |
| Expose an OpenAI-compatible endpoint | [OpenAI-Compatible API](./openai-api.md) |

For shorter, outcome-focused walkthroughs, browse the [task guide index](./guides/README.md).

## Operate Kohaku

| Need | Read |
|---|---|
| Commands and flags | [CLI Reference](./cli-reference.md) |
| In-chat slash commands | [In-Chat Commands](./chat-commands.md) |
| Config, workspace, gateway, sessions, tools, and memory in plain language | [Concepts](./concepts.md) |
| Provider/model matching and selection | [Providers and Models](./providers.md) |
| Setup and runtime diagnosis | [Troubleshooting](./troubleshooting.md) |
| Older development highlights | [Release Archive](./release-archive.md) |

## Reference

Use reference pages to look up an exact option after you know what you are trying to configure:

| Area | Reference |
|---|---|
| Every configuration field and default | [Configuration](./configuration.md) |
| Provider and model behavior | [Providers and Models](./providers.md) |
| Chat channel prerequisites and manual JSON | [Chat Apps](./chat-apps.md) |
| WebSocket authentication and wire protocol | [WebSocket](./websocket.md) |
| Python SDK classes, events, sessions, and hooks | [Python SDK](./python-sdk.md) |
| OpenAI-compatible HTTP routes and payloads | [OpenAI-Compatible API](./openai-api.md) |
| Runtime self-inspection and tuning | [My Tool](./my-tool.md) |

Configuration examples are usually snippets to merge into `~/.kohaku/config.json`, not complete replacement files. The docs use camelCase because Kohaku writes config that way. Keep API keys, bot tokens, and passwords out of issues and public logs.

## Extend or Contribute

These pages explain implementation and extension points. You do not need them to install or operate Kohaku.

| Goal | Read |
|---|---|
| Understand source ownership and runtime flow | [Architecture](./architecture.md) |
| Review feature implementation gaps and acceptance criteria | [Feature implementation audit](./feature-implementation-audit-2026-11-05.md) |
| Review completed feature repairs and their validation limits | [Feature repair results](./audits/feature-implementation-2026-11-05/repair-results.md) |
| Follow the code-quality audit scope and stages | [Code-quality audit charter](./code-quality-audit-2026-11-04.md) |
| Review the A3 audit, candidate design, decisions and merge gates | [A3 architecture record](./architecture/a3/README.md) |
| Review the Kohaku history, prompt navigation, and composer design and track implementation gates | [Chat-page design and audit](./architecture/gui-session-history-design.md) · [Work plan and progress](./architecture/gui-session-history-work-plan.md) |
| Review future agent-managed sessions and the SuperAgent study | [SuperAgent research and session direction](./superagent-research-and-session-direction.md) |
| Review the retired H1/concurrency cleanup | [Telegram H1 and global concurrency cleanup plan](./telegram-h1-concurrency-cleanup-plan.md) |
| Set up a development environment | [Development](./development.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Add a channel package | [Channel Package Guide](./channel-package-guide.md) |
| Build the WebUI source | [WebUI Development](../webui/README.md) |

The A3 record describes a separately pinned candidate and links its immutable audit history.
Its review and qualification states are separate from the staging runtime described by the main architecture page.

If a command or screen does not match these docs, open an issue in [this repository](https://github.com/JadenChun/nanobot-custom/issues). Include your Kohaku version, operating system, and the page that needs correction.
