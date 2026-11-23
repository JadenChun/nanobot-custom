<div align="center">
  <img src="webui/public/brand/kohaku_icon_512.png" alt="Kohaku" width="144">
  <h1>Kohaku</h1>
  <p>A personal AI assistant for work across apps and files.</p>
</div>

Kohaku helps you plan work and complete tasks with language models, tools, and chat integrations. This repository is the Kohaku fork of [HKUDS/nanobot](https://github.com/HKUDS/nanobot). Review [LICENSE](./LICENSE) for license terms.

Kohaku uses an adaptive agent harness. The main agent can handle work directly or ask focused roles to plan, implement, investigate, or review. The runner compacts older tool results near the context limit. The gateway can write rotation-managed logs.

## Install and Run

Install the current source from this repository. Kohaku requires Python 3.11 or newer.

```bash
git clone https://github.com/JadenChun/nanobot-custom.git kohaku
cd kohaku
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Kohaku and set up the first instance:

```bash
python -m pip install -e .
kohaku onboard --wizard
kohaku agent -m "Hello!"
```

Kohaku stores new default configuration and runtime data under `~/.kohaku/`. The default workspace is `~/.kohaku/workspace/`.

If you already use Nanobot, read the [Upgrade to Kohaku guide](./docs/upgrade-to-kohaku.md) before you start a Kohaku process.

The editable install skips the bundled WebUI build. Run these commands to build the WebUI assets and start the browser workbench:

```bash
cd webui
bun install
bun run build
cd ..
kohaku webui
```

For package builds, the [WebUI source guide](./webui/README.md) explains the automatic build hook.

## What Kohaku Provides

- A terminal interface and a browser workbench.
- Chat integrations for platforms such as Telegram, Discord, Slack, and WhatsApp.
- Agent tools for files, shell commands, web research, MCP servers, and scheduled work.
- Separate configuration and workspaces for multiple instances.

## Guides

| Goal | Guide |
|---|---|
| Upgrade an existing Nanobot profile | [Upgrade to Kohaku](./docs/upgrade-to-kohaku.md) |
| Understand workspaces, sessions, tools, and runtime behavior | [Concepts](./docs/concepts.md) |
| Configure providers and other settings | [Configuration](./docs/configuration.md) and [Provider Cookbook](./docs/provider-cookbook.md) |
| Use the browser workbench | [WebUI](./docs/webui.md) |
| Connect a chat platform | [Chat Apps](./docs/chat-apps.md) |
| Configure an MCP tool server | [MCP Tools](./docs/guides/configure-mcp-tools.md) |
| Manage web search and other tools | [Configuration: Web Tools](./docs/configuration.md#web-tools) |
| Run multiple instances | [Multiple Instances](./docs/multiple-instances.md) |
| Deploy a long-running gateway | [Deployment](./docs/deployment.md) |
| Find commands and options | [CLI Reference](./docs/cli-reference.md) |
| Use Kohaku from Python or an API client | [Python SDK](./docs/python-sdk.md) and [OpenAI-Compatible API](./docs/openai-api.md) |
| Understand the source architecture | [Architecture](./docs/architecture.md) |
| Diagnose a setup or runtime issue | [Troubleshooting](./docs/troubleshooting.md) |

## Architecture

Kohaku receives messages through the CLI, WebUI, or chat channels. The agent loop builds context, calls a model provider, runs available tools, and stores session data. Read the [architecture guide](./docs/architecture.md) for source ownership and subsystem details.
