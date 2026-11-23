# Install and Quick Start

Use this guide to install Kohaku from a clone of this repository and get one reply. Do not add other integrations until this path works.

If you are new to terminals or Python, use the [beginner walkthrough](./start-without-technical-background.md).

## What You Need

- Python 3.11 or newer.
- Git to clone this repository.
- Access to an AI provider, company endpoint, or local model server.
- The credential, endpoint URL, and model ID required by that service.

Kohaku stores new default config and workspace data under `~/.kohaku/`. A first install does not adopt an existing Nanobot profile. If you already use Nanobot, follow [Upgrade to Kohaku](./upgrade-to-kohaku.md).

## 1. Install from This Repository

Clone the repository and create a virtual environment:

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

Install Kohaku in editable mode:

```bash
python -m pip install -e .
```

Editable mode points Python to this checkout. For development dependencies, run `python -m pip install -e ".[dev]"` from the repository root.

## 2. Configure a Model

Run the setup wizard:

```bash
kohaku onboard --wizard
```

Select a provider and model. Enter the API key, endpoint, or local model details that the provider requires. Keep credentials private.

The default profile uses these paths:

| Path | Purpose |
|---|---|
| `~/.kohaku/config.json` | Provider, model, channel, tool, and runtime settings |
| `~/.kohaku/workspace/` | Memory, skills, automations, and generated files |
| `~/.kohaku/sessions/<workspace-id>/` | Session history stored outside the workspace |

Read [Configuration](./configuration.md) for advanced settings and exact environment variables.

## 3. Check the Setup

```bash
kohaku status
```

Check that the config, workspace, provider, and model match your choices. This command checks local setup but does not call the model.

## 4. Get the First Reply

Send one message:

```bash
kohaku -m "Hello!"
```

A normal reply proves that Kohaku can load the config, reach the model, and use the workspace.

## 5. Open the WebUI

An editable install skips the packaged WebUI build. Install Bun, then build the WebUI from the repository root:

```bash
cd webui
bun install
bun run build
cd ..
kohaku webui
```

The browser opens the local workbench. Read the [WebUI guide](./webui.md) for browser access, workspaces, Apps, and settings. Read the [WebUI source guide](../webui/README.md) to develop or package the frontend.

## Choose One Next Step

| Goal | Guide |
|---|---|
| Understand workspaces and sessions | [Concepts](./concepts.md) |
| Connect a chat platform | [Chat Apps](./chat-apps.md) |
| Choose a hosted, company, or local model | [Provider Cookbook](./provider-cookbook.md) |
| Configure web search or an MCP server | [Tool setup guides](./guides/README.md) |
| Schedule work | [Automations](./automations.md) |
| Run Kohaku continuously | [Deployment](./deployment.md) |
| Use Kohaku from Python | [Python SDK](./python-sdk.md) |
| Expose an OpenAI-compatible API | [API guide](./openai-api.md) |

## Update a Source Checkout

From the repository root, update the checkout and reinstall it in the same environment:

```bash
git pull --ff-only
python -m pip install -e .
```

If you are moving from Nanobot to Kohaku, follow [Upgrade to Kohaku](./upgrade-to-kohaku.md). Do not use this source update path to migrate an existing profile.
