# Start Without Technical Background

This walkthrough installs Kohaku from a clone of this repository. It helps you get one reply without editing a config file by hand.

If you already use Nanobot, read [Upgrade to Kohaku](./upgrade-to-kohaku.md) before you start. A fresh Kohaku install does not select the old profile automatically.

## What You Need

- A Windows, macOS, or Linux computer.
- Python 3.11 or newer and Git.
- An account or endpoint that can run an AI model.
- The API key, endpoint, and model name required by that service.

An API key is secret. Do not post it in an issue, screenshot, chat, or public config file.

## A Few Useful Words

| Word | Meaning |
|---|---|
| Terminal | A text window where you enter commands |
| Command | One instruction that you enter in the terminal |
| Provider | A service or local server that runs an AI model |
| Model ID | The model name that the provider expects |
| API key | A secret that lets software call a provider |
| Wizard | A setup menu that asks you questions |
| WebUI | The browser page where you use Kohaku |

## 1. Install Python and Git

Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/). Install Git from [git-scm.com](https://git-scm.com/downloads). On Windows, enable **Add python.exe to PATH** if the installer shows that option.

Open a terminal:

| System | How to open it |
|---|---|
| Windows | Press `Win`, type `PowerShell`, and open Windows PowerShell |
| macOS | Press `Command+Space`, type `Terminal`, and press Enter |
| Linux | Open your application menu and search for Terminal |

Check Python:

```bash
python --version
```

The result must show Python 3.11 or newer. On macOS or Linux, use `python3 --version` if `python` is not found. On Windows, use `py --version` if needed.

## 2. Prepare Your Model Details

Before setup, find these details in your provider's account or documentation:

1. The provider or endpoint name.
2. Its API key, if it requires one.
3. Its base URL, if it has one.
4. A model ID that your account can use.

Use a key, endpoint, and model from the same provider. Do not share your key with anyone.

## 3. Clone and Install Kohaku

Clone this repository and create a private Python environment:

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

Install Kohaku:

```bash
python -m pip install -e .
```

## 4. Configure Your Model

Run the Kohaku setup wizard:

```bash
kohaku onboard --wizard
```

Follow the prompts to choose your provider and model. Enter an API key or endpoint only when the provider requires it. Kohaku stores new default settings under `~/.kohaku/`.

Keep your key private. Do not include it in a screenshot or support request.

## 5. Get the First Reply

Check the local setup:

```bash
kohaku status
```

Ask Kohaku one question:

```bash
kohaku -m "Hello!"
```

A normal reply means the setup works. The exact words do not matter.

## 6. Open the Browser Workbench

The editable install does not include a WebUI build. Install [Bun](https://bun.sh/), then build and start the WebUI:

```bash
cd webui
bun install
bun run build
cd ..
kohaku webui
```

Open `http://127.0.0.1:8765` if your browser does not open. Leave the terminal open while you use the WebUI.

## Choose One Next Step

| Goal | Guide |
|---|---|
| Change the model or provider | [Provider Cookbook](./provider-cookbook.md) |
| Connect Telegram, Discord, Slack, or another chat app | [Chat Apps](./chat-apps.md) |
| Add a tool integration | Open **Apps** in the WebUI or follow the [MCP tools guide](./guides/configure-mcp-tools.md) |
| Schedule a task | [Automations](./automations.md) |
| Use a project workspace | [Concepts](./concepts.md) |
| Deploy Kohaku | [Deployment](./deployment.md) |

## If Something Fails

Run these commands one at a time:

```bash
kohaku --version
kohaku status
kohaku -m "Hello!"
```

| What you see | What to check |
|---|---|
| `kohaku: command not found` | Activate the `.venv` environment, then repeat the command. |
| `401` or `unauthorized` | Check that the API key belongs to the selected provider. |
| Model not found | Check the model ID in the provider's account or documentation. |
| Browser does not open | Open `http://127.0.0.1:8765` and keep the terminal running. |
| Browser opens but messages fail | Run `kohaku -m "Hello!"` to check the provider and model. |

For more help, follow the [Troubleshooting guide](./troubleshooting.md). Remove credentials and private account details before you share diagnostic output.
