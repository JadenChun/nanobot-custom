import { invoke } from "@tauri-apps/api/core";

type DesktopState = {
  state: string;
  code: string;
  selected_config: string | null;
  selected_interpreter: string | null;
  selected_workspace: string | null;
  setup_defaults?: {
    config_path: string;
    interpreter_path: string | null;
  } | null;
  tray_available: boolean;
  selection_locked: boolean;
  admission_pending: boolean;
  admission_unknown: boolean;
  stop_unconfirmed: boolean;
  retry_allowed: boolean;
};

declare global {
  interface Window {
    __NANOBOT_DESKTOP_STATE__?: (state: DesktopState) => void;
  }
}

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing startup element: ${id}`);
  return element as T;
};

const splash = $("splash");
const splashStatus = $("splash-status");
const fallback = $("fallback");
const selection = $("selection");
const toggleAdvanced = $("toggle-advanced") as HTMLButtonElement;
const config = $("config") as HTMLInputElement;
const python = $("python") as HTMLInputElement;
const workspace = $("workspace") as HTMLInputElement;
const connect = $("connect") as HTMLButtonElement;
const retry = $("retry") as HTMLButtonElement;
const openChat = $("open-chat") as HTMLButtonElement;
const change = $("change") as HTMLButtonElement;
const quit = $("quit") as HTMLButtonElement;
const detail = $("detail");
const stateLabel = $("state");

let advancedExpanded = false;
let initialized = false;

const copy: Record<string, string> = {
  setup_required: "Could not detect an installed Kohaku Python environment automatically. Choose the paths under Connection settings to continue.",
  hello_required: "Validate the selected installation before starting it.",
  backend_version_unsupported: "This Python installation has an unsupported Nanobot version.",
  backend_capability_unsupported: "This Nanobot installation lacks the desktop lifecycle capability.",
  default_config_backend_unsupported: "Select a newer Nanobot Python installation to create a default config.",
  default_config_write_failed: "Could not create the default config. Check that ~/.nanobot is writable.",
  interpreter_mismatch: "The selected Python interpreter did not launch the installed backend.",
  interpreter_invalid: "Choose the executable bin/python from an installed Nanobot environment.",
  config_invalid: "The selected config file could not be validated.",
  workspace_invalid: "The workspace override must be an absolute path.",
  loopback_required: "Desktop requires loopback addresses (such as 127.0.0.1) for gateway.host and channels.websocket.host.",
  provider_unavailable: "Configure a model provider in your config, then retry.",
  runtime_preflight_refused: "The backend refused startup during its safety checks.",
  startup_admission_refused: "The backend refused desktop startup.",
  gateway_conflict: "Another unmanaged or incompatible gateway is using this instance.",
  gateway_unavailable: "The local server is unavailable. Retry to start it again.",
  gateway_starting: "Starting Kohaku…",
  gateway_stopping: "Stopping Kohaku…",
  gateway_stopped: "The local server has stopped.",
  gateway_ready: "Opening chat…",
  gateway_identity_mismatch: "The responding server did not match this Kohaku instance.",
  webui_auth_failed: "The local server rejected desktop authentication.",
  adapter_unavailable: "The backend process stopped responding. Retry to reconnect.",
  stop_unconfirmed: "Shutdown is still confirming. Retry to check status.",
  singleton_guard: "Could not verify this desktop session. The server was not started.",
  host_lock_busy: "Another Kohaku window is already running.",
  tray_unavailable: "Kohaku is running. Keep this window open or click Open chat.",
  host_lock_unavailable: "Could not establish a private single-instance lock. The server was not started.",
};

function message(code: string): string {
  return copy[code] ?? "Kohaku could not start cleanly. Retry or check Connection settings.";
}

function draw(state: DesktopState, syncSelection = false): void {
  const showSplash =
    state.state === "starting"
    || state.state === "stopping"
    || (state.state === "running" && state.tray_available);

  splash.hidden = !showSplash;
  fallback.hidden = showSplash;

  if (showSplash) {
    splashStatus.textContent =
      state.state === "stopping"
        ? "Stopping Kohaku…"
        : state.state === "running"
          ? "Opening chat…"
          : "Starting Kohaku…";
    return;
  }

  const names: Record<string, string> = {
    setup: "Setup needed",
    running: "Kohaku is running",
    unavailable: "Could not start Kohaku",
    stopped: "Kohaku stopped",
  };
  stateLabel.textContent = names[state.state] ?? "Could not start Kohaku";
  detail.textContent = state.state === "running" && !state.tray_available
    ? message("tray_unavailable")
    : state.admission_unknown
      ? "Reconciling the startup response. Keep this window open while Kohaku checks the running server."
      : state.stop_unconfirmed
        ? message("stop_unconfirmed")
        : message(state.code);

  const hasSelection = state.selected_config !== null && state.selected_interpreter !== null;
  if (syncSelection || !initialized) {
    initialized = true;
    config.defaultValue = state.setup_defaults?.config_path ?? "";
    python.defaultValue = state.setup_defaults?.interpreter_path ?? "";
    config.value = state.selected_config ?? config.defaultValue;
    python.value = state.selected_interpreter ?? python.defaultValue;
    workspace.value = state.selected_workspace ?? "";
    if (state.state === "setup" && !hasSelection) {
      advancedExpanded = true;
    }
  }

  retry.hidden = !state.retry_allowed;
  openChat.hidden = state.state !== "running";
  change.hidden = !hasSelection || state.selection_locked;
  toggleAdvanced.hidden = state.selection_locked && hasSelection;
  selection.hidden = !advancedExpanded || (hasSelection && state.selection_locked);

  config.disabled = state.selection_locked;
  python.disabled = state.selection_locked;
  workspace.disabled = state.selection_locked;
  connect.disabled = state.selection_locked;
  document.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((button) => {
    button.disabled = state.selection_locked;
  });
}

async function refresh(syncSelection = false): Promise<void> {
  try {
    draw(await invoke<DesktopState>("desktop_state"), syncSelection);
  } catch {
    splash.hidden = false;
    fallback.hidden = true;
    splashStatus.textContent = "Starting Kohaku…";
  }
}

window.__NANOBOT_DESKTOP_STATE__ = draw;

toggleAdvanced.addEventListener("click", () => {
  advancedExpanded = !advancedExpanded;
  selection.hidden = !advancedExpanded;
});

document.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((button) => {
  button.addEventListener("click", async () => {
    const kind = button.dataset.pick;
    if (!kind) return;
    try {
      const path = await invoke<string | null>("choose_path", { kind });
      if (path === null) return;
      if (kind === "config") config.value = path;
      else if (kind === "python") python.value = path;
      else workspace.value = path;
    } catch {
      detail.textContent = "The native file picker could not open.";
    }
  });
});

connect.addEventListener("click", async () => {
  connect.disabled = true;
  splash.hidden = false;
  fallback.hidden = true;
  splashStatus.textContent = "Starting Kohaku…";
  try {
    const state = await invoke<DesktopState>("select_backend", {
      configPath: config.value.trim() || config.defaultValue,
      interpreterPath: python.value.trim() || python.defaultValue,
      workspace: workspace.value.trim() || null,
    });
    draw(state, true);
  } catch (error) {
    const code = typeof error === "string" ? error : "config_invalid";
    splash.hidden = true;
    fallback.hidden = false;
    detail.textContent = message(code);
  } finally {
    await refresh();
  }
});

retry.addEventListener("click", async () => {
  retry.disabled = true;
  splash.hidden = false;
  fallback.hidden = true;
  splashStatus.textContent = "Starting Kohaku…";
  try {
    draw(await invoke<DesktopState>("retry_backend"));
  } catch (error) {
    splash.hidden = true;
    fallback.hidden = false;
    detail.textContent = message(typeof error === "string" ? error : "adapter_unavailable");
  } finally {
    retry.disabled = false;
    await refresh();
  }
});

openChat.addEventListener("click", async () => {
  await invoke("open_chat");
});

change.addEventListener("click", async () => {
  try {
    advancedExpanded = true;
    draw(await invoke<DesktopState>("change_selection"), true);
  } catch {
    detail.textContent = "Quit the running server before changing connection settings.";
  }
});

quit.addEventListener("click", async () => {
  quit.disabled = true;
  splash.hidden = false;
  fallback.hidden = true;
  splashStatus.textContent = "Stopping Kohaku…";
  try {
    draw(await invoke<DesktopState>("quit_desktop"));
  } catch (error) {
    splash.hidden = true;
    fallback.hidden = false;
    detail.textContent = message(typeof error === "string" ? error : "stop_unconfirmed");
  } finally {
    quit.disabled = false;
    await refresh();
  }
});

void refresh(true);
window.setInterval(() => void refresh(), 1000);
