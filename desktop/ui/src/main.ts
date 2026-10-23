import { invoke } from "@tauri-apps/api/core";

type DesktopState = {
  state: string;
  code: string;
  selected_config: string | null;
  selected_interpreter: string | null;
  selected_workspace: string | null;
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
  if (!element) throw new Error(`Missing setup element: ${id}`);
  return element as T;
};

const config = $("config") as HTMLInputElement;
const python = $("python") as HTMLInputElement;
const workspace = $("workspace") as HTMLInputElement;
const connect = $("connect") as HTMLButtonElement;
const retry = $("retry") as HTMLButtonElement;
const openChat = $("open-chat") as HTMLButtonElement;
const change = $("change") as HTMLButtonElement;
const quit = $("quit") as HTMLButtonElement;
const detail = $("detail");
const summary = $("summary");
const selected = $("selected");
const stateLabel = $("state");
const indicator = $("indicator");

const copy: Record<string, string> = {
  setup_required: "Choose an existing config file and compatible Python interpreter.",
  hello_required: "Validate the selected installation before starting it.",
  backend_version_unsupported: "This Python installation has an unsupported Nanobot version.",
  backend_capability_unsupported: "This Nanobot installation lacks the desktop lifecycle capability.",
  interpreter_mismatch: "The selected Python interpreter did not launch the installed backend.",
  config_invalid: "The selected config file could not be validated.",
  workspace_invalid: "The workspace override must be an absolute path.",
  loopback_required: "This config does not use a supported loopback gateway.",
  provider_unavailable: "The selected config has an incomplete provider setup.",
  runtime_preflight_refused: "The selected backend refused startup during its safety checks.",
  startup_admission_refused: "The selected backend refused desktop startup.",
  gateway_conflict: "An unmanaged or incompatible gateway is using this instance.",
  gateway_unavailable: "The managed gateway is unavailable. Retry to check it again.",
  gateway_starting: "The managed gateway is starting.",
  gateway_stopping: "Stopping the managed backend. This window stays open until its process exit is confirmed.",
  gateway_stopped: "The managed gateway has stopped.",
  gateway_ready: "The managed gateway is ready.",
  gateway_identity_mismatch: "The responding gateway did not match the selected instance.",
  webui_auth_failed: "The selected gateway rejected desktop authentication.",
  adapter_unavailable: "The selected Python process stopped responding. Retry to reconcile it.",
  stop_unconfirmed: "Shutdown is unconfirmed. The backend may still be running; retry to check.",
  singleton_guard: "The session bus could not verify this desktop process. The backend was not started.",
  host_lock_busy: "Another Nanobot Desktop window is already running.",
  tray_unavailable: "No system tray host is available. Keep this window open for controls.",
  host_lock_unavailable: "This desktop session cannot establish a private single-instance lock. The backend was not started.",
};

function message(code: string): string {
  return copy[code] ?? "Nanobot could not complete this operation. Check the selected installation before retrying.";
}

function draw(state: DesktopState, syncSelection = false): void {
  const names: Record<string, string> = {
    setup: "Setup required",
    starting: "Starting",
    running: "Running",
    unavailable: "Unavailable",
    stopping: "Stopping",
    stopped: "Stopped",
  };
  stateLabel.textContent = names[state.state] ?? "Unavailable";
  detail.textContent = state.state === "running" && !state.tray_available
    ? message("tray_unavailable")
    : state.admission_unknown
      ? "The start response was lost before this desktop could verify the gateway identity. The selection is locked; keep this window open while the original result is reconciled."
      : state.stop_unconfirmed
        ? message("stop_unconfirmed")
      : message(state.code);
  summary.textContent = state.state === "running"
    ? "Your selected Nanobot backend is ready. Chat opens in its own window."
    : "Connect this shell to an existing Nanobot Python installation.";
  indicator.className = "indicator";
  if (state.state === "running") indicator.classList.add("running");
  else if (state.state === "starting" || state.state === "stopping") indicator.classList.add("busy");
  else if (state.state === "unavailable" || state.state === "stopped") indicator.classList.add("error");
  const hasSelection = state.selected_config !== null && state.selected_interpreter !== null;
  if (syncSelection) {
    config.value = state.selected_config ?? "";
    python.value = state.selected_interpreter ?? "";
    workspace.value = state.selected_workspace ?? "";
  }
  selected.textContent = hasSelection
    ? `Config: ${state.selected_config}\nPython: ${state.selected_interpreter}${state.selected_workspace ? `\nWorkspace: ${state.selected_workspace}` : ""}`
    : "";
  retry.hidden = !state.retry_allowed;
  openChat.hidden = state.state !== "running";
  change.hidden = !hasSelection || state.selection_locked;
  $("selection").toggleAttribute("hidden", hasSelection && state.selection_locked);
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
    detail.textContent = "Desktop status is temporarily unavailable.";
  }
}

window.__NANOBOT_DESKTOP_STATE__ = draw;

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
  detail.textContent = "Validating the selected installation…";
  try {
    const state = await invoke<DesktopState>("select_backend", {
      configPath: config.value.trim(),
      interpreterPath: python.value.trim(),
      workspace: workspace.value.trim() || null,
    });
    draw(state, true);
  } catch (error) {
    const code = typeof error === "string" ? error : "config_invalid";
    detail.textContent = message(code);
  } finally {
    await refresh();
  }
});

retry.addEventListener("click", async () => {
  retry.disabled = true;
  try {
    draw(await invoke<DesktopState>("retry_backend"));
  } catch (error) {
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
    draw(await invoke<DesktopState>("change_selection"), true);
  } catch {
    detail.textContent = "Quit the managed backend before changing its selection.";
  }
});

quit.addEventListener("click", async () => {
  quit.disabled = true;
  detail.textContent = "Stopping the verified managed backend…";
  try {
    draw(await invoke<DesktopState>("quit_desktop"));
  } catch (error) {
    detail.textContent = message(typeof error === "string" ? error : "stop_unconfirmed");
  } finally {
    quit.disabled = false;
    await refresh();
  }
});

void refresh(true);
window.setInterval(() => void refresh(), 1000);
