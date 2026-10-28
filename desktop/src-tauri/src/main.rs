use std::{
    env,
    sync::{mpsc::Receiver, Arc},
    time::Duration,
};

use nanobot_desktop::{
    controller::{self, Shared, Snapshot},
    native::{self, HostStart},
    preferences::{self, Preferences},
};
use serde_json::Value;
use tauri::{
    AppHandle, Manager, State, Wry,
};
use tauri_plugin_dialog::DialogExt;

const INSTALL_MANIFEST: &str = include_str!("../resources/backend-compat.json");
const OPERATION_TIMEOUT: Duration = Duration::from_secs(40);

#[tauri::command]
fn desktop_state(shared: State<'_, Arc<Shared>>) -> Snapshot {
    shared.snapshot()
}

#[tauri::command(rename_all = "camelCase")]
async fn choose_path(app: AppHandle<Wry>, kind: String) -> Result<Option<String>, String> {
    let mut dialog = app.dialog().file().set_title(match kind.as_str() {
        "config" => "Select Nanobot config",
        "python" => "Select Python interpreter",
        "workspace" => "Select workspace",
        _ => return Err("selection_invalid".to_owned()),
    });
    let selected = match kind.as_str() {
        "config" => {
            dialog = dialog.add_filter("JSON configuration", &["json"]);
            dialog.blocking_pick_file()
        }
        "python" => dialog.blocking_pick_file(),
        "workspace" => dialog.blocking_pick_folder(),
        _ => return Err("selection_invalid".to_owned()),
    };
    selected
        .map(|path| {
            path.into_path()
                .map(|path| path.to_string_lossy().into_owned())
                .map_err(|_| "selection_invalid".to_owned())
        })
        .transpose()
}

#[tauri::command(rename_all = "camelCase")]
async fn select_backend(
    shared: State<'_, Arc<Shared>>,
    config_path: String,
    interpreter_path: String,
    workspace: Option<String>,
) -> Result<Snapshot, String> {
    let receiver = shared.enqueue_select(Preferences {
        config_path,
        interpreter_path,
        workspace,
    })?;
    await_operation(receiver, OPERATION_TIMEOUT).await
}

#[tauri::command]
async fn retry_backend(shared: State<'_, Arc<Shared>>) -> Result<Snapshot, String> {
    let receiver = shared.enqueue_retry()?;
    await_operation(receiver, OPERATION_TIMEOUT).await
}

#[tauri::command]
async fn open_chat(shared: State<'_, Arc<Shared>>) -> Result<Snapshot, String> {
    let receiver = shared.enqueue_open_chat()?;
    await_operation(receiver, Duration::from_secs(8)).await
}

#[tauri::command]
async fn change_selection(shared: State<'_, Arc<Shared>>) -> Result<Snapshot, String> {
    let receiver = shared.enqueue_change_selection()?;
    await_operation(receiver, Duration::from_secs(8)).await
}

#[tauri::command]
fn quit_desktop(shared: State<'_, Arc<Shared>>) -> Snapshot {
    shared.enqueue_quit()
}

async fn await_operation(
    receiver: Receiver<Result<Snapshot, String>>,
    timeout: Duration,
) -> Result<Snapshot, String> {
    tauri::async_runtime::spawn_blocking(move || {
        receiver.recv_timeout(timeout).map_err(|error| match error {
            std::sync::mpsc::RecvTimeoutError::Timeout => "adapter_timeout".to_owned(),
            std::sync::mpsc::RecvTimeoutError::Disconnected => "desktop_unavailable".to_owned(),
        })?
    })
    .await
    .map_err(|_| "desktop_unavailable".to_owned())?
}

fn select_initial(guard: Option<&native::HostGuard>) -> Result<Option<Preferences>, &'static str> {
    match guard {
        Some(guard) if guard.bus_verified => preferences::load(),
        Some(_) => Err("singleton_guard"),
        None => Err("host_lock_unavailable"),
    }
}

fn register_commands(app: tauri::Builder<Wry>) -> tauri::Builder<Wry> {
    app.invoke_handler(tauri::generate_handler![
        desktop_state,
        choose_path,
        select_backend,
        retry_backend,
        open_chat,
        change_selection,
        quit_desktop
    ])
}

fn run() {
    let app = register_commands(tauri::Builder::default())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            native::focus_existing_window(app);
        }))
        .plugin(
            tauri_plugin_opener::Builder::new()
                .open_js_links_on_click(false)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let guard = match native::acquire_host_guard(&app_handle) {
                Ok(HostStart::Primary(guard)) => Some(guard),
                Ok(HostStart::Secondary) => {
                    app.handle().exit(0);
                    return Ok(());
                }
                Err(_) => None,
            };
            let initial = select_initial(guard.as_ref());
            let shared = controller::initialize(app_handle.clone(), guard, initial);
            app.manage(shared.clone());
            if let Some(window) = app.get_webview_window("chat") {
                if let Ok(url) = window.url() {
                    shared.set_setup_url(url);
                }
                native::set_window_icon(&window)?;
                let shared_for_close = shared.clone();
                let window_for_close = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        native::close_setup(&shared_for_close, &window_for_close);
                    }
                });
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to initialize Nanobot Desktop");
    app.run(|_app, _event| {});
}

fn main() {
    let mut args = env::args_os();
    let _program = args.next();
    match args.next().as_deref() {
        Some(arg) if arg == "--version" => {
            println!("nanobot-desktop {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some(arg) if arg == "--check-install" => {
            if verify_install_manifest().is_ok() {
                println!("nanobot-desktop install metadata available");
            } else {
                eprintln!("nanobot-desktop install metadata invalid");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }
    if env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        // Fedora 44 KDE/Wayland's default WebKit DMA-BUF renderer produced a
        // black WebView in the release-mode qualification run.
        env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    run();
}

fn verify_install_manifest() -> Result<(), ()> {
    let manifest: Value = serde_json::from_str(INSTALL_MANIFEST).map_err(|_| ())?;
    let protocol = manifest
        .get("protocol_version")
        .and_then(Value::as_u64)
        .ok_or(())?;
    let backend_versions = manifest
        .get("backend_versions")
        .and_then(Value::as_array)
        .ok_or(())?;
    let required_capabilities = manifest
        .get("required_capabilities")
        .and_then(Value::as_array)
        .ok_or(())?;
    if protocol != 1
        || !backend_versions
            .iter()
            .any(|version| version.as_str() == Some("0.3.0"))
        || !required_capabilities
            .iter()
            .any(|capability| capability.as_str() == Some("managed-gateway-v1"))
    {
        return Err(());
    }
    Ok(())
}
