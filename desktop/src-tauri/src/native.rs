use std::{
    env, fs,
    io::{Read, Write},
    os::fd::AsRawFd,
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};

use gio::{glib::translate::IntoGlib, prelude::*};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    webview::{NewWindowResponse, WebviewWindow},
    AppHandle, LogicalSize, Manager, Wry,
};
use tauri_plugin_opener::OpenerExt;
use url::Url;
use webkit2gtk::{
    NavigationPolicyDecisionExt, PolicyDecisionExt, PolicyDecisionType, URIRequestExt,
    WebViewExt,
};

use crate::controller::Shared;

const HIDE_SCRIPT: &str = "window.dispatchEvent(new Event('nanobot:desktop-hide'));";

pub struct HostGuard {
    _lock: fs::File,
    pub runtime_dir: PathBuf,
    socket_path: PathBuf,
    pub bus_verified: bool,
}

impl Drop for HostGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.socket_path);
    }
}

pub enum HostStart {
    Primary(HostGuard),
    Secondary,
}

pub fn acquire_host_guard(app: &AppHandle<Wry>) -> Result<HostStart, String> {
    let runtime_dir = runtime_directory()?;
    let lock_path = runtime_dir.join("host.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&lock_path)
        .map_err(|_| "host_lock_unavailable".to_owned())?;
    let metadata = file
        .metadata()
        .map_err(|_| "host_lock_unavailable".to_owned())?;
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("host_lock_unavailable".to_owned());
    }
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
            let _ = activate_existing_host(&runtime_dir.join("activate.sock"));
            return Ok(HostStart::Secondary);
        }
        return Err("host_lock_unavailable".to_owned());
    }

    let socket_path = runtime_dir.join("activate.sock");
    if socket_path.as_os_str().as_encoded_bytes().len() >= 100 {
        return Err("host_lock_unavailable".to_owned());
    }
    let _ = fs::remove_file(&socket_path);
    let bus_verified = owns_instance_name(app);
    Ok(HostStart::Primary(HostGuard {
        _lock: file,
        runtime_dir,
        socket_path,
        bus_verified,
    }))
}

pub fn initialize_primary(app: &AppHandle<Wry>, guard: &HostGuard, shared: Arc<Shared>) {
    if let Ok(listener) = UnixListener::bind(&guard.socket_path) {
        let _ = fs::set_permissions(&guard.socket_path, fs::Permissions::from_mode(0o600));
        let _ = listener.set_nonblocking(true);
        let app = app.clone();
        let shared_for_activation = shared.clone();
        thread::spawn(move || loop {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut command = [0u8; 16];
                    if let Ok(count) = stream.read(&mut command) {
                        if &command[..count] == b"activate\n" {
                            focus_control_window(&app, &shared_for_activation);
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(_) => thread::sleep(Duration::from_millis(250)),
            }
        });
    }
    if guard.bus_verified {
        if install_tray(app, shared.clone()).is_ok() {
            shared.set_tray_available(tray_host_available());
            start_tray_host_monitor(app.clone(), shared);
        } else {
            shared.set_tray_available(false);
        }
    } else {
        shared.set_error("singleton_guard");
    }
}

fn runtime_directory() -> Result<PathBuf, String> {
    let path = PathBuf::from(
        env::var_os("XDG_RUNTIME_DIR").ok_or_else(|| "host_lock_unavailable".to_owned())?,
    );
    if !path.is_absolute() {
        return Err("host_lock_unavailable".to_owned());
    }
    let metadata = fs::symlink_metadata(&path).map_err(|_| "host_lock_unavailable".to_owned())?;
    if !metadata.file_type().is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("host_lock_unavailable".to_owned());
    }
    let app_dir = path.join("nanobot-desktop");
    match fs::create_dir(&app_dir) {
        Ok(()) => {
            fs::set_permissions(&app_dir, fs::Permissions::from_mode(0o700))
                .map_err(|_| "host_lock_unavailable".to_owned())?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err("host_lock_unavailable".to_owned()),
    }
    let metadata = fs::symlink_metadata(&app_dir).map_err(|_| "host_lock_unavailable".to_owned())?;
    if !metadata.file_type().is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("host_lock_unavailable".to_owned());
    }
    Ok(app_dir)
}

fn activate_existing_host(path: &PathBuf) -> bool {
    let Ok(mut stream) = UnixStream::connect(path) else {
        return false;
    };
    stream.write_all(b"activate\n").is_ok()
}

fn owns_instance_name(app: &AppHandle<Wry>) -> bool {
    let name = format!("{}.SingleInstance", app.config().identifier);
    let Ok(bus) = gio::bus_get_sync(gio::BusType::Session, None::<&gio::Cancellable>) else {
        return false;
    };
    let Ok(owner_reply) = bus.call_sync(
        Some("org.freedesktop.DBus"),
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "GetNameOwner",
        Some(&(name.as_str(),).to_variant()),
        None,
        gio::DBusCallFlags::NONE,
        1000,
        None::<&gio::Cancellable>,
    ) else {
        return false;
    };
    let Some((owner,)) = owner_reply.get::<(String,)>() else {
        return false;
    };
    let Ok(pid_reply) = bus.call_sync(
        Some("org.freedesktop.DBus"),
        "/org/freedesktop/DBus",
        "org.freedesktop.DBus",
        "GetConnectionUnixProcessID",
        Some(&(owner.as_str(),).to_variant()),
        None,
        gio::DBusCallFlags::NONE,
        1000,
        None::<&gio::Cancellable>,
    ) else {
        return false;
    };
    pid_reply
        .get::<(u32,)>()
        .is_some_and(|(pid,)| pid == std::process::id())
}

fn tray_host_available() -> bool {
    let Ok(bus) = gio::bus_get_sync(gio::BusType::Session, None::<&gio::Cancellable>) else {
        return false;
    };
    let Ok(reply) = bus.call_sync(
        Some("org.kde.StatusNotifierWatcher"),
        "/StatusNotifierWatcher",
        "org.freedesktop.DBus.Properties",
        "Get",
        Some(
            &(
                "org.kde.StatusNotifierWatcher",
                "IsStatusNotifierHostRegistered",
            )
                .to_variant(),
        ),
        None,
        gio::DBusCallFlags::NONE,
        1000,
        None::<&gio::Cancellable>,
    ) else {
        return false;
    };
    reply
        .get::<(gio::glib::Variant,)>()
        .and_then(|(value,)| value.get::<bool>())
        .unwrap_or(false)
}

fn install_tray(app: &AppHandle<Wry>, shared: Arc<Shared>) -> Result<(), tauri::Error> {
    let status = MenuItem::with_id(app, "status", "Setup required", false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open Chat", true, None::<&str>)?;
    let retry = MenuItem::with_id(app, "retry", "Retry", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Nanobot", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&status, &open, &retry, &separator, &quit])?;
    shared.set_tray_status_item(status);
    let shared_for_menu = shared.clone();
    TrayIconBuilder::with_id("nanobot-desktop-tray")
        .icon(app_icon())
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => focus_control_window(app, &shared_for_menu),
            "retry" => shared_for_menu.enqueue_retry_from_tray(),
            "quit" => shared_for_menu.enqueue_quit_from_tray(),
            _ => {}
        })
        .build(app)?;
    Ok(())
}

fn start_tray_host_monitor(app: AppHandle<Wry>, shared: Arc<Shared>) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(500));
        let available = tray_host_available();
        let previous = shared.tray_available();
        if available == previous {
            continue;
        }
        shared.set_tray_available(available);
        if !available {
            let app_for_window = app.clone();
            let _ = app.run_on_main_thread(move || {
                if let Some(window) = app_for_window.get_webview_window("chat") {
                    let _ = window.unminimize();
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            });
        }
    });
}

pub fn focus_existing_window(app: &AppHandle<Wry>) {
    if let Some(shared) = app.try_state::<Arc<Shared>>() {
        focus_control_window(app, shared.inner());
    } else {
        open_setup(app);
    }
}

fn focus_control_window(app: &AppHandle<Wry>, shared: &Shared) {
    if shared.snapshot().state == "running" {
        if let Some(window) = app.get_webview_window("chat") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
            return;
        }
    }
    show_setup(app, shared);
}

pub fn open_setup(app: &AppHandle<Wry>) {
    if let Some(window) = app.get_webview_window("chat") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Show the connection/status page when no backend is running. The single window
/// is navigated back from the chat origin when necessary.
pub fn show_setup(app: &AppHandle<Wry>, shared: &Shared) {
    if let Some(window) = app.get_webview_window("chat") {
        if shared.snapshot().state != "running" {
            if let Some(setup_url) = shared.setup_url() {
                let on_setup = window
                    .url()
                    .ok()
                    .is_some_and(|current| current == setup_url);
                if !on_setup {
                    let _ = window.navigate(setup_url);
                    let _ = window.set_size(LogicalSize::new(650.0, 720.0));
                    let _ = window.set_title("Nanobot Desktop");
                }
            }
        }
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Return the single window to the setup UI after a selection change.
pub fn return_to_setup(app: &AppHandle<Wry>, shared: &Shared) {
    if let Some(window) = app.get_webview_window("chat") {
        if let Some(setup_url) = shared.setup_url() {
            let _ = window.navigate(setup_url);
        }
        let _ = window.set_size(LogicalSize::new(650.0, 720.0));
        let _ = window.set_title("Nanobot Desktop");
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

pub fn install_hide_on_close(
    app: &AppHandle<Wry>,
    shared: Arc<Shared>,
    window: &WebviewWindow<Wry>,
) {
    let app = app.clone();
    let in_progress = Arc::new(AtomicBool::new(false));
    window.on_window_event(move |event| {
        if let tauri::WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            if let Some(window) = app.get_webview_window("chat") {
                hide_chat(&app, &shared, &window, in_progress.clone());
            }
        }
    });
}

pub fn hide_chat(
    app: &AppHandle<Wry>,
    shared: &Shared,
    window: &WebviewWindow<Wry>,
    in_progress: Arc<AtomicBool>,
) {
    if !try_begin_hide(&in_progress) {
        return;
    }
    let app = app.clone();
    let window = window.clone();
    let tray_available = shared.tray_available();
    thread::spawn(move || {
        if !notify_before_window_action(&window, tray_available, false) && !tray_available {
            open_setup(&app);
        }
        in_progress.store(false, Ordering::Release);
    });
}

fn try_begin_hide(in_progress: &AtomicBool) -> bool {
    in_progress
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

/// Dispatches the fixed frontend cleanup event and waits for WebKit's completion callback
/// before hiding, minimizing, or destroying the retained webview.
pub fn destroy_chat_after_hide(
    _app: &AppHandle<Wry>,
    shared: &Shared,
    window: &WebviewWindow<Wry>,
) -> bool {
    notify_before_window_action(window, shared.tray_available(), true)
}

fn notify_before_window_action(
    window: &WebviewWindow<Wry>,
    tray_available: bool,
    destroy: bool,
) -> bool {
    // 0 = pending, 1 = the callback confirmed execution and performed the action,
    // 2 = the caller timed out, 3 = evaluation failed. A late completion after 2
    // must keep the window reachable rather than hiding it after the timeout.
    let disposition = Arc::new(AtomicU8::new(0));
    let callback_disposition = disposition.clone();
    let callback_window = window.clone();
    let timeout_window = window.clone();
    let (sender, receiver) = mpsc::sync_channel(1);
    if window
        .with_webview(move |platform| {
            platform.inner().evaluate_javascript(
                HIDE_SCRIPT,
                None,
                None,
                None::<&gio::Cancellable>,
                move |result| {
                    let action_window = callback_window.clone();
                    let restore_window = callback_window.clone();
                    let confirmed = finish_hide_callback(
                        &callback_disposition,
                        result.is_ok(),
                        move || {
                            if destroy {
                                let _ = action_window.destroy();
                            } else if tray_available {
                                let _ = action_window.hide();
                            } else {
                                let _ = action_window.minimize();
                            }
                        },
                        move || {
                            let _ = restore_window.unminimize();
                            let _ = restore_window.show();
                        },
                    );
                    let _ = sender.send(confirmed);
                },
            );
        })
        .is_err()
    {
        let _ = disposition.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire);
        let _ = timeout_window.unminimize();
        let _ = timeout_window.show();
        return false;
    }

    match receiver.recv_timeout(Duration::from_secs(3)) {
        Ok(confirmed) => confirmed,
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = disposition.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire);
            let _ = timeout_window.unminimize();
            let _ = timeout_window.show();
            false
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = disposition.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire);
            if disposition.load(Ordering::Acquire) == 1 {
                // The callback executed the event and completed the requested action
                // just before the channel receive timed out.
                true
            } else {
                let _ = timeout_window.unminimize();
                let _ = timeout_window.show();
                false
            }
        }
    }
}

fn finish_hide_callback(
    disposition: &AtomicU8,
    callback_succeeded: bool,
    hide: impl FnOnce(),
    restore: impl FnOnce(),
) -> bool {
    if callback_succeeded
        && disposition
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    {
        hide();
        true
    } else {
        let _ = disposition.compare_exchange(0, 3, Ordering::AcqRel, Ordering::Acquire);
        restore();
        false
    }
}

pub fn install_navigation_policy(
    window: &WebviewWindow<Wry>,
    origin: Url,
    setup_origin: Option<Url>,
    app: AppHandle<Wry>,
) -> Result<(), &'static str> {
    let installed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let installed_for_webview = installed.clone();
    window
        .with_webview(move |platform| {
            let view = platform.inner();
            let signal_id = unsafe {
                gio::glib::gobject_ffi::g_signal_lookup(
                    c"decide-policy".as_ptr(),
                    view.type_().into_glib(),
                )
            };
            let object = view.as_ptr().cast::<gio::glib::gobject_ffi::GObject>();
            let count = if signal_id == 0 {
                0
            } else {
                unsafe {
                    let count = gio::glib::gobject_ffi::g_signal_handlers_block_matched(
                        object,
                        gio::glib::gobject_ffi::G_SIGNAL_MATCH_ID,
                        signal_id,
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    );
                    gio::glib::gobject_ffi::g_signal_handlers_unblock_matched(
                        object,
                        gio::glib::gobject_ffi::G_SIGNAL_MATCH_ID,
                        signal_id,
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    );
                    count
                }
            };
            let disconnected = if count == 1 {
                unsafe {
                    gio::glib::gobject_ffi::g_signal_handlers_disconnect_matched(
                        object,
                        gio::glib::gobject_ffi::G_SIGNAL_MATCH_ID,
                        signal_id,
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    )
                }
            } else {
                0
            };
            if disconnected != 1 {
                return;
            }
            let expected = origin.clone();
            let setup_expected = setup_origin.clone();
            let opener_app = app.clone();
            view.connect_decide_policy(move |_view, decision, decision_type| {
                if decision_type != PolicyDecisionType::NavigationAction {
                    return false;
                }
                let Ok(navigation) = decision
                    .clone()
                    .downcast::<webkit2gtk::NavigationPolicyDecision>()
                else {
                    decision.ignore();
                    return true;
                };
                let Some(action) = navigation.navigation_action() else {
                    decision.ignore();
                    return true;
                };
                let Some(uri) = action.request().and_then(|request| request.uri()) else {
                    decision.ignore();
                    return true;
                };
                let Ok(target) = Url::parse(uri.as_str()) else {
                    decision.ignore();
                    return true;
                };
                if has_credentials(&target) {
                    decision.ignore();
                    return true;
                }
                if target.origin() == expected.origin()
                    || setup_expected
                        .as_ref()
                        .is_some_and(|setup| target.origin() == setup.origin())
                {
                    decision.use_();
                    return true;
                }
                let open_external = action.is_user_gesture() && is_safe_external(&target);
                decision.ignore();
                if open_external {
                    open_external_url(&opener_app, target);
                }
                true
            });
            let expected_for_popup = origin.clone();
            let popup_app = app.clone();
            view.connect_create(move |_view, action| {
                let uri = action.request().and_then(|request| request.uri())?;
                let Ok(target) = Url::parse(uri.as_str()) else {
                    return None;
                };
                if target.origin() != expected_for_popup.origin()
                    && !has_credentials(&target)
                    && action.is_user_gesture()
                    && is_safe_external(&target)
                {
                    open_external_url(&popup_app, target);
                }
                None
            });
            installed_for_webview.store(true, std::sync::atomic::Ordering::Release);
        })
        .map_err(|_| "navigation_policy_install_failed")?;
    if !installed.load(std::sync::atomic::Ordering::Acquire) {
        return Err("navigation_policy_install_failed");
    }
    Ok(())
}

pub fn navigation_allowed(target: &Url, selected: &Url, setup: Option<&Url>) -> bool {
    !has_credentials(target)
        && (target.origin() == selected.origin()
            || setup.is_some_and(|value| target.origin() == value.origin()))
}

fn has_credentials(url: &Url) -> bool {
    !url.username().is_empty() || url.password().is_some()
}

fn is_safe_external(url: &Url) -> bool {
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some()
        && !has_credentials(url)
}

fn open_external_url(app: &AppHandle<Wry>, url: Url) {
    let app = app.clone();
    let href = url.to_string();
    thread::spawn(move || {
        let _ = app.opener().open_url(href, None::<&str>);
    });
}

pub fn app_icon() -> tauri::image::Image<'static> {
    tauri::include_image!("icons/icon.png")
}

pub fn set_window_icon(window: &WebviewWindow<Wry>) -> tauri::Result<()> {
    // Use the packaged theme icon so GTK publishes the icon to the window manager.
    window.gtk_window()?.set_property("icon-name", "nanobot-desktop");
    Ok(())
}

pub fn new_window_response() -> NewWindowResponse<Wry> {
    NewWindowResponse::Deny
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn generated_setup_acl_allows_local_setup_commands_on_chat_only() {
        use std::collections::BTreeMap;
        use tauri::{
            ipc::Origin,
            utils::{
                acl::{capability::Capability, manifest::Manifest, resolved::Resolved},
                platform::Target,
            },
        };

        let capabilities = serde_json::from_str::<BTreeMap<String, Capability>>(include_str!(
            "../gen/schemas/capabilities.json"
        ))
        .expect("generated capabilities should parse");
        let manifests = serde_json::from_str::<BTreeMap<String, Manifest>>(include_str!(
            "../gen/schemas/acl-manifests.json"
        ))
        .expect("generated ACL manifests should parse");
        let resolved = Resolved::resolve(&manifests, capabilities, Target::current())
            .expect("generated ACL should resolve");
        let authority = tauri::runtime_authority!(manifests, resolved);
        let gateway = Origin::Remote {
            url: Url::parse("http://127.0.0.1:18790/").unwrap(),
        };

        for command in [
            "desktop_state",
            "choose_path",
            "select_backend",
            "retry_backend",
            "open_chat",
            "change_selection",
            "quit_desktop",
            "plugin:dialog|open",
        ] {
            assert!(
                authority
                    .resolve_access(command, "chat", "chat", &Origin::Local)
                    .is_some(),
                "the local chat window must be authorized to call {command} during setup"
            );
            assert!(
                authority
                    .resolve_access(command, "chat", "chat", &gateway)
                    .is_none(),
                "the remote gateway page must not be authorized to call {command}"
            );
        }
    }

    #[test]
    fn repeated_close_requests_share_one_hide_guard() {
        let in_progress = AtomicBool::new(false);

        assert!(try_begin_hide(&in_progress));
        assert!(!try_begin_hide(&in_progress));
        in_progress.store(false, Ordering::Release);
        assert!(try_begin_hide(&in_progress));
    }

    #[test]
    fn cleanup_callback_controls_hide_and_late_completion_restores_visibility() {
        use std::sync::atomic::AtomicUsize;

        let completed = AtomicU8::new(0);
        let hides = AtomicUsize::new(0);
        let restores = AtomicUsize::new(0);
        assert!(finish_hide_callback(
            &completed,
            true,
            || { hides.fetch_add(1, Ordering::Relaxed); },
            || { restores.fetch_add(1, Ordering::Relaxed); },
        ));
        assert_eq!(completed.load(Ordering::Acquire), 1);
        assert_eq!(hides.load(Ordering::Relaxed), 1);
        assert_eq!(restores.load(Ordering::Relaxed), 0);

        let failed = AtomicU8::new(0);
        assert!(!finish_hide_callback(
            &failed,
            false,
            || { hides.fetch_add(1, Ordering::Relaxed); },
            || { restores.fetch_add(1, Ordering::Relaxed); },
        ));
        assert_eq!(failed.load(Ordering::Acquire), 3);
        assert_eq!(hides.load(Ordering::Relaxed), 1);
        assert_eq!(restores.load(Ordering::Relaxed), 1);

        let timed_out = AtomicU8::new(2);
        assert!(!finish_hide_callback(
            &timed_out,
            true,
            || { hides.fetch_add(1, Ordering::Relaxed); },
            || { restores.fetch_add(1, Ordering::Relaxed); },
        ));
        assert_eq!(timed_out.load(Ordering::Acquire), 2);
        assert_eq!(hides.load(Ordering::Relaxed), 1);
        assert_eq!(restores.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn navigation_rejects_credentials_even_for_selected_origin() {
        let selected = Url::parse("http://127.0.0.1:8765/").unwrap();
        let credentialed = Url::parse("http://user:pass@127.0.0.1:8765/chat").unwrap();
        assert!(!navigation_allowed(&credentialed, &selected, None));
        assert!(navigation_allowed(&Url::parse("http://127.0.0.1:8765/chat").unwrap(), &selected, None));
    }

    #[test]
    fn navigation_allows_the_setup_origin_only() {
        let selected = Url::parse("http://127.0.0.1:8765/").unwrap();
        let setup = Url::parse("http://tauri.localhost/").unwrap();
        assert!(navigation_allowed(
            &Url::parse("http://tauri.localhost/index.html").unwrap(),
            &selected,
            Some(&setup),
        ));
        assert!(!navigation_allowed(
            &Url::parse("https://example.invalid/").unwrap(),
            &selected,
            Some(&setup),
        ));
    }

    #[test]
    fn navigation_does_not_allow_foreign_or_non_http_origins() {
        let selected = Url::parse("http://127.0.0.1:8765/").unwrap();
        assert!(!navigation_allowed(&Url::parse("https://example.invalid/").unwrap(), &selected, None));
        assert!(!navigation_allowed(&Url::parse("file:///tmp/a").unwrap(), &selected, None));
        assert!(!navigation_allowed(&Url::parse("javascript:alert(1)").unwrap(), &selected, None));
    }

    #[test]
    fn host_guard_holds_its_kernel_lock_until_owner_drop() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("nanobot-host-guard-{}-{unique}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let lock_path = root.join("host.lock");
        let socket_path = root.join("activate.sock");
        let lock = fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&lock_path)
            .unwrap();
        assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        fs::File::create(&socket_path).unwrap();
        let guard = HostGuard {
            _lock: lock,
            runtime_dir: root.clone(),
            socket_path: socket_path.clone(),
            bus_verified: false,
        };
        let observer = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();

        assert_eq!(
            unsafe { libc::flock(observer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            -1
        );
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::EWOULDBLOCK));

        drop(guard);

        assert!(!socket_path.exists());
        assert_eq!(unsafe { libc::flock(observer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        drop(observer);
        fs::remove_dir_all(root).unwrap();
    }
}
