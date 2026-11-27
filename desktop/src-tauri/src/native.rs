use std::{
    fs,
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

#[cfg(any(target_os = "linux", test))]
use std::env;
#[cfg(target_os = "linux")]
use gio::{glib::translate::IntoGlib, prelude::*};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    webview::{NewWindowResponse, WebviewWindow},
    AppHandle, Manager, Wry,
};
use tauri_plugin_opener::OpenerExt;
use url::Url;
#[cfg(target_os = "linux")]
use webkit2gtk::{
    NavigationPolicyDecisionExt, PolicyDecisionExt, PolicyDecisionType, URIRequestExt,
    WebViewExt,
};

use crate::controller::Shared;

const HIDE_SCRIPT: &str = "window.dispatchEvent(new Event('nanobot:desktop-hide'));";
const SHOW_SCRIPT: &str = "window.dispatchEvent(new Event('nanobot:desktop-show'));";

pub struct HostGuard {
    _lock: fs::File,
    pub runtime_dir: PathBuf,
    socket_path: PathBuf,
    pub instance_verified: bool,
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

pub fn acquire_host_guard(_app: &AppHandle<Wry>) -> Result<HostStart, String> {
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
    #[cfg(target_os = "linux")]
    let instance_verified = owns_instance_name(_app);
    // On macOS Launch Services reopens the running bundle. The per-user
    // kernel lock also excludes additional hosts launched as executables.
    #[cfg(target_os = "macos")]
    let instance_verified = true;
    Ok(HostStart::Primary(HostGuard {
        _lock: file,
        runtime_dir,
        socket_path,
        instance_verified,
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
    if guard.instance_verified {
        if install_tray(app, shared.clone()).is_ok() {
            #[cfg(target_os = "linux")]
            {
                shared.set_tray_available(tray_host_available());
                start_tray_host_monitor(shared);
            }
            #[cfg(target_os = "macos")]
            shared.set_tray_available(true);
        } else {
            shared.set_tray_available(false);
        }
    } else {
        shared.set_error("singleton_guard");
    }
}

fn runtime_directory() -> Result<PathBuf, String> {
    #[cfg(target_os = "linux")]
    let path = PathBuf::from(
        env::var_os("XDG_RUNTIME_DIR").ok_or_else(|| "host_lock_unavailable".to_owned())?,
    );
    #[cfg(target_os = "macos")]
    let path = macos_runtime_base()?;
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
    let app_dir = path.join("kohaku-desktop");
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

#[cfg(target_os = "macos")]
fn macos_runtime_base() -> Result<PathBuf, String> {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

    // Query Darwin directly: a shell may set TMPDIR differently from Finder.
    let mut path = [0u8; 1024];
    let len = unsafe {
        libc::confstr(
            libc::_CS_DARWIN_USER_TEMP_DIR,
            path.as_mut_ptr().cast(),
            path.len(),
        )
    };
    if len <= 1 || len > path.len() {
        return Err("host_lock_unavailable".to_owned());
    }
    Ok(PathBuf::from(OsStr::from_bytes(&path[..len - 1])))
}

fn activate_existing_host(path: &PathBuf) -> bool {
    let Ok(mut stream) = UnixStream::connect(path) else {
        return false;
    };
    stream.write_all(b"activate\n").is_ok()
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
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

pub(crate) const GATEWAY_TRAY_ID: &str = "kohaku-desktop-tray";

fn install_tray(app: &AppHandle<Wry>, shared: Arc<Shared>) -> Result<(), tauri::Error> {
    let status = MenuItem::with_id(app, "status", "Setup required", false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open Chat", true, None::<&str>)?;
    let retry = MenuItem::with_id(app, "retry", "Retry", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Kohaku", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&status, &open, &retry, &separator, &quit])?;
    let shared_for_menu = shared.clone();
    let tray = TrayIconBuilder::with_id(GATEWAY_TRAY_ID)
        .icon(app_icon())
        .tooltip("Kohaku gateway")
        .menu(&menu)
        .show_menu_on_left_click(cfg!(target_os = "macos"))
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => focus_control_window(app, &shared_for_menu),
            "retry" => shared_for_menu.enqueue_retry_from_tray(),
            "quit" => shared_for_menu.enqueue_quit_from_tray(),
            _ => {}
        })
        .build(app)?;
    #[cfg(target_os = "macos")]
    tray.set_visible(false)?;
    #[cfg(not(target_os = "macos"))]
    let _ = tray;
    shared.set_tray_status_item(status);
    Ok(())
}

#[cfg(target_os = "linux")]
fn start_tray_host_monitor(shared: Arc<Shared>) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(500));
        let available = tray_host_available();
        if available == shared.tray_available() {
            continue;
        }
        // A disappearing tray host must not pull a closed window back onto the
        // desktop. The tray label still reflects the state, and relaunching the
        // app activates the existing hidden window if the user wants it back.
        shared.set_tray_available(available);
    });
}

/// Bring the retained window back without changing its page, and tell the
/// WebUI that notifications are deliverable again.
pub fn reveal_window(shared: &Shared, window: &WebviewWindow<Wry>) {
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_focus();
    let _ = window.eval(SHOW_SCRIPT);
    shared.set_window_open(true);
}

pub fn focus_existing_window(app: &AppHandle<Wry>) {
    if let Some(shared) = app.try_state::<Arc<Shared>>() {
        focus_control_window(app, shared.inner());
    } else if let Some(window) = app.get_webview_window("chat") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn focus_control_window(app: &AppHandle<Wry>, shared: &Shared) {
    if shared.snapshot().state == "running" {
        if let Some(window) = app.get_webview_window("chat") {
            reveal_window(shared, &window);
            return;
        }
    }
    show_setup(app, shared);
}

pub fn open_setup(app: &AppHandle<Wry>, shared: &Shared) {
    if let Some(window) = app.get_webview_window("chat") {
        reveal_window(shared, &window);
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
                    #[cfg(target_os = "macos")]
                    clear_gateway_origin(app);
                    let _ = window.navigate(setup_url);
                    let _ = window.set_title("Kohaku");
                }
            }
        }
        reveal_window(shared, &window);
    }
}

/// Return the single window to the setup UI after a selection change.
pub fn return_to_setup(app: &AppHandle<Wry>, shared: &Shared) {
    clear_gateway_origin(app);
    if let Some(window) = app.get_webview_window("chat") {
        if let Some(setup_url) = shared.setup_url() {
            let _ = window.navigate(setup_url);
        }
        let _ = window.set_title("Kohaku");
        reveal_window(shared, &window);
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
                hide_chat(&app, shared.clone(), &window, in_progress.clone());
            }
        }
    });
}

pub fn hide_chat(
    app: &AppHandle<Wry>,
    shared: Arc<Shared>,
    window: &WebviewWindow<Wry>,
    in_progress: Arc<AtomicBool>,
) {
    if !try_begin_hide(&in_progress) {
        return;
    }
    let app = app.clone();
    let window = window.clone();
    let tray_available = shared.tray_available();
    // Close the notification window immediately so a status change racing the
    // hide callback cannot reveal the window the user just dismissed.
    shared.set_window_open(false);
    thread::spawn(move || {
        let hidden = notify_before_window_action(&window, tray_available, false);
        if !hidden {
            shared.set_window_open(true);
            if !tray_available {
                open_setup(&app, &shared);
            }
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
    let evaluate = evaluate_hide_script(window, move |succeeded| {
        let action_window = callback_window.clone();
        let restore_window = callback_window.clone();
        let confirmed = finish_hide_callback(
            &callback_disposition,
            succeeded,
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
    });
    if evaluate.is_err() {
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

#[cfg(target_os = "linux")]
fn evaluate_hide_script(
    window: &WebviewWindow<Wry>,
    finished: impl Fn(bool) + Send + 'static,
) -> tauri::Result<()> {
    window.with_webview(move |platform| {
        platform.inner().evaluate_javascript(
            HIDE_SCRIPT,
            None,
            None,
            None::<&gio::Cancellable>,
            move |result| finished(result.is_ok()),
        );
    })
}

#[cfg(target_os = "macos")]
fn evaluate_hide_script(
    window: &WebviewWindow<Wry>,
    finished: impl Fn(bool) + Send + 'static,
) -> tauri::Result<()> {
    // Wry supplies a JSON result, or an empty string on evaluation failure.
    window.eval_with_callback(format!("{HIDE_SCRIPT} true;"), move |value| {
        finished(value == "true");
    })
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

#[cfg(target_os = "linux")]
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
    #[cfg(target_os = "linux")]
    // Use the packaged theme icon so GTK publishes the icon to the window manager.
    window.gtk_window()?.set_property("icon-name", "kohaku-desktop");
    #[cfg(target_os = "macos")]
    let _ = window; // The bundle supplies the Dock and app icon.
    Ok(())
}

pub fn new_window_response() -> NewWindowResponse<Wry> {
    NewWindowResponse::Deny
}

type GatewayNavigation = Arc<std::sync::RwLock<Option<Url>>>;

pub fn initialize_gateway_navigation(app: &AppHandle<Wry>) {
    app.manage(GatewayNavigation::default());
}

pub fn is_gateway_page(app: &AppHandle<Wry>, target: &Url) -> bool {
    app.try_state::<GatewayNavigation>().is_some_and(|selected| {
        selected
            .read()
            .ok()
            .as_deref()
            .and_then(Option::as_ref)
            .is_some_and(|origin| navigation_allowed(target, origin, None))
    })
}

#[cfg(target_os = "macos")]
pub fn macos_menu(app: &AppHandle<Wry>) -> tauri::Result<Menu<Wry>> {
    use tauri::menu::SubmenuBuilder;

    // AppKit's built-in Quit exits immediately. Route the application menu
    // and Cmd+Q through the controller, like the menu bar icon's Quit.
    let quit = MenuItem::with_id(app, "managed-quit", "Quit Kohaku", true, Some("Cmd+Q"))?;
    let application = SubmenuBuilder::new(app, "Kohaku")
        .about(None)
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .item(&quit)
        .build()?;
    let edit = SubmenuBuilder::new(app, "Edit")
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;
    let window = SubmenuBuilder::new(app, "Window")
        .minimize()
        .maximize()
        .close_window()
        .build()?;
    Menu::with_items(app, &[&application, &edit, &window])
}

#[cfg(target_os = "macos")]
pub fn macos_setup_url() -> Url {
    Url::parse("tauri://localhost").expect("valid bundled setup URL")
}

#[cfg(target_os = "macos")]
pub fn create_setup_window(app: &AppHandle<Wry>) -> tauri::Result<()> {
    let selected = app.state::<GatewayNavigation>().inner().clone();
    let setup = macos_setup_url();
    let config = &app.config().app.windows[0];
    let opener_app = app.clone();
    // Returns whether the webview may navigate. Safe off-site links go to the
    // system browser instead.
    let route = move |target: &Url| {
        let action = classify_link(
            target,
            &setup,
            selected.read().ok().as_deref().and_then(Option::as_ref),
        );
        if action == LinkAction::OpenExternal {
            open_external_url(&opener_app, target.clone());
        }
        action == LinkAction::Allow
    };
    let popup_route = route.clone();
    tauri::WebviewWindowBuilder::from_config(app, config)?
        // Wry exposes no user gesture or target frame here, so scripted and
        // subframe navigations to safe off-site links open externally too.
        .on_navigation(move |target| route(target))
        .on_new_window(move |target, _| {
            popup_route(&target);
            new_window_response()
        })
        .build()?;
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
#[derive(Debug, PartialEq, Eq)]
enum LinkAction {
    Allow,
    OpenExternal,
    Deny,
}

#[cfg(any(target_os = "macos", test))]
fn classify_link(target: &Url, setup: &Url, selected: Option<&Url>) -> LinkAction {
    if macos_navigation_allowed(target, setup, selected) {
        LinkAction::Allow
    } else if is_safe_external(target)
        // The chat URL's credential must stay inside the webview, even if the
        // gateway selection changed while that load was pending.
        && !target
            .fragment()
            .is_some_and(|fragment| fragment.contains("bootstrapSecret="))
    {
        LinkAction::OpenExternal
    } else {
        LinkAction::Deny
    }
}

#[cfg(any(target_os = "macos", test))]
fn macos_navigation_allowed(target: &Url, setup: &Url, selected: Option<&Url>) -> bool {
    // URL origins for the custom tauri: scheme are opaque; match the exact
    // bundled setup URL, then use normal origin checks for the HTTP gateway.
    target == setup
        || target.as_str() == "tauri://localhost/"
        || selected.is_some_and(|origin| navigation_allowed(target, origin, None))
}

pub fn allow_gateway_origin(app: &AppHandle<Wry>, origin: Url) -> Result<(), &'static str> {
    let selected = app
        .try_state::<GatewayNavigation>()
        .ok_or("navigation_policy_install_failed")?;
    *selected.write().map_err(|_| "navigation_policy_install_failed")? = Some(origin);
    Ok(())
}

pub fn clear_gateway_origin(app: &AppHandle<Wry>) {
    if let Some(selected) = app.try_state::<GatewayNavigation>() {
        if let Ok(mut selected) = selected.write() {
            *selected = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_policy_allows_only_setup_and_the_current_gateway() {
        let setup = macos_setup_url();
        let old = Url::parse("http://127.0.0.1:18880/").unwrap();
        let current = Url::parse("http://127.0.0.1:18881/").unwrap();

        assert!(macos_navigation_allowed(&setup, &setup, None));
        assert!(!macos_navigation_allowed(&current, &setup, None));
        assert!(macos_navigation_allowed(&current, &setup, Some(&current)));
        assert!(!macos_navigation_allowed(&old, &setup, Some(&current)));
        let other_page = setup.join("other.html").unwrap();
        assert!(!macos_navigation_allowed(&other_page, &setup, Some(&current)));
    }

    #[test]
    fn link_classification_separates_in_app_external_and_denied() {
        let setup = Url::parse("tauri://localhost").unwrap();
        let gateway = Url::parse("http://127.0.0.1:18881/").unwrap();
        let classify = |target: &str, selected: Option<&Url>| {
            classify_link(&Url::parse(target).unwrap(), &setup, selected)
        };

        for target in [
            "tauri://localhost",
            "tauri://localhost/",
            "http://127.0.0.1:18881/",
            "http://127.0.0.1:18881/chat?x=1#/?bootstrapSecret=s",
        ] {
            assert_eq!(classify(target, Some(&gateway)), LinkAction::Allow, "{target}");
        }
        for target in [
            "https://github.com/example/repo/pull/1",
            "http://example.invalid/path?q=1#section",
            "HTTPS://EXAMPLE.INVALID/",
            // Another port is another origin, including on loopback.
            "http://127.0.0.1:3000/",
        ] {
            assert_eq!(classify(target, Some(&gateway)), LinkAction::OpenExternal, "{target}");
            assert_eq!(classify(target, None), LinkAction::OpenExternal, "{target}");
        }
        for target in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,hello",
            "about:blank",
            "blob:https://example.invalid/id",
            "mailto:someone@example.invalid",
            "vscode://file/tmp/a",
            "tauri://localhost/other.html",
            "ftp://example.invalid/",
            "ws://example.invalid/",
            "https://user@example.invalid/",
            "http://user@127.0.0.1:18881/",
        ] {
            assert_eq!(classify(target, Some(&gateway)), LinkAction::Deny, "{target}");
        }
        // A chat URL that no longer matches the selection must not carry its
        // credential into the system browser.
        let chat = "http://127.0.0.1:18881/#/?bootstrapSecret=s";
        assert_eq!(classify(chat, None), LinkAction::Deny);
        let replaced = Url::parse("http://127.0.0.1:18882/").unwrap();
        assert_eq!(classify(chat, Some(&replaced)), LinkAction::Deny);
    }

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
            instance_verified: false,
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
