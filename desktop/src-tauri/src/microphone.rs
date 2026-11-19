use tauri::{webview::WebviewWindow, Manager, Wry};

use crate::native;

#[cfg(target_os = "linux")]
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
#[cfg(target_os = "linux")]
use gio::prelude::*;
#[cfg(target_os = "linux")]
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};
#[cfg(target_os = "linux")]
use url::Url;
#[cfg(target_os = "linux")]
use webkit2gtk::{
    PermissionRequestExt, SettingsExt, UserMediaPermissionRequest, UserMediaPermissionRequestExt,
    WebViewExt,
};

#[cfg(target_os = "linux")]
static PROMPT_PENDING: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "linux")]
fn choice_path() -> Option<PathBuf> {
    Some(
        crate::preferences::preference_path()
            .ok()?
            .with_file_name("microphone"),
    )
}

#[cfg(target_os = "linux")]
fn read_choice() -> Option<bool> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(choice_path()?)
        .ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut choice = Vec::new();
    file.take(2).read_to_end(&mut choice).ok()?;
    match choice.as_slice() {
        b"1" => Some(true),
        b"0" => Some(false),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn write_choice(allowed: bool) -> Option<()> {
    let path = choice_path()?;
    let temporary = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&temporary)
        .ok()?;
    let result = file
        .write_all(if allowed { b"1" } else { b"0" })
        .and_then(|()| file.sync_all())
        .and_then(|()| std::fs::rename(&temporary, &path));
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
        return None;
    }
    File::open(path.parent()?).ok()?.sync_all().ok()
}

#[cfg(target_os = "linux")]
fn prompt(window: &WebviewWindow<Wry>, on_choice: impl FnOnce(bool) + Send + 'static) {
    if PROMPT_PENDING.swap(true, Ordering::AcqRel) {
        on_choice(false);
        return;
    }
    window.dialog()
        .message("Allow Kohaku to use your microphone for voice input?")
        .title("Microphone access")
        .parent(window)
        .buttons(MessageDialogButtons::OkCancelCustom("Allow".into(), "Not now".into()))
        .show(move |allowed| {
            let _ = write_choice(allowed);
            PROMPT_PENDING.store(false, Ordering::Release);
            on_choice(allowed);
        });
}

#[cfg(target_os = "linux")]
pub fn request_on_first_chat(window: &WebviewWindow<Wry>) {
    if read_choice().is_none() {
        prompt(window, |_| {});
    }
}

#[cfg(target_os = "linux")]
pub fn install(window: &WebviewWindow<Wry>) -> tauri::Result<()> {
    let parent = window.clone();
    window.with_webview(move |platform| {
        let view = platform.inner();
        if let Some(settings) = WebViewExt::settings(&view) {
            settings.set_enable_media_stream(true);
        }
        view.connect_permission_request(move |view, request| {
            let Some(media) = request.downcast_ref::<UserMediaPermissionRequest>() else {
                return false;
            };
            let selected = view.uri().and_then(|uri| Url::parse(uri.as_str()).ok());
            let Some(selected) = selected
                .filter(|url| native::is_gateway_page(parent.app_handle(), url))
                .filter(|_| media.is_for_audio_device() && !media.is_for_video_device())
            else {
                request.deny();
                return true;
            };
            if read_choice() == Some(true) {
                request.allow();
                return true;
            }
            let (send, mut answer) = tauri::async_runtime::channel(1);
            prompt(&parent, move |allowed| {
                let _ = send.try_send(allowed);
            });
            let request = request.clone();
            let view = view.downgrade();
            let app = parent.app_handle().clone();
            gio::glib::MainContext::default().spawn_local(async move {
                let allowed = answer.recv().await == Some(true);
                let current = view
                    .upgrade()
                    .and_then(|view| view.uri())
                    .and_then(|uri| Url::parse(uri.as_str()).ok());
                if allowed
                    && current.is_some_and(|url| {
                        url.origin() == selected.origin() && native::is_gateway_page(&app, &url)
                    })
                {
                    request.allow();
                } else {
                    request.deny();
                }
            });
            true
        });
    })
}

#[cfg(target_os = "macos")]
use tauri::webview::{PermissionKind, PermissionResponse, Webview};

#[cfg(target_os = "macos")]
#[link(name = "AVFoundation", kind = "framework")]
extern "C" {
    static AVMediaTypeAudio: *const objc2::runtime::AnyObject;
}

#[cfg(target_os = "macos")]
const NOT_DETERMINED: isize = 0;
#[cfg(target_os = "macos")]
const AUTHORIZED: isize = 3;

#[cfg(target_os = "macos")]
fn macos_status() -> isize {
    // AVFoundation returns an AVAuthorizationStatus, an NSInteger.
    unsafe {
        objc2::msg_send![objc2::class!(AVCaptureDevice),
            authorizationStatusForMediaType: AVMediaTypeAudio]
    }
}

#[cfg(target_os = "macos")]
pub fn request_on_first_chat(_window: &WebviewWindow<Wry>) {
    if macos_status() == NOT_DETERMINED {
        let completed = block2::RcBlock::new(|_: objc2::runtime::Bool| {});
        // AVFoundation copies the completion block for asynchronous delivery.
        unsafe {
            let _: () = objc2::msg_send![objc2::class!(AVCaptureDevice),
                requestAccessForMediaType: AVMediaTypeAudio,
                completionHandler: &*completed];
        }
    }
}

#[cfg(target_os = "macos")]
pub fn macos_permission(view: Webview<Wry>, kind: PermissionKind) -> PermissionResponse {
    match kind {
        PermissionKind::Microphone if view.url().is_ok_and(|url| {
            native::is_gateway_page(view.app_handle(), &url)
        }) => match macos_status() {
            AUTHORIZED => PermissionResponse::Allow,
            NOT_DETERMINED => PermissionResponse::Default,
            _ => PermissionResponse::Deny,
        },
        PermissionKind::Microphone | PermissionKind::Camera | PermissionKind::DisplayCapture => {
            PermissionResponse::Deny
        }
        _ => PermissionResponse::Default,
    }
}
