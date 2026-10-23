use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender},
        Arc, Mutex, RwLock,
    },
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;
use tauri::{
    menu::MenuItem,
    webview::WebviewWindowBuilder,
    AppHandle, Manager, WebviewUrl, Wry,
};
use url::{Host, Url};

use crate::{
    native::{self, HostGuard},
    preferences::{self, Preferences},
    protocol::{
        BeginFailure, BootstrapResult, LifecycleResult, PendingResponse, PipeAdapter,
    },
};

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub state: String,
    pub code: String,
    pub selected_config: Option<String>,
    pub selected_interpreter: Option<String>,
    pub selected_workspace: Option<String>,
    pub effective_workspace: Option<String>,
    pub tray_available: bool,
    pub selection_locked: bool,
    pub admission_pending: bool,
    pub admission_unknown: bool,
    pub stop_unconfirmed: bool,
    pub retry_allowed: bool,
}

impl Snapshot {
    fn empty() -> Self {
        Self {
            state: "setup".to_owned(),
            code: "setup_required".to_owned(),
            selected_config: None,
            selected_interpreter: None,
            selected_workspace: None,
            effective_workspace: None,
            tray_available: false,
            selection_locked: false,
            admission_pending: false,
            admission_unknown: false,
            stop_unconfirmed: false,
            retry_allowed: false,
        }
    }
}

enum Job {
    Select(Preferences, SyncSender<Result<Snapshot, String>>),
    Retry(SyncSender<Result<Snapshot, String>>),
    OpenChat(SyncSender<Result<Snapshot, String>>),
    ChangeSelection(SyncSender<Result<Snapshot, String>>),
    Quit(Instant),
}

#[derive(Default)]
struct AdmissionGate(AtomicU8);

impl AdmissionGate {
    fn try_begin(&self) -> bool {
        self.0
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    fn latch_quit(&self) {
        self.0.store(2, Ordering::SeqCst);
    }

    fn release_if_pending(&self) {
        let _ = self
            .0
            .compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn state(&self) -> u8 {
        self.0.load(Ordering::SeqCst)
    }
}

pub struct Shared {
    snapshot: RwLock<Snapshot>,
    sender: Sender<Job>,
    busy: AtomicBool,
    quit_requested: AtomicBool,
    quit_queued: AtomicBool,
    lifecycle_gate: AdmissionGate,
    bus_verified: bool,
    _host_guard: Option<HostGuard>,
    tray_status_item: Mutex<Option<MenuItem<Wry>>>,
    app: AppHandle<Wry>,
}

impl Shared {
    fn new(app: AppHandle<Wry>, host_guard: Option<HostGuard>) -> (Arc<Self>, Receiver<Job>) {
        let (sender, receiver) = mpsc::channel();
        let bus_verified = host_guard.as_ref().is_some_and(|guard| guard.bus_verified);
        let shared = Arc::new(Self {
            snapshot: RwLock::new(Snapshot::empty()),
            sender,
            busy: AtomicBool::new(false),
            quit_requested: AtomicBool::new(false),
            quit_queued: AtomicBool::new(false),
            lifecycle_gate: AdmissionGate::default(),
            bus_verified,
            _host_guard: host_guard,
            tray_status_item: Mutex::new(None),
            app,
        });
        (shared, receiver)
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
            .read()
            .map(|snapshot| snapshot.clone())
            .unwrap_or_else(|_| Snapshot {
                state: "unavailable".to_owned(),
                code: "desktop_state_unavailable".to_owned(),
                ..Snapshot::empty()
            })
    }

    pub fn set_tray_status_item(&self, item: MenuItem<Wry>) {
        if let Ok(mut slot) = self.tray_status_item.lock() {
            *slot = Some(item);
        }
        self.refresh_tray_label();
    }

    pub fn set_tray_available(&self, available: bool) {
        if let Ok(mut snapshot) = self.snapshot.write() {
            snapshot.tray_available = available;
        }
        self.refresh_tray_label();
    }

    pub fn tray_available(&self) -> bool {
        self.snapshot
            .read()
            .map(|snapshot| snapshot.tray_available)
            .unwrap_or(false)
    }

    pub fn set_error(&self, code: &str) {
        self.update_snapshot("unavailable", code);
    }

    pub fn enqueue_retry_from_tray(self: &Arc<Self>) {
        let _ = self.enqueue_simple(JobKind::Retry);
    }

    pub fn enqueue_quit_from_tray(self: &Arc<Self>) {
        self.enqueue_quit();
    }

    pub fn enqueue_select(
        self: &Arc<Self>,
        preferences: Preferences,
    ) -> Result<Receiver<Result<Snapshot, String>>, String> {
        if !self.bus_verified {
            return Err("singleton_guard".to_owned());
        }
        if self.quit_requested.load(Ordering::Acquire) || self.snapshot().selection_locked {
            return Err("operation_busy".to_owned());
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        self.reserve_job(Job::Select(preferences, reply))?;
        Ok(receiver)
    }

    pub fn enqueue_retry(
        self: &Arc<Self>,
    ) -> Result<Receiver<Result<Snapshot, String>>, String> {
        self.enqueue_simple(JobKind::Retry)
    }

    pub fn enqueue_open_chat(
        self: &Arc<Self>,
    ) -> Result<Receiver<Result<Snapshot, String>>, String> {
        self.enqueue_simple(JobKind::OpenChat)
    }

    pub fn enqueue_change_selection(
        self: &Arc<Self>,
    ) -> Result<Receiver<Result<Snapshot, String>>, String> {
        self.enqueue_simple(JobKind::ChangeSelection)
    }

    pub fn enqueue_quit(self: &Arc<Self>) -> Snapshot {
        self.lifecycle_gate.latch_quit();
        self.quit_requested.store(true, Ordering::Release);
        if !self.quit_queued.swap(true, Ordering::AcqRel) {
            let mut snapshot = self.snapshot();
            snapshot.state = "stopping".to_owned();
            snapshot.code = "gateway_stopping".to_owned();
            snapshot.selection_locked = true;
            snapshot.retry_allowed = false;
            self.replace_snapshot(snapshot);
            let deadline = Instant::now() + Duration::from_secs(25);
            let _ = self.sender.send(Job::Quit(deadline));
        }
        self.snapshot()
    }

    fn enqueue_simple(
        self: &Arc<Self>,
        kind: JobKind,
    ) -> Result<Receiver<Result<Snapshot, String>>, String> {
        if !self.bus_verified {
            return Err("singleton_guard".to_owned());
        }
        if self.quit_requested.load(Ordering::Acquire) {
            let snapshot = self.snapshot();
            if !matches!(kind, JobKind::Retry) || !snapshot.retry_allowed {
                return Err("operation_busy".to_owned());
            }
        } else if self.snapshot().selection_locked
            && matches!(kind, JobKind::ChangeSelection)
        {
            return Err("operation_busy".to_owned());
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        let job = match kind {
            JobKind::Retry => Job::Retry(reply),
            JobKind::OpenChat => Job::OpenChat(reply),
            JobKind::ChangeSelection => Job::ChangeSelection(reply),
        };
        self.reserve_job(job)?;
        Ok(receiver)
    }

    fn reserve_job(&self, job: Job) -> Result<(), String> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "operation_busy".to_owned())?;
        if self.sender.send(job).is_err() {
            self.busy.store(false, Ordering::Release);
            return Err("desktop_unavailable".to_owned());
        }
        Ok(())
    }

    fn latch_admission(&self) -> bool {
        self.lifecycle_gate.try_begin()
    }

    fn release_admission(&self) {
        self.lifecycle_gate.release_if_pending();
    }

    fn replace_snapshot(&self, snapshot: Snapshot) {
        if let Ok(mut current) = self.snapshot.write() {
            *current = snapshot;
        }
        self.refresh_tray_label();
    }

    fn update_snapshot(&self, state: &str, code: &str) {
        if let Ok(mut snapshot) = self.snapshot.write() {
            snapshot.state = state.to_owned();
            snapshot.code = code.to_owned();
        }
        self.refresh_tray_label();
    }

    fn refresh_tray_label(&self) {
        let Some(item) = self
            .tray_status_item
            .lock()
            .ok()
            .and_then(|item| item.clone())
        else {
            return;
        };
        let snapshot = self.snapshot();
        let label = match snapshot.state.as_str() {
            "setup" => "Setup required",
            "starting" => "Starting",
            "running" => "Running",
            "stopping" => "Stopping",
            "stopped" => "Stopped",
            _ => "Unavailable",
        }
        .to_owned();
        let _ = self.app.run_on_main_thread(move || {
            let _ = item.set_text(label);
        });
    }
}

#[derive(Clone, Copy)]
enum JobKind {
    Retry,
    OpenChat,
    ChangeSelection,
}

pub fn initialize(
    app: AppHandle<Wry>,
    guard: Option<HostGuard>,
    initial: Result<Option<Preferences>, &'static str>,
) -> Arc<Shared> {
    let (shared, receiver) = Shared::new(app.clone(), guard);
    if let Some(guard) = shared._host_guard.as_ref() {
        native::initialize_primary(&app, guard, shared.clone());
    }
    start_worker(shared.clone(), receiver, initial);
    shared
}

fn start_worker(
    shared: Arc<Shared>,
    receiver: Receiver<Job>,
    initial: Result<Option<Preferences>, &'static str>,
) {
    let app = shared.app.clone();
    thread::spawn(move || {
        let mut controller = Controller::new(app, shared.clone());
        match initial {
            Ok(Some(preferences)) if shared.bus_verified => {
                controller.start_selected(preferences, false);
            }
            Ok(Some(_)) => shared.set_error("singleton_guard"),
            Ok(None) => {}
            Err(code) => shared.set_error(code),
        }
        loop {
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(job) => {
                    if controller.handle_job(job) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            controller.advance();
            if controller.exit_requested {
                break;
            }
        }
    });
}

struct Controller {
    app: AppHandle<Wry>,
    shared: Arc<Shared>,
    preferences: Option<Preferences>,
    adapter: Option<PipeAdapter>,
    handle: Option<String>,
    origin: Option<Url>,
    effective_workspace: Option<String>,
    pending_start: Option<PendingResponse>,
    admission_pending: bool,
    admission_unknown: bool,
    readiness_deadline: Option<Instant>,
    last_status_poll: Instant,
    uncertain_stop: bool,
    pending_stop: Option<PendingResponse>,
    stop_handle: Option<String>,
    quit_deadline: Option<Instant>,
    quit_ui_expired: bool,
    exit_requested: bool,
    start_open_chat: bool,
    chat_origin: Option<Url>,
}

impl Controller {
    fn new(app: AppHandle<Wry>, shared: Arc<Shared>) -> Self {
        Self {
            app,
            shared,
            preferences: None,
            adapter: None,
            handle: None,
            origin: None,
            effective_workspace: None,
            pending_start: None,
            admission_pending: false,
            admission_unknown: false,
            readiness_deadline: None,
            last_status_poll: Instant::now(),
            uncertain_stop: false,
            pending_stop: None,
            stop_handle: None,
            quit_deadline: None,
            quit_ui_expired: false,
            exit_requested: false,
            start_open_chat: true,
            chat_origin: None,
        }
    }

    fn handle_job(&mut self, job: Job) -> bool {
        match job {
            Job::Select(preferences, reply) => {
                let result = if self.shared.quit_requested.load(Ordering::Acquire) {
                    Err("operation_busy".to_owned())
                } else {
                    self.select_and_start(preferences)
                };
                let snapshot = result.map(|_| self.shared.snapshot());
                let _ = reply.send(snapshot);
                self.shared.busy.store(false, Ordering::Release);
                false
            }
            Job::Retry(reply) => {
                let result = if self.shared.quit_requested.load(Ordering::Acquire) {
                    self.reconcile_quit()
                } else {
                    self.retry()
                };
                let snapshot = result.map(|_| self.shared.snapshot());
                let _ = reply.send(snapshot);
                self.shared.busy.store(false, Ordering::Release);
                false
            }
            Job::OpenChat(reply) => {
                let result = self.show_chat();
                let snapshot = result.map(|_| self.shared.snapshot());
                let _ = reply.send(snapshot);
                self.shared.busy.store(false, Ordering::Release);
                false
            }
            Job::ChangeSelection(reply) => {
                let result = self.change_selection();
                let snapshot = result.map(|_| self.shared.snapshot());
                let _ = reply.send(snapshot);
                self.shared.busy.store(false, Ordering::Release);
                false
            }
            Job::Quit(deadline) => {
                self.quit_deadline = Some(deadline);
                self.advance_quit();
                self.exit_requested
            }
        }
    }

    fn select_and_start(&mut self, preferences: Preferences) -> Result<(), String> {
        if self.is_selection_locked() {
            return Err("gateway_conflict".to_owned());
        }
        let preferences = normalize_selection(preferences)?;
        if self.preferences.as_ref().is_some_and(|old| old != &preferences) {
            if let Some(chat) = self.app.get_webview_window("chat") {
                if !native::destroy_chat_after_hide(&self.app, &self.shared, &chat) {
                    return Err("chat_cleanup_unconfirmed".to_owned());
                }
            }
            self.chat_origin = None;
        }
        if let Some(mut adapter) = self.adapter.take() {
            adapter.shutdown();
        }
        self.start_selected(preferences, true);
        if self.shared.snapshot().state == "starting" || self.shared.snapshot().state == "running" {
            Ok(())
        } else {
            Err(self.shared.snapshot().code)
        }
    }

    fn start_selected(&mut self, preferences: Preferences, persist: bool) {
        if !self.shared.bus_verified {
            self.shared.set_error("singleton_guard");
            return;
        }
        if self.shared.quit_requested.load(Ordering::Acquire)
            || self.pending_start.is_some()
            || self.admission_unknown
            || self.handle.is_some()
        {
            self.publish("unavailable", "operation_busy");
            return;
        }
        self.preferences = Some(preferences.clone());
        self.mark_selection(&preferences, "starting", "gateway_starting");
        if let Err(code) = self.ensure_new_adapter(&preferences) {
            self.admission_pending = false;
            self.publish("unavailable", &code);
            return;
        }
        if persist {
            if let Err(code) = preferences::store(&preferences) {
                self.adapter.take();
                self.admission_pending = false;
                self.publish("unavailable", code);
                return;
            }
        }
        self.preferences = Some(preferences);
        if !self.shared.latch_admission() {
            self.adapter.take();
            self.admission_pending = false;
            self.publish("stopping", "gateway_stopping");
            return;
        }
        self.admission_pending = true;
        self.admission_unknown = false;
        self.start_open_chat = true;
        self.publish("starting", "gateway_starting");
        let dispatch = self
            .adapter
            .as_mut()
            .expect("adapter initialized")
            .dispatch_start();
        match dispatch {
            Ok(pending) => {
                self.pending_start = Some(pending);
                self.readiness_deadline = Some(Instant::now() + Duration::from_secs(30));
            }
            Err(BeginFailure::BeforeDispatch(error)) => {
                self.admission_pending = false;
                self.shared.release_admission();
                self.publish("unavailable", &error.0);
            }
            Err(BeginFailure::DispatchUncertain(pending, _error)) => {
                self.pending_start = Some(pending);
                self.admission_unknown = true;
                self.admission_pending = false;
                self.publish("unavailable", "stop_unconfirmed");
            }
        }
    }

    fn ensure_new_adapter(&mut self, preferences: &Preferences) -> Result<(), String> {
        if !Path::new(&preferences.interpreter_path).is_file() {
            return Err("interpreter_invalid".to_owned());
        }
        let interpreter_metadata = fs::metadata(&preferences.interpreter_path)
            .map_err(|_| "interpreter_invalid".to_owned())?;
        if interpreter_metadata.permissions().mode() & 0o111 == 0 {
            return Err("interpreter_invalid".to_owned());
        }
        if !Path::new(&preferences.config_path).is_file() {
            return Err("config_invalid".to_owned());
        }
        let mut adapter = PipeAdapter::spawn(Path::new(&preferences.interpreter_path))
            .map_err(|error| error.0)?;
        let hello = adapter
            .hello(
                &preferences.config_path,
                preferences.workspace.as_deref(),
            )
            .map_err(|error| error.0)?;
        let expected_config = fs::canonicalize(&preferences.config_path)
            .map_err(|_| "config_invalid".to_owned())?;
        if Path::new(&hello.config_path) != expected_config {
            return Err("config_invalid".to_owned());
        }
        self.effective_workspace = Some(hello.workspace);
        self.adapter = Some(adapter);
        Ok(())
    }

    fn ensure_adapter(&mut self) -> Result<(), String> {
        if self.adapter.is_some() {
            return Ok(());
        }
        let preferences = self
            .preferences
            .clone()
            .ok_or_else(|| "setup_required".to_owned())?;
        self.ensure_new_adapter(&preferences)
    }

    fn retry(&mut self) -> Result<(), String> {
        if !self.shared.bus_verified {
            return Err("singleton_guard".to_owned());
        }
        if self.pending_start.is_some() || self.admission_unknown {
            return Err("stop_unconfirmed".to_owned());
        }
        let Some(preferences) = self.preferences.clone() else {
            return Err("setup_required".to_owned());
        };
        if self.uncertain_stop {
            let handle = self
                .handle
                .clone()
                .ok_or_else(|| "stop_unconfirmed".to_owned())?;
            self.ensure_adapter()?;
            let status = match self
                .adapter
                .as_mut()
                .expect("adapter initialized")
                .status(&handle)
            {
                Ok(status) => status,
                Err(error) => {
                    self.adapter.take();
                    return Err(error.0);
                }
            };
            if confirms_stopped(&status, &handle) {
                self.handle = None;
                self.uncertain_stop = false;
                self.shared.release_admission();
                self.publish("stopped", "gateway_stopped");
                return Ok(());
            }
            self.apply_lifecycle(status);
            if self.uncertain_stop {
                self.publish("unavailable", "stop_unconfirmed");
            }
            return Ok(());
        }
        if let Some(handle) = self.handle.clone() {
            self.ensure_adapter()?;
            let status = self
                .adapter
                .as_mut()
                .expect("adapter initialized")
                .status(&handle)
                .map_err(|error| error.0)?;
            if status.state == "running" {
                self.apply_lifecycle(status);
                if self.app.get_webview_window("chat").is_none() {
                    self.show_chat()?;
                }
                return Ok(());
            }
            if status.state != "stopped" {
                self.apply_lifecycle(status);
                return Ok(());
            }
            if status.handle.as_deref() != Some(handle.as_str()) {
                self.admission_unknown = true;
                self.publish("unavailable", "gateway_identity_mismatch");
                return Err("gateway_identity_mismatch".to_owned());
            }
            self.handle = None;
            self.admission_unknown = false;
            self.shared.release_admission();
        }
        self.start_selected(preferences, false);
        if self.shared.snapshot().state == "running" || self.shared.snapshot().state == "starting" {
            Ok(())
        } else {
            Err(self.shared.snapshot().code)
        }
    }

    fn reconcile_quit(&mut self) -> Result<(), String> {
        if self.pending_start.is_some() || self.pending_stop.is_some() || self.admission_unknown {
            return Err("stop_unconfirmed".to_owned());
        }
        if !self.uncertain_stop {
            return Err("operation_busy".to_owned());
        }
        let handle = self.handle.clone().ok_or_else(|| "stop_unconfirmed".to_owned())?;
        self.ensure_adapter()?;
        let result = match self
            .adapter
            .as_mut()
            .expect("adapter initialized")
            .status(&handle)
        {
            Ok(result) => result,
            Err(error) => {
                self.adapter.take();
                return Err(error.0);
            }
        };
        if confirms_stopped(&result, &handle) {
            self.handle = None;
            self.uncertain_stop = false;
            self.publish("stopped", "gateway_stopped");
            self.exit_after_quit();
            return Ok(());
        }
        self.apply_lifecycle(result);
        self.uncertain_stop = true;
        self.publish("unavailable", "stop_unconfirmed");
        Err("stop_unconfirmed".to_owned())
    }

    fn show_chat(&mut self) -> Result<(), String> {
        if self.shared.snapshot().state != "running" {
            return Err("gateway_unavailable".to_owned());
        }
        let handle = self.handle.as_deref().ok_or_else(|| "gateway_identity_unavailable".to_owned())?;
        let bootstrap = match self
            .adapter
            .as_mut()
            .ok_or_else(|| "adapter_unavailable".to_owned())?
            .bootstrap(handle)
        {
            Ok(bootstrap) => bootstrap,
            Err(error) => {
                self.adapter.take();
                self.publish("unavailable", &error.0);
                return Err(error.0);
            }
        };
        self.open_bootstrapped_chat(bootstrap)
    }

    fn open_bootstrapped_chat(&mut self, bootstrap: BootstrapResult) -> Result<(), String> {
        if self.shared.quit_requested.load(Ordering::Acquire) {
            return Err("operation_busy".to_owned());
        }
        let expected_origin = self
            .origin
            .clone()
            .ok_or_else(|| "gateway_origin_mismatch".to_owned())?;
        if bootstrap.secret.is_empty()
            || bootstrap.secret.len() > 4096
            || bootstrap.secret.chars().any(char::is_control)
        {
            return Err("gateway_origin_mismatch".to_owned());
        }
        let origin = parse_loopback_origin(&bootstrap.origin)?;
        if origin.origin() != expected_origin.origin() {
            return Err("gateway_origin_mismatch".to_owned());
        }
        let ws_url = Url::parse(&bootstrap.ws_url).map_err(|_| "gateway_origin_mismatch".to_owned())?;
        if ws_url.scheme() != "ws"
            || ws_url.host_str() != origin.host_str()
            || ws_url.port_or_known_default() != origin.port_or_known_default()
            || !ws_url.username().is_empty()
            || ws_url.password().is_some()
            || ws_url.query().is_some()
            || ws_url.fragment().is_some()
        {
            return Err("gateway_origin_mismatch".to_owned());
        }
        if let Some(existing) = self.app.get_webview_window("chat") {
            if self.chat_origin.as_ref().is_some_and(|old| old.origin() == origin.origin()) {
                native::focus_existing_window(&self.app);
                return Ok(());
            }
            if !native::destroy_chat_after_hide(&self.app, &self.shared, &existing) {
                return Err("chat_cleanup_unconfirmed".to_owned());
            }
        }
        let preferences = self
            .preferences
            .as_ref()
            .ok_or_else(|| "setup_required".to_owned())?;
        let data_directory = preferences::profile_directory(preferences).map_err(str::to_owned)?;
        let secret = url::form_urlencoded::byte_serialize(bootstrap.secret.as_bytes()).collect::<String>();
        let mut chat_url = origin.clone();
        chat_url.set_fragment(Some(&format!("/?bootstrapSecret={secret}")));
        let app = self.app.clone();
        let origin_for_policy = origin.clone();
        let target_origin = origin.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let app_for_create = self.app.clone();
        let shared_for_event = self.shared.clone();
        let _ = self.app.run_on_main_thread(move || {
            let builder = WebviewWindowBuilder::new(
                &app_for_create,
                "chat",
                WebviewUrl::External(chat_url),
            )
            .title("Nanobot")
            .inner_size(1120.0, 780.0)
            .data_directory(data_directory)
            .on_navigation({
                let selected = target_origin.clone();
                move |target| native::navigation_allowed(target, &selected)
            })
            .on_new_window(|_url, _features| native::new_window_response());
            let chat = match builder.build() {
                Ok(chat) => chat,
                Err(_) => {
                    let _ = sender.send(Err("chat_window_unavailable".to_owned()));
                    return;
                }
            };
            let _ = native::set_window_icon(&chat);
            if native::install_navigation_policy(&chat, origin_for_policy, app.clone()).is_err() {
                let _ = chat.destroy();
                let _ = sender.send(Err("navigation_policy_install_failed".to_owned()));
                return;
            }
            let app_for_close = app.clone();
            let shared_for_close = shared_for_event.clone();
            let close_hide_in_progress = Arc::new(AtomicBool::new(false));
            chat.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    if let Some(window) = app_for_close.get_webview_window("chat") {
                        native::hide_chat(
                            &app_for_close,
                            &shared_for_close,
                            &window,
                            close_hide_in_progress.clone(),
                        );
                    }
                }
            });
            let _ = sender.send(Ok(()));
        });
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| "chat_window_unavailable".to_owned())??;
        self.chat_origin = Some(origin);
        self.publish("running", "gateway_ready");
        if self.shared.tray_available() {
            if let Some(setup) = self.app.get_webview_window("setup") {
                let _ = setup.hide();
            }
        }
        Ok(())
    }

    fn apply_lifecycle(&mut self, result: LifecycleResult) {
        if !valid_code(&result.code) {
            self.publish("unavailable", "protocol_invalid");
            return;
        }
        let Some(state) = allowed_state(&result.state) else {
            self.publish("unavailable", "protocol_invalid");
            return;
        };
        if let Some(handle) = &result.handle {
            if handle.is_empty() || handle.len() > 2048 || handle.chars().any(char::is_control) {
                self.publish("unavailable", "protocol_invalid");
                return;
            }
            if self.handle.as_ref().is_some_and(|known| known != handle) {
                self.admission_unknown = true;
                self.publish("unavailable", "gateway_identity_mismatch");
                return;
            }
            self.handle = Some(handle.clone());
        } else if self.handle.is_some() || state != "stopped" {
            self.publish("unavailable", "gateway_identity_unavailable");
            return;
        }
        match state {
            "running" => {
                let Some(origin) = result.origin.as_deref() else {
                    self.publish("unavailable", "gateway_origin_mismatch");
                    return;
                };
                match parse_loopback_origin(origin) {
                    Ok(origin) => self.origin = Some(origin),
                    Err(code) => {
                        self.publish("unavailable", &code);
                        return;
                    }
                }
                self.admission_pending = false;
                self.admission_unknown = false;
                self.uncertain_stop = false;
                self.readiness_deadline = None;
                if self.shared.quit_requested.load(Ordering::Acquire) {
                    self.publish("stopping", "gateway_stopping");
                    return;
                }
                self.publish("running", &result.code);
                if self.start_open_chat && !self.shared.quit_requested.load(Ordering::Acquire) {
                    if self.show_chat().is_ok() {
                        self.start_open_chat = false;
                    } else {
                        self.publish("unavailable", "gateway_bootstrap_failed");
                    }
                }
            }
            "stopped" => {
                if self.handle.is_none() {
                    self.admission_unknown = true;
                    self.publish("unavailable", "gateway_identity_unavailable");
                    return;
                }
                self.handle = None;
                self.origin = None;
                self.admission_pending = false;
                self.admission_unknown = false;
                self.uncertain_stop = false;
                self.readiness_deadline = None;
                self.shared.release_admission();
                self.publish("stopped", &result.code);
            }
            other => {
                if state == "starting" {
                    self.admission_pending = false;
                    self.admission_unknown = false;
                    if self.readiness_deadline.is_none() {
                        self.readiness_deadline = Some(Instant::now() + Duration::from_secs(30));
                    }
                }
                if self.shared.quit_requested.load(Ordering::Acquire) {
                    self.publish("stopping", "gateway_stopping");
                } else {
                    self.publish(other, &result.code);
                }
            }
        }
        if let Ok(mut snapshot) = self.shared.snapshot.write() {
            snapshot.effective_workspace = self.effective_workspace.clone();
        }
    }

    fn mark_selection(&self, preferences: &Preferences, state: &str, code: &str) {
        let mut snapshot = self.shared.snapshot();
        snapshot.state = state.to_owned();
        snapshot.code = code.to_owned();
        snapshot.selected_config = Some(preferences.config_path.clone());
        snapshot.selected_interpreter = Some(preferences.interpreter_path.clone());
        snapshot.selected_workspace = preferences.workspace.clone();
        snapshot.effective_workspace = self.effective_workspace.clone();
        snapshot.selection_locked = true;
        snapshot.admission_pending = true;
        snapshot.admission_unknown = false;
        snapshot.stop_unconfirmed = false;
        snapshot.retry_allowed = false;
        self.shared.replace_snapshot(snapshot);
    }

    fn change_selection(&mut self) -> Result<(), String> {
        if self.is_selection_locked() {
            return Err("gateway_conflict".to_owned());
        }
        if let Some(chat) = self.app.get_webview_window("chat") {
            if !native::destroy_chat_after_hide(&self.app, &self.shared, &chat) {
                return Err("chat_cleanup_unconfirmed".to_owned());
            }
        }
        preferences::remove().map_err(str::to_owned)?;
        if let Some(mut adapter) = self.adapter.take() {
            adapter.shutdown();
        }
        self.preferences = None;
        self.chat_origin = None;
        self.effective_workspace = None;
        self.shared.release_admission();
        let mut snapshot = Snapshot::empty();
        snapshot.tray_available = self.shared.tray_available();
        self.shared.replace_snapshot(snapshot);
        native::open_setup(&self.app);
        Ok(())
    }

    fn advance(&mut self) {
        self.advance_start_response();
        self.advance_stop_response();

        let now = Instant::now();
        if self.pending_start.is_none()
            && self.pending_stop.is_none()
            && !self.shared.quit_requested.load(Ordering::Acquire)
            && !self.uncertain_stop
            && self.handle.is_some()
            && now.duration_since(self.last_status_poll) >= Duration::from_secs(1)
        {
            self.poll_known_status();
        }

        self.advance_quit();
        if self.quit_deadline.is_some_and(|deadline| now >= deadline) && !self.quit_ui_expired {
            self.quit_ui_expired = true;
            if self.shared.quit_requested.load(Ordering::Acquire) && !self.exit_requested {
                self.publish("unavailable", "stop_unconfirmed");
                native::open_setup(&self.app);
            }
        }
        if self.readiness_deadline.is_some_and(|deadline| now >= deadline)
            && self.handle.is_some()
            && !self.shared.quit_requested.load(Ordering::Acquire)
        {
            self.readiness_deadline = None;
            self.publish("unavailable", "gateway_unavailable");
        }
    }

    fn advance_start_response(&mut self) {
        let Some(pending) = self.pending_start else {
            return;
        };
        if self.admission_pending && Instant::now() >= pending.deadline {
            self.admission_pending = false;
            self.admission_unknown = true;
            self.publish("unavailable", "stop_unconfirmed");
        }
        let response = match self.adapter.as_mut() {
            Some(adapter) => adapter.poll_lifecycle(pending),
            None => Err(crate::protocol::AdapterFailure("adapter_unavailable".to_owned())),
        };
        match response {
            Ok(None) => {}
            Ok(Some(result)) => {
                self.pending_start = None;
                self.admission_pending = false;
                if captured_start_handle(&result).is_none() {
                    self.admission_unknown = true;
                    self.publish("unavailable", "stop_unconfirmed");
                    return;
                }
                self.admission_unknown = false;
                self.apply_lifecycle(result);
                if self.shared.quit_requested.load(Ordering::Acquire) {
                    self.start_open_chat = false;
                    self.advance_quit();
                }
            }
            Err(_error) => {
                self.pending_start = None;
                self.admission_pending = false;
                self.admission_unknown = true;
                self.publish("unavailable", "stop_unconfirmed");
            }
        }
    }

    fn advance_stop_response(&mut self) {
        let Some(pending) = self.pending_stop else {
            return;
        };
        let response = match self.adapter.as_mut() {
            Some(adapter) => adapter.poll_lifecycle(pending),
            None => Err(crate::protocol::AdapterFailure("adapter_unavailable".to_owned())),
        };
        match response {
            Ok(None) => {
                if Instant::now() >= pending.deadline {
                    self.uncertain_stop = true;
                    self.publish("unavailable", "stop_unconfirmed");
                }
            }
            Ok(Some(result)) => {
                self.pending_stop = None;
                let expected = self.stop_handle.as_deref();
                if expected.is_none_or(|handle| !is_bound_lifecycle(&result, handle)) {
                    self.uncertain_stop = true;
                    self.publish("unavailable", "stop_unconfirmed");
                    return;
                }
                if result.state == "stopped" {
                    self.handle = None;
                    self.stop_handle = None;
                    self.uncertain_stop = false;
                    self.admission_unknown = false;
                    self.readiness_deadline = None;
                    if self.shared.quit_requested.load(Ordering::Acquire) {
                        self.exit_after_quit();
                    } else {
                        self.publish("stopped", "gateway_stopped");
                    }
                } else {
                    self.apply_lifecycle(result);
                    self.uncertain_stop = true;
                    self.publish("unavailable", "stop_unconfirmed");
                }
            }
            Err(_error) => {
                self.pending_stop = None;
                self.uncertain_stop = true;
                self.adapter.take();
                self.publish("unavailable", "stop_unconfirmed");
            }
        }
    }

    fn poll_known_status(&mut self) {
        self.last_status_poll = Instant::now();
        let Some(handle) = self.handle.clone() else {
            return;
        };
        let result = self
            .adapter
            .as_mut()
            .map(|adapter| adapter.status(&handle));
        match result {
            Some(Ok(result)) => self.apply_lifecycle(result),
            Some(Err(error)) => {
                self.adapter.take();
                self.publish("unavailable", &error.0);
            }
            None => self.publish("unavailable", "adapter_unavailable"),
        }
    }

    fn advance_quit(&mut self) {
        if !self.shared.quit_requested.load(Ordering::Acquire) || self.exit_requested {
            return;
        }
        if self.admission_unknown {
            self.publish("unavailable", "stop_unconfirmed");
            return;
        }
        if self.pending_start.is_some() || self.admission_pending || self.pending_stop.is_some() {
            if self.pending_start.is_some() || self.admission_pending {
                self.publish("stopping", "gateway_stopping");
            }
            return;
        }
        if self.uncertain_stop {
            self.publish("unavailable", "stop_unconfirmed");
            return;
        }
        let Some(handle) = self.handle.clone() else {
            self.exit_after_quit();
            return;
        };
        if self.adapter.is_none() {
            if let Err(code) = self.ensure_adapter() {
                self.uncertain_stop = true;
                self.stop_handle = Some(handle);
                self.publish("unavailable", &code);
                return;
            }
        }
        let Some(adapter) = self.adapter.as_mut() else {
            self.uncertain_stop = true;
            self.stop_handle = Some(handle);
            self.publish("unavailable", "stop_unconfirmed");
            return;
        };
        let deadline = Instant::now() + Duration::from_secs(25);
        match adapter.dispatch_stop_until(&handle, deadline) {
            Ok(pending) => {
                self.pending_stop = Some(pending);
                self.stop_handle = Some(handle);
                self.publish("stopping", "gateway_stopping");
            }
            Err(BeginFailure::DispatchUncertain(pending, _error)) => {
                self.pending_stop = Some(pending);
                self.stop_handle = Some(handle);
                self.uncertain_stop = true;
                self.publish("stopping", "gateway_stopping");
            }
            Err(BeginFailure::BeforeDispatch(_error)) => {
                self.uncertain_stop = true;
                self.stop_handle = Some(handle);
                self.publish("unavailable", "stop_unconfirmed");
            }
        }
    }

    fn is_selection_locked(&self) -> bool {
        self.handle.is_some()
            || self.pending_start.is_some()
            || self.admission_pending
            || self.admission_unknown
            || self.pending_stop.is_some()
            || self.uncertain_stop
            || self.shared.quit_requested.load(Ordering::Acquire)
    }

    fn publish(&self, state: &str, code: &str) {
        let mut snapshot = self.shared.snapshot();
        let reveal_status = snapshot.state == "running" && state != "running";
        snapshot.state = state.to_owned();
        snapshot.code = code.to_owned();
        snapshot.selection_locked = self.is_selection_locked();
        snapshot.admission_pending = self.admission_pending || self.pending_start.is_some();
        snapshot.admission_unknown = self.admission_unknown;
        snapshot.stop_unconfirmed =
            self.admission_unknown || self.uncertain_stop || self.quit_ui_expired;
        snapshot.retry_allowed = self.preferences.is_some()
            && self.pending_start.is_none()
            && self.pending_stop.is_none()
            && !self.admission_pending
            && !self.admission_unknown
            && (if self.shared.quit_requested.load(Ordering::Acquire) {
                self.uncertain_stop
            } else {
                state == "unavailable" || state == "stopped"
            });
        snapshot.effective_workspace = self.effective_workspace.clone();
        self.shared.replace_snapshot(snapshot);
        if reveal_status {
            native::open_setup(&self.app);
        }
    }

    fn exit_after_quit(&mut self) {
        if let Some(mut adapter) = self.adapter.take() {
            adapter.shutdown();
        }
        self.shared.busy.store(false, Ordering::Release);
        self.shared.quit_queued.store(false, Ordering::Release);
        self.exit_requested = true;
        let _ = self.app.run_on_main_thread({
            let app = self.app.clone();
            move || app.exit(0)
        });
    }
}

fn normalize_selection(mut value: Preferences) -> Result<Preferences, String> {
    value.validate().map_err(str::to_owned)?;
    let config = fs::canonicalize(&value.config_path).map_err(|_| "config_invalid".to_owned())?;
    if !config.is_file() {
        return Err("config_invalid".to_owned());
    }
    value.config_path = config.to_string_lossy().into_owned();
    if let Some(workspace) = &value.workspace {
        if Path::new(workspace).exists() {
            value.workspace = Some(
                fs::canonicalize(workspace)
                    .map_err(|_| "workspace_invalid".to_owned())?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    Ok(value)
}

fn allowed_state(state: &str) -> Option<&'static str> {
    match state {
        "starting" => Some("starting"),
        "running" => Some("running"),
        "unavailable" => Some("unavailable"),
        "stopped" => Some("stopped"),
        _ => None,
    }
}

fn valid_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_lowercase()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn valid_handle(value: &str) -> bool {
    !value.is_empty() && value.len() <= 2048 && !value.chars().any(char::is_control)
}

fn captured_start_handle(result: &LifecycleResult) -> Option<&str> {
    if !valid_code(&result.code) || allowed_state(&result.state).is_none() {
        return None;
    }
    result.handle.as_deref().filter(|handle| valid_handle(handle))
}

fn confirms_stopped(result: &LifecycleResult, original_handle: &str) -> bool {
    result.state == "stopped" && is_bound_lifecycle(result, original_handle)
}

fn is_bound_lifecycle(result: &LifecycleResult, original_handle: &str) -> bool {
    valid_code(&result.code)
        && allowed_state(&result.state).is_some()
        && result.handle.as_deref() == Some(original_handle)
}

fn parse_loopback_origin(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|_| "gateway_origin_mismatch".to_owned())?;
    if url.scheme() != "http"
        || url.username() != ""
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.host().is_some_and(|host| match host {
            Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
            Host::Ipv4(address) => address.is_loopback(),
            Host::Ipv6(address) => address.is_loopback(),
        })
        || url.port_or_known_default().is_none()
    {
        return Err("gateway_origin_mismatch".to_owned());
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_origin_is_loopback_only_and_exact_http() {
        assert!(parse_loopback_origin("http://127.0.0.1:8765/").is_ok());
        assert!(parse_loopback_origin("http://[::1]:8765/").is_ok());
        assert!(parse_loopback_origin("http://localhost:8765/").is_ok());
        assert!(parse_loopback_origin("https://127.0.0.1:8765/").is_err());
        assert!(parse_loopback_origin("http://example.invalid:8765/").is_err());
        assert!(parse_loopback_origin("http://user:pass@127.0.0.1:8765/").is_err());
        assert!(parse_loopback_origin("http://127.0.0.1:8765/path").is_err());
    }

    #[test]
    fn quit_latch_prevents_a_start_after_quit_and_survives_late_admission() {
        let gate = AdmissionGate::default();
        gate.latch_quit();
        assert!(!gate.try_begin());

        let racing = AdmissionGate::default();
        assert!(racing.try_begin());
        racing.latch_quit();
        racing.release_if_pending();
        assert_eq!(racing.state(), 2);
        assert!(!racing.try_begin());
    }

    #[test]
    fn initial_start_response_must_carry_a_valid_identity() {
        let mut result = LifecycleResult {
            state: "starting".to_owned(),
            code: "gateway_starting".to_owned(),
            origin: None,
            handle: None,
        };
        assert!(captured_start_handle(&result).is_none());

        result.handle = Some("process-birth-selector-fingerprint".to_owned());
        assert_eq!(captured_start_handle(&result), result.handle.as_deref());
        result.handle = Some("bad\nhandle".to_owned());
        assert!(captured_start_handle(&result).is_none());
    }

    #[test]
    fn quit_confirmation_requires_the_original_handle() {
        let result = LifecycleResult {
            state: "stopped".to_owned(),
            code: "gateway_stopped".to_owned(),
            origin: None,
            handle: Some("original".to_owned()),
        };
        assert!(confirms_stopped(&result, "original"));
        assert!(!confirms_stopped(&result, "replacement"));
    }
}
