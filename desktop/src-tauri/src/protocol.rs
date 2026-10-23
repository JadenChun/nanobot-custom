use std::{
    io::{self, BufRead, BufReader, Read},
    os::fd::AsRawFd,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const MAX_FRAME_BYTES: usize = 65_536;
pub const BACKEND_VERSION: &str = "0.3.0";
pub const BACKEND_CAPABILITY: &str = "managed-gateway-v1";

#[derive(Debug)]
pub struct AdapterFailure(pub String);

impl AdapterFailure {
    fn new(code: &str) -> Self {
        Self(code.to_owned())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WireError {
    code: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResponse {
    v: u8,
    id: u64,
    ok: bool,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<WireError>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelloResult {
    pub protocol_version: u8,
    pub backend_version: String,
    pub python_executable: String,
    pub capabilities: Vec<String>,
    pub config_path: String,
    pub workspace: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleResult {
    pub state: String,
    pub code: String,
    pub origin: Option<String>,
    pub handle: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapResult {
    pub origin: String,
    pub secret: String,
    pub ws_url: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingResponse {
    id: u64,
    pub deadline: Instant,
}

#[derive(Debug)]
pub enum BeginFailure {
    BeforeDispatch(AdapterFailure),
    DispatchUncertain(PendingResponse, AdapterFailure),
}

impl BeginFailure {
    pub fn into_failure(self) -> AdapterFailure {
        match self {
            Self::BeforeDispatch(error) | Self::DispatchUncertain(_, error) => error,
        }
    }
}

pub struct PipeAdapter {
    child: Child,
    stdin: Option<ChildStdin>,
    responses: Receiver<Result<Vec<u8>, AdapterFailure>>,
    request_id: u64,
    outstanding: Option<PendingResponse>,
    interpreter: String,
}

impl PipeAdapter {
    pub fn spawn(interpreter: &Path) -> Result<Self, AdapterFailure> {
        if !interpreter.is_absolute() || !interpreter.is_file() {
            return Err(AdapterFailure::new("interpreter_invalid"));
        }
        let mut command = Command::new(interpreter);
        command
            .args(["-I", "-m", "nanobot.desktop"])
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command.env_remove("PYTHONPATH");
        let mut child = command.spawn().map_err(|_| AdapterFailure::new("adapter_start_failed"))?;
        let stdin = child.stdin.take().ok_or_else(|| AdapterFailure::new("adapter_start_failed"))?;
        let stdout = child.stdout.take().ok_or_else(|| AdapterFailure::new("adapter_start_failed"))?;
        if let Some(mut stderr) = child.stderr.take() {
            thread::spawn(move || {
                let mut buffer = [0u8; 4096];
                loop {
                    match stderr.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
            });
        }
        let (sender, responses) = mpsc::channel();
        thread::spawn(move || read_responses(stdout, sender));
        Ok(Self {
            child,
            stdin: Some(stdin),
            responses,
            request_id: 0,
            outstanding: None,
            interpreter: interpreter.to_string_lossy().into_owned(),
        })
    }

    pub fn hello(
        &mut self,
        config_path: &str,
        workspace: Option<&str>,
    ) -> Result<HelloResult, AdapterFailure> {
        let response = self.request(
            "hello",
            json!({"config_path": config_path, "workspace": workspace}),
            Duration::from_secs(5),
        )?;
        let value: HelloResult = serde_json::from_value(response)
            .map_err(|_| AdapterFailure::new("protocol_invalid"))?;
        if value.protocol_version != 1 {
            return Err(AdapterFailure::new("protocol_version_unsupported"));
        }
        if value.backend_version != BACKEND_VERSION {
            return Err(AdapterFailure::new("backend_version_unsupported"));
        }
        if value.python_executable != self.interpreter {
            return Err(AdapterFailure::new("interpreter_mismatch"));
        }
        if !value.capabilities.iter().any(|capability| capability == BACKEND_CAPABILITY) {
            return Err(AdapterFailure::new("backend_capability_unsupported"));
        }
        if !Path::new(&value.config_path).is_absolute() || !Path::new(&value.workspace).is_absolute() {
            return Err(AdapterFailure::new("protocol_invalid"));
        }
        Ok(value)
    }

    pub fn start(&mut self, handle: Option<&str>) -> Result<LifecycleResult, AdapterFailure> {
        let params = handle.map_or_else(|| json!({}), |handle| json!({"handle": handle}));
        self.lifecycle("start", params, Duration::from_secs(5))
    }

    pub fn begin_start(&mut self) -> Result<PendingResponse, AdapterFailure> {
        self.begin_request("start", json!({}), Duration::from_secs(5))
    }

    pub fn dispatch_start(&mut self) -> Result<PendingResponse, BeginFailure> {
        self.begin_request_inner("start", json!({}), Instant::now() + Duration::from_secs(5))
    }

    pub fn dispatch_stop_until(
        &mut self,
        handle: &str,
        deadline: Instant,
    ) -> Result<PendingResponse, BeginFailure> {
        self.begin_request_inner("stop", json!({"handle": handle}), deadline)
    }

    pub fn begin_status_until(
        &mut self,
        handle: &str,
        deadline: Instant,
    ) -> Result<PendingResponse, BeginFailure> {
        self.begin_request_inner("status", json!({"handle": handle}), deadline)
    }

    pub fn poll_lifecycle(
        &mut self,
        pending: PendingResponse,
    ) -> Result<Option<LifecycleResult>, AdapterFailure> {
        self.poll_request(pending)?
            .map(|value| {
                serde_json::from_value(value).map_err(|_| AdapterFailure::new("protocol_invalid"))
            })
            .transpose()
    }

    pub fn status(&mut self, handle: &str) -> Result<LifecycleResult, AdapterFailure> {
        self.lifecycle(
            "status",
            json!({"handle": handle}),
            Duration::from_secs(3),
        )
    }

    pub fn bootstrap(&mut self, handle: &str) -> Result<BootstrapResult, AdapterFailure> {
        let value = self.request(
            "bootstrap",
            json!({"handle": handle}),
            Duration::from_secs(3),
        )?;
        serde_json::from_value(value).map_err(|_| AdapterFailure::new("protocol_invalid"))
    }

    pub fn stop(&mut self, handle: &str) -> Result<LifecycleResult, AdapterFailure> {
        self.lifecycle(
            "stop",
            json!({"handle": handle}),
            Duration::from_secs(25),
        )
    }

    pub fn shutdown(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn lifecycle(
        &mut self,
        op: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<LifecycleResult, AdapterFailure> {
        let value = self.request(op, params, timeout)?;
        serde_json::from_value(value).map_err(|_| AdapterFailure::new("protocol_invalid"))
    }

    fn request(&mut self, op: &str, params: Value, timeout: Duration) -> Result<Value, AdapterFailure> {
        let pending = self.begin_request(op, params, timeout)?;
        self.wait_request(pending)
    }

    fn begin_request(
        &mut self,
        op: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<PendingResponse, AdapterFailure> {
        self.begin_request_inner(op, params, Instant::now() + timeout)
            .map_err(BeginFailure::into_failure)
    }

    fn begin_request_inner(
        &mut self,
        op: &str,
        params: Value,
        deadline: Instant,
    ) -> Result<PendingResponse, BeginFailure> {
        if self.outstanding.is_some() {
            return Err(BeginFailure::BeforeDispatch(AdapterFailure::new("operation_busy")));
        }
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or_else(|| BeginFailure::BeforeDispatch(AdapterFailure::new("protocol_invalid")))?;
        let id = self.request_id;
        let mut frame = serde_json::to_vec(&json!({"v": 1, "id": id, "op": op, "params": params}))
            .map_err(|_| BeginFailure::BeforeDispatch(AdapterFailure::new("protocol_invalid")))?;
        if frame.len() + 1 > MAX_FRAME_BYTES {
            return Err(BeginFailure::BeforeDispatch(AdapterFailure::new("protocol_invalid")));
        }
        frame.push(b'\n');
        let pending = PendingResponse { id, deadline };
        let input = self
            .stdin
            .as_mut()
            .ok_or_else(|| BeginFailure::BeforeDispatch(AdapterFailure::new("adapter_unavailable")))?;
        self.outstanding = Some(pending);
        if let Err(error) = write_deadline(input, &frame, deadline) {
            return Err(BeginFailure::DispatchUncertain(pending, error));
        }
        Ok(pending)
    }

    fn wait_request(&mut self, pending: PendingResponse) -> Result<Value, AdapterFailure> {
        if self.outstanding != Some(pending) {
            return Err(AdapterFailure::new("protocol_invalid"));
        }
        let remaining = pending.deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| AdapterFailure::new("adapter_timeout"))?;
        let response = self
            .responses
            .recv_timeout(remaining)
            .map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => AdapterFailure::new("adapter_timeout"),
                mpsc::RecvTimeoutError::Disconnected => AdapterFailure::new("adapter_unavailable"),
            })??;
        self.decode_response(pending.id, response)
    }

    fn poll_request(&mut self, pending: PendingResponse) -> Result<Option<Value>, AdapterFailure> {
        if self.outstanding != Some(pending) {
            return Err(AdapterFailure::new("protocol_invalid"));
        }
        match self.responses.try_recv() {
            Ok(Ok(response)) => self.decode_response(pending.id, response).map(Some),
            Ok(Err(error)) => {
                self.outstanding = None;
                Err(error)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.outstanding = None;
                Err(AdapterFailure::new("adapter_unavailable"))
            }
        }
    }

    fn decode_response(&mut self, id: u64, response: Vec<u8>) -> Result<Value, AdapterFailure> {
        self.outstanding = None;
        let value: WireResponse = serde_json::from_slice(&response)
            .map_err(|_| AdapterFailure::new("protocol_invalid"))?;
        if value.v != 1 || value.id != id {
            return Err(AdapterFailure::new("protocol_invalid"));
        }
        if value.ok {
            if value.error.is_some() {
                return Err(AdapterFailure::new("protocol_invalid"));
            }
            value.result.ok_or_else(|| AdapterFailure::new("protocol_invalid"))
        } else {
            if value.result.is_some() {
                return Err(AdapterFailure::new("protocol_invalid"));
            }
            let error = value.error.ok_or_else(|| AdapterFailure::new("protocol_invalid"))?;
            if !valid_error_code(&error.code) {
                return Err(AdapterFailure::new("protocol_invalid"));
            }
            Err(AdapterFailure(error.code))
        }
    }
}

fn write_deadline(
    input: &impl AsRawFd,
    bytes: &[u8],
    deadline: Instant,
) -> Result<(), AdapterFailure> {
    let descriptor = input.as_raw_fd();
    let current = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if current < 0 || unsafe { libc::fcntl(descriptor, libc::F_SETFL, current | libc::O_NONBLOCK) } < 0 {
        return Err(AdapterFailure::new("adapter_unavailable"));
    }
    let mut offset = 0;
    while offset < bytes.len() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| AdapterFailure::new("adapter_timeout"))?;
        let wait_ms = remaining.as_millis().clamp(1, i32::MAX as u128) as i32;
        let mut poll = libc::pollfd {
            fd: descriptor,
            events: libc::POLLOUT,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll, 1, wait_ms) };
        if ready == 0 {
            return Err(AdapterFailure::new("adapter_timeout"));
        }
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(AdapterFailure::new("adapter_unavailable"));
        }
        let written = unsafe {
            libc::write(
                descriptor,
                bytes[offset..].as_ptr().cast(),
                bytes.len() - offset,
            )
        };
        if written > 0 {
            offset += written as usize;
            continue;
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted || error.kind() == io::ErrorKind::WouldBlock {
            continue;
        }
        return Err(AdapterFailure::new("adapter_unavailable"));
    }
    Ok(())
}

impl Drop for PipeAdapter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn valid_error_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_lowercase()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn read_responses(stdout: impl Read, sender: Sender<Result<Vec<u8>, AdapterFailure>>) {
    let mut reader = BufReader::new(stdout);
    loop {
        match read_frame(&mut reader) {
            Ok(Some(frame)) => {
                if sender.send(Ok(frame)).is_err() {
                    return;
                }
            }
            Ok(None) => {
                let _ = sender.send(Err(AdapterFailure::new("adapter_unavailable")));
                return;
            }
            Err(_) => {
                let _ = sender.send(Err(AdapterFailure::new("protocol_invalid")));
                return;
            }
        }
    }
}

fn read_frame(reader: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut frame = Vec::with_capacity(1024);
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if frame.is_empty() {
                Ok(None)
            } else {
                Err(io::ErrorKind::UnexpectedEof.into())
            };
        }
        let to_newline = available.iter().position(|byte| *byte == b'\n');
        let count = to_newline.map_or(available.len(), |at| at + 1);
        let room = MAX_FRAME_BYTES + 1 - frame.len();
        let count = count.min(room);
        frame.extend_from_slice(&available[..count]);
        reader.consume(count);
        if frame.len() > MAX_FRAME_BYTES {
            return Err(io::ErrorKind::InvalidData.into());
        }
        if frame.last() == Some(&b'\n') {
            return Ok(Some(frame));
        }
        if frame.len() == MAX_FRAME_BYTES {
            let next = reader.fill_buf()?;
            if next.is_empty() {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            if next[0] != b'\n' {
                return Err(io::ErrorKind::InvalidData.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;

    #[test]
    fn frame_reader_accepts_limit_and_rejects_over_limit() {
        let accepted = format!("{}\n", "x".repeat(MAX_FRAME_BYTES - 1));
        let mut reader = BufReader::new(accepted.as_bytes());
        assert_eq!(read_frame(&mut reader).unwrap().unwrap().len(), MAX_FRAME_BYTES);

        let rejected = format!("{}\n", "x".repeat(MAX_FRAME_BYTES));
        let mut reader = BufReader::new(rejected.as_bytes());
        assert_eq!(read_frame(&mut reader).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn error_codes_are_bounded_and_sanitized() {
        assert!(valid_error_code("gateway_unavailable"));
        assert!(!valid_error_code("https://example.invalid"));
        assert!(!valid_error_code(""));
    }

    #[test]
    fn a_blocked_pipe_write_obeys_the_request_deadline() {
        let (writer, _reader) = UnixStream::pair().unwrap();
        let payload = vec![b'x'; 8 * 1024 * 1024];
        let started = Instant::now();
        let result = write_deadline(
            &writer,
            &payload,
            started + Duration::from_millis(100),
        );
        assert_eq!(result.unwrap_err().0, "adapter_timeout");
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
