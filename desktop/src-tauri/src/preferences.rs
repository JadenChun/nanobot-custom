use std::{
    env, fs,
    io::{Read, Write},
    os::fd::AsRawFd,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
#[cfg(any(target_os = "linux", test))]
use sha2::{Digest, Sha256};

const MAX_PREFERENCES_BYTES: u64 = 16 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Preferences {
    pub config_path: String,
    pub interpreter_path: String,
    #[serde(default)]
    pub workspace: Option<String>,
}

impl Preferences {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.config_path.len() > 4096
            || !Path::new(&self.config_path).is_absolute()
            || self.interpreter_path.len() > 4096
            || !Path::new(&self.interpreter_path).is_absolute()
            || self.workspace.as_ref().is_some_and(|path| {
                path.len() > 4096 || !Path::new(path).is_absolute()
            })
        {
            return Err("selection_invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SetupDefaults {
    pub config_path: String,
    pub interpreter_path: Option<String>,
}

fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from).filter(|path| path.is_absolute())
}

fn config_under(home: &Path) -> PathBuf {
    home.join(".kohaku").join("config.json")
}

fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.mode() & 0o111 != 0)
}

fn resolve_release_python(root: &Path, selector: &str, relative_python: &str) -> Option<PathBuf> {
    let release = fs::canonicalize(root.join(selector)).ok()?;
    // Resolve the release, not its interpreter symlink: Python needs the
    // venv's bin path to find its pyvenv.cfg.
    let python = release.join(relative_python);
    is_executable_file(&python).then_some(python)
}

fn setup_defaults_under(home: &Path) -> Option<SetupDefaults> {
    let managed = home.join(".local/share/kohaku");
    let release_root = home.join(".local/share/kohaku-release");
    let interpreter_path = resolve_release_python(&managed, "desktop-current", "env/bin/python")
        .or_else(|| resolve_release_python(&managed, "current", "env/bin/python"))
        .or_else(|| resolve_release_python(&release_root, "desktop-current", "venv/bin/python"))
        .or_else(|| resolve_release_python(&release_root, "current", "venv/bin/python"))
        .or_else(|| {
            let direct = release_root.join("venv/bin/python");
            is_executable_file(&direct).then_some(direct)
        })
        .and_then(|path| path.into_os_string().into_string().ok());
    Some(SetupDefaults {
        config_path: config_under(home).into_os_string().into_string().ok()?,
        interpreter_path,
    })
}

fn bundled_interpreter_near_exe() -> Option<String> {
    let exe = env::current_exe().ok()?;
    let bin_dir = exe.parent()?;
    let release_dir = bin_dir.parent()?;
    for rel in ["venv/bin/python", "env/bin/python"] {
        let candidate = release_dir.join(rel);
        if is_executable_file(&candidate) {
            return candidate.into_os_string().into_string().ok();
        }
    }
    None
}

pub fn setup_defaults() -> Option<SetupDefaults> {
    let mut defaults = setup_defaults_under(&home_directory()?)?;
    if defaults.interpreter_path.is_none() {
        defaults.interpreter_path = bundled_interpreter_near_exe();
    }
    Some(defaults)
}

pub fn missing_default_config(path: &Path) -> bool {
    home_directory().is_some_and(|home| config_under(&home) == path)
        && fs::symlink_metadata(path)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}

pub fn preference_path() -> Result<PathBuf, &'static str> {
    let base = if let Some(path) = env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(path)
    } else {
        let home = PathBuf::from(env::var_os("HOME").ok_or("preferences_unavailable")?);
        #[cfg(target_os = "linux")]
        let path = home.join(".config");
        #[cfg(target_os = "macos")]
        let path = home.join("Library").join("Application Support");
        path
    };
    if !base.is_absolute() {
        return Err("preferences_unavailable");
    }
    ensure_base_directory(&base)?;
    let app_dir = base.join("kohaku-desktop");
    ensure_private_directory(&app_dir)?;
    Ok(app_dir.join("preferences.json"))
}

pub fn load() -> Result<Option<Preferences>, &'static str> {
    let path = preference_path()?;
    let mut options = fs::OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = match options.open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("preferences_unavailable"),
    };
    let metadata = file.metadata().map_err(|_| "preferences_unavailable")?;
    validate_private_file(&metadata)?;
    if metadata.len() > MAX_PREFERENCES_BYTES {
        return Err("preferences_invalid");
    }
    let mut data = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_PREFERENCES_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|_| "preferences_unavailable")?;
    if data.len() as u64 > MAX_PREFERENCES_BYTES {
        return Err("preferences_invalid");
    }
    let value: Preferences = serde_json::from_slice(&data).map_err(|_| "preferences_invalid")?;
    value.validate()?;
    Ok(Some(value))
}

pub fn store(value: &Preferences) -> Result<(), &'static str> {
    value.validate()?;
    let path = preference_path()?;
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec(value).map_err(|_| "preferences_invalid")?;
    if bytes.len() as u64 > MAX_PREFERENCES_BYTES {
        return Err("preferences_invalid");
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&temp)
        .map_err(|_| "preferences_unavailable")?;
    let metadata = file.metadata().map_err(|_| "preferences_unavailable")?;
    validate_private_file(&metadata)?;
    file.write_all(&bytes).map_err(|_| "preferences_unavailable")?;
    file.sync_all().map_err(|_| "preferences_unavailable")?;
    drop(file);
    fs::rename(&temp, &path).map_err(|_| "preferences_unavailable")?;
    let parent = fs::File::open(path.parent().ok_or("preferences_unavailable")?)
        .map_err(|_| "preferences_unavailable")?;
    parent.sync_all().map_err(|_| "preferences_unavailable")?;
    Ok(())
}

pub fn remove() -> Result<(), &'static str> {
    let path = preference_path()?;
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("preferences_unavailable"),
    }
}

#[cfg(target_os = "linux")]
pub fn profile_directory(value: &Preferences) -> Result<PathBuf, &'static str> {
    let base = if let Some(path) = env::var_os("XDG_DATA_HOME") {
        PathBuf::from(path)
    } else {
        PathBuf::from(env::var_os("HOME").ok_or("profile_unavailable")?)
            .join(".local")
            .join("share")
    };
    if !base.is_absolute() {
        return Err("profile_unavailable");
    }
    ensure_base_directory(&base).map_err(|_| "profile_unavailable")?;
    let root = base.join("nanobot-desktop");
    ensure_private_directory(&root).map_err(|_| "profile_unavailable")?;
    let mut digest = Sha256::new();
    digest.update(value.config_path.as_bytes());
    digest.update([0]);
    if let Some(workspace) = &value.workspace {
        digest.update(workspace.as_bytes());
    }
    let profile = root.join(format!("{:x}", digest.finalize()));
    ensure_private_directory(&profile).map_err(|_| "profile_unavailable")?;
    Ok(profile)
}

fn ensure_base_directory(path: &Path) -> Result<(), &'static str> {
    if !path.exists() {
        fs::create_dir_all(path).map_err(|_| "preferences_unavailable")?;
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| "preferences_unavailable")?;
    if !metadata.file_type().is_dir() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err("preferences_unavailable");
    }
    Ok(())
}

fn ensure_private_directory(path: &Path) -> Result<(), &'static str> {
    match fs::create_dir(path) {
        Ok(()) => {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .map_err(|_| "preferences_unavailable")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err("preferences_unavailable"),
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| "preferences_unavailable")?;
    if !metadata.file_type().is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("preferences_unavailable");
    }
    Ok(())
}

fn validate_private_file(metadata: &fs::Metadata) -> Result<(), &'static str> {
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("preferences_unavailable");
    }
    Ok(())
}

pub fn lock_is_exclusive(file: &fs::File) -> bool {
    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_suggests_the_standard_config_and_prefers_the_desktop_install() {
        let home = env::temp_dir().join(format!(
            "nanobot-desktop-defaults-{}",
            std::process::id()
        ));
        fs::create_dir(&home).unwrap();
        let defaults = setup_defaults_under(&home).unwrap();
        assert_eq!(Path::new(&defaults.config_path), home.join(".kohaku/config.json"));
        assert!(!home.join(".kohaku").exists());
        assert!(defaults.interpreter_path.is_none());
        for selector in ["current", "desktop-current"] {
            let python = home.join(".local/share/kohaku").join(selector).join("env/bin/python");
            fs::create_dir_all(python.parent().unwrap()).unwrap();
            fs::write(&python, []).unwrap();
            fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
            let selected = setup_defaults_under(&home).unwrap().interpreter_path.unwrap();
            assert_eq!(
                Path::new(&selected),
                fs::canonicalize(python.parent().unwrap()).unwrap().join("python")
            );
        }
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn selectors_must_be_absolute_and_preserve_null_workspace() {
        let valid = Preferences {
            config_path: "/tmp/selected/config.json".into(),
            interpreter_path: "/tmp/venv/bin/python".into(),
            workspace: None,
        };
        assert_eq!(valid.validate(), Ok(()));
        assert_eq!(valid.workspace, None);
        let invalid = Preferences {
            config_path: "relative.json".into(),
            ..valid
        };
        assert_eq!(invalid.validate(), Err("selection_invalid"));
    }

    #[test]
    fn config_identity_differs_for_explicit_workspace_selector() {
        let original = Preferences {
            config_path: "/tmp/selected/config.json".into(),
            interpreter_path: "/tmp/venv/bin/python".into(),
            workspace: None,
        };
        let explicit = Preferences {
            workspace: Some("/tmp/selected/workspace".into()),
            ..original.clone()
        };
        let first = profile_key(&original);
        let second = profile_key(&explicit);
        assert_ne!(first, second);
    }

    fn profile_key(value: &Preferences) -> String {
        let mut digest = Sha256::new();
        digest.update(value.config_path.as_bytes());
        digest.update([0]);
        if let Some(workspace) = &value.workspace {
            digest.update(workspace.as_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}
