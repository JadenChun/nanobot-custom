use std::{
    env, fs,
    io::{Read, Write},
    os::fd::AsRawFd,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
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

pub fn preference_path() -> Result<PathBuf, &'static str> {
    let base = if let Some(path) = env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(path)
    } else {
        PathBuf::from(env::var_os("HOME").ok_or("preferences_unavailable")?)
            .join(".config")
    };
    if !base.is_absolute() {
        return Err("preferences_unavailable");
    }
    ensure_base_directory(&base)?;
    let app_dir = base.join("nanobot-desktop");
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
