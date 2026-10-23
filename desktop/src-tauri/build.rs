fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(
            tauri_build::AppManifest::new().commands(&[
                "desktop_state",
                "choose_path",
                "select_backend",
                "retry_backend",
                "open_chat",
                "change_selection",
                "quit_desktop",
            ]),
        ),
    )
    .expect("failed to build Tauri metadata");
}
