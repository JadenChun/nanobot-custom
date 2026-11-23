# Upgrade to Kohaku

Use this guide when you have an existing Nanobot profile. Kohaku does not adopt the old default profile automatically.

## Prepare the Existing Profile

1. Stop Nanobot gateways, desktop apps, scheduled jobs, and other processes that can write to the profile.
2. Record the absolute path to each config file, workspace, and configured state root.
3. Snapshot each config profile and every configured workspace and state root.
4. Store the snapshot with the same access controls as the original data.

Do not print secrets or copy private key files. Keep existing secret-store references and permissions in place.

The Kohaku default profile uses `~/.kohaku/`. Existing Nanobot data can use other paths, including `~/.nanobot/`. A fresh Kohaku install does not copy or adopt that data.

## Inspect and Select the Existing Profile

Use the absolute paths that you recorded. This example shows a legacy profile under `~/.nanobot/`:

```bash
kohaku status --config "$HOME/.nanobot/config.json" --workspace "$HOME/.nanobot/workspace"
```

Review the status output before you start the agent. Then use the same `--config` and `--workspace` paths for Kohaku commands that should use this profile.

Keep frozen session state and absolute security bindings exact. Do not move, rewrite, or normalize their paths during adoption. This includes workspace restrictions, allowed roots, and trusted proxy bindings.

New Kohaku configuration uses `KOHAKU_*` environment variable names. Use the [Configuration guide](./configuration.md#runtime-environment-variables) to check each setting. Do not assume that an old `NANOBOT_*` name still works.

## Install a Matched Build

Use a tested Kohaku shell and Python wheel pair from the same release. Use the matching terminal sidecar when your setup needs one. Do not mix builds from different releases.

After installation, run `kohaku status` with the recorded paths. Confirm that Kohaku reads the intended config and workspace before you start a gateway or connect a channel.

## Update Integrations

External Python applications must change their imports to the Kohaku package. For example:

```python
from kohaku import Kohaku
```

Use `kohaku` for the command-line interface. New installs use `~/.kohaku/` and `KOHAKU_*` settings.

The new Android app ID is `ai.kohaku.mobile`. Install a fresh Kohaku APK and pair it with the gateway again. The new app does not reuse the old Nanobot app data or pairing.

Some workspace files and protocol values keep the historical `nanobot` spelling for compatibility. Keep those identifiers unchanged. Do not rename persisted data or wire tokens by hand.

## Roll Back

1. Stop every Kohaku process that can write to the profile.
2. Restore the snapshot to the original absolute config, workspace, and state paths.
3. Reinstall the tested shell and wheel pair that you used before the upgrade.
4. Start the old build only after you confirm that it uses the restored paths.

Restore all recorded roots. Restoring only `config.json` can leave sessions, workspace markers, credentials, or other state out of sync.
