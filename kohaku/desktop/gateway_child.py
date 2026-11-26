"""Isolated managed gateway runner used only by the private desktop adapter."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from kohaku.channels.websocket.runtime import WebSocketConfig
from kohaku.cli.gateway_runtime import _run_gateway
from kohaku.config.loader import load_config, resolve_config_env_vars, set_config_path
from kohaku.config.schema import Config
from kohaku.gateway.runtime import GatewayInstance
from kohaku.providers.factory import validate_provider_setup
from kohaku.runtime_composition import RuntimePreflightError, runtime_preflight
from kohaku.security.network import is_loopback_host


def _arguments(argv: list[str]) -> tuple[Path, str | None] | None:
    config_path: Path | None = None
    workspace: str | None = None
    index = 0
    while index < len(argv):
        option = argv[index]
        if option not in {"--config", "--workspace"} or index + 1 >= len(argv):
            return None
        value = argv[index + 1]
        if not value or value.startswith("-"):
            return None
        if option == "--config":
            if config_path is not None:
                return None
            config_path = Path(value)
        else:
            if workspace is not None:
                return None
            workspace = value
        index += 2
    if config_path is None or not config_path.is_absolute():
        return None
    if workspace is not None and not Path(workspace).is_absolute():
        return None
    return config_path, workspace


def _validate(config: Config) -> None:
    websocket = WebSocketConfig.model_validate(config.channels.websocket)
    if (
        not websocket.enabled
        or not is_loopback_host(config.gateway.host)
        or not is_loopback_host(websocket.host)
        or websocket.unix_socket_path
        or websocket.public_ws_url
        or websocket.ssl_certfile
        or websocket.ssl_keyfile
        or not (websocket.token_issue_secret.strip() or websocket.token.strip())
        or config.gateway.port == websocket.port
    ):
        raise ValueError("desktop_config_invalid")
    validate_provider_setup(config)
    if config.runtime_data_dir is None:
        raise ValueError("config_unavailable")
    sessions_root = config.runtime_data_dir / "sessions"
    runtime_preflight(config, sessions_root=sessions_root)


def _set_macos_background_policy() -> None:
    """Keep AppKit's screen queries from making this helper a foreground app."""
    try:
        appkit = import_module("AppKit")
        policy: object = getattr(appkit, "NSApplicationActivationPolicyProhibited", None)
        application_class: object = getattr(appkit, "NSApplication", None)
        shared_application: object = getattr(application_class, "sharedApplication", None)
        if isinstance(policy, int) and callable(shared_application):
            application: object = shared_application()
            set_policy: object = getattr(application, "setActivationPolicy_", None)
            if callable(set_policy) and set_policy(policy) is True:
                return
    except ImportError:
        # PyObjC is optional; desktop-use retains its normal missing-dependency error.
        return
    except Exception:
        # A presentation failure must not prevent the gateway or scheduler from starting.
        pass
    sys.stderr.write("Kohaku Desktop gateway child: macos_background_policy_unavailable\n")


def main(argv: list[str] | None = None) -> int:
    """Start the configured gateway after independent child-side admission."""
    parsed = _arguments(sys.argv[1:] if argv is None else argv)
    if parsed is None:
        sys.stderr.write("Kohaku Desktop gateway child: invalid_arguments\n")
        return 2
    config_path, workspace_override = parsed
    try:
        config_path = config_path.expanduser().resolve(strict=True)
        if not config_path.is_file():
            raise OSError
        config = resolve_config_env_vars(load_config(config_path), config_path=config_path)
        set_config_path(config_path)
        workspace_selector = None
        if workspace_override is not None:
            workspace_selector = str(Path(workspace_override).expanduser().resolve(strict=False))
            config.agents.defaults.workspace = workspace_selector
        config.agents.defaults.workspace = str(config.workspace_path.resolve(strict=False))
        _validate(config)
        instance = GatewayInstance.resolve(
            config_path=config_path,
            workspace=workspace_selector,
        )
        if sys.platform == "darwin" and config.tools.desktop_use.enabled:
            _set_macos_background_policy()
        _run_gateway(
            config,
            gateway_instance=instance,
            webui_bundle_mode="skip",
            desktop_managed=True,
        )
    except RuntimePreflightError:
        sys.stderr.write("Kohaku Desktop gateway child: runtime_preflight_refused\n")
        return 3
    except Exception:
        sys.stderr.write("Kohaku Desktop gateway child: startup_failed\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
