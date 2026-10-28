"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import (
    ProgressEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    replace_outbound_event,
)
from nanobot.bus.queue import MessageBus
from nanobot.channels._setup import channel_setup_spec
from nanobot.channels.base import BaseChannel
from nanobot.channels.contracts import (
    channel_default_config,
    channel_instance_specs,
    channel_runtime_name,
    resolve_channel_action_target,
)
from nanobot.channels.registry import channel_default_enabled
from nanobot.config.schema import Config
from nanobot.context_repo import ContextRepoManager
from nanobot.utils.restart import (
    RestartNotice,
    consume_restart_notice_from_env,
    format_restart_completed_message,
)

if TYPE_CHECKING:
    from nanobot.cron.service import CronService
    from nanobot.session.manager import SessionManager
    from nanobot.triggers.local_store import LocalTriggerStore


def _default_webui_dist() -> Path | None:
    """Return the absolute path to the bundled webui dist directory if it exists."""
    try:
        import nanobot.web as web_pkg  # type: ignore[import-not-found]
    except ImportError:
        return None
    candidate = Path(web_pkg.__file__).resolve().parent / "dist"
    return candidate if candidate.is_dir() else None


# Retry delays for message sending (exponential backoff: 1s, 2s, 4s)
_SEND_RETRY_DELAYS = (1, 2, 4)
_OUTBOUND_CONCURRENCY = 32
_OUTBOUND_PENDING_LIMIT = 256
_RESTART_NOTICE_START_TIMEOUT_S = 30.0
_RESTART_NOTICE_START_POLL_S = 0.25
ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE = 1000

_CleanupResult = TypeVar("_CleanupResult")


async def _join_owned_cleanup(task: asyncio.Task[_CleanupResult]) -> _CleanupResult:
    """Join same-loop cleanup even after repeated owner cancellation.

    Cancellation is reported only after the owned task releases its resources;
    a simultaneous cleanup failure is retained rather than masked.
    """
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                interrupted = True
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as exc:
        if interrupted:
            raise BaseExceptionGroup(
                "channel cleanup failed after cancellation", [asyncio.CancelledError(), exc],
            ) from None
        raise
    if interrupted:
        raise asyncio.CancelledError()
    return result

_BOOL_CAMEL_ALIASES: dict[str, str] = {
    "send_progress": "sendProgress",
    "send_tool_hints": "sendToolHints",
    "show_reasoning": "showReasoning",
}

def _default_channel_config(name: str) -> dict[str, Any] | None:
    from nanobot.channels.registry import load_channel_plugin

    plugin = load_channel_plugin(name)
    if not plugin.default_enabled:
        return None
    return channel_default_config(plugin)


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        *,
        session_manager: "SessionManager | None" = None,
        cron_service: CronService | None = None,
        local_trigger_store: LocalTriggerStore | None = None,
        webui_runtime_model_name: Callable[[], str | None] | None = None,
        webui_refresh_runtime_config: Callable[[], None] | None = None,
        webui_cron_pending_job_ids: Callable[[str], set[str]] | None = None,
        webui_local_trigger_pending_ids: Callable[[str], set[str]] | None = None,
        webui_static_dist: bool = True,
        webui_runtime_surface: str = "browser",
        webui_runtime_capabilities: dict[str, Any] | None = None,
        webui_mcp_runtime_status: Callable[[], Mapping[str, str]] | None = None,
        webui_mcp_reload: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        webui_skill_state_action: Callable[[set[str]], None] | None = None,
        webui_recovery_action: (
            Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None
        ) = None,
        config_path: Path | None = None,
    ):
        if config_path is None:
            from nanobot.config.loader import get_config_path

            config_path = get_config_path()
        self.config = config
        self._config_path = config_path.expanduser().resolve(strict=False)
        self.bus = bus
        self._session_manager = session_manager
        self._cron_service = cron_service
        self._local_trigger_store = local_trigger_store
        self._webui_runtime_model_name = webui_runtime_model_name
        self._webui_refresh_runtime_config = webui_refresh_runtime_config
        self._webui_cron_pending_job_ids = webui_cron_pending_job_ids
        self._webui_local_trigger_pending_ids = webui_local_trigger_pending_ids
        self._webui_static_dist = webui_static_dist
        self._webui_runtime_surface = webui_runtime_surface
        self._webui_runtime_capabilities = dict(webui_runtime_capabilities or {})
        self._webui_mcp_runtime_status = webui_mcp_runtime_status
        self._webui_mcp_reload = webui_mcp_reload
        self._webui_skill_state_action = webui_skill_state_action
        self._webui_recovery_action = webui_recovery_action
        self.channels: dict[str, BaseChannel] = {}
        self._channel_owners: dict[str, str] = {}
        self._channel_runtime_specs: dict[str, tuple[str, str]] = {}
        self._channel_errors: dict[str, str] = {}
        self._channel_tasks: dict[str, asyncio.Task[None]] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._outbound_tasks: dict[asyncio.Task[None], tuple[str, str]] = {}
        self._outbound_tails: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._outbound_slots = asyncio.Semaphore(_OUTBOUND_PENDING_LIMIT)
        self._outbound_sends = asyncio.Semaphore(_OUTBOUND_CONCURRENCY)
        self._stopping_channels: set[str] = set()
        self._started = False
        self._acquired = False
        self._acquiring = False
        self._channel_stop_tasks: dict[str, tuple[BaseChannel, asyncio.Task[bool]]] = {}
        self._restart_notice_task: asyncio.Task[None] | None = None
        self._origin_reply_fingerprints: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    def _channel_section(
        self,
        name: str,
        *,
        config: Config | None = None,
        default_sections: dict[str, Any] | None = None,
        default_enabled: bool | None = None,
    ) -> Any:
        config = config or self.config
        section = getattr(config.channels, name, None)
        if default_enabled is None:
            default_enabled = channel_default_enabled(name)
        if section is not None or not default_enabled:
            return section
        if default_sections is None:
            return _default_channel_config(name)
        if name not in default_sections:
            default = _default_channel_config(name)
            if default is not None:
                default_sections[name] = default
        return default_sections.get(name)

    def _build_channel(
        self,
        name: str,
        cls: type[BaseChannel],
        section: Any,
        *,
        runtime_name: str | None = None,
    ) -> BaseChannel:
        kwargs: dict[str, Any] = {}
        if cls.name == "websocket":
            from nanobot.channels.websocket.runtime import WebSocketConfig
            from nanobot.webui.gateway_services import build_gateway_services

            parsed = WebSocketConfig.model_validate(section)
            static_path = _default_webui_dist() if self._webui_static_dist else None
            workspace = Path(self.config.workspace_path)
            gateway = build_gateway_services(
                config=parsed,
                bus=self.bus,
                session_manager=self._session_manager,
                static_dist_path=static_path,
                workspace_path=workspace,
                default_restrict_to_workspace=self.config.tools.restrict_to_workspace,
                configured_exec_sandbox=bool(self.config.tools.exec.sandbox),
                context_manager=ContextRepoManager.from_config(
                    context_paths=self.config.agents.defaults.context_paths,
                    context_repos=self.config.agents.defaults.context_repos,
                ),
                config_path=self._config_path,
                disabled_skills=set(self.config.agents.defaults.disabled_skills),
                runtime_model_name=self._webui_runtime_model_name,
                refresh_runtime_config=self._webui_refresh_runtime_config,
                runtime_surface=self._webui_runtime_surface,
                runtime_capabilities_overrides=self._webui_runtime_capabilities,
                cron_service=self._cron_service,
                local_trigger_store=self._local_trigger_store,
                cron_pending_job_ids=self._webui_cron_pending_job_ids,
                local_trigger_pending_ids=self._webui_local_trigger_pending_ids,
                channel_feature_action=self.apply_channel_feature_action,
                channel_runtime_status=self.get_status,
                mcp_runtime_status=self._webui_mcp_runtime_status,
                mcp_reload=self._webui_mcp_reload,
                skill_state_action=self._webui_skill_state_action,
                recovery_action=self._webui_recovery_action,
                logger=logger,
            )
            kwargs["gateway"] = gateway
        channel = cls(section, self.bus, **kwargs)
        if runtime_name and runtime_name != channel.name:
            channel.name = runtime_name
        progress_default, tool_hints_default = channel.progress_transport_defaults() or (
            self.config.channels.send_progress,
            self.config.channels.send_tool_hints,
        )
        channel.send_progress = self._resolve_bool_override(
            section, "send_progress", progress_default,
        )
        channel.send_tool_hints = self._resolve_bool_override(
            section, "send_tool_hints", tool_hints_default,
        )
        channel.show_reasoning = self._resolve_bool_override(
            section, "show_reasoning", self.config.channels.show_reasoning,
        )
        return channel

    async def acquire(self) -> ChannelManager:
        """Initialize enabled runtimes from dependency-free channel descriptors."""
        if self._acquired:
            return self
        if self._acquiring:
            raise RuntimeError("channel manager acquisition is already in progress")
        snapshots = (
            dict(self.channels),
            dict(self._channel_owners),
            dict(self._channel_runtime_specs),
            dict(self._channel_errors),
        )
        self._acquiring = True
        try:
            return await self._acquire_channels()
        except BaseException:
            channels, owners, runtime_specs, errors = snapshots
            self.channels.clear()
            self.channels.update(channels)
            self._channel_owners.clear()
            self._channel_owners.update(owners)
            self._channel_runtime_specs.clear()
            self._channel_runtime_specs.update(runtime_specs)
            self._channel_errors.clear()
            self._channel_errors.update(errors)
            self._acquired = False
            raise
        finally:
            self._acquiring = False

    async def _acquire_channels(self) -> ChannelManager:
        from nanobot.channels.registry import (
            discover_legacy_plugins,
            discover_plugins,
            packaged_channel_names,
        )
        from nanobot.optional_features import check_enabled_channel_dependencies

        reserved_names = set(packaged_channel_names())
        plugins = discover_plugins()
        extras = self.config.channels.model_extra or {}
        shared_names = set(self.config.channels.__class__.model_fields)
        external_enabled = {
            name
            for name, section in extras.items()
            if name not in shared_names
            and isinstance(section, Mapping)
            and cast(Mapping[object, object], section).get("enabled") is True
            and name not in plugins
        }
        legacy_plugins = discover_legacy_plugins(
            external_enabled,
            reserved_names=reserved_names,
        )
        plugins.update(legacy_plugins)
        default_sections: dict[str, Any] = {}
        activations: dict[str, tuple[Any, list[tuple[str, Any]]]] = {}
        enabled_names: set[str] = set()
        for name, plugin in plugins.items():
            section = self._channel_section(
                name,
                default_sections=default_sections,
                default_enabled=plugin.default_enabled,
            )
            if section is None:
                continue
            if name == "telegram" and not plugin.management.multi_instance:
                self._validate_telegram_allow_from_config(
                    section,
                    default_enabled=plugin.default_enabled,
                )
            try:
                channel_setup_spec(name, plugin=plugin)
                specs = channel_instance_specs(plugin, section)
                if name == "telegram":
                    for spec in specs:
                        self._validate_telegram_allow_from_config(
                            spec.config,
                            default_enabled=True,
                        )
                runtime_specs = [
                    (channel_runtime_name(plugin, spec.instance_id), spec)
                    for spec in specs
                ]
            except Exception as exc:
                if name in legacy_plugins:
                    raise ValueError(
                        f"Enabled external channel '{name}' has invalid configuration: {exc}"
                    ) from exc
                logger.warning("Could not inspect {} channel activation: {}", name, exc)
                continue
            if not runtime_specs:
                continue
            collisions = sorted(
                set(self._channel_runtime_specs)
                & {runtime_name for runtime_name, _spec in runtime_specs}
            )
            if collisions:
                logger.warning(
                    "{} channel runtime name(s) are already claimed: {}",
                    name,
                    ", ".join(collisions),
                )
                continue
            for runtime_name, spec in runtime_specs:
                self._channel_runtime_specs[runtime_name] = (name, spec.instance_id)
            activations[name] = (plugin, runtime_specs)
            enabled_names.add(name)

        dependency_errors = check_enabled_channel_dependencies(enabled_names, plugins)
        invalid_external = sorted(set(dependency_errors) & set(legacy_plugins))
        if invalid_external:
            raise ValueError(
                "Enabled external channel dependencies are unavailable: "
                + ", ".join(invalid_external)
            )
        for name, error in dependency_errors.items():
            self._mark_channel_error(name, error)

        # Import and validate enabled external adapters before constructing any
        # channel runtime. Disabled and unconfigured entry points never reach
        # this point and therefore never execute plugin code.
        external_classes = {
            name: legacy_plugins[name].load_channel_class()
            for name in sorted(set(activations) & set(legacy_plugins))
        }

        acquired: list[BaseChannel] = []
        pending_channels: list[tuple[str, str, BaseChannel]] = []
        try:
            for name, (plugin, runtime_specs) in activations.items():
                if name in dependency_errors:
                    continue
                cls = external_classes.get(name) or plugin.load_channel_class()
                for runtime_name, spec in runtime_specs:
                    channel = self._build_channel(
                        name,
                        cls,
                        spec.config,
                        runtime_name=runtime_name,
                    )
                    acquired.append(channel)
                    pending_channels.append((name, runtime_name, channel))
            for name, runtime_name, channel in pending_channels:
                self.channels[runtime_name] = channel
                self._channel_owners[runtime_name] = name
                logger.info("{} channel enabled as {}", channel.display_name, runtime_name)
            self._validate_allow_from()
            self._acquired = True
            return self
        except BaseException as exc:
            self.channels.clear()
            self._channel_owners.clear()
            rollback = asyncio.create_task(self._rollback_acquired_channels(acquired, exc))
            await _join_owned_cleanup(rollback)
            raise AssertionError("acquisition rollback returned without raising")

    async def _rollback_acquired_channels(
        self,
        acquired: list[BaseChannel],
        acquisition_exc: BaseException,
    ) -> None:
        cleanup_errors: list[BaseException] = []
        for channel in reversed(acquired):
            try:
                await channel.stop()
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if cleanup_errors:
            values = [acquisition_exc, *cleanup_errors]
            if all(isinstance(value, Exception) for value in values):
                raise ExceptionGroup(
                    "channel acquisition and rollback failed",
                    values,  # type: ignore[arg-type]
                )
            raise BaseExceptionGroup("channel acquisition and rollback failed", values)
        raise acquisition_exc

    def _mark_channel_error(self, owner: str, message: str) -> None:
        self._mark_runtime_error(
            (
                runtime_name
                for runtime_name, (runtime_owner, _instance_id)
                in self._channel_runtime_specs.items()
                if runtime_owner == owner
            ),
            message,
        )

    def _mark_runtime_error(self, runtime_names: Iterable[str], message: str) -> None:
        for runtime_name in runtime_names:
            self._channel_errors[runtime_name] = message

    @staticmethod
    def _validate_telegram_allow_from_config(
        config: Any,
        *,
        default_enabled: bool,
    ) -> None:
        if isinstance(config, dict):
            config_data = cast(dict[str, Any], config)
            enabled = config_data.get("enabled", default_enabled)
            if "allow_from" in config_data:
                allow = config_data.get("allow_from")
            else:
                allow = config_data.get("allowFrom")
        else:
            enabled = getattr(config, "enabled", default_enabled)
            allow = getattr(config, "allow_from", None)
        if enabled is not False and not allow:
            raise SystemExit(
                "Error: Telegram allowFrom requires at least one sender ID/username "
                'or "*". Pairing does not replace sender authorization.'
            )

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            cfg = ch.config
            if isinstance(cfg, dict):
                config_data = cast(dict[str, Any], cfg)
                if "allow_from" in config_data:
                    allow = config_data.get("allow_from")
                else:
                    allow = config_data.get("allowFrom")
            else:
                allow = getattr(cfg, "allow_from", None)

            owners = getattr(self, "_channel_owners", {})
            owner = owners.get(name, name)
            if owner == "telegram":
                self._validate_telegram_allow_from_config(
                    cfg,
                    default_enabled=True,
                )
            if allow is None:
                # allowFrom omitted → pairing-only mode.  Unapproved senders
                # receive a pairing code instead of being silently ignored.
                logger.info(
                    '"{}" has no allowFrom; unapproved users will receive a pairing code',
                    name,
                )

    def _should_send_progress(self, channel_name: str, *, tool_hint: bool = False) -> bool:
        """Return whether progress (or tool-hints) may be sent to *channel_name*."""
        ch = self.channels.get(channel_name)
        if ch is None:
            logger.debug("Progress check for unknown channel: {}", channel_name)
            return False
        return ch.send_tool_hints if tool_hint else ch.send_progress

    def _resolve_bool_override(self, section: Any, key: str, default: bool) -> bool:
        """Return *key* from *section* if it is a bool, otherwise *default*.

        For dict configs also checks the camelCase alias (e.g. ``sendProgress``
        for ``send_progress``) so raw JSON/TOML configs work alongside
        Pydantic models.
        """
        if isinstance(section, dict):
            section_data = cast(dict[str, Any], section)
            value = section_data.get(key)
            if value is None:
                camel = _BOOL_CAMEL_ALIASES.get(key)
                if camel:
                    value = section_data.get(camel)
            return value if isinstance(value, bool) else default
        value = getattr(section, key, None)
        return value if isinstance(value, bool) else default

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        errors = getattr(self, "_channel_errors", None)
        if errors is None:
            errors = self._channel_errors = {}
        errors.pop(name, None)
        try:
            await channel.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            public_error = channel.start_error_message(exc)
            errors[name] = public_error or "Channel failed to start. Check gateway logs."
            if public_error:
                logger.error("Failed to start channel {}: {}", name, public_error)
            else:
                logger.exception("Failed to start channel {}", name)
            raise

    def _start_channel_task(self, name: str, channel: BaseChannel) -> asyncio.Task[None]:
        logger.info("Starting {} channel...", name)
        task = asyncio.create_task(self._start_channel(name, channel))
        self._channel_tasks[name] = task
        return task

    async def _stop_channel(self, name: str) -> bool:
        return await self._stop_channel_runtime(name)

    async def _stop_channel_runtime(self, name: str) -> bool:
        channel = self.channels.get(name)
        if channel is None:
            self._channel_tasks.pop(name, None)
            return False
        stop_tasks = getattr(self, "_channel_stop_tasks", None)
        if stop_tasks is None:
            stop_tasks = self._channel_stop_tasks = {}
        previous = stop_tasks.get(name)
        if previous is not None and previous[0] is channel:
            if previous[1].done():
                return False
            return await _join_owned_cleanup(previous[1])

        # Keep the instance itself, not its recyclable object ID. One entry per
        # runtime slot also lets a replacement acquire independent ownership.
        task = self._channel_tasks.pop(name, None)
        cleanup = asyncio.create_task(self._finish_channel_stop(name, channel, task))
        stop_tasks[name] = (channel, cleanup)
        return await _join_owned_cleanup(cleanup)

    async def _finish_channel_stop(
        self, name: str, channel: BaseChannel, task: asyncio.Task[None] | None,
    ) -> bool:
        errors: list[BaseException] = []
        self._stopping_channels.add(name)
        try:
            await self._cancel_outbound(name)
        except BaseException as exc:
            errors.append(exc)
        try:
            await channel.stop()
            logger.info("Stopped {} channel", name)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                errors.append(exc)
            else:
                # Some channels await an already-cancelled server in stop().
                # This is not owner cancellation (the owner awaits shielded
                # cleanup), but its start task still has to be joined below.
                logger.debug("Channel {} stop task was already cancelled", name)
        except BaseException as exc:
            logger.exception("Error stopping {}", name)
            errors.append(exc)

        if task is not None and not task.done():
            task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )
        self._stopping_channels.discard(name)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            if all(isinstance(error, Exception) for error in errors):
                raise ExceptionGroup(
                    f"failed to stop {name} channel",
                    errors,  # type: ignore[arg-type]
                )
            raise BaseExceptionGroup(f"failed to stop {name} channel", errors)
        return True

    async def apply_channel_feature_action(
        self,
        action: str,
        name: str,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a WebUI channel enable/disable action without restarting the gateway.

        Returns a small transport-neutral result. ``handled=False`` means the
        optional feature is not a channel and should keep the default feature
        response semantics.
        """
        name = name.strip()
        instance_id = (instance_id or "").strip() or None
        if not name:
            return {"handled": False}

        from nanobot.channels.registry import discover_plugins

        plugin = discover_plugins({name}).get(name)
        if plugin is None:
            return {"handled": False}
        if "always_enabled" in plugin.capabilities:
            return {
                "handled": True,
                "ok": False,
                "requires_restart": True,
                "message": f"{plugin.display_name} is always enabled and is applied on restart.",
            }

        from nanobot.config.loader import load_config

        proposed_config = load_config()
        section = self._channel_section(
            name,
            config=proposed_config,
            default_enabled=plugin.default_enabled,
        )
        if action == "enable" and name == "telegram" and not plugin.management.multi_instance:
            self._validate_telegram_allow_from_config(
                section,
                default_enabled=plugin.default_enabled,
            )
        channel_setup_spec(name, plugin=plugin)
        instance_id = resolve_channel_action_target(instance_id)

        if action == "disable":
            self.config = proposed_config
            runtime_name = channel_runtime_name(plugin, instance_id)
            runtime_names = (
                [runtime_name]
                if self._channel_owners.get(runtime_name) == name
                else []
            )
            stopped = False
            for runtime_name in runtime_names:
                try:
                    stopped = await self._stop_channel(runtime_name) or stopped
                except BaseException:
                    logger.exception("Failed to stop {} channel after settings change", runtime_name)
                    self._channel_errors[runtime_name] = (
                        "Channel failed to stop. Check gateway logs."
                    )
                    return {
                        "handled": True,
                        "ok": False,
                        "requires_restart": False,
                        "message": f"{name} channel failed to stop. Check gateway logs.",
                    }
                self.channels.pop(runtime_name, None)
                self._channel_owners.pop(runtime_name, None)
                self._channel_stop_tasks.pop(runtime_name, None)
            self._channel_runtime_specs.pop(runtime_name, None)
            self._channel_errors.pop(runtime_name, None)
            return {
                "handled": True,
                "ok": True,
                "requires_restart": False,
                "message": f"{name} channel stopped." if stopped else f"{name} channel disabled.",
            }

        if action != "enable":
            return {"handled": True, "ok": False, "requires_restart": True}

        specs = channel_instance_specs(plugin, section) if section is not None else []
        specs = [spec for spec in specs if spec.instance_id == instance_id]
        if name == "telegram":
            for spec in specs:
                self._validate_telegram_allow_from_config(
                    spec.config,
                    default_enabled=True,
                )
        if not specs:
            return {
                "handled": True,
                "ok": False,
                "requires_restart": True,
                "message": f"{name} channel config was not enabled.",
            }

        self.config = proposed_config

        runtime_specs = [
            (channel_runtime_name(plugin, spec.instance_id), spec)
            for spec in specs
        ]
        collisions = [
            runtime_name
            for runtime_name, _spec in runtime_specs
            if (
                runtime_name in self.channels
                and self._channel_owners.get(runtime_name) != name
            )
        ]
        if collisions:
            return {
                "handled": True,
                "ok": False,
                "requires_restart": True,
                "message": (
                    "Channel runtime name(s) already owned by another channel: "
                    + ", ".join(sorted(collisions))
                ),
            }
        for runtime_name, spec in runtime_specs:
            self._channel_runtime_specs[runtime_name] = (name, spec.instance_id)

        try:
            cls = plugin.load_channel_class()
        except Exception:
            self._mark_runtime_error(
                (runtime_name for runtime_name, _spec in runtime_specs),
                "Channel runtime could not be loaded. Check gateway logs.",
            )
            return {
                "handled": True,
                "ok": False,
                "requires_restart": False,
                "message": f"{name} channel could not be loaded. Check gateway logs.",
            }

        try:
            built = [
                (
                    runtime_name,
                    self._build_channel(
                        name,
                        cls,
                        spec.config,
                        runtime_name=runtime_name,
                    ),
                )
                for runtime_name, spec in runtime_specs
            ]
        except Exception:
            self._mark_runtime_error(
                (runtime_name for runtime_name, _spec in runtime_specs),
                "Channel runtime could not be built. Check gateway logs.",
            )
            logger.exception("Failed to build {} channel after settings change", name)
            return {
                "handled": True,
                "ok": False,
                "requires_restart": False,
                "message": f"{name} channel could not be started. Check gateway logs.",
            }

        runtime_names_to_replace = {runtime_name for runtime_name, _channel in built}
        for runtime_name in sorted(runtime_names_to_replace):
            if runtime_name not in self.channels:
                continue
            try:
                await self._stop_channel(runtime_name)
            except BaseException:
                logger.exception("Failed to replace {} channel after settings change", runtime_name)
                self._channel_errors[runtime_name] = (
                    "Channel failed to stop. Check gateway logs."
                )
                return {
                    "handled": True,
                    "ok": False,
                    "requires_restart": False,
                    "message": f"{name} channel failed to stop. Check gateway logs.",
                }
            self.channels.pop(runtime_name, None)
            self._channel_owners.pop(runtime_name, None)
            self._channel_stop_tasks.pop(runtime_name, None)

        for runtime_name, channel in built:
            self.channels[runtime_name] = channel
            self._channel_owners[runtime_name] = name
            self._channel_errors.pop(runtime_name, None)
            if self._started:
                self._start_channel_task(runtime_name, channel)
            logger.info("{} channel applied without restart", runtime_name)
        if self._started:
            await asyncio.sleep(0)
        failed = [
            runtime_name
            for runtime_name, _channel in built
            if runtime_name in self._channel_errors
        ]
        return {
            "handled": True,
            "ok": not failed,
            "requires_restart": False,
            "message": (
                f"{cls.display_name} channel failed to start. Check gateway logs."
                if failed
                else f"{cls.display_name} channel applied without restart."
            ),
        }

    async def start_all(self) -> None:
        """Start all channels, rolling back every earlier start on failure."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        self._started = True
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())
        tasks: list[asyncio.Task[None]] = []
        attempted: list[str] = []
        supervisor: asyncio.Future[list[None]] | None = None
        try:
            for name, channel in self.channels.items():
                attempted.append(name)
                task = self._start_channel_task(name, channel)
                tasks.append(task)
                # Channel start coroutines normally remain pending. Give each
                # one a turn so immediate setup failures are transactional.
                await asyncio.sleep(0)
                if task.done():
                    task.result()
            self._restart_notice_task = self._notify_restart_done_if_needed()
            supervisor = asyncio.gather(*tasks)
            await asyncio.shield(supervisor)
        except BaseException as start_exc:
            current = asyncio.current_task()
            graceful_shutdown = (
                isinstance(start_exc, asyncio.CancelledError) and not self._started
                and current is not None and not current.cancelling()
            )
            rollback = asyncio.create_task(
                self._rollback_startup(
                    attempted,
                    supervisor,
                    start_exc,
                    graceful_shutdown=graceful_shutdown,
                ),
            )
            await _join_owned_cleanup(rollback)
            return

    async def _rollback_startup(
        self,
        attempted: list[str],
        supervisor: asyncio.Future[list[None]] | None,
        start_exc: BaseException,
        *,
        graceful_shutdown: bool,
    ) -> None:
        errors = await self._cancel_restart_notice()
        for name in reversed(attempted):
            try:
                await self._stop_channel_runtime(name)
            except BaseException as exc:
                errors.append(exc)
        if supervisor is not None:
            await asyncio.gather(supervisor, return_exceptions=True)
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            results = await asyncio.gather(self._dispatch_task, return_exceptions=True)
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )
            if not graceful_shutdown:
                self._dispatch_task = None
        self._started = False
        if errors:
            values = errors if graceful_shutdown else [start_exc, *errors]
            if len(values) == 1:
                raise values[0]
            if all(isinstance(value, Exception) for value in values):
                raise ExceptionGroup(
                    "channel start and rollback failed",
                    values,  # type: ignore[arg-type]
                )
            raise BaseExceptionGroup("channel start and rollback failed", values)
        if graceful_shutdown:
            return
        raise start_exc

    def _notify_restart_done_if_needed(self) -> asyncio.Task[None] | None:
        """Schedule restart completion after the target channel starts."""
        notice = consume_restart_notice_from_env()
        if not notice:
            return None
        return asyncio.create_task(self._send_restart_notice_when_started(notice))

    async def _cancel_restart_notice(self) -> list[BaseException]:
        task = getattr(self, "_restart_notice_task", None)
        self._restart_notice_task = None
        if task is None:
            return []
        if not task.done():
            task.cancel()
        # Native gather constructs a list of results/exceptions; the getattr
        # input loses that result type without changing the construction.
        results = cast(list[object], await asyncio.gather(task, return_exceptions=True))
        return [
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]

    async def _send_restart_notice_when_started(
        self,
        notice: RestartNotice,
        *,
        timeout_s: float = _RESTART_NOTICE_START_TIMEOUT_S,
        poll_s: float = _RESTART_NOTICE_START_POLL_S,
    ) -> None:
        """Deliver a restart notice after the target channel starts."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        target = self.channels.get(notice.channel)
        if target is None:
            logger.warning("Restart notice target channel is not enabled: {}", notice.channel)
            return
        if notice.channel == "websocket":
            # Reconnect and recovery are already represented by WebSocket
            # protocol state. A generic restart-complete notice must not
            # masquerade as a recovery transition and overwrite a real
            # awaiting-user checkpoint in connected clients.
            return

        while not target.is_running:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Restart notice target did not start: {}:{}",
                    notice.channel,
                    notice.chat_id,
                )
                return
            await asyncio.sleep(min(poll_s, remaining))

        await self._send_with_retry(
            target,
            OutboundMessage(
                channel=notice.channel,
                chat_id=notice.chat_id,
                content=format_restart_completed_message(notice.started_at_raw),
                metadata=dict(notice.metadata or {}),
            ),
            deadline=deadline,
        )

    async def stop_all(self) -> None:
        """Stop every owned resource once and surface ordered cleanup failures."""
        await _join_owned_cleanup(asyncio.create_task(self._stop_all_owned()))

    async def _stop_all_owned(self) -> None:
        logger.info("Stopping all channels...")
        self._started = False
        errors = await self._cancel_restart_notice()

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            results = await asyncio.gather(self._dispatch_task, return_exceptions=True)
            errors.extend(
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            )

        # Stop all channels
        for name in reversed(list(self.channels)):
            try:
                await self._stop_channel(name)
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors and all(isinstance(error, Exception) for error in errors):
            raise ExceptionGroup(
                "channel manager shutdown failed",
                errors,  # type: ignore[arg-type]
            )
        if errors:
            raise BaseExceptionGroup("channel manager shutdown failed", errors)

    @staticmethod
    def _fingerprint_content(content: str) -> str:
        normalized = " ".join(content.split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def _remember_origin_reply_fingerprint(
        self,
        key: tuple[str, str, str],
        fingerprint: str,
    ) -> None:
        self._origin_reply_fingerprints[key] = fingerprint
        self._origin_reply_fingerprints.move_to_end(key)
        while len(self._origin_reply_fingerprints) > ORIGIN_REPLY_FINGERPRINTS_MAX_SIZE:
            self._origin_reply_fingerprints.popitem(last=False)

    def _should_suppress_outbound(self, msg: OutboundMessage) -> bool:
        metadata = msg.metadata or {}
        if isinstance(msg.event, ProgressEvent):
            return False
        fingerprint = self._fingerprint_content(msg.content)
        if not fingerprint:
            return False

        origin_message_id = metadata.get("origin_message_id")
        if isinstance(origin_message_id, str) and origin_message_id:
            key = (msg.channel, msg.chat_id, origin_message_id)
            if self._origin_reply_fingerprints.get(key) == fingerprint:
                self._origin_reply_fingerprints.move_to_end(key)
                return True
            self._remember_origin_reply_fingerprint(key, fingerprint)

        message_id = metadata.get("message_id")
        if isinstance(message_id, str) and message_id:
            key = (msg.channel, msg.chat_id, message_id)
            self._remember_origin_reply_fingerprint(key, fingerprint)

        return False

    async def _cancel_outbound(self, channel: str | None = None) -> None:
        tasks = [task for task, key in self._outbound_tasks.items()
                 if channel is None or key[0] == channel]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _queue_outbound(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Bound scheduled sends; preserve FIFO per destination, not across chats.

        Retry waits occupy only their destination and one send slot. Saturation
        pauses dispatch rather than creating unbounded tasks or dropping messages.
        """
        await self._outbound_slots.acquire()
        if msg.channel in self._stopping_channels or self.channels.get(msg.channel) is not channel:
            self._outbound_slots.release()
            return
        key = (msg.channel, msg.chat_id)
        previous = self._outbound_tails.get(key)

        async def send() -> None:
            if previous is not None:
                await asyncio.shield(asyncio.gather(previous, return_exceptions=True))
            async with self._outbound_sends:
                await self._send_with_retry(channel, msg)

        task = asyncio.create_task(send(), name=f"outbound-{msg.channel}-{msg.chat_id}")
        self._outbound_tasks[task] = key
        self._outbound_tails[key] = task

        def finished(done: asyncio.Task[None]) -> None:
            self._outbound_tasks.pop(done, None)
            if self._outbound_tails.get(key) is done:
                self._outbound_tails.pop(key, None)
            self._outbound_slots.release()
            if not done.cancelled() and (error := done.exception()) is not None:
                logger.error("Outbound delivery to {}:{} failed: {}", *key, error)

        task.add_done_callback(finished)

    async def _dispatch_outbound(self) -> None:
        try:
            await self._dispatch_outbound_loop()
        finally:
            await self._cancel_outbound()

    async def _dispatch_outbound_loop(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        # Buffer for messages that couldn't be processed during delta coalescing
        # (since asyncio.Queue doesn't support push_front)
        pending: list[OutboundMessage] = []

        while True:
            try:
                # First check pending buffer before waiting on queue
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await asyncio.wait_for(
                        self.bus.consume_outbound(),
                        timeout=1.0
                    )

                event = msg.event
                progress_event = event if isinstance(event, ProgressEvent) else None
                if (progress_event and not progress_event.tool_hint
                        and self.config.channels.task_update_mode != "verbose"):
                    continue
                if progress_event and (
                    progress_event.reasoning_delta
                    or progress_event.reasoning_end
                    or progress_event.reasoning
                ):
                    # Reasoning rides its own plugin channel: only delivered
                    # when the destination channel opts in via ``show_reasoning``
                    # and overrides the streaming primitives. Channels without
                    # a low-emphasis UI affordance keep the base no-op and the
                    # content silently drops here.
                    channel = self.channels.get(msg.channel)
                    if channel is not None and channel.show_reasoning:
                        await self._queue_outbound(channel, msg)
                    continue

                if progress_event:
                    if progress_event.tool_hint and not self._should_send_progress(
                        msg.channel, tool_hint=True,
                    ):
                        continue
                    if not progress_event.tool_hint and not self._should_send_progress(
                        msg.channel, tool_hint=False,
                    ):
                        continue

                if isinstance(event, RetryWaitEvent):
                    continue

                if (
                    isinstance(event, RuntimeModelUpdatedEvent)
                    and msg.channel == "websocket"
                    and "websocket" not in self.channels
                ):
                    continue

                # Coalesce consecutive stream delta messages for the same (channel, chat_id)
                # to reduce API calls and improve streaming latency
                if isinstance(event, StreamDeltaEvent):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)
                    event = msg.event

                channel = self.channels.get(msg.channel)
                if channel:
                    # Duplicate suppression is scoped to a known source message
                    # so repeated content from separate turns is still delivered.
                    if (
                        not isinstance(
                            event,
                            StreamDeltaEvent | StreamEndEvent | StreamedResponseEvent,
                        )
                    ):
                        if self._should_suppress_outbound(msg):
                            logger.info("Suppressing duplicate outbound message to {}:{}", msg.channel, msg.chat_id)
                            continue
                    await self._queue_outbound(channel, msg)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _send_reasoning_delta(
        channel: BaseChannel,
        msg: OutboundMessage,
        event: ProgressEvent,
    ) -> None:
        await channel.send_reasoning_delta(
            msg.chat_id,
            msg.content,
            msg.metadata,
            stream_id=event.stream_id,
        )

    @staticmethod
    async def _send_reasoning_end(
        channel: BaseChannel,
        msg: OutboundMessage,
        event: ProgressEvent,
    ) -> None:
        await channel.send_reasoning_end(
            msg.chat_id,
            msg.metadata,
            stream_id=event.stream_id,
        )

    @staticmethod
    async def _send_stream_event(
        channel: BaseChannel,
        msg: OutboundMessage,
        event: StreamDeltaEvent | StreamEndEvent,
    ) -> None:
        kwargs: dict[str, Any] = {
            "stream_id": event.stream_id,
            "stream_end": isinstance(event, StreamEndEvent),
            "resuming": event.resuming if isinstance(event, StreamEndEvent) else False,
        }
        if isinstance(event, StreamEndEvent) and event.merge_next:
            try:
                signature = inspect.signature(channel.send_delta)
                if (
                    "merge_next" in signature.parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values()
                    )
                ):
                    kwargs["merge_next"] = True
            except (TypeError, ValueError):
                pass
        await channel.send_delta(
            msg.chat_id,
            msg.content,
            msg.metadata,
            **kwargs,
        )

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send one outbound message without retry policy."""
        event = msg.event
        if isinstance(event, ProgressEvent) and event.reasoning_end:
            await ChannelManager._send_reasoning_end(channel, msg, event)
        elif isinstance(event, ProgressEvent) and event.reasoning_delta:
            await ChannelManager._send_reasoning_delta(channel, msg, event)
        elif isinstance(event, ProgressEvent) and event.reasoning:
            # BaseChannel translates one-shot reasoning to a single delta +
            # end pair so plugins only implement the streaming primitives.
            await channel.send_reasoning(msg)
        elif isinstance(event, ProgressEvent) and event.file_edit_events:
            await channel.send_file_edit_events(
                msg.chat_id,
                event.file_edit_events,
                msg.metadata,
            )
        elif isinstance(event, StreamDeltaEvent):
            await ChannelManager._send_stream_event(channel, msg, event)
        elif isinstance(event, StreamEndEvent):
            await ChannelManager._send_stream_event(channel, msg, event)
        elif not isinstance(event, StreamedResponseEvent):
            await channel.send(msg)

    def _coalesce_stream_deltas(
        self, first_msg: OutboundMessage
    ) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """Merge consecutive stream deltas for the same (channel, chat_id, stream_id).

        This reduces the number of API calls when the queue has accumulated multiple
        deltas, which happens when LLM generates faster than the channel can process.

        Returns:
            tuple of (merged_message, list_of_non_matching_messages)
        """
        first_event = first_msg.event
        first_stream_id = first_event.stream_id if isinstance(first_event, StreamDeltaEvent) else None
        target_key = (first_msg.channel, first_msg.chat_id, first_stream_id)
        combined_content = first_msg.content
        final_event: StreamDeltaEvent | StreamEndEvent = (
            first_event
            if isinstance(first_event, StreamDeltaEvent)
            else StreamDeltaEvent(stream_id=first_stream_id)
        )
        non_matching: list[OutboundMessage] = []

        # Only merge consecutive deltas. As soon as we hit any other message,
        # stop and hand that boundary back to the dispatcher via `pending`.
        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Check if this message belongs to the same stream
            next_event = next_msg.event
            next_stream_id = (
                next_event.stream_id
                if isinstance(next_event, StreamDeltaEvent | StreamEndEvent)
                else None
            )
            same_target = (
                next_msg.channel,
                next_msg.chat_id,
                next_stream_id,
            ) == target_key
            is_delta = isinstance(next_event, StreamDeltaEvent)
            is_end = isinstance(next_event, StreamEndEvent)

            if same_target and (is_delta or (is_end and next_msg.content)):
                # Accumulate content
                combined_content += next_msg.content
                # If we see stream_end, remember it and stop coalescing this stream
                if isinstance(next_event, StreamEndEvent):
                    final_event = StreamEndEvent(
                        stream_id=next_stream_id,
                        resuming=next_event.resuming,
                        merge_next=next_event.merge_next,
                    )
                    # Stream ended - stop coalescing this stream
                    break
            else:
                # First non-matching message defines the coalescing boundary.
                non_matching.append(next_msg)
                break

        merged = replace_outbound_event(first_msg, final_event, content=combined_content)
        return merged, non_matching

    async def _send_with_retry(
        self,
        channel: BaseChannel,
        msg: OutboundMessage,
        *,
        deadline: float | None = None,
    ) -> None:
        """Send a message with retry on failure using exponential backoff.

        When deadline is provided, retry until that monotonic time instead of
        stopping at the configured attempt limit.

        Note: CancelledError is re-raised to allow graceful shutdown.
        """
        if isinstance(msg.event, (StreamDeltaEvent, StreamEndEvent)):
            msg = replace(msg, metadata={**msg.metadata, "_delivery_id": uuid.uuid4().hex})
        max_attempts = max(self.config.channels.send_max_retries, 1)
        attempt = 0

        while True:
            attempt += 1
            try:
                await self._send_once(channel, msg)
                return  # Send succeeded
            except asyncio.CancelledError:
                raise  # Propagate cancellation for graceful shutdown
            except Exception as e:
                if not channel.should_retry_send_error(e):
                    logger.error(
                        "Send to {} failed with a non-retryable {}: {}",
                        msg.channel,
                        type(e).__name__,
                        e,
                    )
                    return
                loop = asyncio.get_running_loop()
                exhausted = (
                    attempt >= max_attempts
                    if deadline is None
                    else loop.time() >= deadline
                )
                if exhausted:
                    logger.exception(
                        "Failed to send to {} after {} attempts",
                        msg.channel, attempt,
                    )
                    return
                delay = _SEND_RETRY_DELAYS[min(attempt - 1, len(_SEND_RETRY_DELAYS) - 1)]
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - loop.time()))
                attempt_label = str(attempt)
                if deadline is None:
                    attempt_label = f"{attempt}/{max_attempts}"
                logger.warning(
                    "Send to {} failed (attempt {}): {}, retrying in {}s",
                    msg.channel, attempt_label, type(e).__name__, delay,
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise  # Propagate cancellation during sleep

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Return actual runtime state, including enabled runtimes that failed."""
        owners = getattr(self, "_channel_owners", {})
        runtime_specs = dict(getattr(self, "_channel_runtime_specs", {}))
        for runtime_name in self.channels:
            runtime_specs.setdefault(
                runtime_name,
                (owners.get(runtime_name, runtime_name), "default"),
            )
        tasks = getattr(self, "_channel_tasks", {})
        errors = getattr(self, "_channel_errors", {})
        status: dict[str, Any] = {}
        for runtime_name, (owner, instance_id) in runtime_specs.items():
            channel = self.channels.get(runtime_name)
            task = tasks.get(runtime_name)
            error = errors.get(runtime_name)
            running = bool(channel and channel.is_running)
            if error:
                state = "failed"
            elif running:
                state = "running"
            elif task is not None and not task.done():
                state = "starting"
            else:
                state = "stopped"
            status[runtime_name] = {
                "enabled": True,
                "running": running,
                "state": state,
                "owner": owner,
                "instance_id": instance_id,
            }
            if error:
                status[runtime_name]["error"] = error
        return status

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
