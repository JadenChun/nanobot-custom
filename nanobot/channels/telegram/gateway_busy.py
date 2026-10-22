"""Explicit gateway composition for the selected regular Telegram busy inbox."""

from __future__ import annotations

import asyncio

from nanobot.agent.busy_control import BusyControlRouter, ProviderControlModel
from nanobot.agent.loop import AgentLoop
from nanobot.channels.telegram.busy_inbox import TelegramBusyInbox
from nanobot.channels.telegram.runtime import TelegramChannel
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import build_provider_snapshot
from nanobot.session.busy_inbox import SessionBusyInbox


class GatewayTelegramBusy:
    def __init__(self, config: Config, agent: AgentLoop, channel: TelegramChannel, *,
                 control_provider: LLMProvider | None = None) -> None:
        options = config.gateway.telegram_busy
        if not options.enabled:
            raise ValueError("Telegram busy input requires explicit enablement")
        if config.agents.defaults.unified_session:
            raise ValueError("selected Telegram busy input requires unifiedSession=false; cross-channel ownership is not enabled")
        owners = set(options.owner_sender_ids)
        if not channel.config.allow_from or any(value not in owners for value in channel.config.allow_from):
            raise ValueError("Telegram allowFrom must contain only the configured busy owner numeric IDs")
        if not agent.sessions.get_or_create(options.session_key).policy.persist:
            raise ValueError("Telegram busy input requires a persistent regular session")
        self.options, self.agent, self.channel = options.model_copy(deep=True), agent, channel
        selected = set(agent.tools.tool_names)

        def scope(names: list[str] | None) -> tuple[str, ...]:
            result = tuple(sorted(selected)) if names is None else tuple(names)
            if len(set(result)) != len(result) or not set(result).issubset(selected):
                raise ValueError("Telegram busy scope contains duplicate or unavailable tools")
            return result

        self.tool_scope, self.automation_scope = scope(options.tool_scope), scope(options.automation_scope)
        if control_provider is None:
            control_config = config.model_copy(deep=True)
            control_config.agents.defaults.fallback = []
            control_config.agents.defaults.fallback_models = []
            preset = ModelPresetConfig(model=options.control_model, provider=options.control_provider,
                                       max_tokens=options.limits.max_output_tokens, temperature=0)
            control_provider = build_provider_snapshot(control_config, preset=preset).provider
        if control_provider is agent.provider:
            raise ValueError("busy control requires a distinct provider instance")
        self.router = BusyControlRouter(ProviderControlModel(control_provider, options.control_model), options.limits)
        self.edge: TelegramBusyInbox | None = None
        self._ready, self._failed = asyncio.Event(), asyncio.Event()
        self._failure: Exception | None = None
        channel.install_authenticated_startup(self.on_authenticated)

    async def on_authenticated(self, channel: TelegramChannel) -> None:
        try:
            account_id = channel.authenticated_account_id
            if channel is not self.channel or account_id is None:
                raise ValueError("Telegram busy input requires the authenticated configured account")
            if self.edge is not None:
                if self.edge.account_id != account_id or not channel.owns_busy_inbox(self.edge):
                    raise ValueError("Telegram busy owner changed during channel restart")
                return
            inbox = SessionBusyInbox(self.agent.sessions, owner_id="telegram:" + account_id,
                session_key=self.options.session_key, limits=self.options.inbox)
            self.edge = self.agent.register_telegram_busy_inbox(channel, inbox, self.router,
                account_id=account_id, owner_sender_ids=frozenset(self.options.owner_sender_ids),
                owner_chat_ids=frozenset(self.options.owner_chat_ids),
                tool_scope=self.tool_scope, automation_scope=self.automation_scope)
            self.edge.consumer.wake()
            self.edge.wake_notices()
            self._ready.set()
        except Exception as exc:
            self._failure = exc
            self._failed.set()
            self._ready.set()
            raise

    async def run(self) -> None:
        async with asyncio.timeout(60):
            await self._ready.wait()
        await self._failed.wait()
        assert self._failure is not None
        raise self._failure

    async def aclose(self) -> None:
        self.channel.remove_authenticated_startup(self.on_authenticated)
        # AgentLoop owns the registered edge/consumer and closes them through
        # its existing lifecycle. Close also handles pre-registration failure.
        await self.router.aclose()
