"""Effect-fenced message tests using the canonical turn and owned graph."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.loop import AgentLoop, _AdmittedTurnIO, _TurnExecutionOptions
from nanobot.agent.turn import DeliveryTarget, TurnRequest, TurnResult
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import GenerationSettings, LLMProvider
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory


async def make_loop(
    root: Path, acquire: AsyncAdmittedFactory, *, keys: tuple[str, ...] = (),
) -> AgentLoop:
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False

    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    return await acquire(root, build, keys=keys)


async def execute_message(
    loop: AgentLoop, message: InboundMessage, *, ephemeral: bool = False,
    session_key: str | None = None,
    pending_queue: asyncio.Queue[InboundMessage] | None = None,
) -> TurnResult:
    @asynccontextmanager
    async def queue_scope(key: str) -> AsyncIterator[_AdmittedTurnIO]:
        yield _AdmittedTurnIO(
            pending_queue=pending_queue,
            delivery=loop.turn_delivery_factory.create(message, key),
        )

    return await loop.execute_turn(
        TurnRequest(
            content=message.content, session_key=session_key or message.session_key,
            sender_id=message.sender_id, source=loop._inbound_turn_source(message),
            media=tuple(message.media), metadata=dict(message.metadata),
            route=DeliveryTarget(channel=message.channel, chat_id=message.chat_id),
        ),
        _options=_TurnExecutionOptions(
            inbound=message, ephemeral=ephemeral, raise_errors=True,
            trusted_foreground_activity=loop._trusted_foreground_inbound(message),
            admission_scope=queue_scope if pending_queue is not None else None,
        ),
    )
