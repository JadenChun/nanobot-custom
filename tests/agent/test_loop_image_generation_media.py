from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.config.loader import set_config_path
from nanobot.config.schema import ImageGenerationToolConfig, ProviderConfig, ToolsConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.providers.image_generation import GeneratedImageResponse
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.test_migration_engine import ScriptedProvider

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeImageClient:
    def __init__(self, **kwargs: Any) -> None:
        pass

    async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
        return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})


@pytest.mark.asyncio
async def test_outbound_no_longer_carries_generated_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_async: AsyncAdmittedFactory,
) -> None:
    """Media delivery is now the LLM's responsibility via the message tool."""
    set_config_path(tmp_path / "config.json")
    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name: FakeImageClient if name == "openrouter" else None,
    )
    provider = ScriptedProvider([
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call_img",
                        name="generate_image",
                        arguments={"prompt": "draw a tiny icon"},
                    )
                ],
            ),
            LLMResponse(content="Done", finish_reason="stop"),
    ])
    provider.generation = replace(provider.generation, max_tokens=4096)

    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            restrict_to_workspace=True, session_ttl_minutes=0,
            tools_config=ToolsConfig(
                image_generation=ImageGenerationToolConfig(enabled=True),
            ),
            image_generation_provider_config=ProviderConfig(api_key="sk-or-test"),
        )

    loop = await admitted_async(tmp_path, build)

    result = await loop.process_direct(
        "draw an icon", session_key="websocket:chat-image",
        channel="websocket", sender_id="user", chat_id="chat-image",
    )

    assert result is not None
    assert result.content == "Done"
    # OutboundMessage no longer carries generated media —
    # the LLM sends images via the message tool instead.
    assert result.media == []


def test_image_loop_construction_rejects_pending_startup_receipt(tmp_path: Path) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
    from nanobot.bus.queue import MessageBus

    workspace = tmp_path / "workspace"
    pending = workspace / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")
    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=MessageBus(), provider=ScriptedProvider([]), workspace=workspace,
            sessions_root=tmp_path / "sessions",
        )
