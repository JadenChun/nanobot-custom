"""Resolve persisted image configuration source without losing legacy defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.agent.tools.image_generation import ImageGenerationToolConfig
    from nanobot.config.schema import ImageConfig, ProviderConfig, ToolsConfig


@dataclass(frozen=True)
class EffectiveImageConfiguration:
    config: ImageGenerationToolConfig
    providers: dict[str, ProviderConfig]
    legacy: ImageConfig | None = None


def effective_image_configuration(tools: ToolsConfig, providers: dict[str, ProviderConfig] | None = None) -> EffectiveImageConfiguration:
    from nanobot.agent.tools.image_generation import ImageGenerationToolConfig
    from nanobot.config.schema import ProviderConfig
    from nanobot.providers.codex_cli_image import resolve_command

    if tools.image_generation.configuration_source == 'upstream':
        return EffectiveImageConfiguration(tools.image_generation, dict(providers or {}))
    legacy = tools.image
    provider = (legacy.provider or 'auto').lower()
    key = legacy.api_key or os.environ.get('OPENROUTER_API_KEY', '')
    if provider == 'auto':
        provider = 'codex_cli' if resolve_command(legacy.codex_command) else 'openrouter'
    if provider in {'codex', 'codex-cli', 'codex_cli'}:
        provider = 'codex_cli'
    enabled = provider == 'codex_cli' or bool(key)
    effective = ImageGenerationToolConfig(
        configuration_source='legacy', enabled=enabled, provider=provider,
        model=legacy.codex_model if provider == 'codex_cli' else legacy.model,
        default_aspect_ratio='16:9', default_image_size='',
    )
    return EffectiveImageConfiguration(effective, {
        'openrouter': ProviderConfig(api_key=key, api_base=legacy.api_base),
    }, legacy)
