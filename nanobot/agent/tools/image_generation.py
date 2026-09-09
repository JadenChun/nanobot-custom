"""Image generation tool."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger
from pydantic import Field

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.agent.tools.image_configuration import effective_image_configuration
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import (
    ArraySchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.agent.turn import TurnContext
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_ACK,
    RUNTIME_CONTROL_IMAGE_GENERATION_RELOAD,
    InboundMessage,
)
from nanobot.bus.queue import MessageBus
from nanobot.config.paths import get_media_dir
from nanobot.config_base import Base
from nanobot.providers.image_generation import (
    ImageGenerationError,
    ImageGenerationProvider,
    get_image_gen_provider,
    image_gen_provider_configs,
)
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.security.workspace_policy import WorkspaceBoundaryError, resolve_allowed_path
from nanobot.utils.artifacts import (
    ArtifactError,
    generated_image_tool_result,
    store_generated_image_artifact,
)
from nanobot.utils.helpers import detect_image_mime

if TYPE_CHECKING:
    from nanobot.agent.tools.context import ToolContext
    from nanobot.config.schema import ImageConfig, ProviderConfig


class ImageGenerationToolConfig(Base):
    """Image generation tool configuration."""
    configuration_source: Literal["legacy", "upstream"] = "upstream"
    enabled: bool = False
    provider: str = "openrouter"
    model: str = "openai/gpt-5.4-image-2"
    default_aspect_ratio: str = "1:1"
    default_image_size: str = "1K"
    max_images_per_turn: int = Field(default=4, ge=1, le=8)
    save_dir: str = "generated"


@tool_parameters(
    tool_parameters_schema(
        prompt=StringSchema(
            "Detailed image generation or edit prompt. Include style, subject, composition, colors, and constraints.",
            min_length=1,
        ),
        output_path=StringSchema("Optional legacy output file destination; requires count=1."),
        reference_images=ArraySchema(
            StringSchema("Local path of an existing image artifact or user-provided image to use as an edit reference."),
            description="Optional local image paths. Use generated artifact paths for iterative edits.",
        ),
        aspect_ratio=StringSchema(
            "Optional output aspect ratio, e.g. 1:1, 16:9, 9:16, 4:3.",
        ),
        image_size=StringSchema(
            "Optional output size hint supported by the configured provider, e.g. 1K, 2K, 4K, or 1024x1024.",
        ),
        count=IntegerSchema(
            description="Number of images to generate in this turn.",
            minimum=1,
            maximum=8,
        ),
        required=["prompt"],
    )
)
class ImageGenerationTool(Tool):
    """Generate persistent image artifacts through the configured image provider."""

    config_key = "image_generation"

    @classmethod
    def config_cls(cls):
        return ImageGenerationToolConfig

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return effective_image_configuration(ctx.config, ctx.image_generation_provider_configs).config.enabled

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        effective = effective_image_configuration(ctx.config, ctx.image_generation_provider_configs)
        return cls(
            workspace=ctx.workspace, config=effective.config, provider_configs=effective.providers,
            legacy_config=effective.legacy, write_tool=WriteFileTool.create(ctx),
        )

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: ImageGenerationToolConfig,
        provider_config: ProviderConfig | None = None,
        provider_configs: dict[str, ProviderConfig] | None = None,
        legacy_config: ImageConfig | None = None,
        write_tool: WriteFileTool | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser()
        self.config = config
        self.legacy_config = legacy_config
        self._write_tool = write_tool or WriteFileTool(workspace=self.workspace, allowed_dir=self.workspace)
        self.provider_configs = dict(provider_configs or {})
        if provider_config is not None and "openrouter" not in self.provider_configs:
            self.provider_configs["openrouter"] = provider_config

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Generate or edit images and store them as persistent artifacts. "
            "Returns artifact ids and local paths. For edits, pass prior generated image paths "
            "or user image paths as reference_images."
        )

    def _provider_config(self) -> ProviderConfig | None:
        return self.provider_configs.get(self.config.provider)

    def _provider_client(self) -> ImageGenerationProvider | None:
        if self.legacy_config is not None and self.config.provider == "codex_cli":
            from nanobot.providers.codex_cli_image import CodexCLIImageGenerationClient
            return CodexCLIImageGenerationClient(self.legacy_config)
        provider = self._provider_config()
        cls = get_image_gen_provider(self.config.provider)
        if cls is None:
            return None
        kwargs: dict[str, Any] = {
            "api_key": provider.api_key if provider and isinstance(provider.api_key, str) else None,
            "api_base": provider.api_base if provider and isinstance(provider.api_base, str) else None,
            "extra_headers": provider.extra_headers
            if provider and isinstance(provider.extra_headers, dict) else None,
            "extra_body": provider.extra_body
            if provider and isinstance(provider.extra_body, dict) else None,
            "proxy": provider.proxy if provider and isinstance(provider.proxy, str) else None,
        }
        if self.legacy_config is not None and self.config.provider == "openrouter":
            kwargs["legacy_request"] = True
        return cls(**kwargs)

    def _resolve_reference_image(self, value: str) -> str:
        access = current_tool_workspace(self.workspace, restrict_to_workspace=True)
        workspace = access.project_path or self.workspace
        try:
            resolved = resolve_allowed_path(
                value,
                workspace=workspace,
                allowed_root=access.allowed_root,
                extra_allowed_roots=[get_media_dir()] if access.allowed_root is not None else None,
                strict=True,
            )
        except WorkspaceBoundaryError as exc:
            raise ImageGenerationError(
                "reference_images must be inside the workspace or nanobot media directory"
            ) from exc
        except OSError as exc:
            raise ImageGenerationError(f"reference image not found: {value}") from exc
        if not resolved.is_file():
            raise ImageGenerationError(f"reference image is not a file: {value}")
        resolved = self._write_tool._resolve_read(str(resolved))
        raw = resolved.read_bytes()
        if detect_image_mime(raw) is None:
            raise ImageGenerationError(f"unsupported reference image: {value}")
        return str(resolved)

    def _resolve_reference_images(self, values: list[str] | None) -> list[str]:
        if not values:
            return []
        return [self._resolve_reference_image(value) for value in values if value]

    async def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        count: int | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        requested = 1 if count is None else count
        if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= self.config.max_images_per_turn:
            return ToolResult.error("Error: count is outside tools.imageGeneration.maxImagesPerTurn")
        if output_path is not None and requested != 1:
            return ToolResult.error("Error: output_path requires count=1")
        artifacts: list[dict[str, Any]] = []
        try:
            destination = self._output_destination(output_path) if output_path is not None else None
            refs = self._resolve_reference_images(reference_images)
            client = self._provider_client()
            if client is None:
                return ToolResult.error(f"Error: unsupported image generation provider '{self.config.provider}'")
            while len(artifacts) < requested:
                response = await client.generate(
                    prompt=prompt,
                    model=self.config.model,
                    reference_images=refs,
                    aspect_ratio=aspect_ratio or self.config.default_aspect_ratio,
                    image_size=image_size or self.config.default_image_size,
                )
                if not response.images:
                    raise ImageGenerationError("image provider returned no images")
                for image_data_url in response.images:
                    artifact = store_generated_image_artifact(
                        image_data_url,
                        prompt=prompt,
                        model=self.config.model,
                        source_images=refs,
                        save_dir=self.config.save_dir,
                        provider=self.config.provider,
                    )
                    artifacts.append(artifact)
                    if len(artifacts) >= requested:
                        break
            result = json.loads(generated_image_tool_result(artifacts))
            if destination is not None:
                await self._publish_output(output_path, destination, Path(artifacts[0]["path"]))
                result.update(path=str(destination), size_bytes=destination.stat().st_size)
                if self.config.provider == "codex_cli":
                    result.update({key: response.raw[key] for key in ("provider", "source_path", "prompt")
                                   if key in response.raw})
            return json.dumps(result, ensure_ascii=False)
        except (ArtifactError, ImageGenerationError, OSError, ValueError) as exc:
            retained = f"; retained artifacts: {json.dumps(artifacts)}" if artifacts else ""
            return ToolResult.error(f"Error: {exc}{retained}")

    def _output_destination(self, value: str) -> Path:
        if not value.strip():
            raise ImageGenerationError("output_path must not be empty")
        requested = Path(value).expanduser()
        workspace = self.workspace.resolve()
        generated = workspace / "generated_images"
        if not requested.is_absolute():
            requested = generated / requested.name
        elif requested.is_relative_to(workspace) and not requested.is_relative_to(generated):
            requested = generated / requested.name
        return self._write_tool._resolve_write(str(requested))

    async def _publish_output(self, value: str, destination: Path, artifact: Path) -> None:
        async with self._write_tool._mutation_guard(destination):
            if self._output_destination(value) != destination:
                raise ImageGenerationError("output destination changed during generation")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(artifact.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                self._write_tool._record_touch(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> Any:
        token = self._write_tool._lock_owner_var.set(context.lock_owner or context.run_id)
        try:
            return await self.execute(**kwargs)
        finally:
            self._write_tool._lock_owner_var.reset(token)


async def reload_image_generation_tool(state: Any, registry: ToolRegistry) -> dict[str, Any]:
    """Apply the persisted image configuration to the running agent."""
    try:
        from nanobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config())
        effective = effective_image_configuration(config.tools, image_gen_provider_configs(config))
        tool_config = effective.config
        provider_configs = effective.providers
    except Exception as exc:
        logger.warning("Image generation hot reload could not read config: {}", exc)
        return {
            "ok": False,
            "message": "Could not reload image generation config.",
            "requires_restart": True,
            "error": str(exc),
        }

    previous = registry.get("generate_image")
    if previous is not None and not isinstance(previous, ImageGenerationTool):
        raise ValueError("Image reload conflicts with a noncanonical tool")
    from nanobot.agent.tools.context import ToolContext

    dependencies = getattr(state, "_fork_tool_dependencies", None)
    if previous is None and dependencies is None and getattr(state, "resource_policy", None) is not None:
        return {"ok": False, "requires_restart": True,
                "message": "Image reload requires the owning tool construction dependencies."}
    writer = previous._write_tool if previous else WriteFileTool.create(ToolContext(
        config=config.tools, workspace=str(state.workspace), fork_dependencies=dependencies,
        file_state_store=getattr(state, "_file_state_store", None),
    ))
    next_tool = (
        ImageGenerationTool(
            workspace=state.workspace, config=tool_config, provider_configs=provider_configs,
            legacy_config=effective.legacy, write_tool=writer,
        ) if tool_config.enabled else None
    )

    state.tools_config.image_generation = tool_config
    state._image_generation_provider_configs = provider_configs
    if next_tool is not None:
        if previous is None:
            registry.register(next_tool)
        else:
            registry.replace(next_tool)
    else:
        registry.unregister("generate_image")

    logger.info(
        "Image generation config reloaded: enabled={} provider={} model={}",
        tool_config.enabled,
        tool_config.provider,
        tool_config.model,
    )
    return {
        "ok": True,
        "message": "Image generation settings applied without restarting nanobot.",
        "enabled": tool_config.enabled,
        "provider": tool_config.provider,
        "model": tool_config.model,
        "requires_restart": False,
    }


async def request_image_generation_reload(
    bus: MessageBus,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Ask the running agent loop to refresh its image generation tool."""
    loop = asyncio.get_running_loop()
    ack: asyncio.Future[dict[str, Any]] = loop.create_future()
    await bus.publish_inbound(
        InboundMessage(
            channel="system",
            sender_id="webui-settings",
            chat_id="runtime",
            content=RUNTIME_CONTROL_IMAGE_GENERATION_RELOAD,
            metadata={
                INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_IMAGE_GENERATION_RELOAD,
                RUNTIME_CONTROL_ACK: ack,
            },
        )
    )
    try:
        result = await asyncio.wait_for(ack, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "message": "Image generation hot reload timed out.",
            "requires_restart": True,
        }
    if not isinstance(cast(object, result), dict):
        return {
            "ok": False,
            "message": "Image generation hot reload returned an unexpected response.",
            "requires_restart": True,
        }
    return result


async def handle_runtime_control(
    state: Any,
    msg: InboundMessage,
    registry: ToolRegistry,
) -> bool:
    """Handle an in-process image generation reload request."""
    metadata = msg.metadata
    if metadata.get(INBOUND_META_RUNTIME_CONTROL) != RUNTIME_CONTROL_IMAGE_GENERATION_RELOAD:
        return False

    ack = metadata.get(RUNTIME_CONTROL_ACK)
    try:
        result = await reload_image_generation_tool(state, registry)
    except Exception as exc:
        logger.exception("Image generation hot reload failed")
        result = {
            "ok": False,
            "message": "Image generation hot reload failed.",
            "requires_restart": True,
            "error": str(exc),
        }
    if isinstance(ack, asyncio.Future) and not ack.done():
        cast(asyncio.Future[Any], ack).set_result(result)
    return True
