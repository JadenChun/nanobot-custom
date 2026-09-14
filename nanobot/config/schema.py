"""Configuration schema using Pydantic."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from nanobot.config.timezone import detect_system_timezone
from nanobot.config_base import Base
from nanobot.cron.types import CronSchedule

if TYPE_CHECKING:
    from nanobot.agent.tools.cli_apps import CliAppsToolConfig
    from nanobot.agent.tools.filesystem import FileToolsConfig
    from nanobot.agent.tools.image_generation import ImageGenerationToolConfig
    from nanobot.agent.tools.self import MyToolConfig
    from nanobot.agent.tools.shell import ExecToolConfig
    from nanobot.agent.tools.web import WebToolsConfig


class MaxTokensConfig(Base):
    """Token limits configuration."""

    input: int = 120000
    output: int = 4096

class FallbackEntry(Base):
    """One entry in the provider fallback chain."""

    provider: str  # Provider name, e.g. "gemini"
    model: str

class ContextRepoConfig(Base):
    """Optional advanced overrides for an attached context repository."""

    path: str
    read_only: bool = False
    auto_sync: bool = True
    credential_profile: str | None = None
    manifest: str = "nanobot.context.json"
    target_repo_paths: dict[str, str] = Field(default_factory=dict)

class CrawlerAgentConfig(Base):
    """Dedicated low-cost model used by the foreground crawler role."""

    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    max_tokens: MaxTokensConfig = Field(
        default_factory=lambda: MaxTokensConfig(input=20000, output=2000)
    )
    max_tool_iterations: int = Field(default=20, ge=1, le=100)
    reasoning_effort: str | None = "low"

class AgentBrowserConfig(Base):
    """Agent Browser tool configuration."""

    enabled: bool = True
    package: str = "agent-browser"
    timeout: int = 180
    max_output_chars: int = 12000

class AgentDeviceConfig(Base):
    """Agent Device tool configuration."""

    enabled: bool = True
    package: str = "agent-device"
    timeout: int = 180
    max_output_chars: int = 12000

class DesktopUseConfig(Base):
    """Desktop-use tool configuration (pure-Python, stateless).

    Stateless desktop driver: no daemon, no AX tree. Each call is a
    one-shot OS-level event (CGEvent on macOS). The model works from
    screenshots + pixel coordinates, like Anthropic's ``computer-use-demo``
    and OpenAI's Operator agent.

    Disabled by default. macOS requires the PyObjC frameworks
    (``pip install 'nanobot-ai[desktop]'``) plus Accessibility and Screen
    Recording permissions in System Settings > Privacy & Security.
    Windows / Linux backends are not yet implemented.
    """

    enabled: bool = False
    screenshot_delay: float = 0.5  # seconds to wait before capturing (let UI settle)
    typing_delay_ms: int = 12  # per-character delay during `type`
    scaling_enabled: bool = True  # downscale coords to XGA/WXGA/FWXGA
    max_output_chars: int = 12000
    humanize_typing: bool = True

class ImageConfig(Base):
    """Image generation configuration."""
    provider: str = "codex_cli"
    model: str = "bytedance-seed/seedream-4.5"
    api_key: str = ""
    api_base: str = "https://openrouter.ai/api/v1"
    codex_command: str = "codex"
    codex_model: str = "gpt-5.4-mini"
    codex_timeout: int = 300
    codex_home: str = ""

class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def preserve_fork_activation_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        websocket = data.get("websocket")
        if websocket is None:
            data["websocket"] = {"enabled": False}
        elif isinstance(websocket, dict):
            data["websocket"] = {"enabled": False, **websocket}
        if "taskUpdateMode" not in data and "task_update_mode" not in data and "sendProgress" in data:
            data["taskUpdateMode"] = "verbose" if data["sendProgress"] else "result"
        return data

    websocket: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    task_update_mode: Literal["result", "verbose"] = "verbose"

    send_progress: bool = True  # stream agent's text progress to the channel
    send_tool_hints: bool = False  # stream tool-call hints (e.g. read_file("…"))
    show_reasoning: bool = True  # surface model reasoning when channel implements it
    extract_document_text: bool = True  # Deprecated and ignored; documents are read on demand
    send_max_retries: int = Field(default=3, ge=0, le=10)  # Max delivery attempts (initial send included)
    transcription_provider: str = "groq"  # Deprecated: use top-level transcription.provider
    transcription_language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")  # Deprecated: use top-level transcription.language


class TranscriptionConfig(Base):
    """Cross-channel audio transcription configuration."""

    enabled: bool = True
    provider: str | None = None  # Validated by nanobot.audio.transcription_registry.
    model: str | None = None
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")
    max_duration_sec: int = Field(default=120, ge=1, le=600)
    max_upload_mb: int = Field(default=25, ge=1, le=100)


class DreamConfig(Base):
    """Dream memory consolidation configuration."""

    _HOUR_MS = 3_600_000

    enabled: bool = False  # Register the periodic Dream consolidation job on startup
    interval_h: int = Field(default=2, ge=1)  # Every 2 hours by default
    cron: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )  # Legacy cron expression override
    model_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelOverride", "model", "model_override"),
    )  # Model preset name for Dream sessions

    def build_schedule(self, timezone: str) -> CronSchedule:
        """Build the runtime schedule, preferring the legacy cron override if present."""
        if self.cron:
            return CronSchedule(kind="cron", expr=self.cron, tz=timezone)
        return CronSchedule(kind="every", every_ms=self.interval_h * self._HOUR_MS)

    def describe_schedule(self) -> str:
        """Return a human-readable summary for logs and startup output."""
        if self.cron:
            return f"cron {self.cron} (legacy)"
        hours = self.interval_h
        return f"every {hours}h"


class InlineFallbackConfig(Base):
    """One inline fallback model configuration."""

    model: str
    provider: str
    max_tokens: int | None = None
    context_window_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None


FallbackCandidate = str | InlineFallbackConfig


class ModelPresetConfig(Base):
    """A named set of model + generation parameters for quick switching."""

    model: str
    provider: str = "auto"
    max_tokens: int = 8192
    context_window_tokens: int = 200_000
    temperature: float = 0.1
    reasoning_effort: str | None = None

    def to_generation_settings(self) -> Any:
        from nanobot.providers.base import GenerationSettings
        return GenerationSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )


class AgentDefaults(Base):
    """Default agent configuration."""

    fallback: list[FallbackEntry] = Field(default_factory=list)
    context_paths: list[str] = Field(default_factory=list)
    context_repos: list[str | ContextRepoConfig] = Field(default_factory=list)
    tool_result_clearing_keep: int = 3
    consolidation_trigger_ratio: float = 0.5
    consolidation_target_ratio: float = 0.3

    @model_validator(mode="before")
    @classmethod
    def normalize_fork_tokens(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("maxTokens", data.get("max_tokens"))
        legacy_input = data.pop("maxInputTokens", None)
        if isinstance(raw, dict):
            explicit = data.get("contextWindowTokens", data.get("context_window_tokens"))
            fallback = legacy_input if legacy_input is not None else explicit
            total = raw.get("input", fallback if fallback is not None else 120000)
            data.pop("maxTokens", None)
            data.pop("max_tokens", None)
            data["max_tokens"] = raw.get("output", 4096)
            data["context_window_tokens"] = total
            data.pop("contextWindowTokens", None)
        elif legacy_input is not None:
            # Staging gave maxInputTokens precedence over contextWindowTokens.
            # Remove the camel-case alias before supplying the canonical field,
            # otherwise Pydantic would prefer the stale alias over this value.
            data.pop("contextWindowTokens", None)
            data["context_window_tokens"] = legacy_input
        if "context_path" in data or "contextPath" in data:
            path = data.pop("context_path", data.pop("contextPath", None))
            if path:
                paths = list(data.get("context_paths", data.get("contextPaths", [])))
                paths.append(path)
                data.pop("contextPaths", None)
                data["context_paths"] = paths
        return data

    workspace: str = "~/.nanobot/workspace"
    model_preset: str | None = None  # Active preset name — takes precedence over fields below
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    max_tokens: int = 4096
    context_window_tokens: int = 120_000
    temperature: float = 0.1
    fallback_models: list[FallbackCandidate] = Field(default_factory=list)
    max_tool_iterations: int = 200
    max_concurrent_subagents: int = Field(default=4, ge=1)
    max_tool_result_chars: int = 16_000
    provider_retry_mode: Literal["standard", "persistent"] = "standard"
    tool_hint_max_length: int = Field(
        default=40,
        ge=20,
        le=500,
        validation_alias=AliasChoices("toolHintMaxLength"),
        serialization_alias="toolHintMaxLength",
    )  # Max characters for tool hint display (e.g. "$ cd …/project && npm test")
    reasoning_effort: str | None = None  # low / medium / high / xhigh / max / adaptive / none — LLM thinking effort; None preserves the provider default
    timezone: str = "UTC"  # Effective IANA timezone, e.g. "Asia/Shanghai"
    timezone_mode: Literal["auto", "manual"] = "manual"
    bot_name: str = "nanobot"  # Display name shown in CLI prompts (e.g. "{name} is thinking...")
    bot_icon: str = "🐈"  # Short icon (emoji or text) shown next to the bot name in CLI; "" to omit
    unified_session: bool = False  # Share one session across all channels (single-user multi-device)
    disabled_skills: list[str] = Field(default_factory=list)  # Skill names to exclude from loading (e.g. ["summarize", "skill-creator"])
    session_ttl_minutes: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("idleCompactAfterMinutes", "sessionTtlMinutes"),
        serialization_alias="idleCompactAfterMinutes",
    )  # Auto-compact idle threshold in minutes (0 = disabled)
    idle_compact_check_interval_seconds: int = Field(
        default=60,
        ge=0,
    )  # Minimum interval in seconds between scans for idle sessions
    dream: DreamConfig = Field(default_factory=DreamConfig)

    @model_validator(mode="before")
    @classmethod
    def resolve_timezone(cls, value: object) -> object:
        """Detect new defaults server-side while preserving configured timezones."""
        if not isinstance(value, dict):
            return value

        data = dict(cast(dict[str, object], value))
        timezone_mode = data.get("timezoneMode", data.get("timezone_mode"))
        if timezone_mode is None:
            timezone_mode = "manual"
            data["timezoneMode"] = timezone_mode
        if timezone_mode == "auto":
            data["timezone"] = detect_system_timezone()
        return data

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError:
            raise ValueError(f"unknown timezone {value!r}") from None
        return value


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    crawler: CrawlerAgentConfig = Field(default_factory=CrawlerAgentConfig)


class ProviderConfig(Base):
    """LLM provider configuration."""

    # User-facing name for dynamic custom providers.
    display_name: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    api_keys: list[str] = Field(default_factory=list, repr=False)
    rate_limit: int = 0
    timeout: float = 60.0

    @property
    def effective_keys(self) -> list[str]:
        if self.api_keys:
            return [key for key in self.api_keys if key]
        return [self.api_key] if self.api_key else []

    api_key: str | None = Field(default=None, repr=False)
    api_base: str | None = None
    api_type: Literal["auto", "chat_completions", "responses"] = "auto"  # Request API surface
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)
    extra_body: dict[str, Any] | None = None  # Extra provider request fields; shape depends on provider/API surface
    extra_query: dict[str, str] | None = None  # Extra query params (e.g. api-version for Azure-style gateways)
    proxy: str | None = None  # Explicit HTTP proxy; image downloads trust its DNS and egress
    thinking_style: str | None = None  # Thinking/reasoning style for custom providers

    # Valid values mirror the keys of _THINKING_STYLE_MAP in
    # nanobot/providers/openai_compat_provider.py. Kept duplicated here to
    # avoid an import cycle (schema.py must not import from providers/).
    _VALID_THINKING_STYLES: ClassVar[tuple[str, ...]] = (
        "thinking_type",
        "enable_thinking",
        "reasoning_split",
    )

    @field_validator("thinking_style")
    @classmethod
    def _validate_thinking_style(cls, v: str | None) -> str | None:
        if not v:  # None or "" -> no injection, valid (backwards compatible)
            return v
        if v not in cls._VALID_THINKING_STYLES:
            raise ValueError(
                f"Invalid thinking_style {v!r}. "
                f"Must be one of: {', '.join(repr(s) for s in cls._VALID_THINKING_STYLES)} "
                f"(or empty/omitted)."
            )
        return v


class BedrockProviderConfig(ProviderConfig):
    """AWS Bedrock Runtime provider configuration."""

    region: str | None = None  # AWS region, falls back to AWS_REGION/AWS_DEFAULT_REGION/profile
    profile: str | None = None  # Optional AWS shared config profile


class ProvidersConfig(Base):
    """Configuration for LLM providers.

    Supports custom providers via extra fields — any additional field
    becomes an OpenAI-compatible custom provider.
    """

    model_config = ConfigDict(extra="allow")

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # Any OpenAI-compatible endpoint
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI (model = deployment name)
    bedrock: BedrockProviderConfig = Field(default_factory=BedrockProviderConfig)  # AWS Bedrock Converse
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    orcarouter: ProviderConfig = Field(default_factory=ProviderConfig)  # OrcaRouter API gateway
    assemblyai: ProviderConfig = Field(default_factory=ProviderConfig)  # AssemblyAI voice transcription
    huggingface: ProviderConfig = Field(default_factory=ProviderConfig)
    skywork: ProviderConfig = Field(default_factory=ProviderConfig)  # Skywork / APIFree API gateway
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    modelscope: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama local models
    lm_studio: ProviderConfig = Field(default_factory=ProviderConfig)  # LM Studio local models
    atomic_chat: ProviderConfig = Field(default_factory=ProviderConfig)  # Atomic Chat local models
    ovms: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenVINO Model Server (OVMS)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    kimi_coding: ProviderConfig = Field(default_factory=ProviderConfig)  # Kimi Coding Plan (Anthropic Messages API)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax_anthropic: ProviderConfig = Field(default_factory=ProviderConfig)  # MiniMax Anthropic endpoint (thinking)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    stepfun: ProviderConfig = Field(default_factory=ProviderConfig)  # Step Fun (阶跃星辰) — LLM + ASR (set apiBase to Plan URL for ASR)
    xiaomi_mimo: ProviderConfig = Field(default_factory=ProviderConfig)  # Xiaomi MIMO (小米)
    longcat: ProviderConfig = Field(default_factory=ProviderConfig)  # LongCat
    ant_ling: ProviderConfig = Field(default_factory=ProviderConfig)  # Ant Ling
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # SiliconFlow (硅基流动)
    edenai: ProviderConfig = Field(default_factory=ProviderConfig)  # Eden AI API gateway
    novita: ProviderConfig = Field(default_factory=ProviderConfig)  # Novita AI
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine (火山引擎)
    volcengine_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # VolcEngine Coding Plan
    byteplus: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus (VolcEngine international)
    byteplus_coding_plan: ProviderConfig = Field(default_factory=ProviderConfig)  # BytePlus Coding Plan
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # OpenAI Codex (OAuth)
    xai_grok: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # xAI Grok (OAuth)
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig, exclude=True)  # Github Copilot (OAuth)
    qianfan: ProviderConfig = Field(default_factory=ProviderConfig)  # Qianfan (百度千帆)
    nvidia: ProviderConfig = Field(default_factory=ProviderConfig)  # NVIDIA NIM (nvapi- keys)
    opencode: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenCode Zen (canonical provider id)
    opencode_zen: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenCode Zen (curated coding models)
    opencode_go: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenCode Go (low-cost coding models)

    @model_validator(mode="after")
    def convert_extra_providers(self):
        """Convert extra fields (custom providers) to ProviderConfig objects."""
        if self.model_extra:
            from nanobot.providers.registry import find_by_name

            for key, value in self.model_extra.items():
                if spec := find_by_name(key):
                    raise ValueError(
                        f"providers.{key} conflicts with built-in provider {spec.name!r}; "
                        "use the built-in provider key or choose a different custom provider name"
                    )
                if isinstance(value, dict):
                    self.model_extra[key] = ProviderConfig.model_validate(value)
        return self

    @model_validator(mode="after")
    def _validate_api_type_scope(self) -> "ProvidersConfig":
        for name in self.__class__.model_fields:
            if name == "openai":
                continue
            provider = getattr(self, name, None)
            if isinstance(provider, ProviderConfig) and provider.api_type != "auto":
                raise ValueError("providers.<name>.api_type is only supported for providers.openai")
        for provider in (self.model_extra or {}).values():
            if isinstance(provider, ProviderConfig) and provider.api_type != "auto":
                raise ValueError("providers.<name>.api_type is only supported for providers.openai")
        return self


class HeartbeatConfig(Base):
    keep_recent_messages: int = 8
    """Heartbeat service configuration (now backed by cron)."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes


class ApiConfig(Base):
    """OpenAI-compatible API server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 8900
    timeout: float = 120.0  # Per-request timeout in seconds.
    api_key: str = Field(default="", repr=False)

    @model_validator(mode="after")
    def wildcard_host_requires_auth(self) -> "ApiConfig":
        if self.host not in ("0.0.0.0", "::"):
            return self
        if self.api_key.strip():
            return self
        raise ValueError(
            "host is 0.0.0.0 (all interfaces) but api_key is not set "
            "- set api.api_key to prevent unauthenticated access"
        )


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "127.0.0.1"  # Safer default: local-only bind.
    port: int = 18790
    restart_mode: Literal["auto", "exec", "spawn", "exit"] = "auto"
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    auth: Literal["oauth"] | None = None  # Remote MCP OAuth; tokens are stored outside config
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    cwd: str = ""  # Stdio: working directory for MCP server runtime artifacts
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all capabilities (tools, resources, prompts); any restriction = only listed tools, no resources/prompts


def _lazy_default(module_path: str, class_name: str) -> Any:
    """Deferred import helper for ToolsConfig default factories."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


class ToolsConfig(Base):
    """Tools configuration.

    Field types for tool-specific sub-configs are resolved via model_rebuild()
    at the bottom of this file so tool config classes can stay next to their
    tool implementations.
    """

    @classmethod
    def model_rebuild(cls, *, force: bool = False, raise_errors: bool = True,
                      _parent_namespace_depth: int = 2,
                      _types_namespace: Mapping[str, Any] | None = None) -> bool | None:
        namespace = dict(_types_namespace or {})
        namespace.update(_tool_config_types())
        return super().model_rebuild(force=force, raise_errors=raise_errors,
            _parent_namespace_depth=_parent_namespace_depth, _types_namespace=namespace)

    @model_validator(mode="before")
    @classmethod
    def _select_image_configuration(cls, data: Any) -> Any:
        if isinstance(data, dict) and "image_generation" not in data and "imageGeneration" not in data:
            data = dict(data)
            data["image_generation"] = {"configuration_source": "legacy"}
        return data

    agent_browser: AgentBrowserConfig = Field(default_factory=AgentBrowserConfig)
    agent_device: AgentDeviceConfig = Field(default_factory=AgentDeviceConfig)
    desktop_use: DesktopUseConfig = Field(default_factory=DesktopUseConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)

    web: WebToolsConfig = Field(default_factory=lambda: _lazy_default("nanobot.agent.tools.web", "WebToolsConfig"))
    exec: ExecToolConfig = Field(default_factory=lambda: _lazy_default("nanobot.agent.tools.shell", "ExecToolConfig"))
    file: FileToolsConfig = Field(default_factory=lambda: _lazy_default("nanobot.agent.tools.filesystem", "FileToolsConfig"))
    cli_apps: CliAppsToolConfig = Field(default_factory=lambda: _lazy_default("nanobot.agent.tools.cli_apps", "CliAppsToolConfig"))
    my: MyToolConfig = Field(default_factory=lambda: _lazy_default("nanobot.agent.tools.self", "MyToolConfig"))
    image_generation: ImageGenerationToolConfig = Field(
        default_factory=lambda: _lazy_default("nanobot.agent.tools.image_generation", "ImageGenerationToolConfig"),
    )
    max_session_messages_per_minute: int = Field(default=6, ge=1)
    restrict_to_workspace: bool = False  # policy intent: keep tool access inside workspace when possible
    webui_allow_local_service_access: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "webuiAllowLocalServiceAccess",
            "webui_allow_local_service_access",
            "allowLocalPreviewAccess",
            "allow_local_preview_access",
        ),
    )  # allow WebUI Full Access shell checks against localhost services; legacy allowLocalPreviewAccess still reads
    webui_allow_remote_package_install: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "webuiAllowRemotePackageInstall",
            "webui_allow_remote_package_install",
        ),
    )  # allow non-local WebUI clients to install optional packages and agent skills
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    ssrf_whitelist: list[str] = Field(default_factory=list)  # CIDR ranges to exempt from SSRF blocking (e.g. ["100.64.0.0/10"] for Tailscale)


class ArtifactsConfig(Base):
    """Opt-in private storage for source-bound context artifacts."""

    enabled: StrictBool = False
    root: str | None = None
    max_blob_bytes: StrictInt = Field(default=64 * 1024 * 1024, gt=0)
    max_namespace_bytes: StrictInt = Field(default=1024 * 1024 * 1024, gt=0)

    @field_validator("root")
    @classmethod
    def _validate_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("artifacts.root must be an absolute path")
        path = Path(value).expanduser()
        if not path.is_absolute() or path == Path(path.anchor):
            raise ValueError("artifacts.root must be an absolute non-root path")
        # Normalize lexical components without resolving symlinks.  The
        # factory rejects symlinked roots/components before it opens or creates
        # private state, so resolving here would hide an unsafe configuration.
        return os.path.abspath(os.fspath(path))

    @model_validator(mode="after")
    def _validate_limits(self) -> "ArtifactsConfig":
        if self.max_namespace_bytes < self.max_blob_bytes:
            raise ValueError("artifacts.maxNamespaceBytes must be at least artifacts.maxBlobBytes")
        return self


class Config(BaseSettings):
    """Root configuration for nanobot."""

    _source_path: Path | None = PrivateAttr(default=None)

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    model_presets: dict[str, ModelPresetConfig] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("modelPresets", "model_presets"),
        serialization_alias="modelPresets",
    )

    @classmethod
    def model_rebuild(cls, *, force: bool = False, raise_errors: bool = True,
                      _parent_namespace_depth: int = 2,
                      _types_namespace: Mapping[str, Any] | None = None) -> bool | None:
        namespace = dict(_types_namespace or {})
        namespace.update(_tool_config_types())
        return super().model_rebuild(force=force, raise_errors=raise_errors,
            _parent_namespace_depth=_parent_namespace_depth, _types_namespace=namespace)

    def __init__(self, **values: Any) -> None:
        if not type(self).__pydantic_complete__:
            _resolve_tool_config_refs()
        super().__init__(**values)

    def bind_source_path(self, path: Path) -> None:
        """Record the config file that owns instance-level runtime data."""
        self._source_path = path.expanduser().resolve(strict=False)

    @property
    def runtime_data_dir(self) -> Path | None:
        """Return the active instance data directory when loaded from a config path."""
        return self._source_path.parent if self._source_path is not None else None

    @model_validator(mode="after")
    def _validate_model_preset(self) -> "Config":
        # Keep persisted names accepted by previous releases loadable. New
        # names are normalized and checked case-insensitively at mutation
        # boundaries, where conflicts can be reported without breaking startup.
        if "default" in self.model_presets:
            raise ValueError("model_preset name 'default' is reserved for agents.defaults")
        name = self.agents.defaults.model_preset
        if name and name != "default" and name not in self.model_presets:
            raise ValueError(f"model_preset {name!r} not found in model_presets")
        dream_name = self.agents.defaults.dream.model_override
        if dream_name and dream_name != "default" and dream_name not in self.model_presets:
            raise ValueError(f"Dream model preset {dream_name!r} not found in model_presets")
        if self.agents.defaults.fallback and self.agents.defaults.fallback_models:
            raise ValueError("Configure either legacy fallback or fallback_models, not both")
        for fallback in self.agents.defaults.fallback_models:
            if isinstance(fallback, str) and fallback not in self.model_presets:
                raise ValueError(f"fallback_models entry {fallback!r} not found in model_presets")
        return self

    def resolve_default_preset(self) -> ModelPresetConfig:
        """Return the implicit `default` preset from agents.defaults fields."""
        d = self.agents.defaults
        return ModelPresetConfig(
            model=d.model, provider=d.provider, max_tokens=d.max_tokens,
            context_window_tokens=d.context_window_tokens,
            temperature=d.temperature, reasoning_effort=d.reasoning_effort,
        )

    def resolve_preset(self, name: str | None = None) -> ModelPresetConfig:
        """Return effective model params from a named preset or the implicit default."""
        name = self.agents.defaults.model_preset if name is None else name
        if not name or name == "default":
            return self.resolve_default_preset()
        if name not in self.model_presets:
            raise KeyError(f"model_preset {name!r} not found in model_presets")
        return self.model_presets[name]

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> tuple["ProviderConfig | None", str | None]:
        """Match provider config and its registry name. Returns (config, spec_name)."""
        from nanobot.providers.registry import (
            PROVIDERS,
            find_by_name,
        )

        resolved = preset or self.resolve_preset()
        forced = resolved.provider

        def _custom_provider_by_name(name: str) -> tuple[ProviderConfig, str] | None:
            normalized = name.replace("-", "_").lower()
            for attr_name, provider in (self.providers.model_extra or {}).items():
                if not isinstance(provider, ProviderConfig):
                    continue
                if attr_name.replace("-", "_").lower() == normalized:
                    return provider, attr_name
            return None

        if forced != "auto":
            spec = find_by_name(forced)
            if spec:
                p = getattr(self.providers, spec.name, None)
                return (p, spec.name) if p else (None, None)
            custom = _custom_provider_by_name(forced)
            if custom is not None:
                return custom
            return None, None

        model_lower = (model or resolved.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")
        prefixed_provider = find_by_name(model_prefix) if model_prefix else None

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # Explicit provider prefix wins — prevents `github-copilot/...codex` matching openai_codex.
        for spec in PROVIDERS:
            if spec.is_transcription_only:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or spec.is_direct or p.effective_keys:
                    return p, spec.name

        # Check for custom provider by prefix (e.g., "companyProxy/gpt-4").
        # Return the matching provider even when apiBase is missing, so a
        # malformed explicit prefix fails instead of falling through to a
        # different custom provider.
        if model_prefix:
            custom = _custom_provider_by_name(normalized_prefix)
            if custom is not None:
                return custom

        # Match by keyword (order follows PROVIDERS registry)
        for spec in PROVIDERS:
            if spec.is_transcription_only:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                # Local providers (Ollama, vLLM, …) keep model-family keywords
                # like "nemotron" or "llama" to enable bare-model auto-routing,
                # but those keywords collide with cloud-hosted variants of the
                # same family (e.g. `nvidia/nemotron-...` via OpenRouter). Only
                # honor a local keyword match when the user has actually
                # configured that local endpoint via `api_base` — mirrors the
                # gate already used by the local-fallback loop below.
                if spec.is_local:
                    # A qualified model belongs to its explicit provider or a
                    # gateway fallback, never to a different local provider
                    # whose model-family keyword happens to match.
                    foreign_prefix = bool(
                        prefixed_provider is not None and prefixed_provider.name != spec.name
                    )
                    if not p.api_base or foreign_prefix:
                        continue
                if spec.is_oauth or spec.is_local or spec.is_direct or p.effective_keys:
                    return p, spec.name

        # Fallback: configured local providers can route models without
        # provider-specific keywords (for example plain "llama3.2" on Ollama).
        # Prefer providers whose detect_by_base_keyword matches the configured api_base
        # (e.g. Ollama's "11434" in "http://localhost:11434") over plain registry order.
        local_fallback: tuple[ProviderConfig, str] | None = None
        if prefixed_provider is None:
            for spec in PROVIDERS:
                if not spec.is_local:
                    continue
                p = getattr(self.providers, spec.name, None)
                if not (p and p.api_base):
                    continue
                if spec.detect_by_base_keyword and spec.detect_by_base_keyword in p.api_base:
                    return p, spec.name
                if local_fallback is None:
                    local_fallback = (p, spec.name)
        if local_fallback:
            return local_fallback

        # Fallback: gateways first, then others (follows registry order)
        # OAuth providers are NOT valid fallbacks — they require explicit model selection
        for spec in PROVIDERS:
            if spec.is_oauth or spec.is_transcription_only:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.effective_keys:
                return p, spec.name

        # Final fallback: check for any configured custom provider
        for attr_name, p in (self.providers.model_extra or {}).items():
            if isinstance(p, ProviderConfig) and p.api_base:
                return p, attr_name

        return None, None

    def get_provider(
        self,
        model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> ProviderConfig | None:
        """Get matched provider config (api_key, api_base, extra_headers). Falls back to first available."""
        p, _ = self._match_provider(model, preset=preset)
        return p

    def get_provider_name(
        self,
        model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> str | None:
        """Get the registry name of the matched provider (e.g. "deepseek", "openrouter")."""
        _, name = self._match_provider(model, preset=preset)
        return name

    def get_api_key(
        self,
        model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model, preset=preset)
        return p.effective_keys[0] if p and p.effective_keys else None

    def get_api_base(
        self,
        model: str | None = None,
        *,
        preset: ModelPresetConfig | None = None,
    ) -> str | None:
        """Get API base URL for the given model, falling back to the provider default when present."""
        from nanobot.providers.registry import find_by_name

        p, name = self._match_provider(model, preset=preset)
        if p and p.api_base:
            return p.api_base
        if name:
            spec = find_by_name(name)
            if spec and spec.default_api_base:
                return spec.default_api_base
        return None

    model_config = SettingsConfigDict(
        env_prefix="NANOBOT_",
        env_nested_delimiter="__",
    )


_TOOL_CONFIG_MODULES = {
    "CliAppsToolConfig": "nanobot.agent.tools.cli_apps",
    "FileToolsConfig": "nanobot.agent.tools.filesystem",
    "ImageGenerationToolConfig": "nanobot.agent.tools.image_generation",
    "MyToolConfig": "nanobot.agent.tools.self",
    "ExecToolConfig": "nanobot.agent.tools.shell",
    "WebFetchConfig": "nanobot.agent.tools.web",
    "WebSearchConfig": "nanobot.agent.tools.web",
    "WebToolsConfig": "nanobot.agent.tools.web",
}


def __getattr__(name: str) -> Any:
    """Resolve historical aliases after a tool-first import cycle has unwound."""
    from importlib import import_module

    module_name = _TOOL_CONFIG_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def _tool_config_types() -> dict[str, Any]:
    """Collect canonical types without recursively rebuilding Pydantic models."""
    namespace = {}
    for name in _TOOL_CONFIG_MODULES:
        try:
            namespace[name] = globals().get(name) or __getattr__(name)
        except AttributeError as exc:
            # A tool may currently be importing this schema before defining its
            # config class. Defer the eager attempt; first-use rebuild retries.
            raise ImportError(f'Tool configuration {name} is not yet available') from exc
    return namespace


def _resolve_tool_config_refs() -> None:
    """Resolve both public schemas once canonical tool classes are importable."""
    namespace = _tool_config_types()
    ToolsConfig.model_rebuild(_types_namespace=namespace)
    Config.model_rebuild(_types_namespace=namespace)


# Eagerly resolve when the import chain allows it (no circular deps at this
# point).  If it fails (first import triggers a cycle), the rebuild will
# happen lazily when Config/ToolsConfig is first used at runtime.
try:
    _resolve_tool_config_refs()
except ImportError:
    pass
