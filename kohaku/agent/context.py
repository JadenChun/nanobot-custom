"""Context builder for assembling agent prompts."""

import platform
import uuid
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from kohaku.agent.memory import MemoryStore
from kohaku.agent.skills import SkillsLoader
from kohaku.agent.tools import image_generation as image_generation_tools
from kohaku.agent.tools import mcp as mcp_tools
from kohaku.agent.tools import sessions as session_tools
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.attachments.context import AttachmentContext
from kohaku.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)
from kohaku.context.budget import TokenMeter
from kohaku.context.compiler import PromptCompiler, truncate_section_to_tokens
from kohaku.context.project_docs import discover_project_instructions
from kohaku.context.types import (
    SYSTEM_BLOCKS_META,
    SYSTEM_SEPARATOR,
    CompiledPrompt,
    PromptSection,
    PromptSourceFile,
    SessionPromptSnapshot,
    TurnKind,
)
from kohaku.context_repo import ContextRepoManager
from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.harness.claude_sdk import claude_sdk_available
from kohaku.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    RUNTIME_CONTEXT_TAG,
    RuntimeContextBlock,
    append_runtime_context,
    runtime_metadata_value,
    wrap_runtime_context_lines,
)
from kohaku.security.folder_grants import FolderGrantView
from kohaku.security.workspace_access import (
    WorkspaceScopeResolver,
    current_workspace_scope,
)
from kohaku.session.keys import last_channel_from_metadata
from kohaku.session.manager import Session
from kohaku.session.source_projection import CheckpointNode, RuntimeNode, SourceNode
from kohaku.session.source_request import RequestNode, SourceTranscript
from kohaku.session.summary import SessionSummary
from kohaku.utils.helpers import current_time_str, load_bundled_template
from kohaku.utils.prompt_templates import render_template


class _LegacyChatIdDefault(Enum):
    OMITTED = "omitted"


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Retain structured references to other sessions."""
    return session_tools.session_extra(metadata)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    if msg.metadata.get(INBOUND_META_RUNTIME_CONTROL) == RUNTIME_CONTROL_SESSION_DISCARD:
        await state.discard_session(msg.session_key)
        return True
    return await image_generation_tools.handle_runtime_control(state, msg, tools)


@dataclass(frozen=True, slots=True)
class PersistedPromptContextResolver:
    """Restore prompt routing context when no inbound message is available."""

    workspace_scopes: WorkspaceScopeResolver
    unified_session: bool = False

    def __call__(self, session: Session) -> tuple[str | None, Path]:
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        if self.unified_session:
            route = last_channel_from_metadata(session.metadata)
            if route is not None:
                channel = route[0]
        scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=None,
            session_metadata=session.metadata,
        )
        return channel, scope.project_path


@dataclass(frozen=True, slots=True)
class TranscriptInput:
    """Raw turn inputs from which ``ContextBuilder`` assembles a transcript."""

    history: list[dict[str, Any]]
    current_message: str | None
    media: Sequence[str] | None = None
    current_role: str = "user"
    session_summary: SessionSummary | None = None
    runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None
    include_skill_context: bool = True

    @property
    def message_count(self) -> int:
        """Number of boundary-preserving messages in the assembled transcript."""
        summary_count = int(
            self.session_summary is not None
            and self.session_summary["text"] != "(nothing)"
        )
        return 1 + summary_count + len(self.history) + (self.current_message is not None)


# Recognize earlier scaffold files after bundled defaults change.
_LEGACY_BOOTSTRAP_DIGESTS: dict[str, frozenset[str]] = {
    "memory/MEMORY.md": frozenset({
        "6632103aee2f6cec744dee940e69a2e5080ea85dbbd026bc484f73272f8f23fc",
    }),
    "USER.md": frozenset({
        "2cdf850d2f0b25bd27cb357f8b7a60034b56de4eeaf3de8fa1eda34eb1d3a094",
    }),
    "AGENTS.md": frozenset({
        "0fc3ed4f125fb51540ada6189db8b0dbd653cfd36ea87395fc27e01a56762218",
        "68b91afdd8958d20de7e913cba598d0bc2f4aaf0cdd70d73d306e1816a4ca906",
    }),
    "TOOLS.md": frozenset({
        "4c56f2fc92ac5dbd43c011f62acb148b76872ce1a72717552bf9ea02a75deafd",
    }),
    "SOUL.md": frozenset({
        "59cb9a0583a0ae7401160af13f95c5627be3dbaef2aec2ec210a361abdfc5623",
    }),
    "IDENTITY.md": frozenset({
        "8519eb8862f0bb9c3774d8f607e14370213039c38739fec45b021ac7a02ef694",
    }),
}


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _SKIPPABLE_DEFAULTS = {"AGENTS.md", "USER.md", "TOOLS.md", "IDENTITY.md"}
    _RUNTIME_CONTEXT_TAG = RUNTIME_CONTEXT_TAG

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        *,
        context_paths: list[Path] | None = None,
        context_manager: ContextRepoManager | None = None,
        sessions_root: Path | None = None,
        resource_policy: ResourceAccessPolicy | None = None,
        folder_grants: FolderGrantView | None = None,
        attachment_context: AttachmentContext | None = None,
    ):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace, sessions_root=sessions_root)
        self.context_manager = context_manager or ContextRepoManager.from_config(context_paths=context_paths)
        self.context_paths = self.context_manager.paths
        self._prompt_snapshots: dict[str, SessionPromptSnapshot] = {}
        self.resource_policy = resource_policy
        self.folder_grants = folder_grants
        self.attachments = attachment_context or AttachmentContext()
        self.skills = SkillsLoader(
            workspace,
            disabled_skills=set(disabled_skills) if disabled_skills else None,
            extra_paths=self.context_manager.skill_roots(),
            resource_policy=resource_policy,
        )

    def refresh_prompt_snapshot(self, session_key: str) -> None:
        """Start a fresh prompt epoch on the next admitted turn."""
        self._prompt_snapshots.pop(session_key, None)

    def compile_system_prompt(
        self,
        *,
        channel: str | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        tool_names: frozenset[str] | None = None,
        tools: ToolRegistry | None = None,
        session: Session | None = None,
    ) -> CompiledPrompt:
        """Compile the portable prompt and retain its provider cache boundaries."""
        managed = bool(self.resource_policy and self.resource_policy.managed_mode)
        scope = current_workspace_scope() if managed else None
        project = (scope.project_path if scope else workspace or self.workspace).resolve()
        snapshot = self._prompt_snapshot(project, managed, include_memory, tools, session)
        effective_tools = frozenset(tools.tool_names) if tools is not None else tool_names
        root = self.workspace if managed else (workspace or self.workspace)
        sections = [
            PromptSection(
                "identity", "static",
                self._get_identity(workspace=root, include_memory=include_memory),
                priority=10,
            ),
            PromptSection(
                "format_hint", "static",
                render_template("agent/_snippets/format_hint.md", channel=channel or "")
                if session is None else "",
                priority=15,
            ),
            PromptSection(
                "operating_principles", "static",
                render_template("agent/tool_contract.md"),
                max_tokens=800, priority=20,
            ),
            PromptSection(
                "untrusted_content", "static",
                render_template("agent/_snippets/untrusted_content.md"),
                priority=30,
            ),
            PromptSection(
                "sandbox", "static",
                self._managed_prompt_metadata() if managed else self._workspace_access_section(),
                priority=40,
            ),
            PromptSection(
                "repository_discovery", "static",
                "Use `repository_catalog` to discover configured projects. "
                "Read the selected project's instructions and use its authorized paths.",
                requires_all_tools=frozenset({"repository_catalog"}),
                priority=45,
            ),
            PromptSection(
                "project_selection", "static",
                "Use `select_workspace` to change the default folder for relative paths. "
                "Selection does not grant access.",
                requires_all_tools=frozenset({"select_workspace"}),
                priority=46,
            ),
            PromptSection(
                "folder_requests", "static",
                "Use `request_folder_access` when a task needs an unapproved folder.",
                requires_all_tools=frozenset({"request_folder_access"}),
                priority=47,
            ),
            PromptSection(
                "chat_delivery", "static",
                "Reply to the current chat normally. Use `message` (load via `tool_search` "
                "if deferred) only for a separate delivery or media attachment.",
                requires_all_tools=frozenset({"message"}),
                priority=48,
            ),
            PromptSection(
                "memory_recall", "static",
                "Use `memory_search` to check past sessions, user preferences, and prior "
                "decisions before guessing.",
                requires_all_tools=frozenset({"memory_search"}),
                priority=49,
            ),
            PromptSection(
                "scheduling_guidance", "static",
                "For reminders or scheduled tasks, load and use `cron` (follow the `cron` "
                "skill). For periodic silent checks that notify only on actionable changes, "
                "update `HEARTBEAT.md`. Do not just write reminders to `MEMORY.md`.",
                requires_all_tools=frozenset({"cron"}),
                priority=50,
            ),
            PromptSection(
                "coding_delegation", "static",
                (
                    "For substantial coding, codebase investigation, debugging, or implementation "
                    "work, load the `delegation` tools via `tool_search` and use "
                    "`harness=\"claude_sdk\"` so Claude Code handles the engineering work; "
                    "handle quick, self-contained reads or edits directly."
                    if claude_sdk_available()
                    else "For substantial multi-file work or broad investigation, load the "
                    "`delegation` tools via `tool_search` to keep the main context focused."
                ),
                requires_any_tools=frozenset({"delegate_task", "explore"}),
                priority=51,
            ),
            PromptSection(
                "web_research", "static",
                "For current, time-sensitive, or unfamiliar external facts (news, prices, "
                "schedules, releases, policies, or APIs), search the web and open primary "
                "sources when details matter instead of guessing from memory. Cross-check "
                "important claims, apply the findings directly to the user's request, and "
                "cite key source URLs.",
                requires_any_tools=frozenset({"web_search", "web_fetch"}),
                priority=52,
            ),
            *snapshot.sections,
            *(self._tool_namespace_sections(tools) if tools is not None else ()),
        ]
        return PromptCompiler.compile_system_prompt(sections, tool_names=effective_tools)

    def build_system_prompt(
        self, *, channel: str | None = None, workspace: Path | None = None,
        include_memory: bool = True, tool_names: frozenset[str] | None = None,
        session: Session | None = None,
    ) -> str:
        """Return the flattened prompt for compatibility callers."""
        return self.compile_system_prompt(
            channel=channel, workspace=workspace, include_memory=include_memory,
            tool_names=tool_names, session=session,
        ).full_system_text

    def _prompt_snapshot(
        self,
        project: Path,
        managed: bool,
        include_memory: bool,
        tools: ToolRegistry | None,
        session: Session | None,
    ) -> SessionPromptSnapshot:
        retain = session is not None and include_memory and session.policy.persist
        if retain and session is not None:
            cached = self._prompt_snapshots.get(session.key)
            if cached is not None and (
                cached.workspace == project
                and cached.generation == session.generation
                and cached.incarnation == session.incarnation
            ):
                for source in cached.source_files:
                    self._validate_prompt_source(source, managed=managed)
                if tools is not None and not any(s.id.startswith("mcp:") for s in cached.sections):
                    mcp_sections = self._mcp_prompt_sections(tools)
                    if mcp_sections:
                        cached = SessionPromptSnapshot(
                            cached.workspace, cached.generation, cached.incarnation,
                            (*cached.sections, *mcp_sections), cached.source_files,
                        )
                        self._prompt_snapshots[session.key] = cached
                session.prompt_snapshot = cached
                return cached

        sections, source_files = self._snapshot_sections(project, managed, include_memory, tools)
        snapshot = SessionPromptSnapshot(
            project, session.generation if session else 0,
            session.incarnation if session else "", tuple(sections), tuple(source_files),
        )
        if retain and session is not None:
            self._prompt_snapshots[session.key] = snapshot
            session.prompt_snapshot = snapshot
        return snapshot

    def _snapshot_sections(
        self, project: Path, managed: bool, include_memory: bool, tools: ToolRegistry | None,
    ) -> tuple[list[PromptSection], list[PromptSourceFile]]:
        sections: list[PromptSection] = []
        source_files: list[PromptSourceFile] = []

        def remember(path: Path, root: Path, *, allow_folder_grants: bool = True) -> Path:
            source = PromptSourceFile(path, root, allow_folder_grants)
            resolved = self._validate_prompt_source(source, managed=managed)
            source_files.append(source)
            return resolved

        def skip_default(path: Path, text: str) -> bool:
            return path.name in self._SKIPPABLE_DEFAULTS and self._is_template_content(text, path.name)

        def validate_project_file(path: Path) -> None:
            remember(path, path.parent)

        agent_home = self.workspace.expanduser().resolve()
        sections.extend(self._bootstrap_sections(
            include_memory,
            lambda path: remember(path, agent_home, allow_folder_grants=False),
        ))
        if project != agent_home:
            sections.append(PromptSection(
                "current_project", "session_snapshot",
                f"# Current Project\n\nWorking directory: {project}\n"
                "Use it as the default root for project files and relative tool paths.",
                priority=60,
            ))
        instructions = discover_project_instructions(
            self.workspace, project, validator=validate_project_file,
            skip_content=skip_default,
        )
        if instructions:
            sections.append(PromptSection(
                "project_instructions", "session_snapshot", instructions,
                priority=70, max_tokens=8000,
            ))

        if include_memory:
            legacy_memory = self.memory.memory_file
            for path in (legacy_memory.with_name("memory_summary.md"), legacy_memory):
                if not path.is_file():
                    continue
                source = remember(path, agent_home, allow_folder_grants=False)
                memory = source.read_text(encoding="utf-8")
                if not memory.strip() or self._is_template_content(memory, "memory/MEMORY.md"):
                    continue
                sections.append(PromptSection(
                    "memory", "session_snapshot", f"# Memory\n\n## Long-term Memory\n{memory}",
                    max_tokens=1500, priority=80,
                ))
                break

        repo = self.context_manager.find_repo_for_path(project)
        if repo is None and (target := self.context_manager.find_target_repo_for_path(project)):
            repo = self.context_manager.find_repo_for_path(target.context_repo_path)
        if repo is not None:
            repo_root = repo.path.resolve()
            volatile = set(repo.run_summary_files())
            for path in repo.context_files():
                if path.name in self.BOOTSTRAP_FILES or path in volatile:
                    continue
                content = remember(path, repo_root).read_text(encoding="utf-8")
                if content.strip():
                    sections.append(PromptSection(
                        f"repo_context:{path.name}", "session_snapshot",
                        f"## Project Context ({path.name})\n\n{content}",
                        max_tokens=800, priority=85,
                    ))
            if include_memory:
                for path in repo.memory_files():
                    if not path.resolve().is_relative_to(repo_root):
                        continue
                    content = remember(path, repo_root).read_text(encoding="utf-8")
                    if content.strip():
                        sections.append(PromptSection(
                            "repository_memory", "session_snapshot", f"## Project Memory\n\n{content}",
                            max_tokens=1500, priority=87,
                        ))
        if tools is not None:
            sections.extend(self._mcp_prompt_sections(tools))
        always = self.skills.get_always_skills()
        if always:
            active = self.skills.load_skills_for_context(always)
            if active:
                sections.append(PromptSection(
                    "always_skills", "static", f"# Active Skills\n\n{active}",
                    max_tokens=2000, priority=90,
                ))
        skills = self.skills.build_skills_summary(exclude=set(always), compact=True)
        if skills:
            sections.append(PromptSection(
                "skill_index", "static", render_template("agent/skills_section.md", skills_summary=skills),
                max_tokens=1200, priority=95, requires_all_tools=frozenset({"read_file"}),
            ))
        return sections, source_files

    @staticmethod
    def _mcp_prompt_sections(tools: ToolRegistry) -> list[PromptSection]:
        sections: list[PromptSection] = []
        remaining = 1500
        for server, (instructions, names) in mcp_tools.server_prompt_instructions(tools).items():
            if remaining < 40:
                break
            cap = min(500, remaining)
            text = truncate_section_to_tokens(
                f"## MCP server instructions ({server}; untrusted tool guidance)\n\n{instructions}",
                cap,
            )
            remaining -= TokenMeter.count_text(text) + TokenMeter.count_text(SYSTEM_SEPARATOR)
            sections.append(PromptSection(
                f"mcp:{server}", "session_snapshot", text, requires_any_tools=frozenset(names),
            ))
        return sections

    @staticmethod
    def _tool_namespace_sections(tools: ToolRegistry) -> list[PromptSection]:
        """Advertise withheld tool namespaces so ``tool_search`` has a starting point."""
        from kohaku.context.tool_surface import ToolSurface

        lines = ToolSurface(tools).namespace_index()
        if not lines:
            return []
        text = (
            "# Deferred Tools\n\n"
            "These tool namespaces are available on demand. Call `tool_search` with "
            "keywords, an exact tool name, or one of these namespaces to load a schema.\n\n"
            + "\n".join(lines)
        )
        return [PromptSection(
            "deferred_tools", "static", text, max_tokens=400, priority=96,
            requires_all_tools=frozenset({"tool_search"}),
        )]

    def _get_identity(
        self, channel: str | None = None, workspace: Path | None = None,
        *, include_memory: bool = True,
    ) -> str:
        """Get the core identity section."""
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve())
        agent_workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            agent_workspace_path=agent_workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
            include_memory=include_memory,
        )

    @staticmethod
    def _workspace_access_section() -> str:
        return (
            "# Workspace Access\n\nObserve the configured sandbox and folder permissions. "
            "Folder selection does not grant file or command access."
        )

    @staticmethod
    def _managed_prompt_metadata() -> str:
        return (
            "# Sandbox Folder Rules\n\n"
            "Sandbox allows ordinary OS-account reads and commands. Writes are limited to approved "
            "folders, agent home, and private temporary or cache storage. More specific read-only "
            "rules override writable parents. Full access uses normal OS permissions. "
            "Kohaku's private runtime files and configured context policies remain protected."
        )

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _bootstrap_sections(
        self, include_memory: bool, validate: Callable[[Path], Path],
    ) -> list[PromptSection]:
        sections: list[PromptSection] = []
        for name, cap in (("SOUL.md", 400), ("USER.md", 500),
                          ("TOOLS.md", None), ("IDENTITY.md", None)):
            if name == "USER.md" and not include_memory:
                continue
            path = self.workspace / name
            if not path.is_file():
                continue
            text = validate(path).read_text(encoding="utf-8")
            if name == "SOUL.md" and (
                self._is_template_content(text, "legacy/SOUL.md")
                or self._matches_legacy_digest(text, "SOUL.md")
            ):
                text = load_bundled_template("SOUL.md") or text
            if not text.strip() or (
                name in self._SKIPPABLE_DEFAULTS and self._is_template_content(text, name)
            ):
                continue
            sections.append(PromptSection(
                name, "session_snapshot", f"## {name}\n\n{text}", max_tokens=cap,
            ))
        return sections

    def _validate_prompt_source(self, source: PromptSourceFile, *, managed: bool) -> Path:
        resolved = source.path.resolve()
        if not resolved.is_relative_to(source.root):
            if not source.allow_folder_grants:
                raise PermissionError("Prompt file resolves outside agent home.")
            raise PermissionError("Prompt file resolves outside its source folder.")
        policy = self.resource_policy
        if managed and policy is not None:
            policy.validate_path(
                resolved,
                "read",
                managed=True,
                allow_folder_grants=source.allow_folder_grants and policy.folder_grants_allowed,
            )
        return resolved

    @staticmethod
    def _matches_legacy_digest(content: str, template_path: str) -> bool:
        digests = _LEGACY_BOOTSTRAP_DIGESTS.get(template_path)
        return bool(digests and sha256(content.strip().encode("utf-8")).hexdigest() in digests)

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        content = content.strip()
        template = load_bundled_template(template_path)
        if template is not None and content == template.strip():
            return True
        return ContextBuilder._matches_legacy_digest(content, template_path)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str | None,
        *,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None | _LegacyChatIdDefault = _LegacyChatIdDefault.OMITTED,
        current_role: str = "user",
        session_summary: SessionSummary | None = None,
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        tool_names: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper for callers that need merged adjacent roles."""
        blocks = runtime_context_blocks
        if (current_message is not None and current_role == "user"
                and not isinstance(chat_id, _LegacyChatIdDefault)):
            # Only explicit legacy calls acquire routing/time context. Canonical
            # transcripts remain metadata-free unless their caller supplies it.
            lines = [f"Current Time: {runtime_metadata_value(current_time_str(self.timezone))}"]
            if channel and chat_id:
                lines.extend([
                    f"Channel: {runtime_metadata_value(channel)}",
                    f"Chat ID: {runtime_metadata_value(chat_id)}",
                ])
            blocks = [
                RuntimeContextBlock("legacy_session", wrap_runtime_context_lines(lines)),
                *(runtime_context_blocks or ()),
            ]
        messages = self.build_transcript(
            TranscriptInput(
                history=history,
                current_message=current_message,
                media=media,
                current_role=current_role,
                session_summary=session_summary,
                runtime_context_blocks=blocks,
            ),
            channel=channel,
            workspace=workspace,
            include_memory=include_memory,
            tool_names=tool_names,
        )
        if current_message is None:
            return messages
        current = messages[-1]
        if len(messages) < 2 or messages[-2].get("role") != current.get("role"):
            return messages

        merged = dict(messages[-2])
        merged["content"] = self._merge_message_content(
            merged.get("content"),
            current.get("content"),
        )
        current_meta = current.get("_meta")
        if current.get("role") == "user" and isinstance(current_meta, dict):
            internal_meta = dict(merged.get("_meta") or {})
            internal_meta.update(cast(dict[str, Any], current_meta))
            merged["_meta"] = internal_meta
        return [*messages[:-2], merged]

    def build_transcript(
        self,
        transcript: TranscriptInput,
        *,
        channel: str | None = None,
        workspace: Path | None = None,
        include_memory: bool = True,
        tool_names: frozenset[str] | None = None,
        tools: ToolRegistry | None = None,
        session: Session | None = None,
    ) -> list[dict[str, Any]]:
        """Build a model transcript while preserving the fresh-turn boundary."""
        root = workspace or self.workspace
        messages = [self._transcript_system_message(
            channel=channel, workspace=root, include_memory=include_memory,
            tool_names=tool_names, session=session, tools=tools,
        )]
        summary = transcript.session_summary
        if summary and summary["text"] != "(nothing)":
            summary_text = (
                f"Previous conversation summary (last active {summary['last_active']}):\n"
                f"{summary['text']}"
            )
            messages.append(CheckpointNode("legacy-summary", summary_text).message())
        messages.extend(transcript.history)
        if transcript.current_message is None:
            return messages

        current = self.build_current_message(
            transcript.current_message,
            media=list(transcript.media) if transcript.media else None,
            current_role=transcript.current_role,
            runtime_context_blocks=transcript.runtime_context_blocks,
            include_skill_context=transcript.include_skill_context,
        )
        messages.append(current)
        return messages

    def build_source_transcript(
        self, transcript: TranscriptInput, *, history: SourceTranscript,
        input_event_id: str | None, channel: str | None = None,
        workspace: Path | None = None, include_memory: bool = True,
        tool_names: frozenset[str] | None = None,
        system_prompt_override: str | None = None,
        include_skill_context: bool = True,
        tools: ToolRegistry | None = None,
        session: Session | None = None,
    ) -> SourceTranscript:
        """Bind source identities at construction, never by message offsets."""
        if system_prompt_override is None:
            compiled = self.compile_system_prompt(
                channel=channel, workspace=workspace or self.workspace,
                include_memory=include_memory, tool_names=tool_names, session=session, tools=tools,
            )
            system = self._system_message(compiled)
        else:
            system = {'role': 'system', 'content': system_prompt_override}
        nodes: list[RequestNode] = [RuntimeNode(uuid.uuid4().hex, system), *history.nodes]
        if transcript.current_message is not None:
            current = self.build_current_message(
                transcript.current_message, media=list(transcript.media) if transcript.media else None,
                current_role=transcript.current_role, runtime_context_blocks=transcript.runtime_context_blocks,
                include_skill_context=include_skill_context and transcript.include_skill_context)
            nodes.append(SourceNode(input_event_id, current) if input_event_id is not None
                         else RuntimeNode(uuid.uuid4().hex, current))
        return SourceTranscript(tuple(nodes))

    @staticmethod
    def _system_message(compiled: CompiledPrompt) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "system", "content": compiled.full_system_text}
        if len(blocks := compiled.system_blocks()) > 1:
            message["_meta"] = {SYSTEM_BLOCKS_META: blocks}
        return message

    def _transcript_system_message(
        self,
        *,
        channel: str | None,
        workspace: Path,
        include_memory: bool,
        tool_names: frozenset[str] | None,
        session: Session | None,
        tools: ToolRegistry | None,
    ) -> dict[str, Any]:
        """Compile both tiers from the effective tool registry once."""
        compiled = self.compile_system_prompt(
            channel=channel, workspace=workspace, include_memory=include_memory,
            tool_names=tool_names, session=session, tools=tools,
        )
        return self._system_message(compiled)

    def build_current_message(
        self,
        current_message: str,
        *,
        media: list[str] | None = None,
        current_role: str = "user",
        runtime_context_blocks: Sequence[RuntimeContextBlock] | None = None,
        include_skill_context: bool = True,
    ) -> dict[str, Any]:
        """Build only the fresh turn message without merging it into history."""
        content = self.build_user_content(current_message, image_paths=media)
        envelope = PromptCompiler.build_turn_envelope(
            (runtime_context_blocks or ()) if current_role == "user" else (),
            turn_kind=TurnKind.USER if current_role == "user" else TurnKind.SYSTEM,
            include_skill_context=include_skill_context, user_text=current_message,
            explicit_skill=self.skills.build_explicit_skill_runtime_context,
        )
        merged, runtime_context_meta = append_runtime_context(content, envelope.blocks)
        current: dict[str, Any] = {"role": current_role, "content": merged}
        if current_role == "user" and runtime_context_meta is not None:
            current["_meta"] = {
                RUNTIME_CONTEXT_MESSAGE_META: runtime_context_meta,
            }
        return current

    def build_user_content(
        self,
        text: str,
        image_paths: list[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content from prefiltered image paths."""
        return self.attachments.user_content_from_paths(text, image_paths)
