"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path, skill_paths: list[Path] | None = None, context_path: Path | None = None):
        self.workspace = workspace
        self.context_path = context_path
        self.memory = MemoryStore(workspace)

        # If a context repo is configured, include its skills dir in extra paths
        all_skill_paths = list(skill_paths or [])
        if context_path and (context_path / "skills").is_dir():
            all_skill_paths.append(context_path / "skills")

        self.skills = SkillsLoader(workspace, extra_paths=all_skill_paths or None)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # Context repo memory (read-only, supplemental)
        if self.context_path:
            ctx_memory_file = self.context_path / "memory" / "MEMORY.md"
            if ctx_memory_file.exists():
                ctx_memory = ctx_memory_file.read_text(encoding="utf-8")
                if ctx_memory.strip():
                    parts.append(f"# Context Memory\n\n{ctx_memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        context_section = ""
        if self.context_path:
            ctx_path = str(self.context_path.expanduser().resolve())
            context_section = (
                f"\n## Context Repository\n"
                f"A shared context repository is loaded from: {ctx_path}\n"
                f"- Context memory, skills, and bootstrap files from this repo supplement your workspace.\n"
                f"- Context repo files are read-only - write your own data to the workspace.\n"
            )

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md
{context_section}
Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel.

## Tool Call Guidelines
- Before calling tools, you may briefly state your intent (e.g. "Let me check that"), but NEVER predict or describe the expected result before receiving it.
- Before modifying a file, read it first to confirm its current content.
- Do not assume a file or directory exists — use list_dir or read_file to verify.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.

## Memory
- Remember important facts: write to {workspace_path}/memory/MEMORY.md
- Recall past events: grep {workspace_path}/memory/HISTORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace and context repo."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        # Load additional bootstrap files from context repo (won't override workspace files)
        if self.context_path:
            loaded_names = {f for f in self.BOOTSTRAP_FILES if (self.workspace / f).exists()}
            for filename in self.BOOTSTRAP_FILES:
                if filename not in loaded_names:
                    ctx_file = self.context_path / filename
                    if ctx_file.exists():
                        content = ctx_file.read_text(encoding="utf-8")
                        parts.append(f"## {filename} (context)\n\n{content}")

            # Load any extra .md files from context repo root (not in BOOTSTRAP_FILES)
            if self.context_path.is_dir():
                for md_file in sorted(self.context_path.glob("*.md")):
                    if md_file.name not in self.BOOTSTRAP_FILES and md_file.name != "README.md":
                        content = md_file.read_text(encoding="utf-8")
                        parts.append(f"## {md_file.name} (context)\n\n{content}")

        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded media (images, video, audio)."""
        if not media:
            return text

        media_parts: list[dict[str, Any]] = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime:
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            if mime.startswith("image/"):
                media_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            elif mime.startswith("video/"):
                media_parts.append({"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}})
            elif mime.startswith("audio/"):
                ext = p.suffix.lstrip(".") or mime.split("/")[-1]
                media_parts.append({"type": "input_audio", "input_audio": {"data": b64, "format": ext}})

        if not media_parts:
            return text
        return media_parts + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result — plain string or multimodal content blocks.

        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Always include content — some providers (e.g. StepFun) reject
        # assistant messages that omit the key entirely.
        msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
