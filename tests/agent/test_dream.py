"""Tests for Dream memory consolidation — build_dream_prompt and cursor management."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.config.schema import ModelPresetConfig
from nanobot.providers.base import LLMResponse
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    default_workspace_scope,
    reset_workspace_scope,
)
from nanobot.utils.prompt_templates import render_template
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync


def _dream_loop(
    inputs: AgentInputs, provider: MagicMock, hooks: list[AgentHook] | None = None,
) -> AgentLoop:
    loop = AgentLoop(
        bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
        context_window_tokens=32_000, hooks=hooks,
        session_manager=inputs.sessions, tool_registry=inputs.tools,
        cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
        tools_config=inputs.config.tools, restrict_to_workspace=True,
        enable_legacy_auto_compact=False,
    )
    assert loop.context.memory.workspace == inputs.config.workspace_path
    assert loop.consolidator.store is loop.context.memory
    return loop


def _dream_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.supports_tools = True
    provider.generation = MagicMock(max_tokens=4096)
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=[], finish_reason="stop", usage=None)
    )
    return provider


def _dream_store(loop: AgentLoop) -> MemoryStore:
    store = loop.context.memory
    store.write_soul("# Soul")
    store.write_memory("# Memory")
    return store


def _identity_content(_context: AgentHookContext, content: str | None) -> str | None:
    return content


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_memory("# Memory\n- Project X active")
    return s


class TestBuildDreamPrompt:
    def test_returns_none_when_no_history(self, store):
        assert store.build_dream_prompt() is None

    def test_returns_prompt_with_history(self, store):
        store.append_history("hello")
        result = store.build_dream_prompt()
        assert result is not None
        prompt, cursor = result
        assert cursor > 0
        assert "## Conversation History" in prompt
        assert "hello" in prompt

    def test_cursor_advances_only_new_entries(self, store):
        store.append_history("first")
        r1 = store.build_dream_prompt()
        assert r1 is not None
        _, c1 = r1

        # Cursor not yet advanced — same entries are still available
        assert store.build_dream_prompt() is not None

        # Advance cursor
        store.set_last_dream_cursor(c1)
        # Now no new entries
        assert store.build_dream_prompt() is None

        # Add new entry
        store.append_history("second")
        r2 = store.build_dream_prompt()
        assert r2 is not None
        _, c2 = r2
        assert c2 > c1

    def test_prompt_includes_skill_creator_path(self, store):
        store.append_history("test")
        result = store.build_dream_prompt()
        assert result is not None
        prompt, _ = result
        assert "skill-creator" in prompt

    def test_prompt_does_not_duplicate_current_memory_file_contents(self, store):
        store.append_history("hello")
        result = store.build_dream_prompt()
        assert result is not None
        prompt, _ = result
        assert "## Current Memory Files" not in prompt
        assert "Project X active" not in prompt
        assert "Helpful" not in prompt

    def test_workspace_dream_prompt_overrides_default(self, store):
        store.dream_prompt_file.parent.mkdir(parents=True)
        store.dream_prompt_file.write_text(
            "Custom Dream prompt.",
            encoding="utf-8",
        )
        store.append_history("keep this fact")

        result = store.build_dream_prompt()

        assert result is not None
        prompt, _ = result
        assert prompt.startswith("Custom Dream prompt.")
        assert "memory consolidation engine" not in prompt
        assert "## Conversation History" in prompt
        assert "keep this fact" in prompt

    def test_workspace_dream_prompt_override_is_capped(self, store):
        store.dream_prompt_file.parent.mkdir(parents=True)
        store.dream_prompt_file.write_text("x" * 40_000, encoding="utf-8")
        store.append_history("keep this fact")

        result = store.build_dream_prompt()

        assert result is not None
        prompt, _ = result
        assert "x" * 40_000 not in prompt
        assert "... (truncated)" in prompt
        assert "## Conversation History" in prompt
        assert "keep this fact" in prompt

    def test_empty_workspace_dream_prompt_uses_default(self, store):
        store.dream_prompt_file.parent.mkdir(parents=True)
        store.dream_prompt_file.write_text("  \n", encoding="utf-8")
        store.append_history("test")

        result = store.build_dream_prompt()

        assert result is not None
        prompt, _ = result
        assert "memory consolidation engine" in prompt

    def test_truncates_long_entries_at_1000_chars(self, store):
        long_content = "x" * 2000
        store.append_history(long_content)
        result = store.build_dream_prompt()
        assert result is not None
        prompt, _ = result
        assert long_content not in prompt
        assert "x" * 1000 in prompt
        assert "x" * 1001 not in prompt

    def test_batches_oldest_unprocessed_entries_first(self, store):
        for i in range(25):
            store.append_history(f"entry-{i + 1:02d}")

        result = store.build_dream_prompt(max_entries=20)
        assert result is not None
        prompt, cursor = result

        assert cursor == 20
        assert "entry-01" in prompt
        assert "entry-20" in prompt
        assert "entry-21" not in prompt

        store.set_last_dream_cursor(cursor)
        next_result = store.build_dream_prompt(max_entries=20)
        assert next_result is not None
        next_prompt, next_cursor = next_result
        assert next_cursor == 25
        assert "entry-21" in next_prompt
        assert "entry-25" in next_prompt

    def test_skips_malformed_history_entries(self, store):
        """Dream prompt building should tolerate externally corrupted JSONL rows."""
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "usable memory"}\n',
            encoding="utf-8",
        )

        result = store.build_dream_prompt()

        assert result is not None
        prompt, cursor = result
        assert cursor == 2
        assert "usable memory" in prompt

    def test_dream_prompt_consumes_consolidator_attribute_tags(self):
        prompt = render_template(
            "agent/dream.md",
            strip=True,
            skill_creator_path="skills/skill-creator/SKILL.md",
        )

        assert "History attribute tags" in prompt
        assert "[skip]: audit-only" in prompt
        assert "[correction]: replace the older conflicting fact" in prompt
        assert "Always strip these bracketed tags from saved memory content" in prompt


class TestDreamRunCompletion:
    """The runner's terminal state gates Dream cursor advancement."""

    class _Resp:
        def __init__(self, stop_reason: str = "completed") -> None:
            self.metadata = {"_stop_reason": stop_reason}

    def test_completed_stop_reason_completes(self):
        assert MemoryStore.dream_run_completed(self._Resp())

    @pytest.mark.parametrize(
        "stop_reason",
        ["error", "tool_error", "max_iterations", "cancelled"],
    )
    def test_non_completed_stop_reason_blocks(self, stop_reason: str):
        assert not MemoryStore.dream_run_completed(self._Resp(stop_reason))

    def test_missing_response_metadata_blocks(self):
        assert not MemoryStore.dream_run_completed(None)

    def test_incompletion_reason_names_the_cause(self):
        assert MemoryStore.dream_incompletion_reason(
            self._Resp("max_iterations")
        ) == "stop_reason: max_iterations"
        assert MemoryStore.dream_incompletion_reason(None) == (
            "stop_reason: missing response metadata"
        )


class TestDreamTools:
    def test_dream_tools_are_restricted_to_file_edits(self, store):
        tools = store.build_dream_tools()

        assert set(tools.tool_names) == {
            "apply_patch",
            "edit_file",
            "read_file",
            "write_file",
        }

    @pytest.mark.asyncio
    async def test_dream_can_edit_canonical_memory_files(self, store):
        tools = store.build_dream_tools()

        memory_result = await tools.execute(
            "apply_patch",
            {
                "edits": [
                    {
                        "path": "memory/MEMORY.md",
                        "action": "replace",
                        "old_text": "Project X active",
                        "new_text": "Project Y active",
                    }
                ]
            },
        )
        soul_result = await tools.execute(
            "edit_file",
            {
                "path": "SOUL.md",
                "old_text": "Helpful",
                "new_text": "Precise",
            },
        )
        user_result = await tools.execute(
            "write_file",
            {
                "path": "USER.md",
                "content": "# User Profile\n\n- **Name**: Ada\n",
            },
        )

        assert "Patch applied" in memory_result
        assert "Successfully edited" in soul_result
        assert "Successfully wrote" in user_result
        assert "Project Y active" in store.memory_file.read_text(encoding="utf-8")
        assert "Precise" in store.soul_file.read_text(encoding="utf-8")
        assert "**Name**: Ada" in store.user_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_dream_can_write_workspace_skills(self, store):
        tools = store.build_dream_tools()
        target = store.workspace / "skills" / "demo" / "SKILL.md"

        result = await tools.execute(
            "write_file",
            {
                "path": "skills/demo/SKILL.md",
                "content": "---\nname: demo\ndescription: Demo skill.\n---\n\nUse when needed.\n",
            },
        )

        assert "Successfully wrote" in result
        assert target.read_text(encoding="utf-8").startswith("---\nname: demo")

    @pytest.mark.asyncio
    async def test_dream_tools_keep_internal_write_scope_under_full_access(self, store):
        tools = store.build_dream_tools()
        scope = default_workspace_scope(store.workspace, restrict_to_workspace=False)
        outside = store.workspace.parent / f"{store.workspace.name}-outside"
        outside.mkdir()
        outside_target = outside / "escape.txt"
        skill_target = store.workspace / "skills" / "scoped" / "SKILL.md"

        token = bind_workspace_scope(scope)
        try:
            outside_result = await tools.execute(
                "write_file",
                {"path": str(outside_target), "content": "owned"},
            )
            skill_result = await tools.execute(
                "apply_patch",
                {
                    "edits": [
                        {
                            "path": "skills/scoped/SKILL.md",
                            "action": "add",
                            "new_text": "---\nname: scoped\n---\n",
                        }
                    ]
                },
            )
        finally:
            reset_workspace_scope(token)

        assert "outside allowed directory" in outside_result
        assert not outside_target.exists()
        assert "Patch applied" in skill_result
        assert skill_target.read_text(encoding="utf-8").startswith("---\nname: scoped")

    @pytest.mark.asyncio
    async def test_dream_cannot_modify_memory_internal_files(self, store):
        tools = store.build_dream_tools()
        store.history_file.write_text("before\n", encoding="utf-8")
        store._dream_cursor_file.write_text("1", encoding="utf-8")

        history_result = await tools.execute(
            "apply_patch",
            {
                "edits": [
                    {
                        "path": "memory/history.jsonl",
                        "action": "replace",
                        "old_text": "before",
                        "new_text": "after",
                    }
                ]
            },
        )
        cursor_result = await tools.execute(
            "edit_file",
            {
                "path": "memory/.dream_cursor",
                "old_text": "1",
                "new_text": "2",
            },
        )
        history_write_result = await tools.execute(
            "write_file",
            {
                "path": "memory/history.jsonl",
                "content": "after\n",
            },
        )

        assert "outside allowed directory" in history_result
        assert "outside allowed directory" in cursor_result
        assert "outside allowed directory" in history_write_result
        assert store.history_file.read_text(encoding="utf-8") == "before\n"
        assert store._dream_cursor_file.read_text(encoding="utf-8") == "1"

    @pytest.mark.asyncio
    async def test_dream_cannot_create_children_under_canonical_files(self, store):
        tools = store.build_dream_tools()

        memory_child = store.memory_file / "evil.txt"
        user_child = store.user_file / "evil.txt"
        memory_result = await tools.execute(
            "apply_patch",
            {
                "edits": [
                    {
                        "path": "memory/MEMORY.md/evil.txt",
                        "action": "add",
                        "new_text": "owned",
                    }
                ]
            },
        )
        user_result = await tools.execute(
            "write_file",
            {
                "path": "USER.md/evil.txt",
                "content": "owned",
            },
        )

        assert "outside allowed directory" in memory_result
        assert "outside allowed directory" in user_result
        assert not memory_child.exists()
        assert not user_child.exists()


class TestEphemeralDirect:
    """Tests for the ephemeral flag that skips history.jsonl writes for Dream."""

    @pytest.fixture
    def _make_loop(self, tmp_path, admitted_sync):
        """The synchronous preset case retains one helper-owned Runner."""
        provider = _dream_provider()
        loop = admitted_sync(tmp_path, lambda inputs: _dream_loop(inputs, provider))
        return loop, _dream_store(loop)

    @pytest.fixture
    async def _make_loop_async(self, tmp_path, admitted_async):
        """Acquire and close real durable services on the test's event loop."""
        provider = _dream_provider()
        loop = await admitted_async(
            tmp_path, lambda inputs: _dream_loop(inputs, provider),
            keys=("dream:test", "cli:normal", "dream:check", "dream:error",
                  "dream:handled-tool-error"),
        )
        return loop, _dream_store(loop)

    def test_dream_runtime_uses_preset_without_changing_default(self, _make_loop):
        loop, _ = _make_loop
        loop.runtime_resolver._model_presets = {
            "dream": ModelPresetConfig(model="dream-model"),
        }
        loop.dream_model_preset = "dream"

        runtime = loop.dream_runtime()

        assert runtime is not None
        assert runtime.model == "dream-model"
        assert runtime.model_preset == "dream"
        assert loop.model == "test-model"
        assert loop.model_preset is None

    async def test_ephemeral_skips_raw_archive(self, tmp_path, _make_loop_async):
        """When ephemeral=True, raw_archive must not be called."""
        from unittest.mock import patch

        loop, store = _make_loop_async

        with patch.object(loop.context.memory, "raw_archive") as mock_archive:
            await loop.process_direct(
                "test", session_key="dream:test", ephemeral=True,
            )
            mock_archive.assert_not_called()

    async def test_non_ephemeral_runs_normally(self, tmp_path, _make_loop_async):
        """Without ephemeral, the normal path returns the model response."""
        loop, store = _make_loop_async
        response = await loop.process_direct("test", session_key="cli:normal")

        assert response is not None
        assert response.content == "done"
        loop.provider.chat_with_retry.assert_awaited()

    async def test_ephemeral_sets_ctx_flag(self, tmp_path, _make_loop_async):
        """Verify that ephemeral=True is forwarded to TurnContext."""
        from unittest.mock import patch

        loop, store = _make_loop_async

        captured = {}

        original_save = loop._persist_turn

        async def patched_save(ctx):
            captured["ephemeral"] = ctx.ephemeral
            return await original_save(ctx)

        with patch.object(loop, "_persist_turn", side_effect=patched_save):
            await loop.process_direct(
                "test", session_key="dream:check", ephemeral=True,
            )

        assert captured.get("ephemeral") is True

    async def test_default_ephemeral_is_false(self, tmp_path, _make_loop_async):
        """By default ephemeral is False in TurnContext."""
        from unittest.mock import patch

        loop, store = _make_loop_async

        captured = {}

        original_save = loop._persist_turn

        async def patched_save(ctx):
            captured["ephemeral"] = ctx.ephemeral
            return await original_save(ctx)

        with patch.object(loop, "_persist_turn", side_effect=patched_save):
            await loop.process_direct("test", session_key="cli:normal")

        assert captured.get("ephemeral") is False

    async def test_ephemeral_response_reports_stop_reason(self, tmp_path, _make_loop_async):
        loop, store = _make_loop_async
        loop.provider.chat_with_retry.return_value = LLMResponse(
            content="provider error",
            finish_reason="error",
        )

        resp = await loop.process_direct(
            "test", session_key="dream:error", ephemeral=True,
        )

        assert resp is not None
        assert resp.metadata["_stop_reason"] == "error"
        assert MemoryStore.dream_run_completed(resp) is False

    async def test_completed_response_after_tool_error_is_success(self, _make_loop_async):
        """Historical recovery-name lineage: a structured edit error is terminal."""
        from unittest.mock import AsyncMock

        from nanobot.providers.base import ToolCallRequest

        loop, store = _make_loop_async
        before_soul = store.soul_file.read_bytes()
        before_cursor = store.get_last_dream_cursor()
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(
                content="trying an edit",
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id="call_edit",
                    name="edit_file",
                    arguments={
                        "path": "SOUL.md",
                        "old_text": "text that is not present",
                        "new_text": "replacement",
                    },
                )],
                usage=None,
            ),
            LLMResponse(content="done", finish_reason="stop", tool_calls=[], usage=None),
        ])

        resp = await loop.process_direct(
            "test",
            session_key="dream:handled-tool-error",
            ephemeral=True,
            tools=store.build_dream_tools(),
        )

        assert resp is not None
        assert resp.metadata["_stop_reason"] == "tool_error"
        assert MemoryStore.dream_run_completed(resp) is False
        loop.provider.chat_with_retry.assert_awaited_once()
        assert store.soul_file.read_bytes() == before_soul
        assert store.get_last_dream_cursor() == before_cursor

    async def test_dream_turn_can_skip_unbatched_recent_history(self, tmp_path, admitted_async):
        """Dream must only see the batch selected by build_dream_prompt."""
        from unittest.mock import MagicMock

        captured: dict[str, list[dict]] = {}
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.supports_tools = True
        provider.generation = MagicMock(max_tokens=4096)

        async def chat_with_retry(**kwargs):
            captured["messages"] = kwargs["messages"]
            return LLMResponse(content="done", finish_reason="stop")

        provider.chat_with_retry = chat_with_retry
        loop = await admitted_async(
            tmp_path, lambda inputs: _dream_loop(inputs, provider), keys=("dream:test",),
        )
        store = loop.context.memory
        for i in range(60):
            store.append_history(f"entry-{i + 1:02d}")

        result = store.build_dream_prompt(max_entries=20)
        assert result is not None
        prompt, cursor = result
        assert cursor == 20

        await loop.process_direct(
            prompt,
            session_key="dream:test",
            ephemeral=True,
            tools=store.build_dream_tools(),
        )

        messages = captured["messages"]
        system_prompt = messages[0]["content"]
        request_text = "\n".join(str(message.get("content", "")) for message in messages)
        assert "# Recent History" not in system_prompt
        assert "entry-01" in request_text
        assert "entry-20" in request_text
        assert "entry-21" not in request_text
        assert "entry-60" not in request_text

    async def test_dream_turn_injects_memory_files_once_and_persists_session(
        self, tmp_path, admitted_async,
    ):
        """Dream gets durable files from system context without losing its session record."""
        from unittest.mock import MagicMock

        markers = {
            "SOUL.md": "DREAM_SOUL_MARKER",
            "USER.md": "DREAM_USER_MARKER",
            "memory/MEMORY.md": "DREAM_MEMORY_MARKER",
        }
        captured: dict[str, list[dict]] = {}
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.supports_tools = True
        provider.generation = MagicMock(max_tokens=4096)

        async def chat_with_retry(**kwargs):
            captured["messages"] = kwargs["messages"]
            return LLMResponse(content="done", finish_reason="stop")

        provider.chat_with_retry = chat_with_retry
        loop = await admitted_async(
            tmp_path, lambda inputs: _dream_loop(inputs, provider),
            keys=("dream:single-memory-copy",),
        )
        store = loop.context.memory
        store.write_soul(markers["SOUL.md"])
        store.write_user(markers["USER.md"])
        store.write_memory(markers["memory/MEMORY.md"])
        store.append_history("history-marker")
        (store.workspace / "AGENTS.md").write_text("DREAM_AGENTS_MARKER", encoding="utf-8")

        result = store.build_dream_prompt()
        assert result is not None
        prompt, _ = result
        session_key = "dream:single-memory-copy"

        await loop.process_direct(
            prompt,
            session_key=session_key,
            ephemeral=True,
            tools=store.build_dream_tools(),
        )

        messages = captured["messages"]
        system_prompt = str(messages[0]["content"])
        request_text = "\n".join(str(message.get("content", "")) for message in messages)
        for marker in [*markers.values(), "DREAM_AGENTS_MARKER"]:
            assert marker in system_prompt
            assert request_text.count(marker) == 1
        assert loop.sessions._get_session_path(session_key).exists()


class TestEphemeralHooks:
    """When ephemeral=True, extra hooks must not fire."""

    @pytest.fixture
    async def _make_loop_with_spy(self, tmp_path, admitted_async):
        """Build an AgentLoop with a spy hook to verify hook firing behavior."""
        from unittest.mock import AsyncMock, MagicMock

        from nanobot.agent.hook import AgentHook
        from nanobot.providers.base import LLMResponse

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.supports_tools = True
        provider.generation = MagicMock(max_tokens=4096)
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="done", finish_reason="stop", tool_calls=[], usage=None,
            )
        )

        spy = MagicMock(spec=AgentHook)
        spy.wants_streaming.return_value = False
        spy.before_iteration = AsyncMock()
        spy.after_iteration = AsyncMock()
        spy.finalize_content.side_effect = _identity_content

        loop = await admitted_async(
            tmp_path, lambda inputs: _dream_loop(inputs, provider, hooks=[spy]),
            keys=("dream:hook-test", "cli:normal"),
        )

        return loop, spy

    async def test_extra_hooks_skipped_when_ephemeral(self, tmp_path, _make_loop_with_spy):
        """When ephemeral=True, extra hooks must not fire."""
        loop, spy = _make_loop_with_spy

        await loop.process_direct(
            "test", session_key="dream:hook-test", ephemeral=True,
        )
        spy.before_iteration.assert_not_called()
        spy.after_iteration.assert_not_called()

    async def test_extra_hooks_fire_for_normal_sessions(self, tmp_path, _make_loop_with_spy):
        """Without ephemeral, extra hooks should fire normally."""
        import json
        from unittest.mock import patch

        from nanobot.session.manager import Session

        loop, spy = _make_loop_with_spy
        checkpoints = []
        original_save = loop.sessions.save_runtime_checkpoint

        def save_checkpoint(session: Session) -> None:
            original_save(session)
            path = loop.sessions._get_runtime_checkpoint_path(session.key)
            checkpoints.append(json.loads(path.read_text(encoding="utf-8")))

        with patch.object(loop.sessions, "save_runtime_checkpoint", side_effect=save_checkpoint):
            response = await loop.process_direct("test", session_key="cli:normal")
        spy.before_iteration.assert_called()
        assert response is not None and response.content == "done"
        assert any(record["checkpoint"].get("assistant_message", {}).get("content") == "done"
                   for record in checkpoints)
        loop.sessions.invalidate("cli:normal")
        assert loop.sessions.get_or_create("cli:normal").get_history()[-1]["content"] == "done"


@pytest.mark.parametrize("content", ["done", None])
def test_spy_identity_content_implements_hook_default(content):
    context = AgentHookContext(iteration=0, messages=[])
    assert _identity_content(context, content) == AgentHook().finalize_content(context, content)

class TestDreamCommitMessage:
    def test_commit_message_reflects_real_diff_not_narrative(self, tmp_path):
        """The Dream commit message must mirror the real git diff and ignore the
        LLM's narrative, so ``/dream-log`` can never lie.

        Regression for the hallucinated-commit bug: commit ``a72ca2a`` claimed a
        "Medical Research" section that never reached the diff.
        """
        from dulwich.objects import Blob, Commit, Tree
        from dulwich.repo import Repo

        store = MemoryStore(tmp_path)
        store.write_soul("# Soul")
        store.write_memory("# Memory")
        store.git.init()
        store.git.auto_commit("initial state")

        # A real edit to a tracked content file.
        store.write_memory("# Memory\n- DMSO research notes")

        # A lying narrative the old code would have appended verbatim.
        lying = "Added a Medical Research section (Mastic Gum, DMSO) to MEMORY.md"
        diff_body = store.dream_content_diff()
        assert diff_body, "real edit must be detected"
        assert "DMSO research notes" in diff_body
        assert lying not in diff_body

        msg = MemoryStore.build_dream_commit_message(
            "dream: periodic memory consolidation", diff_body,
        )
        assert lying not in msg
        assert "DMSO research notes" in msg

        sha = store.git.auto_commit(msg)
        assert sha is not None
        with Repo(str(tmp_path)) as repo:
            head = repo.head()
            assert head.decode("ascii").startswith(sha)
            commit = repo[head]
            assert isinstance(commit, Commit)
            log = commit.message.decode("utf-8").strip()
            tree = repo[commit.tree]
            assert isinstance(tree, Tree)
            _, blob_id = tree.lookup_path(repo.__getitem__, b"memory/MEMORY.md")
            blob = repo[blob_id]
            assert isinstance(blob, Blob)
            assert blob.data == store.memory_file.read_bytes()
            assert b"DMSO research notes" in blob.data
        assert "dream: periodic memory consolidation" in log
        assert "DMSO research notes" in log
        assert lying not in log

    def test_commit_message_is_bare_prefix_when_no_changes(self, tmp_path):
        """A no-op Dream run yields only the prefix — never a narrated summary."""
        store = MemoryStore(tmp_path)
        store.write_soul("# Soul")
        store.write_memory("# Memory")
        store.git.init()
        store.git.auto_commit("initial state")

        # No edits at all.
        assert store.dream_content_diff() == ""
        msg = MemoryStore.build_dream_commit_message(
            "dream: manual run", store.dream_content_diff(),
        )
        assert msg == "dream: manual run"

    def test_build_commit_message_ignores_none_and_empty_body(self):
        assert MemoryStore.build_dream_commit_message("dream: x", "") == "dream: x"
        assert MemoryStore.build_dream_commit_message("dream: x", None) == "dream: x"
        assert MemoryStore.build_dream_commit_message("dream: x", "  ") == "dream: x"
        assert (
            MemoryStore.build_dream_commit_message("dream: x", "SOUL.md: +1 -0")
            == "dream: x\n\nSOUL.md: +1 -0"
        )


class TestDreamContentDiff:
    """The ground-truth signal that gates cursor advance and commit messages."""

    def test_empty_when_git_not_initialized(self, store):
        assert store.dream_content_diff() == ""

    def test_empty_when_no_tracked_changes(self, store):
        store.git.init()
        store.git.auto_commit("initial")
        assert store.dream_content_diff() == ""

    def test_ignores_platform_line_ending_normalization(self, store, monkeypatch):
        from dulwich import porcelain
        from dulwich.config import ConfigFile
        from dulwich.objects import Blob, Commit, Tree
        from dulwich.repo import Repo

        system_config = store.workspace / "system.gitconfig"
        global_config = store.workspace / "global.gitconfig"
        system_config.touch()
        global_config.touch()
        config = ConfigFile()
        config.set((b"core",), b"autocrlf", b"false")
        config.write_to_path(system_config)
        monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_config))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

        store.soul_file.write_bytes(b"# Soul\r\n- Helpful")
        store.memory_file.write_bytes(b"# Memory\r\n- Project X active")
        store.git.init()

        paths = (b"SOUL.md", b"memory/MEMORY.md")
        with Repo(str(store.workspace)) as repo:
            assert repo.get_config_stack().get((b"core",), b"autocrlf") == b"false"
            commit = repo[repo.head()]
            assert isinstance(commit, Commit)
            tree = repo[commit.tree]
            assert isinstance(tree, Tree)
            index = repo.open_index()
            for path in paths:
                _, blob_id = tree.lookup_path(repo.__getitem__, path)
                blob = repo[blob_id]
                assert isinstance(blob, Blob)
                worktree = (store.workspace / path.decode("utf-8")).read_bytes()
                assert b"\r\n" in worktree
                assert blob.data == worktree
                assert index[path].sha == blob_id
        status = porcelain.status(str(store.workspace))
        # Equivalent to porcelain restricted to the original two exact paths.
        assert all(
            path not in staged and path not in status.unstaged and path not in status.untracked
            for staged in status.staged.values() for path in paths
        )
        assert store.dream_content_diff() == ""

        # No re-staging: HEAD and index keep CRLF while raw worktree becomes LF.
        for path in paths:
            target = store.workspace / path.decode("utf-8")
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
        with Repo(str(store.workspace)) as repo:
            index = repo.open_index()
            for path in paths:
                blob = repo[index[path].sha]
                assert isinstance(blob, Blob)
                worktree = (store.workspace / path.decode("utf-8")).read_bytes()
                assert b"\r\n" in blob.data
                assert b"\r\n" not in worktree
                assert blob.data.replace(b"\r\n", b"\n") == worktree
        changed_status = porcelain.status(str(store.workspace))
        assert all(path in changed_status.unstaged for path in paths)
        assert not any(path in staged for staged in changed_status.staged.values() for path in paths)
        assert store.dream_content_diff() == ""

    @pytest.mark.parametrize(
        ("before", "after", "changed"),
        [
            (b"# Memory\r\n", b"# Memory\n", False),
            (b"# Memory\n", b"# Memory", True),
            (b"# Memory\r", b"# Memory\n", True),
        ],
    )
    def test_only_ignores_crlf_lf_changes(self, store, before, after, changed):
        store.memory_file.write_bytes(before)
        store.git.init()

        store.memory_file.write_bytes(after)

        assert bool(store.dream_content_diff()) is changed

    def test_reflects_real_content_edits(self, store):
        store.git.init()
        store.git.auto_commit("initial")
        store.write_memory("# Memory\n- DMSO research notes")
        diff = store.dream_content_diff()
        assert diff
        assert "memory/MEMORY.md" in diff
        assert "DMSO research notes" in diff

    def test_ignores_cursor_only_changes(self, store):
        """Advancing the cursor must not count as a productive content edit."""
        store.git.init()
        store.git.auto_commit("initial")
        store.set_last_dream_cursor(99)  # only memory/.dream_cursor changes
        assert store.dream_content_diff() == ""
