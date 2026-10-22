"""Request views never act as durable session archival authority."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider


def test_message_only_consolidation_hook_is_retired() -> None:
    assert not hasattr(ContextBudgetManager, "consolidate")


@pytest.mark.parametrize("over_budget", [False, True])
async def test_request_reduction_leaves_durable_memory_untouched(
    tmp_path: Path, over_budget: bool,
) -> None:
    store = MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    store.write_memory("The user's durable preference must survive request fitting.")
    store.append_history("An already archived event.", source="chat")
    paths = [store.memory_file, store.history_file]
    before = [(path.read_bytes(), path.stat()) for path in paths]
    journal = store.read_unprocessed_history(since_cursor=0)
    directory_entries = sorted(path.name for path in store.memory_dir.iterdir())
    provider = MagicMock(spec=LLMProvider)
    # This is an optional provider counter, not a base LLMProvider member.
    provider.estimate_prompt_tokens = MagicMock(side_effect=(
        [(800, "fixture"), (800, "fixture"), (100, "fixture")]
        if over_budget else [(100, "fixture")]
    ))
    manager = ContextBudgetManager(
        ContextBudget(max_tokens=1000, output_reserve=200, safety_buffer=100),
        memory_store=store,
        provider=provider,
        model="fixture-model",
        tool_registry=ToolRegistry(),
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Request-local prefix."},
        {"role": "user", "content": "Old request."},
        {"role": "assistant", "content": "Old answer."},
        {"role": "user", "content": "Current request."},
    ]

    reduced = await manager.enforce_budget(messages, preserve_last_n_turns=1)

    assert reduced is messages
    assert reduced[-1] == {"role": "user", "content": "Current request."}
    assert len(reduced) == (2 if over_budget else 4)
    for path, (data, identity) in zip(paths, before, strict=True):
        current = path.stat()
        assert path.read_bytes() == data
        assert (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_ctime_ns) == (
            identity.st_dev, identity.st_ino, identity.st_mtime_ns, identity.st_ctime_ns,
        )
    assert store.read_unprocessed_history(since_cursor=0) == journal
    assert sorted(path.name for path in store.memory_dir.iterdir()) == directory_entries
    provider.chat.assert_not_called()
    provider.chat_with_retry.assert_not_called()
