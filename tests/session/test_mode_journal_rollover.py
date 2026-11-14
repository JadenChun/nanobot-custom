from pathlib import Path

import pytest

import nanobot.session.mode_storage as mode_storage
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source import new_source_event


@pytest.mark.asyncio
async def test_mode_journal_rollover_preserves_histories_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 2
    monkeypatch.setattr(mode_storage, "MODE_JOURNAL_LIMIT", limit)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_root = tmp_path / "sessions"
    histories = {
        f"cli:rollover-{index}": [
            ("user", f"user message {index}"),
            ("assistant", f"assistant message {index}"),
        ]
        for index in range(3)
    }

    lease = await DurableSessionAuthority(workspace, sessions_root=sessions_root).acquire()
    try:
        manager = lease.manager
        for key, messages in histories.items():
            manager.admit_persistent_write(key)
            manager.reset_session(
                key,
                replacement_events=[
                    new_source_event({"role": role, "content": content}, run_id=key)
                    for role, content in messages
                ],
            )
        journals = manager.sessions_dir / ".session-mode"
    finally:
        await lease.close()

    reopened = await DurableSessionAuthority(workspace, sessions_root=sessions_root).acquire()
    try:
        for key, expected in histories.items():
            snapshot = reopened.manager.read_session_snapshot(key)
            assert snapshot is not None
            assert [(message["role"], message["content"]) for message in snapshot.messages] == expected
    finally:
        await reopened.close()

    assert len(list((journals / "journals").iterdir())) <= limit
    assert any((journals / "retained-journals").iterdir())
