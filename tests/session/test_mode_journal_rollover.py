import asyncio
from pathlib import Path

import nanobot.session.mode_storage as mode_storage
from nanobot.session.runtime_authority import DurableSessionAuthority


def test_mode_journal_rolls_completed_proof_at_capacity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mode_storage, "MODE_JOURNAL_LIMIT", 2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        try:
            manager = lease.manager
            for index in range(5):
                session = manager.reset_session(f"cli:rollover-{index}")
                assert session is not None
                assert session.generation == 0
                assert manager.read_session_snapshot(f"cli:rollover-{index}") is not None

            resolver = manager._mode_resolver
            live = resolver.storage.list(".session-mode/journals")
            assert len(live) <= 2

            retained_dir = ".session-mode/retained-journals"
            assert resolver.storage.exists(retained_dir)
            retained = resolver.storage.list(retained_dir)
            assert len(retained) >= 3

            for operation_id in retained:
                journal = resolver._journal_records(operation_id)
                assert journal
                assert journal[-1]["phase"] in {"complete", "aborted"}
        finally:
            await lease.close()

    asyncio.run(exercise())
