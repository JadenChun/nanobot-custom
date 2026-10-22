from pathlib import Path

from nanobot.session.execution_owner import acquire_execution
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source import new_source_event


async def test_new_canonical_session_is_immediately_usable_in_its_runtime(
    tmp_path: Path,
) -> None:
    session_keys = ("cli:first-turn-one", "cli:first-turn-two")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_root = tmp_path / "state" / "sessions"
    authority = DurableSessionAuthority(workspace, sessions_root=sessions_root)
    lease = await authority.acquire()
    bindings: dict[str, tuple[str, int, str, str, int]] = {}
    identities: dict[str, tuple[str, int]] = {}
    history: dict[str, str] = {}

    try:
        manager = lease.manager
        for session_key in session_keys:
            admitted = manager.admit_persistent_write(session_key)
            assert admitted.state == "active"
            assert admitted.mode == "canonical"
            assert admitted.binding is not None
            bindings[session_key] = admitted.binding

            backend = manager.source_backend(session_key)
            assert backend is manager._jsonl_store
            history[session_key] = f"first turn for {session_key}"

            # This is the same admission and execution boundary used before a
            # persistent first turn reaches its provider.
            async with acquire_execution(
                backend, session_key, run_id=f"first-turn-{session_key}"
            ) as owner:
                assert owner.key == session_key
                snapshot = owner.read_snapshot()
                assert snapshot.state.session_key == session_key
                identities[session_key] = (
                    snapshot.state.incarnation,
                    snapshot.state.generation,
                )
                backend.source.append_source_events(
                    session_key,
                    snapshot.state,
                    [
                        new_source_event(
                            {"role": "user", "content": history[session_key]},
                            run_id=session_key,
                        )
                    ],
                    owner=owner,
                )

            session = manager.read_session_snapshot(session_key)
            assert session is not None
            assert session.key == session_key
            assert [item["content"] for item in session.messages] == [history[session_key]]
    finally:
        assert lease.close is not None
        await lease.close()

    reopened = await DurableSessionAuthority(workspace, sessions_root=sessions_root).acquire()
    try:
        manager = reopened.manager
        for session_key in session_keys:
            resolved = manager.resolve_session_mode(session_key)
            assert resolved.mode == "canonical"
            assert resolved.binding == bindings[session_key]
            snapshot = manager._jsonl_store.source.load_source_snapshot(session_key)
            assert (snapshot.state.incarnation, snapshot.state.generation) == identities[session_key]
            session = manager.read_session_snapshot(session_key)
            assert session is not None
            assert [item["content"] for item in session.messages] == [history[session_key]]
    finally:
        assert reopened.close is not None
        await reopened.close()
