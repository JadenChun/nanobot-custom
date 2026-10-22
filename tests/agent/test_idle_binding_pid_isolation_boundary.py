"""Real snapshot GC must not enter another module's registry PID trace."""

import gc
import os
import traceback
from pathlib import Path
from types import CodeType

import pytest

from nanobot.agent import memory_recovery_storage as storage
from tests.agent.idle_acceptance_support import NetworkDenial
from tests.agent.test_idle_binding_owner_operations import (
    test_existing_public_revalidate_refuses_invalid_inputs_without_effects as original_revalidate,
)


@pytest.mark.parametrize("kind", ["hostile", "subclass", "none", "object"])
def test_real_snapshot_gc_does_not_contaminate_registry_pid_probe(tmp_path, monkeypatch, kind):
    denial = NetworkDenial()
    denial.install(monkeypatch)
    gc.collect()
    workspace = tmp_path / "memory-workspace"
    workspace.mkdir()
    source_path = tmp_path / "source-case"
    source_path.mkdir()
    collecting = [False]
    origins = []
    original_pid = os.getpid
    forget_code = next(
        value
        for value in storage.WorkspaceMemoryAuthority._register_target_snapshot_locked.__code__.co_consts
        if isinstance(value, CodeType) and value.co_name == "forget"
    )
    forget_lines = {line for _, _, line in forget_code.co_lines() if line is not None}

    def observe_real_pid():
        if collecting[0]:
            origins.append([
                (str(Path(frame.filename).resolve()), frame.name, frame.lineno)
                for frame in traceback.extract_stack()
                if frame.name == "forget"
            ])
        return original_pid()

    monkeypatch.setattr(os, "getpid", observe_real_pid)
    with storage.WorkspaceMemoryAuthority(workspace) as authority:
        snapshots = [authority.snapshot_target("MEMORY.md") for _ in range(7)]
        assert len(authority._target_snapshot_issuance) == 7
        cycle = [snapshots]
        cycle.append(cycle)
        retained_cycles = [cycle]
        del cycle, snapshots

        class CollectAtProbeSetup(pytest.MonkeyPatch):
            def setattr(self, target, name, value, raising=True):
                super().setattr(target, name, value, raising=raising)
                if name == "load_source_snapshot":
                    retained_cycles.clear()
                    collecting[0] = True
                    try:
                        gc.collect()
                    finally:
                        collecting[0] = False
                    assert authority._target_snapshot_issuance == {}
                    assert len(origins) == 14
                    assert all(
                        len(stack) == 1
                        and stack[0][0] == str(Path(storage.__file__).resolve())
                        and stack[0][1] == "forget"
                        and stack[0][2] in forget_lines
                        for stack in origins
                    )
                    print("REAL_SNAPSHOT_GC_PID_CALL_ORIGINS", origins)

        probe = CollectAtProbeSetup()
        try:
            original_revalidate(source_path, probe, kind)
        finally:
            probe.undo()
        assert authority._target_snapshot_issuance == {}
        assert len(origins) == 14
    assert denial.attempts == []
