"""Requirement case U3: two explicit instances keep one usage store each."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.llm_usage import (
    get_llm_usage_store,
    llm_usage_store_path,
    record_llm_call,
)
from nanobot.llm_usage.models import LLMCallRecord
from nanobot.providers.base import LLMUsage
from nanobot.webui.settings_system import settings_usage_payload


def _config(root: Path) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    path = root / "config.json"
    path.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(workspace)}}}),
        encoding="utf-8",
    )
    return path


def _call(marker: str) -> LLMCallRecord:
    return LLMCallRecord(
        started_at_ms=int(time.time() * 1000),
        duration_ms=10,
        provider="usage-probe",
        model=marker,
        source="user",
        stream=False,
        finish_reason="stop",
        usage=LLMUsage.reported(input_tokens=5, output_tokens=5),
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_two_explicit_instances_keep_separate_usage_stores(tmp_path: Path) -> None:
    root_a = tmp_path / "rootA"
    root_b = tmp_path / "rootB"
    root_a.mkdir()
    root_b.mkdir()
    config_a = _config(root_a)
    config_b = _config(root_b)
    instance_a = load_config(config_a)
    instance_b = load_config(config_b)
    assert instance_a.runtime_data_dir is not None
    assert instance_b.runtime_data_dir is not None
    store_a = get_llm_usage_store(llm_usage_store_path(instance_a.runtime_data_dir))
    store_b = get_llm_usage_store(llm_usage_store_path(instance_b.runtime_data_dir))

    record_llm_call(_call("caseA"), data_dir=instance_a.runtime_data_dir)
    rows_a = store_a.recent_calls(limit=10)
    assert len(rows_a) == 1
    assert rows_a[0]["model"] == "caseA"
    root_a_before = _snapshot(root_a)

    record_llm_call(_call("caseB"), data_dir=instance_b.runtime_data_dir)
    rows_b = store_b.recent_calls(limit=10)
    assert len(rows_b) == 1
    assert rows_b[0]["model"] == "caseB"

    assert len(store_a.recent_calls(limit=10)) == 1
    assert all(row["model"] != "caseB" for row in store_a.recent_calls(limit=10))
    assert (root_a / "llm_usage.sqlite3").is_file()
    assert (root_b / "llm_usage.sqlite3").is_file()
    assert _snapshot(root_a) == root_a_before

    assert settings_usage_payload(instance_a)["requests_30d"] == 1
    assert settings_usage_payload(instance_b)["requests_30d"] == 1
