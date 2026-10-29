from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nanobot.config.schema import ContextRepoConfig
from nanobot.context_repo.manager import ContextRepoRuntimeConfig


def test_allow_exec_defaults_true_and_serializes_camel_case(tmp_path: Path) -> None:
    config = ContextRepoConfig(path=str(tmp_path))
    runtime = ContextRepoRuntimeConfig.from_raw(str(tmp_path))

    assert config.allow_exec is True
    assert config.model_dump(by_alias=True)["allowExec"] is True
    assert runtime is not None
    assert runtime.allow_exec is True


@pytest.mark.parametrize("name", ["allowExec", "allow_exec"])
def test_allow_exec_aliases_preserve_false(tmp_path: Path, name: str) -> None:
    config = ContextRepoConfig.model_validate({"path": str(tmp_path), name: False})
    runtime = ContextRepoRuntimeConfig.from_raw({"path": str(tmp_path), name: False})

    assert config.allow_exec is False
    assert config.model_dump(by_alias=True)["allowExec"] is False
    assert runtime is not None
    assert runtime.allow_exec is False


@pytest.mark.parametrize("value", ["false", 0, None, [], {}])
def test_allow_exec_schema_rejects_non_boolean_values(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValidationError, match="allowExec|allow_exec"):
        ContextRepoConfig.model_validate({"path": str(tmp_path), "allowExec": value})


@pytest.mark.parametrize("name", ["allowExec", "allow_exec"])
@pytest.mark.parametrize("value", ["false", 0, None, [], {}])
def test_allow_exec_raw_mapping_rejects_non_boolean_values(
    tmp_path: Path, name: str, value: object,
) -> None:
    with pytest.raises(ValueError, match="allowExec|allow_exec"):
        ContextRepoRuntimeConfig.from_raw({"path": str(tmp_path), name: value})


@pytest.mark.parametrize("name", ["allowExec", "allow_exec"])
def test_allow_exec_object_normalization_preserves_false(tmp_path: Path, name: str) -> None:
    raw = SimpleNamespace(path=str(tmp_path), **{name: False})

    runtime = ContextRepoRuntimeConfig.from_raw(raw)

    assert runtime is not None
    assert runtime.allow_exec is False


@pytest.mark.parametrize("name", ["allowExec", "allow_exec"])
@pytest.mark.parametrize("value", ["false", 0, None, [], {}])
def test_allow_exec_object_normalization_rejects_non_boolean_values(
    tmp_path: Path, name: str, value: object,
) -> None:
    raw = SimpleNamespace(path=str(tmp_path), **{name: value})

    with pytest.raises(ValueError, match="allowExec|allow_exec"):
        ContextRepoRuntimeConfig.from_raw(raw)
