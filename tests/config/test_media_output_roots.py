from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.config.schema import ToolsConfig


def test_media_output_roots_normalize_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "comfy-output"
    config = ToolsConfig(media_output_roots=[str(root)])
    assert config.media_output_roots == [str(root)]
    assert ToolsConfig.model_validate(
        {"mediaOutputRoots": [str(root)]}
    ).media_output_roots == [str(root)]


@pytest.mark.parametrize("value", ["relative/output", "/", ""])
def test_media_output_roots_reject_unsafe_paths(value: str) -> None:
    with pytest.raises(ValidationError):
        ToolsConfig(media_output_roots=[value])
