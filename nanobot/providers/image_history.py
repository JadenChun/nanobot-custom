"""Portable model-input image window; never modify the retained transcript."""

from collections.abc import Mapping
from copy import deepcopy
from typing import TypeGuard

IMAGE_HISTORY_OMISSION = (
    "[Earlier image omitted from model context — do not describe or reference it]"
)


def _blocks(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _image(value: object) -> bool:
    return _mapping(value) and value.get("type") == "image_url"


def recent_image_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep every image in the newest image-bearing message, including tool output.

    Text-only follow-ups do not advance the window. Replacement blocks contain
    neither source paths nor guesses about omitted content. All other fields,
    tool structure and retained image metadata survive in a detached copy.
    """
    prepared = deepcopy(messages)
    newest = None
    for index, message in enumerate(prepared):
        content = message.get("content")
        if _blocks(content) and any(_image(block) for block in content):
            newest = index
    if newest is not None:
        for message in prepared[:newest]:
            content = message.get("content")
            if _blocks(content):
                message["content"] = [
                    {"type": "text", "text": IMAGE_HISTORY_OMISSION} if _image(block) else block
                    for block in content
                ]
    return prepared
