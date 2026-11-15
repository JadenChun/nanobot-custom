"""Portable model-input image window; never modify the retained transcript."""

from collections.abc import Mapping
from copy import deepcopy
from typing import TypeGuard

IMAGE_HISTORY_OMISSION = (
    "[Earlier image omitted from model context; no image bytes are present. "
    "Reread its saved file with read_file under current access if needed; do not "
    "claim visual details from this placeholder]"
)
IMAGE_HISTORY_DEFERRED = (
    "[Image candidate not reviewed in this request. Use read_file under current "
    "access to inspect its saved file when needed.]"
)


def _blocks(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _image(value: object) -> bool:
    return _mapping(value) and value.get("type") == "image_url"


def _newest_image_batch_start(messages: list[dict[str, object]], newest: int) -> int:
    """Keep a parallel tool-call batch together when its results carry images.

    Provider protocols emit one tool message per call, so a single assistant
    batch that read three images arrives as three adjacent tool messages.
    Keeping only the newest message silently drops the other two images.
    """
    if messages[newest].get("role") != "tool":
        return newest
    start = newest
    while start > 0 and messages[start - 1].get("role") == "tool":
        start -= 1
    return start


def recent_image_messages(
    messages: list[dict[str, object]],
    *,
    max_tool_images: int | None = None,
) -> list[dict[str, object]]:
    """Keep every image in the newest image-bearing batch, including tool output.

    A batch is either one message or the contiguous tool-result group from one
    parallel tool call. Text-only follow-ups do not advance the window. A caller
    can limit images in a newest tool batch while retaining every tool message.
    Replacement blocks contain neither source paths nor guesses about omitted
    content. All other fields, tool structure and retained image metadata
    survive in a detached copy.
    """
    prepared = deepcopy(messages)
    newest = None
    for index, message in enumerate(prepared):
        content = message.get("content")
        if _blocks(content) and any(_image(block) for block in content):
            newest = index
    if newest is not None:
        batch_start = _newest_image_batch_start(prepared, newest)
        if max_tool_images is not None and prepared[newest].get("role") == "tool":
            retained_tool_images = 0
            for message in prepared[batch_start:newest + 1]:
                content = message.get("content")
                if message.get("role") != "tool" or not _blocks(content):
                    continue
                projected: list[object] = []
                for block in content:
                    if _image(block):
                        if retained_tool_images < max_tool_images:
                            retained_tool_images += 1
                            projected.append(block)
                        else:
                            projected.append({"type": "text", "text": IMAGE_HISTORY_DEFERRED})
                    else:
                        projected.append(block)
                message["content"] = projected
        for message in prepared[:batch_start]:
            content = message.get("content")
            if _blocks(content):
                message["content"] = [
                    {"type": "text", "text": IMAGE_HISTORY_OMISSION} if _image(block) else block
                    for block in content
                ]
    return prepared
