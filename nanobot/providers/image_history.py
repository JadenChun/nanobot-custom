"""Portable model-input image window; never modify the retained transcript."""

from collections.abc import Mapping
from copy import deepcopy
from typing import TypeGuard

from nanobot.utils.media_tokens import MAX_IMAGE_TOKENS, estimate_image_tokens, inline_media_details

IMAGE_HISTORY_OMISSION = (
    "[Earlier image omitted from model context; no image bytes are present. "
    "Reread its saved file with read_file under current access if needed; do not "
    "claim visual details from this placeholder]"
)
IMAGE_HISTORY_DEFERRED = (
    "[Image candidate unavailable in this request due to the review limit. "
    "Use the available images and any earlier observations.]"
)


def _blocks(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _image(value: object) -> bool:
    return _mapping(value) and value.get("type") == "image_url"


def _tool_image_cost(block: object) -> tuple[int, int]:
    value = block.get("image_url") if _mapping(block) else None
    url = value.get("url") if _mapping(value) else value
    if not isinstance(url, str):
        return MAX_IMAGE_TOKENS, 0
    details = inline_media_details(url)
    if details is None or details[0] != "image" or details[1] is None:
        return MAX_IMAGE_TOKENS, len(url)
    return estimate_image_tokens(*details[1]), len(url)


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
    max_tool_image_bytes: int | None = None,
) -> list[dict[str, object]]:
    """Keep every image in the newest image-bearing batch, including tool output.

    A batch is either one message or the contiguous tool-result group from one
    parallel tool call. Text-only follow-ups do not advance the window. A caller
    can limit the newest tool batch by count, visual tokens and bytes. Keep
    the first image available for the transport's normal request validation.
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
            retained_tokens = 0
            retained_bytes = 0
            for message in prepared[batch_start:newest + 1]:
                content = message.get("content")
                if message.get("role") != "tool" or not _blocks(content):
                    continue
                projected: list[object] = []
                for block in content:
                    if _image(block):
                        tokens, size = _tool_image_cost(block)
                        within_budget = retained_tokens + tokens <= MAX_IMAGE_TOKENS and (
                            max_tool_image_bytes is None
                            or retained_bytes + size <= max_tool_image_bytes
                        )
                        if retained_tool_images < max_tool_images and (
                            retained_tool_images == 0 or within_budget
                        ):
                            retained_tool_images += 1
                            retained_tokens += tokens
                            retained_bytes += size
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
