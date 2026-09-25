"""Pure validation and append proposal for history recovery marker frames."""

from __future__ import annotations

from typing import Literal

_MARKER_PREFIX = "<!-- nanobot-consolidation:"
_MARKER_SUFFIX = " -->"
_INVALID_FRAME_MESSAGE = "invalid history marker frame"


def classify_history_marker_frame(
    existing_history: bytes,
    marker_id: str,
    history_entry: str,
) -> tuple[Literal["absent", "present"], bytes | None]:
    """Classify a marker in HISTORY.md and propose exact staging append bytes.

    A present marker must be a single complete record with the receipt's exact
    entry. Any ambiguous or partial occurrence raises the same content-free
    ``ValueError``. An absent result includes the original bytes unchanged,
    followed by the frame as the staging append routine writes it.
    """
    if (
        not isinstance(existing_history, bytes)
        or not isinstance(marker_id, str)
        or not isinstance(history_entry, str)
        or not marker_id
        or not history_entry.rstrip()
        or any(
            not character.isprintable() or character.isspace() or character in "<>"
            for character in marker_id
        )
    ):
        raise ValueError(_INVALID_FRAME_MESSAGE)

    try:
        text = existing_history.decode("utf-8")
        marker = f"{_MARKER_PREFIX}{marker_id}{_MARKER_SUFFIX}"
        expected_frame = f"{marker}\n{history_entry}".rstrip().encode("utf-8") + b"\n\n"
    except UnicodeError:
        raise ValueError(_INVALID_FRAME_MESSAGE) from None

    marker_positions: list[int] = []
    search_from = 0
    while True:
        position = text.find(_MARKER_PREFIX, search_from)
        if position < 0:
            break

        id_start = position + len(_MARKER_PREFIX)
        line_end = text.find("\n", id_start)
        if line_end < 0:
            line_end = len(text)
        marker_line = text[id_start:line_end]
        if marker_line == marker_id + _MARKER_SUFFIX:
            marker_positions.append(position)
        elif marker_line.endswith(_MARKER_SUFFIX):
            other_id = marker_line[: -len(_MARKER_SUFFIX)]
            if not other_id or any(
                not character.isprintable() or character.isspace() or character in "<>"
                for character in other_id
            ):
                raise ValueError(_INVALID_FRAME_MESSAGE)
        elif marker_id.startswith(marker_line) or marker_line.startswith(marker_id):
            # A truncated ID or altered target marker cannot be treated as absent.
            raise ValueError(_INVALID_FRAME_MESSAGE)
        else:
            # An incomplete marker fragment cannot be safely classified.
            raise ValueError(_INVALID_FRAME_MESSAGE)

        search_from = id_start

    if marker_positions:
        if len(marker_positions) != 1:
            raise ValueError(_INVALID_FRAME_MESSAGE)

        position = marker_positions[0]
        at_record_start = position == 0 or text[:position].endswith("\n\n")
        frame_ends_at_eof = position + len(expected_frame) == len(existing_history)
        if (
            not at_record_start
            or not frame_ends_at_eof
            or not existing_history.startswith(expected_frame, position)
        ):
            raise ValueError(_INVALID_FRAME_MESSAGE)
        return "present", None

    # Keep the prefix byte-for-byte: _append_history_unlocked does not insert
    # or normalize a separator before the supplied record. Reject an unterminated
    # record so the proposed marker remains recognizable on a later replay.
    if existing_history and not existing_history.endswith(b"\n\n"):
        raise ValueError(_INVALID_FRAME_MESSAGE)

    output = existing_history + expected_frame
    return "absent", output
