from __future__ import annotations

import pytest

from nanobot.agent.memory_recovery_marker import classify_history_marker_frame

MARKER_ID = "0123456789abcdef01234567"
ENTRY = "[2026-09-25T00:00:00Z] Recovered history"
MARKER = f"<!-- nanobot-consolidation:{MARKER_ID} -->"


def frame(entry: str = ENTRY) -> bytes:
    return (f"{MARKER}\n{entry}").rstrip().encode("utf-8") + b"\n\n"


def assert_absent_proposal_replays(history: bytes, entry: str = ENTRY) -> bytes:
    status, output = classify_history_marker_frame(history, MARKER_ID, entry)

    assert status == "absent"
    assert output is not None
    assert classify_history_marker_frame(output, MARKER_ID, entry) == ("present", None)
    return output


def test_absent_marker_proposes_frame_for_empty_history() -> None:
    assert assert_absent_proposal_replays(b"") == frame()


def test_absent_marker_uses_staging_rstrip_for_receipt_entry() -> None:
    entry = "[記録] Café 🌱  \n\t"

    assert assert_absent_proposal_replays(b"", entry) == frame(entry)


def test_absent_marker_preserves_nonempty_history_bytes_exactly() -> None:
    history = b"# Existing history\r\n\r\nOld entry\n\n"

    assert assert_absent_proposal_replays(history) == history + frame()
    assert history == b"# Existing history\r\n\r\nOld entry\n\n"


def test_nonempty_history_without_record_separator_is_rejected() -> None:
    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(b"Existing record\n", MARKER_ID, ENTRY)


def test_present_requires_exact_complete_frame_as_file_suffix() -> None:
    history = b"# Existing history\n\n" + frame()

    status, output = classify_history_marker_frame(history, MARKER_ID, ENTRY)

    assert (status, output) == ("present", None)


def test_nonterminal_target_frame_is_rejected() -> None:
    history = b"# Existing history\n\n" + frame() + b"Later entry\n\n"

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)


def test_other_complete_marker_does_not_block_absent_result() -> None:
    other_marker = f"<!-- nanobot-consolidation:{'f' * 24} -->\nOther receipt"
    history = other_marker.encode("utf-8") + b"\n\n"

    assert assert_absent_proposal_replays(history) == history + frame()


def test_present_frame_respects_staging_rstrip_with_multibyte_entry() -> None:
    entry = "[記録] Café 🌱  \n\t"
    history = (f"{MARKER}\n{entry}").rstrip().encode("utf-8") + b"\n\n"

    status, output = classify_history_marker_frame(history, MARKER_ID, entry)

    assert (status, output) == ("present", None)


def test_duplicate_target_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(frame() + b"Between records\n\n" + frame(), MARKER_ID, ENTRY)


def test_conflicting_entry_for_target_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(frame("a different entry"), MARKER_ID, ENTRY)


def test_marker_like_text_embedded_in_prose_is_rejected() -> None:
    history = f"A note mentions {MARKER} inside prose.\n\n".encode("utf-8")

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)


def test_partial_marker_frame_is_rejected() -> None:
    history = f"{MARKER}\n{ENTRY}".encode("utf-8")

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)


def test_complete_frame_inside_record_is_rejected() -> None:
    history = b"Existing record\n" + frame()

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)


def test_partial_marker_fragment_is_rejected() -> None:
    history = f"Old record\n\n<!-- nanobot-consolidation:{MARKER_ID}".encode("utf-8")

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)


def test_invalid_utf8_is_rejected_with_content_free_error() -> None:
    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(b"old record\xff", MARKER_ID, ENTRY)


@pytest.mark.parametrize(
    ("marker_id", "entry"),
    [("", ENTRY), ("bad\nid", ENTRY), (MARKER_ID, "  ")],
)
def test_malformed_arguments_use_fixed_error(marker_id: str, entry: str) -> None:
    with pytest.raises(ValueError, match="^invalid history marker frame$") as error:
        classify_history_marker_frame(b"", marker_id, entry)

    assert str(error.value) == "invalid history marker frame"


def test_malformed_mutable_history_input_is_rejected_without_mutation() -> None:
    history = bytearray(b"# Existing history\n\n")
    before = history[:]

    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(history, MARKER_ID, ENTRY)  # type: ignore[arg-type]

    assert history == before
