"""Malformed-input and fault preservation for recovery defensive views."""

from types import SimpleNamespace

import pytest

from nanobot.agent import memory_recovery_manifest as manifest
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_marker import classify_history_marker_frame


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("value", [None, 0, False, object(), bytearray(b"value")])
def test_marker_invalid_types_keep_content_free_refusal(position, value):
    args = [b"", "marker", "entry"]
    args[position] = value
    with pytest.raises(ValueError, match="^invalid history marker frame$") as error:
        classify_history_marker_frame(*args)
    assert str(error.value) == "invalid history marker frame"


@pytest.mark.parametrize("method", ["__bool__", "rstrip", "__iter__"])
def test_marker_string_subclass_faults_keep_identity_and_order(method):
    fault = KeyboardInterrupt("synthetic fault")
    trace = []

    def fail(self, *args):
        trace.append(method)
        raise fault

    text = type("FaultText", (str,), {method: fail})("marker")
    marker, entry = ("marker", text) if method == "rstrip" else (text, "entry")
    with pytest.raises(KeyboardInterrupt) as error:
        classify_history_marker_frame(b"", marker, entry)
    assert error.value is fault
    assert trace == [method]


@pytest.mark.parametrize("history", [
    b"", b"Retained history\n\n", "Retained 記録\n\n".encode(),
    "🌱 Café\n\n".encode(), "中文 café 🌱\r\n\n".encode(),
])
def test_marker_subclasses_preserve_exact_append_and_replay(history):
    class Text(str):
        pass

    status, output = classify_history_marker_frame(history, Text("marker"), Text("entry 🌱  \n"))
    assert status == "absent"
    assert output == history + "<!-- nanobot-consolidation:marker -->\nentry 🌱\n\n".encode()
    assert classify_history_marker_frame(output, Text("marker"), Text("entry 🌱  \n")) == ("present", None)


@pytest.mark.parametrize("history", [b"Retained\n\n", "Retained 記録 🌱\n\n".encode()])
@pytest.mark.parametrize("marker,entry", [("marker", "entry"), ("記録🌱", "café 中文 🌱")])
def test_complete_unicode_prefix_frame_replays(history, marker, entry):
    frame = f"<!-- nanobot-consolidation:{marker} -->\n{entry}\n\n".encode()
    assert classify_history_marker_frame(history + frame, marker, entry) == ("present", None)


@pytest.mark.parametrize("history", [b"Retained\n\n", "Retained 記録 🌱\n\n".encode()])
@pytest.mark.parametrize("damage", ["truncated", "nonterminal", "duplicate", "entry", "separator"])
def test_unicode_prefix_damaged_frames_still_refuse(history, damage):
    frame = "<!-- nanobot-consolidation:記録🌱 -->\ncafé 中文 🌱\n\n".encode()
    value = {
        "truncated": history + frame[:-1],
        "nonterminal": history + frame + b"later\n\n",
        "duplicate": history + frame + frame,
        "entry": history + frame.replace("café".encode(), b"changed"),
        "separator": history[:-2] + frame,
    }[damage]
    with pytest.raises(ValueError, match="^invalid history marker frame$"):
        classify_history_marker_frame(value, "記録🌱", "café 中文 🌱")


@pytest.mark.parametrize("value", [None, object(), "namespace", SimpleNamespace(
    kind="transactions", identity="a" * 64,
)])
def test_manifest_wrong_namespace_keeps_invalid_refusal(value):
    with pytest.raises(manifest.MemoryManifestError) as error:
        manifest.MemoryRecoveryManifest(value)
    assert error.value.code == "invalid"
    assert str(error.value) == "memory recovery manifest input is invalid"
    assert error.value.__cause__ is None and error.value.__suppress_context__


@pytest.mark.parametrize("field", ["kind", "identity"])
@pytest.mark.parametrize("exception_type", [AttributeError, ValueError, KeyboardInterrupt])
def test_manifest_namespace_accessor_faults_keep_refusal_or_system_identity(
    monkeypatch, field, exception_type,
):
    fault = exception_type("synthetic namespace fault")
    namespace = object.__new__(storage.ImmutableEvidenceNamespace)
    trace = []
    original = storage.ImmutableEvidenceNamespace.__getattribute__

    def read(self, name):
        if name in {"kind", "identity"}:
            trace.append(name)
            if name == field:
                raise fault
            return "transactions" if name == "kind" else "a" * 64
        return original(self, name)

    monkeypatch.setattr(storage.ImmutableEvidenceNamespace, "__getattribute__", read)
    if exception_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt) as error:
            manifest.MemoryRecoveryManifest(namespace)
        assert error.value is fault
    else:
        with pytest.raises(manifest.MemoryManifestError) as error:
            manifest.MemoryRecoveryManifest(namespace)
        assert error.value.code == "invalid"
        assert error.value.__cause__ is None and error.value.__suppress_context__
        assert error.value.__context__ is fault
    assert trace == (["kind"] if field == "kind" else ["kind", "identity"])
