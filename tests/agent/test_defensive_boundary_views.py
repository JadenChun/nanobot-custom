"""Preserve runtime rejection at concretely typed, defensive input boundaries."""

from types import SimpleNamespace

import pytest

from nanobot.agent.capabilities import role_capabilities, role_tool_names
from nanobot.agent.context_governance import ContextGovernor
from nanobot.agent.memory import MemoryStore
from nanobot.agent.model_runtime import ModelRuntimeResolver
from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.agent.tools.trend_vpngate import VpnGateDiscoveryError, parse_api_response
from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.sdk.clients import SessionClient
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_verification import verify_published_artifact
from nanobot.session.mode_storage import SessionModeError, SessionModeStorage
from nanobot.session.source import SourceValidationError
from nanobot.session.source_root import SourceRootAuthority, SourceRootAuthorityError


class _NotText:
    def __repr__(self):
        return "not-text"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"invalid input read: {name}")


def _forbid(*args, **kwargs):
    raise AssertionError("boundary accessed an effect or later authority")


NOT_TEXT = [None, 7, True, b"text", [], _NotText()]


@pytest.mark.parametrize("role", NOT_TEXT + ["", "unknown"])
def test_role_refusal_keeps_nontext_inputs_out_of_normalization(role):
    with pytest.raises(ValueError, match="unknown delegated role"):
        role_capabilities(role)


def test_role_mcp_filter_keeps_single_iteration_and_original_allowlist():
    trace = []

    def names():
        for value in (None, [], 1, "", "allowed", "other"):
            trace.append(value)
            yield value

    original = role_capabilities("planner").tools
    assert role_tool_names("planner", read_only_mcp_tools=names()) == original | {
        "allowed", "other",
    }
    assert trace == [None, [], 1, "", "allowed", "other"]
    assert role_tool_names("worker", read_only_mcp_tools=_NotText()) == role_capabilities("worker").tools


@pytest.mark.parametrize("summary", NOT_TEXT + ["", "   "])
def test_source_summary_refusal_precedes_proof_and_writer(summary):
    transcript = SimpleNamespace(messages=lambda: [])
    source = SimpleNamespace(has_pending_requests=False, projection_epoch=1, transcript=transcript)
    compaction = SimpleNamespace(raw_messages=[], stream_id=None, writer=_NotText())
    candidate = SimpleNamespace(transcript=transcript, stream_id=None)
    attempt = SimpleNamespace(candidate=candidate, projection_epoch=1)
    state = SimpleNamespace(source=source, source_compaction=compaction)
    governor = object.__new__(ContextGovernor)
    with pytest.raises(SourceValidationError, match="Invalid bounded source summary"):
        governor.publish_source_compaction(state, attempt, boundary=1, summary=summary,
                                           auxiliary_usage=None, proof=_NotText())


@pytest.mark.parametrize("source", NOT_TEXT[1:])
def test_history_source_refusal_precedes_normalization_lock_and_write(source, monkeypatch):
    store = object.__new__(MemoryStore)
    monkeypatch.setattr(MemoryStore, "_normalize_history_entry", _forbid)
    with pytest.raises(TypeError, match="history source must be a string or None"):
        store.append_history("entry", source=source)


@pytest.mark.parametrize("name", NOT_TEXT + ["", "   "])
def test_qualified_model_name_refusal_precedes_snapshot_or_provider(name):
    resolver = object.__new__(ModelRuntimeResolver)
    with pytest.raises(ValueError, match="model_preset must be a non-empty string"):
        resolver.resolve_qualified_preset(name, _NotText())


@pytest.mark.parametrize("count", [True, False, "1", 1.5, [], _NotText()])
async def test_image_count_refusal_precedes_config_and_provider(count):
    tool = object.__new__(ImageGenerationTool)
    result = await tool.execute("prompt", count=count)
    assert "count is outside tools.imageGeneration.maxImagesPerTurn" in result


@pytest.mark.parametrize("text", NOT_TEXT + ["", "   "])
def test_vpngate_text_refusal_precedes_size_count_and_csv(text):
    with pytest.raises(VpnGateDiscoveryError, match="empty relay list"):
        parse_api_response(text, max_candidates=_NotText())


@pytest.mark.parametrize("count", [None, True, False, "1", 1.5, [], _NotText()])
def test_vpngate_count_refusal_retains_prior_text_phase(count):
    trace = []

    class Text(str):
        def strip(self, *args, **kwargs):
            trace.append("strip")
            return super().strip(*args, **kwargs)

        def encode(self, *args, **kwargs):
            trace.append("encode")
            return super().encode(*args, **kwargs)

    with pytest.raises(VpnGateDiscoveryError, match="max_candidates must be an integer"):
        parse_api_response(Text("not a relay"), max_candidates=count)
    assert trace == ["strip", "encode"]


@pytest.mark.parametrize("parts,reads", [(None, 2), (7, 2), ({}, 2), ("text", 1), (b"text", 1)])
def test_receipt_sequence_refusal_keeps_short_circuit_and_field_reads(parts, reads, monkeypatch):
    trace = []
    original = ChannelTextReceipt.__getattribute__

    def read(self, name):
        if name == "parts":
            trace.append(name)
        return original(self, name)

    monkeypatch.setattr(ChannelTextReceipt, "__getattribute__", read)
    with pytest.raises(ValueError, match="parts must be an ordered sequence"):
        ChannelTextReceipt("telegram", "target", "a" * 64, "b" * 64, parts)
    assert trace == ["parts"] * reads


def test_receipt_sequence_retains_part_identity_and_four_reads(monkeypatch):
    trace = []
    original = ChannelTextReceipt.__getattribute__

    def read(self, name):
        if name == "parts":
            trace.append(name)
        return original(self, name)

    part = ChannelTextPartReceipt(0, 0, 0, 1, 2)
    monkeypatch.setattr(ChannelTextReceipt, "__getattribute__", read)
    receipt = ChannelTextReceipt("telegram", "target", "a" * 64, "b" * 64, [part])
    # Three validation reads plus the original to_json read for the size bound.
    assert trace == ["parts"] * 4
    assert receipt.parts == (part,) and receipt.parts[0] is part


@pytest.mark.parametrize("key", NOT_TEXT + [""])
@pytest.mark.parametrize("entry", ["ingest", "restore"])
async def test_sdk_key_refusal_precedes_loop_resolution_and_session_effects(key, entry):
    client = SessionClient(_forbid)
    with pytest.raises(ValueError, match="must include a session key"):
        if entry == "ingest":
            await client.ingest(key, _NotText())
        else:
            await client.restore(SimpleNamespace(key=key, messages=_NotText()))


@pytest.mark.parametrize("descriptor", NOT_TEXT)
def test_artifact_descriptor_refusal_precedes_all_authority_and_io(descriptor):
    with pytest.raises(ArtifactValidationError, match="Invalid artifact verification input"):
        verify_published_artifact(descriptor, _NotText(), _NotText(), store=_NotText(),
                                  workspace=None, policy=None, resolve_read=_forbid)


@pytest.mark.parametrize("name", NOT_TEXT + ["", "/absolute"])
def test_mode_relative_name_refusal_precedes_split_and_storage(name):
    with pytest.raises(SessionModeError):
        SessionModeStorage._parts(name)


@pytest.mark.parametrize("data", NOT_TEXT + [bytearray(b"x"), b"long"])
def test_mode_data_refusal_precedes_parent_open_and_write(data, monkeypatch):
    storage = object.__new__(SessionModeStorage)
    monkeypatch.setattr(SessionModeStorage, "_parent", _forbid)
    with pytest.raises(SessionModeError):
        storage.write("file", data, limit=2)


@pytest.mark.parametrize("name", NOT_TEXT + ["", "/absolute"])
def test_source_name_refusal_retains_prior_backend_authority(name, monkeypatch):
    trace = []
    root = object.__new__(SourceRootAuthority)
    root._backend = object()
    monkeypatch.setattr(SourceRootAuthority, "_require_backend",
                        lambda self, backend: trace.append("backend"))
    with pytest.raises(SourceRootAuthorityError, match="invalid relative source name"):
        root._open_relative(name, 0)
    assert trace == ["backend"]


@pytest.mark.parametrize("text", NOT_TEXT)
def test_source_text_refusal_retains_backend_and_write_authority(text, monkeypatch):
    from nanobot.session import mode_storage

    trace = []
    root = object.__new__(SourceRootAuthority)
    root._backend = object()
    monkeypatch.setattr(SourceRootAuthority, "_require_backend",
                        lambda self, backend: trace.append("backend"))
    monkeypatch.setattr(mode_storage, "require_namespace_write_access",
                        lambda backend: trace.append("write-authority"))
    monkeypatch.setattr(SourceRootAuthority, "_open_relative", _forbid)
    with pytest.raises(SourceRootAuthorityError, match="invalid source replacement"):
        root.write_text_atomic("checkpoint.json", text)
    assert trace == ["backend", "write-authority"]
