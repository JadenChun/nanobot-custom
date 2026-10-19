"""Preservation checks for existing CLI callbacks, not runtime admission."""

from __future__ import annotations

import inspect
import io
import json
import sys

import pytest
from rich.console import Console

from nanobot.cli import gateway, log_control
from nanobot.cli.migration_preflight import _Parser, main
from nanobot.config.schema import Config


@pytest.fixture
def gateway_logging(monkeypatch: pytest.MonkeyPatch):
    captured = []

    def replace(sink: object, **kwargs: object) -> int:
        captured.append((sink, kwargs))
        return 17

    monkeypatch.setattr(gateway, "replace_cli_console_sink", replace)
    app = gateway.create_gateway_app(
        console=Console(file=io.StringIO()),
        log_handler_id=17,
        load_runtime_config=lambda _config, _workspace: Config(),
        run_gateway=lambda _config, **_kwargs: None,
    )
    assert app.registered_callback is not None
    callback = app.registered_callback.callback
    assert inspect.isfunction(callback)
    configure = inspect.getclosurevars(callback).nonlocals["configure_logging"]
    assert callable(configure)
    return configure, captured


def test_gateway_nonverbose_does_not_replace_sink(gateway_logging):
    configure, captured = gateway_logging
    configure(False)
    assert captured == []


def test_gateway_verbose_keeps_sink_and_options(gateway_logging):
    configure, captured = gateway_logging
    configure(True)
    assert len(captured) == 1
    sink, kwargs = captured[0]
    assert sink is sys.stderr
    assert kwargs["level"] == "DEBUG"
    assert kwargs["colorize"] is None
    assert kwargs["format"] == (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<cyan>{extra[channel]}</cyan> | "
        "<level>{message}</level>"
    )
    assert callable(kwargs["filter"])
    assert kwargs["filter"].__name__ == "<lambda>"


@pytest.mark.parametrize("value", ["telegram", True, 1, ["channel"]])
def test_gateway_filter_returns_existing_truthy_channel_identity(gateway_logging, value):
    configure, captured = gateway_logging
    configure(True)
    callback = captured[0][1]["filter"]
    extra = {"channel": value, "other": "kept"}
    record = {"extra": extra}
    assert callback(record) is value
    assert record["extra"] is extra
    assert extra == {"channel": value, "other": "kept"}


@pytest.mark.parametrize("value", [None, False, 0, "", []])
def test_gateway_filter_keeps_falsey_value_but_returns_true(gateway_logging, value):
    configure, captured = gateway_logging
    configure(True)
    callback = captured[0][1]["filter"]
    extra = {"channel": value}
    assert callback({"extra": extra}) is True
    assert extra["channel"] is value


def test_gateway_filter_adds_default_in_place(gateway_logging):
    configure, captured = gateway_logging
    configure(True)
    callback = captured[0][1]["filter"]
    extra = {"other": "kept"}
    result = callback({"extra": extra})
    assert result == "-"
    assert result is extra["channel"]
    assert extra == {"other": "kept", "channel": "-"}


def test_gateway_filter_repeats_native_setdefault_on_each_call(gateway_logging):
    configure, captured = gateway_logging
    configure(True)
    callback = captured[0][1]["filter"]
    calls = []

    class Extra(dict):
        def setdefault(self, key, default=None):
            calls.append((key, default))
            return super().setdefault(key, default)

    extra = Extra()
    assert callback({"extra": extra}) == "-"
    assert callback({"extra": extra}) == "-"
    assert calls == [("channel", "-"), ("channel", "-")]


@pytest.mark.parametrize(
    "record, error",
    [({}, KeyError), ({"extra": None}, AttributeError), (None, TypeError)],
)
def test_gateway_filter_retains_native_failures(gateway_logging, record, error):
    configure, captured = gateway_logging
    configure(True)
    callback = captured[0][1]["filter"]
    with pytest.raises(error):
        callback(record)


@pytest.fixture
def console_sink(monkeypatch: pytest.MonkeyPatch):
    captured = []
    enabled = []
    cleared = []

    def replace(sink: object, **kwargs: object) -> int:
        captured.append((sink, kwargs))
        return 18

    monkeypatch.setattr(log_control, "replace_cli_console_sink", replace)
    monkeypatch.setattr(log_control, "clear_compatibility_log_file", lambda: cleared.append(True))
    monkeypatch.setattr(log_control.logger, "enable", enabled.append)
    log_control.configure_agent_runtime_logs(show_console=True, log_file=None)
    assert len(captured) == 1
    callback, options = captured[0]
    assert callable(callback)
    assert callback.__name__ == "<lambda>"
    assert options == {"format": log_control._COMPATIBILITY_LOG_FORMAT}
    assert enabled == ["nanobot"]
    assert cleared == [True]
    return callback


def test_console_sink_uses_late_stderr_and_exact_native_write_result(console_sink, monkeypatch):
    calls = []

    class Message:
        def __str__(self) -> str:
            calls.append("str")
            return "Unicode: \N{PILE OF POO}\n"

    class Writer:
        def write(self, text: str) -> int:
            calls.append(("write", text))
            return 213

    with monkeypatch.context() as context:
        context.setattr(sys, "stderr", Writer())
        result = console_sink(Message())
    assert result == 213
    assert calls == ["str", ("write", "Unicode: \N{PILE OF POO}\n")]


def test_console_sink_propagates_conversion_failure_without_write(console_sink, monkeypatch):
    error = ValueError("synthetic conversion refusal")
    writes = []

    class Message:
        def __str__(self) -> str:
            raise error

    class Writer:
        def write(self, text: str) -> int:
            writes.append(text)
            return 0

    with monkeypatch.context() as context:
        context.setattr(sys, "stderr", Writer())
        with pytest.raises(ValueError) as caught:
            console_sink(Message())
    assert caught.value is error
    assert writes == []


def test_console_sink_propagates_write_failure(console_sink, monkeypatch):
    error = OSError("synthetic write refusal")
    writes = []

    class Writer:
        def write(self, text: str) -> int:
            writes.append(text)
            raise error

    with monkeypatch.context() as context:
        context.setattr(sys, "stderr", Writer())
        with pytest.raises(OSError) as caught:
            console_sink("unchanged")
    assert caught.value is error
    assert writes == ["unchanged"]


@pytest.mark.parametrize("message", ["", "private-token", "路径", "line\nsecret"])
def test_parser_error_is_fixed_and_does_not_emit_input(message, capsys):
    parser = _Parser(add_help=False, allow_abbrev=False)
    with pytest.raises(ValueError) as caught:
        parser.error(message)
    assert str(caught.value) == "invalid arguments"
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


@pytest.mark.parametrize("args", [[], ["--private-token=secret"], ["--help"]])
def test_preflight_invalid_arguments_remain_content_free(args, capsys):
    assert main(args) == 2
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["result"] == "NOT_READY"
    assert report["activation"] == "NOT_AUTHORIZED"
    assert "secret" not in output.out
    assert output.err == ""
