from io import StringIO
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from nanobot.cli import stream as stream_mod
from nanobot.cli import terminal


@pytest.fixture(autouse=True)
def restore_terminal_state(monkeypatch):
    """Keep each command unit from leaking terminal singletons to other tests."""
    monkeypatch.setattr(terminal, "_prompt_session", None)
    monkeypatch.setattr(terminal, "_saved_term_attrs", None)


@pytest.fixture
def mock_prompt_session():
    """Mock the global prompt session."""
    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock()
    with patch("nanobot.cli.terminal._prompt_session", mock_session), \
         patch("nanobot.cli.terminal.patch_stdout"):
        yield mock_session


@pytest.mark.asyncio
async def test_read_interactive_input_async_returns_input(mock_prompt_session):
    """Test that _read_interactive_input_async returns the user input from prompt_session."""
    mock_prompt_session.prompt_async.return_value = "hello world"

    result = await terminal._read_interactive_input_async()

    assert result == "hello world"
    mock_prompt_session.prompt_async.assert_called_once()
    args, _ = mock_prompt_session.prompt_async.call_args
    assert isinstance(args[0], HTML)  # Verify HTML prompt is used


@pytest.mark.asyncio
async def test_read_interactive_input_async_handles_eof(mock_prompt_session):
    """Test that EOFError converts to KeyboardInterrupt."""
    mock_prompt_session.prompt_async.side_effect = EOFError()

    with pytest.raises(KeyboardInterrupt):
        await terminal._read_interactive_input_async()


def test_init_prompt_session_creates_session(tmp_path):
    """Test that _init_prompt_session initializes the global session."""
    history_file = tmp_path / "history" / "cli_history"
    with patch("nanobot.cli.terminal.PromptSession") as mock_session, \
         patch("nanobot.cli.terminal.SafeFileHistory") as mock_history, \
         patch("nanobot.config.paths.get_cli_history_path", return_value=history_file):
        terminal._init_prompt_session()

        assert terminal._prompt_session is not None
        mock_session.assert_called_once()
        _, kwargs = mock_session.call_args
        assert kwargs["multiline"] is True
        assert kwargs["enable_open_in_editor"] is False
        assert kwargs["key_bindings"] is not None
        assert kwargs["history"] is mock_history.return_value
        mock_history.assert_called_once_with(str(history_file))
        assert history_file.parent.is_dir()


def test_thinking_spinner_pause_stops_and_restarts():
    """Pause should stop the active spinner and restart it afterward."""
    spinner = MagicMock()
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    thinking = stream_mod.ThinkingSpinner(console=mock_console)
    with thinking:
        with thinking.pause():
            pass

    assert spinner.method_calls == [
        call.start(),
        call.stop(),
        call.start(),
        call.stop(),
    ]


def test_print_cli_progress_line_pauses_spinner_before_printing():
    """CLI progress output should pause spinner to avoid garbled lines."""
    order: list[str] = []
    spinner = MagicMock()
    spinner.start.side_effect = lambda: order.append("start")
    spinner.stop.side_effect = lambda: order.append("stop")
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    with patch.object(terminal.console, "print", side_effect=lambda *_args, **_kwargs: order.append("print")):
        thinking = stream_mod.ThinkingSpinner(console=mock_console)
        with thinking:
            terminal._print_cli_progress_line("tool running", thinking)

    assert order == ["start", "stop", "print", "start", "stop"]


@pytest.mark.asyncio
async def test_print_interactive_progress_line_pauses_spinner_before_printing():
    """Interactive progress output should also pause spinner cleanly."""
    order: list[str] = []
    spinner = MagicMock()
    spinner.start.side_effect = lambda: order.append("start")
    spinner.stop.side_effect = lambda: order.append("stop")
    mock_console = MagicMock()
    mock_console.status.return_value = spinner

    async def fake_print(_text: str) -> None:
        order.append("print")

    with patch("nanobot.cli.terminal._print_interactive_line", side_effect=fake_print):
        thinking = stream_mod.ThinkingSpinner(console=mock_console)
        with thinking:
            await terminal._print_interactive_progress_line("tool running", thinking)

    assert order == ["start", "stop", "print", "start", "stop"]


def test_response_renderable_uses_text_for_explicit_plain_rendering():
    status = (
        "🐈 nanobot v0.1.4.post5\n"
        "🧠 Model: MiniMax-M2.7\n"
        "📊 Tokens: 20639 in / 29 out"
    )

    renderable = terminal._response_renderable(
        status,
        render_markdown=True,
        metadata={"render_as": "text"},
    )

    assert renderable.__class__.__name__ == "Text"


def test_response_renderable_preserves_normal_markdown_rendering():
    renderable = terminal._response_renderable("**bold**", render_markdown=True)

    assert renderable.__class__.__name__ == "Markdown"


def test_response_renderable_without_metadata_keeps_markdown_path():
    help_text = "🐈 nanobot commands:\n/status — Show bot status\n/help — Show available commands"

    renderable = terminal._response_renderable(help_text, render_markdown=True)

    assert renderable.__class__.__name__ == "Markdown"


@pytest.mark.asyncio
async def test_stream_renderer_keeps_live_content_without_reprint():
    """Erase the transient display, then persist one native captured final render."""
    display = StringIO()
    console = Console(file=display, force_terminal=False, width=80)
    live_instances: list[object] = []
    events: list[str] = []

    class FakeLive:
        def __init__(self, renderable, *, console, auto_refresh, transient):
            self.renderable = renderable
            self.console = console
            self.auto_refresh = auto_refresh
            self.transient = transient
            self.started = False
            self.stopped = False
            self.updated: list[object] = []
            self.refreshed = 0
            live_instances.append(self)

        def start(self):
            self.started = True
            events.append("start")

        def update(self, renderable):
            self.updated.append(renderable)
            events.append("update")

        def refresh(self):
            self.refreshed += 1
            events.append("refresh")

        def stop(self):
            self.stopped = True
            events.append("stop")

    class RecordingOutput(StringIO):
        def write(self, value):
            events.append("write")
            return super().write(value)

        def flush(self):
            events.append("flush")
            return super().flush()

    final_output = RecordingOutput()
    with patch("nanobot.cli.stream._make_console", return_value=console), \
         patch("nanobot.cli.stream.Live", FakeLive), \
         patch("sys.stdout", final_output):
        renderer = stream_mod.StreamRenderer(
            render_markdown=False,
            show_spinner=False,
        )
        await renderer.on_delta("hello\nworld")
        await renderer.on_end()

    assert len(live_instances) == 1
    live = live_instances[0]
    assert live.started is True
    assert live.stopped is True
    # The current owner erases Live before writing one persistent final render.
    assert live.transient is True
    # The renderable passed to Live should contain the full content
    assert isinstance(live.renderable, Text)
    assert live.renderable.plain == "hello\nworld"
    assert live.refreshed == 3
    assert len(live.updated) == 1
    assert isinstance(live.updated[0], Text)
    assert live.updated[0].plain == "hello\nworld"
    assert events == ["start", "refresh", "refresh", "update", "refresh", "stop", "write", "flush"]
    assert renderer._live is None
    assert final_output.getvalue() == "hello\nworld\n"
    assert "hello\nworld" not in display.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("render_markdown", (False, True))
@pytest.mark.parametrize("chunks", (("hello world",), ("hello ", "world")))
async def test_native_live_persists_final_render_once(chunks, render_markdown):
    """Real Rich Live/capture keeps no duplicate body in the display or stdout."""
    display, final_output, expected_output = StringIO(), StringIO(), StringIO()
    console = Console(file=display, force_terminal=False, width=80)
    expected_console = Console(file=expected_output, force_terminal=False, width=80)
    content = "".join(chunks)
    expected_console.print(Markdown(content) if render_markdown else Text(content))
    with patch("nanobot.cli.stream._make_console", return_value=console), \
         patch("sys.stdout", final_output):
        renderer = stream_mod.StreamRenderer(render_markdown=render_markdown, show_spinner=False)
        for chunk in chunks:
            await renderer.on_delta(chunk)
        live = renderer._live
        assert isinstance(live, Live)
        assert live.is_started is True
        assert live.transient is True
        with patch.object(live, "stop", wraps=live.stop) as stop:
            await renderer.on_end()
        stop.assert_called_once_with()

    assert live.is_started is False
    assert renderer._live is None
    assert renderer.streamed is True
    assert final_output.getvalue() == expected_output.getvalue()
    assert final_output.getvalue().count("hello world") == 1
    assert "hello world" not in display.getvalue()
