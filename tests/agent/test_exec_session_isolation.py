from types import SimpleNamespace
from unittest.mock import MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import Config
from tests.agent import p6_admitted_fixtures

admitted_sync = p6_admitted_fixtures.admitted_sync


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096, temperature=0.1, reasoning_effort=None)
    return provider


def test_agent_loops_do_not_share_exec_session_managers(tmp_path, admitted_sync):
    def build(inputs):
        return AgentLoop.from_config(
            inputs.config, inputs.bus, provider=_provider(), model="test-model",
            context_window_tokens=4096, session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
        )

    loop_a = admitted_sync(tmp_path, build)
    loop_b = admitted_sync(tmp_path, build)

    exec_a = loop_a.tools.get("exec")
    stdin_a = loop_a.tools.get("exec_session")
    list_a = loop_a.tools.get("list_exec_sessions")
    exec_b = loop_b.tools.get("exec")

    assert exec_a._session_manager is loop_a._exec_session_manager
    assert stdin_a._manager is loop_a._exec_session_manager
    assert list_a._manager is loop_a._exec_session_manager
    assert exec_b._session_manager is loop_b._exec_session_manager
    assert loop_a._exec_session_manager is not loop_b._exec_session_manager


def test_disabled_exec_omits_all_exec_session_tools(tmp_path, admitted_sync):
    config = Config()
    config.tools.exec.enable = False

    def build(inputs):
        return AgentLoop.from_config(
            inputs.config, inputs.bus, provider=_provider(), model="test-model",
            context_window_tokens=4096, session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
        )

    loop = admitted_sync(tmp_path, build, config=config)

    for name in ("exec", "exec_session", "list_exec_sessions"):
        assert loop.tools.get(name) is None
    assert loop.tools.get("read_file") is not None
