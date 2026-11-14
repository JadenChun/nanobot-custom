"""Independent factory-level check of the accepted gateway-wide child cap."""
import asyncio

from nanobot.agent.child_sessions import ChildAdmission
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


async def test_two_loops_share_one_gateway_tree_and_capacity(engine_config):
    sessions = SessionManager(
        engine_config.workspace_path,
        sessions_root=engine_config.runtime_data_dir / "sessions",
    )
    loop_a = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
        session_manager=sessions,
    )
    loop_b = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
        session_manager=sessions,
    )
    try:
        tree_a = loop_a._fork_dependencies.foreground_manager._delegation_tree
        tree_b = loop_b._fork_dependencies.foreground_manager._delegation_tree
        assert tree_a is not None and tree_a is tree_b
        upgrade(loop_a, "cli:shared-a")
        upgrade(loop_b, "cli:shared-b")
        upgrade(loop_a, "cli:shared-c")
        parents = [
            (loop_a, "cli:shared-a"),
            (loop_b, "cli:shared-b"),
        ]
        reserved = []
        for index in range(4):
            loop, key = parents[index % 2]
            parent = loop.sessions.get_or_create(key)
            child = await tree_a.reserve(
                parent=parent,
                parent_key=key,
                assignment=f"hold shared gateway slot {index}",
                attempt_id=f"shared-slot-{index}",
            )
            assert isinstance(child, ChildAdmission)
            reserved.append(child)
        refused = await tree_a.reserve(
            parent=loop_a.sessions.get_or_create("cli:shared-c"),
            parent_key="cli:shared-c",
            assignment="refuse the fifth shared slot",
            attempt_id="shared-slot-refused",
        )
        assert not hasattr(refused, "child_id")
        assert refused.reason == "gateway_child_limit"
    finally:
        await loop_b.aclose()
        await loop_a.aclose()


async def test_three_roots_share_one_four_child_gateway_cap(engine_config):
    roots = ("GATEWAY_CAP_ALPHA", "GATEWAY_CAP_BETA", "GATEWAY_CAP_GAMMA")

    class CapacityProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([])
            self.root_calls = {root: 0 for root in roots}
            self.child_calls = 0
            self.four_started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            if "Worker Role" in str(messages):
                self.child_calls += 1
                if self.child_calls == 4:
                    self.four_started.set()
                await self.release.wait()
                return LLMResponse(content="The bounded child task is settled.")
            root = next(root for root in roots if root in str(messages))
            self.root_calls[root] += 1
            if self.root_calls[root] != 1:
                return LLMResponse(content="The original chat remains usable.")
            amount = 1 if root == roots[2] else 2
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id=f"cap-{root}-{index}", name="delegate_task", arguments={
                    "contract": f"Inspect the bounded {root} task {index} and report.",
                    "write_scope": [f"out-{root}-{index}.txt"], "background": True,
                },
            ) for index in range(amount)])

    provider = CapacityProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    keys = [f"cli:independent-cap-{index}" for index in range(3)]
    for key in keys:
        upgrade(loop, key)
    try:
        for index in range(2):
            result = await loop.execute_turn(TurnRequest(content=roots[index], session_key=keys[index]))
            assert result.error is None, result.error
        await asyncio.wait_for(provider.four_started.wait(), timeout=3)
        last = await loop.execute_turn(TurnRequest(content=roots[2], session_key=keys[2]))
        assert last.error is None, last.error
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        children = [child for key in keys for child in tree.children_of(key)]
        assert len(children) == 4, "a fifth child reservation must be refused across independent roots"
        held = [child for child in children if child.holds_capacity]
        assert len(held) == 4, "a fifth child reservation must be refused across independent roots"
        assert provider.child_calls == 4, "a refused child must never reach its provider"
        assert all(child.state == "active" for child in held)
        assert len(tree.children_of(keys[2])) == 0
        provider.release.set()
        for child in held:
            settled = await tree.wait_child(child.child_id, timeout=3)
            assert settled is not None and not settled.holds_capacity
    finally:
        provider.release.set()
        await loop.aclose()
