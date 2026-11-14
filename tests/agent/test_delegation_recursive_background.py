"""Retired recursive/background delegation regressions.

The retired inline delegation path injected ``ForegroundAgentManager.runner``
and nested roles through it. Normal subagents now run canonical AgentLoop turns
on source-backed child sessions, so these inline-path cases are obsolete.

Coverage now lives in the canonical suite:

- Recursive child/grandchild identities, depths, and depth-limit refusal:
  ``tests/agent/test_delegation_child_sessions.py::
  test_siblings_have_distinct_identities_and_grandchild_is_admitted`` and
  ``test_depth_and_capacity_refusals_leave_no_live_child``.
- Child history kept in the child source, not the parent:
  ``tests/agent/test_migration_source_engine.py::
  test_source_nested_planner_stream_does_not_leak_into_root_history``.
- Background child continuation with independent history:
  ``tests/agent/test_delegation_child_authority.py::
  test_background_child_retains_parent_task_authority_until_settled``.
- Repeated attempt returns the same admitted child:
  ``tests/agent/test_delegation_child_sessions.py::
  test_repeated_attempt_returns_same_child``.
"""
